#!/usr/bin/env python3
"""
Local-worker ntuple production driver for L1DeepMET.

Given a named sample (from `sample_catalog/<campaign>.json`) and an
event budget, this script:

  1. Resolves the sample → list of MINIAOD input files via the catalog.
  2. For each input file, spawns a `cmsRun runPerformanceNTuple.py` child
     with `inputFiles=…&maxEvents=N` such that the total adds up to the
     requested budget (events-per-file = budget / n_files, rounded up).
  3. Runs up to `--workers` children in parallel (Python multiprocessing).
  4. Stages outputs to `/ceph/.../<tag>/<sample>/FP/<campaign_tag>/perfNano_<i>.root`
     and writes a provenance JSON at `<tag>/<sample>/provenance.json`.
  5. Maintains a per-sample state.json so the run is **resumable** — restart
     after a crash and only failed/missing children re-run.

This script is NOT a CRAB submitter. It assumes the MINIAOD inputs are
either locally on /ceph (catalog `add-local` populated them) or reachable
via XRootD with a valid VOMS proxy. HTCondor support is Phase 3.

Usage
─────
  scripts/ntuple_produce.py \\
      --sample TT_PU200 \\
      --campaign Phase2Spring24 \\
      --tag 26May19_150X_extended_v0 \\
      --n-events 50000 \\
      --output-root /ceph/cms/store/user/dprimosc/l1deepmet \\
      --workers 4 \\
      [--dry-run]                # print plan, don't launch cmsRun
      [--cmssw /path/to/CMSSW_15_1_0_pre4]   # default: read from .env

  scripts/ntuple_produce.py status --tag 26May19_150X_extended_v0 \\
      --sample TT_PU200

The cmsRun recipe used is the patched runPerformanceNTuple.py from
external/FastPUPPI (run `scripts/apply_ntuple_recipe.sh apply` once before).
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = REPO_ROOT / "sample_catalog"
SUBMODULE_PY = REPO_ROOT / "external" / "FastPUPPI" / "NtupleProducer" / "python"
RECIPE_PATH = SUBMODULE_PY / "runPerformanceNTuple.py"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _load_catalog(campaign: str) -> dict:
    p = CATALOG_DIR / f"{campaign}.json"
    if not p.exists():
        raise SystemExit(f"No catalog at {p} — run `ntuple_catalog.py refresh` first.")
    with p.open() as f:
        return json.load(f)


def _git_sha(path: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return "<unknown>"


def _patch_applied() -> bool:
    """True iff the L1DeepMET extension is currently applied to the submodule."""
    script = REPO_ROOT / "scripts" / "apply_ntuple_recipe.sh"
    if not script.exists():
        return False
    try:
        r = subprocess.run([str(script), "check"], capture_output=True, text=True, check=True)
        return r.stdout.strip() == "applied"
    except subprocess.CalledProcessError:
        return False


def _build_plan(catalog: dict, sample: str, budget: int) -> list[tuple[str, int]]:
    """Plan = list of (input_file, max_events_for_this_file) summing to `budget`."""
    rec = catalog["samples"].get(sample)
    if rec is None:
        raise SystemExit(f"{sample!r} not in catalog.")
    files = rec.get("files", [])
    if not files:
        raise SystemExit(
            f"{sample!r} has no files catalogued (placeholder entry). "
            f"Run `ntuple_catalog.py refresh` with grid access, or `add-local` if files exist."
        )
    n_files = len(files)
    per_file = max(1, (budget + n_files - 1) // n_files)  # ceil
    return [(f, per_file) for f in files]


def _state_path(tag_root: Path, sample: str) -> Path:
    return tag_root / sample / "state.json"


def _load_state(tag_root: Path, sample: str) -> dict:
    p = _state_path(tag_root, sample)
    if not p.exists():
        return {"jobs": {}}  # job_id -> {"input": ..., "status": "ok"|"failed"|"running", "log": ...}
    with p.open() as f:
        return json.load(f)


def _save_state(tag_root: Path, sample: str, state: dict) -> None:
    p = _state_path(tag_root, sample)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(p)


def _job_id(input_file: str, idx: int) -> str:
    base = Path(input_file).stem
    return f"{idx:04d}_{base}"


# XRootD redirector for /store/ LFNs that aren't on the local file system.
# Override with $L1DEEPMET_XROOTD_REDIRECTOR to use a closer one (e.g. UCSD
# T2's redirector.t2.ucsd.edu, or root://xcache-redirector.t2.ucsd.edu/ for
# the caching layer).
_DEFAULT_XROOTD_REDIRECTOR = "root://cmsxrootd.fnal.gov/"


def _input_url(input_file: str) -> str:
    """Build a cmsRun-friendly URL from a catalog entry.

    - ``/store/...`` paths are CMS LFNs — wrap with an XRootD redirector
      (the global US one by default; tunable via env var).
    - Anything else is assumed local and gets the ``file:`` prefix.
    """
    if input_file.startswith("/store/"):
        redirector = os.environ.get(
            "L1DEEPMET_XROOTD_REDIRECTOR", _DEFAULT_XROOTD_REDIRECTOR
        ).rstrip("/")
        return f"{redirector}/{input_file.lstrip('/')}"
    return f"file:{input_file}"


def _run_one(args_tuple) -> tuple[str, dict]:
    """Worker: cmsRun one input → one perfNano output. Returns (job_id, result)."""
    job_id, input_file, max_events, output_dir, recipe_path = args_tuple
    out_file = Path(output_dir) / f"perfNano_{job_id}.root"
    log_file = Path(output_dir) / f"perfNano_{job_id}.log"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # cmsRun command: pass inputFiles + maxEvents on the command line.
    cmd = [
        "cmsRun", str(recipe_path),
        f"inputFiles={_input_url(input_file)}",
        f"maxEvents={max_events}",
        f"outputFile={out_file}",
    ]
    env = os.environ.copy()
    env["L1DEEPMET_EXTENDED"] = "1"   # the recipe checks this
    t0 = datetime.datetime.now(datetime.timezone.utc)
    try:
        with open(log_file, "w") as logf:
            r = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                               check=False, timeout=3600 * 6)   # 6 hr per file
        ok = (r.returncode == 0) and out_file.exists()
    except subprocess.TimeoutExpired:
        ok = False
        r = None
    elapsed = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    return job_id, {
        "input": input_file,
        "output": str(out_file),
        "status": "ok" if ok else "failed",
        "log": str(log_file),
        "max_events": max_events,
        "wall_s": elapsed,
        "finished_at": _now_iso(),
    }


def cmd_run(args) -> None:
    catalog = _load_catalog(args.campaign)
    plan = _build_plan(catalog, args.sample, args.n_events)

    tag_root = Path(args.output_root) / args.tag
    out_dir = tag_root / args.sample / "FP" / catalog["campaign"]
    out_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(tag_root, args.sample)

    # Compose job list, skipping already-completed jobs (resumable)
    to_run = []
    for idx, (input_file, max_events) in enumerate(plan):
        jid = _job_id(input_file, idx)
        if state["jobs"].get(jid, {}).get("status") == "ok":
            continue
        to_run.append((jid, input_file, max_events, str(out_dir), str(RECIPE_PATH)))

    # Provenance
    provenance = {
        "tag": args.tag,
        "sample": args.sample,
        "campaign": catalog["campaign"],
        "dataset": catalog["samples"][args.sample]["dataset"],
        "n_events_target": args.n_events,
        "n_files": len(plan),
        "produced_at": _now_iso(),
        "produced_by": os.environ.get("USER", "<unknown>"),
        "cmssw_version": os.environ.get("CMSSW_VERSION", "<not in cmsenv>"),
        "fastpuppi_sha": _git_sha(REPO_ROOT / "external" / "FastPUPPI"),
        "repo_sha": _git_sha(REPO_ROOT),
        "patch_applied": _patch_applied(),
        "recipe": str(RECIPE_PATH),
        "extended_branches_enabled": True,
    }
    with (tag_root / args.sample / "provenance.json").open("w") as f:
        json.dump(provenance, f, indent=2)

    print(f"━━━ {args.sample} ({catalog['campaign']}) ━━━")
    print(f"  output:  {out_dir}")
    print(f"  plan:    {len(plan)} files × ≤{plan[0][1] if plan else 0} ev/file = {args.n_events} ev")
    print(f"  pending: {len(to_run)} jobs (skipping {len(plan) - len(to_run)} already done)")
    print(f"  workers: {args.workers}")
    print(f"  recipe:  {RECIPE_PATH}")
    print(f"  patch:   {'applied' if provenance['patch_applied'] else 'NOT APPLIED'}")
    print()

    if not provenance["patch_applied"] and not args.dry_run:
        print("⚠ WARNING: patch is NOT applied. The recipe will emit the upstream-default")
        print("  branches only — none of the L1DeepMET extensions will be in the output.")
        print("  Run: scripts/apply_ntuple_recipe.sh apply")
        print("  (Continue anyway, the script will not block.)")
    if args.dry_run:
        print("(dry-run — nothing launched)")
        return

    if not to_run:
        print("Nothing to do.")
        return

    # Run in parallel
    with mp.Pool(args.workers) as pool:
        for i, (jid, result) in enumerate(pool.imap_unordered(_run_one, to_run)):
            state["jobs"][jid] = result
            _save_state(tag_root, args.sample, state)
            mark = "✓" if result["status"] == "ok" else "✗"
            print(f"  [{i+1:3d}/{len(to_run)}] {mark} {jid}  ({result['wall_s']:.0f}s)")

    n_ok = sum(1 for j in state["jobs"].values() if j["status"] == "ok")
    n_fail = sum(1 for j in state["jobs"].values() if j["status"] == "failed")
    print(f"\nDone: {n_ok} ok, {n_fail} failed.")


def cmd_status(args) -> None:
    tag_root = Path(args.output_root) / args.tag
    if args.sample:
        samples = [args.sample]
    else:
        samples = [d.name for d in tag_root.iterdir() if d.is_dir()] if tag_root.exists() else []
    for s in samples:
        state = _load_state(tag_root, s)
        n_ok = sum(1 for j in state["jobs"].values() if j["status"] == "ok")
        n_fail = sum(1 for j in state["jobs"].values() if j["status"] == "failed")
        n_total = len(state["jobs"])
        print(f"  {s:<28}  {n_ok}/{n_total} ok, {n_fail} failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="Launch (or resume) ntuple production for one sample")
    p.add_argument("--sample", required=True)
    p.add_argument("--campaign", default="Phase2Spring24")
    p.add_argument("--tag", required=True, help="Output tag, e.g. 26May19_150X_extended_v0")
    p.add_argument("--n-events", type=int, required=True)
    p.add_argument("--output-root", default="/ceph/cms/store/user/dprimosc/l1deepmet")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="Inspect job state.json for a tag")
    p.add_argument("--tag", required=True)
    p.add_argument("--sample", default=None)
    p.add_argument("--output-root", default="/ceph/cms/store/user/dprimosc/l1deepmet")
    p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
