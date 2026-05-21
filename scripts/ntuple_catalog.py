#!/usr/bin/env python3
"""
Sample catalog for L1DeepMET ntuple production.

Maintains data/sample_catalog/<campaign>.json, the single source of truth for
"which DAS dataset and which MINIAOD files belong to which named sample".
Once the catalog is built (via `refresh`, which needs grid access), all
downstream pipeline steps (ntuple_produce.py, preprocess.py) read it
without touching the grid.

Subcommands
───────────
  refresh   Query DAS for every sample in the recipe and overwrite the
            catalog. Requires a valid VOMS proxy. Skipped if --dry-run.
  list      Print all catalogued samples.
  show      Print a single sample's full record (DAS path + every input
            file URL + n_events).
  add-local Populate the catalog from already-staged ntuple files on
            /ceph (lets us record what we already have without grid).

Schema (data/sample_catalog/<campaign>.json)
────────────────────────────────────────────
  {
    "campaign": "Phase2Spring24",
    "refreshed_at": "2026-05-19T…",
    "samples": {
      "TT_PU200": {
        "dataset": "/TTTo…/Phase2Spring24DIGIRECOMiniAOD-PU200_…/GEN-SIM-DIGI-RAW-MINIAOD",
        "files":   ["/store/mc/…/file1.root", "/store/mc/…/file2.root", …],
        "n_events": 200000,
        "campaign": "Phase2Spring24"
      },
      …
    }
  }
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# NB: the repo's data/ is a symlink to /ceph — keep the catalog OUT of it so
# it tracks in git, not on a shared scratch mount.
CATALOG_DIR = REPO_ROOT / "sample_catalog"


# ─── Recipe — which samples we want in each campaign ─────────────────────────
# Tier-1 samples are required; tier-2 nice-to-have; tier-3 signal-tail.
# DAS query patterns are deliberately broad to catch the latest version
# (powheg-pythia8 vs amcatnloFXFX, etc.). The `refresh` subcommand picks the
# most recent matching dataset.
RECIPES: dict[str, dict[str, dict]] = {
    "Phase2Spring24": {
        "TT_PU200": {
            "tier": 1,
            "query": "/TTTo*_TuneCP5_14TeV*/Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 200_000,
            "doc": "Bulk MET training source — semileptonic ttbar gives broad real MET",
        },
        "VBFHToInvisible_PU200": {
            "tier": 1,
            "query": "/VBF_HToInvisible*_14TeV*/Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 80_000,
            "doc": "Clean genuine-MET signal — VBF Higgs → invisibles",
        },
        "MinBias_PU200": {
            "tier": 1,
            "query": "/MinBias_TuneCP5_14TeV*/Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 50_000,
            "doc": "Pure pileup — no genuine MET; establishes the false-positive rate floor",
        },
        "SingleNeutrino_PU200": {
            "tier": 1,
            "query": "/SingleNeutrino*/Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 30_000,
            "doc": "Zero genuine MET — defines model noise floor",
        },
        "WJetsToLNu_PU200": {
            "tier": 2,
            "query": "/WJetsToLNu_TuneCP5_14TeV*/Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 100_000,
            "doc": "W → ℓν: broad MET spectrum, both real signal + pileup contributions",
        },
        "DYToLL_PU200": {
            "tier": 2,
            "query": "/DYToLL_M-50_TuneCP5_14TeV*/Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 50_000,
            "doc": "Z + jets, no genuine MET — control sample for response calibration",
        },
        # Tier 3 SUSY signal — optional, big MET tail
        "SMS_T1tttt_PU200": {
            "tier": 3,
            "query": "/SMS-T1tttt_TuneCP5_14TeV*/Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 50_000,
            "doc": "SUSY high-MET tail — currently under-represented",
        },
    },
    # Phase2Spring23: matches the existing 25Jul8 production. Same sample
    # list, same targets; queries point at the 131X mcRun4 v5 / D95-detector
    # campaign which is what FastPUPPI 14_0_X's runInputs131X.py expects
    # (Geometry D95 + GlobalTag 131X_mcRun4_realistic_v9).
    "Phase2Spring23": {
        # Every sample has both noPU and PU200 variants in Phase2Spring23;
        # for MET reconstruction we ALWAYS want PU200, so every query is
        # gated on `*PU200*` in the campaign segment. Without this filter,
        # datasets[-1] picks alphabetically-last (noPU sorts after PU200
        # because lowercase 'n' > uppercase 'P').
        "TT_PU200": {
            "tier": 1,
            "query": "/TTTo*_TuneCP5_14TeV*/Phase2Spring23*PU200*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 200_000,
            "doc": "Bulk MET training source — semileptonic ttbar gives broad real MET",
        },
        "VBFHToInvisible_PU200": {
            "tier": 1,
            # Phase2Spring23 names it VBFHToInvisible (no underscore between
            # VBF and H), unlike Spring24 which uses VBF_HToInvisible.
            "query": "/VBFHToInvisible*_14TeV*/Phase2Spring23*PU200*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 80_000,
            "doc": "Clean genuine-MET signal — VBF Higgs → invisibles",
        },
        "MinBias_PU200": {
            "tier": 1,
            "query": "/MinBias_TuneCP5_14TeV*/Phase2Spring23*PU200*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 50_000,
            "doc": "Pure pileup — no genuine MET; establishes the false-positive rate floor",
        },
        "SingleNeutrino_PU200": {
            "tier": 1,
            "query": "/SingleNeutrino*/Phase2Spring23*PU200*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 30_000,
            "doc": "Zero genuine MET — defines model noise floor",
        },
        "WJetsToLNu_PU200": {
            "tier": 2,
            "query": "/WJetsToLNu_TuneCP5_14TeV*/Phase2Spring23*PU200*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 100_000,
            "doc": "W → ℓν: broad MET spectrum, both real signal + pileup contributions",
        },
        "DYToLL_PU200": {
            "tier": 2,
            "query": "/DYToLL_M-50_TuneCP5_14TeV*/Phase2Spring23*PU200*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 50_000,
            "doc": "Z + jets, no genuine MET — control sample for response calibration",
        },
        "SMS_T1tttt_PU200": {
            "tier": 3,
            "query": "/SMS-T1tttt_TuneCP5_14TeV*/Phase2Spring23*PU200*/GEN-SIM-DIGI-RAW-MINIAOD",
            "target_events": 50_000,
            "doc": "SUSY high-MET tail — currently under-represented",
        },
    },
}


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _catalog_path(campaign: str) -> Path:
    return CATALOG_DIR / f"{campaign}.json"


def _load(campaign: str) -> dict:
    p = _catalog_path(campaign)
    if not p.exists():
        return {"campaign": campaign, "refreshed_at": None, "samples": {}}
    with p.open() as f:
        return json.load(f)


def _save(catalog: dict) -> None:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    p = _catalog_path(catalog["campaign"])
    with p.open("w") as f:
        json.dump(catalog, f, indent=2)
    print(f"wrote {p}", file=sys.stderr)


def _das_query(query: str) -> list[str]:
    """Run dasgoclient and return matching file paths."""
    cmd = ["dasgoclient", "--query", query]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    except FileNotFoundError:
        raise SystemExit("dasgoclient not in PATH — load CMSSW first (cmsenv).")
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"DAS query failed: {query}")
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def cmd_refresh(args) -> None:
    if args.campaign not in RECIPES:
        raise SystemExit(f"Unknown campaign {args.campaign!r}. Options: {list(RECIPES)}")
    recipe = RECIPES[args.campaign]
    catalog = {"campaign": args.campaign, "refreshed_at": _now_iso(), "samples": {}}

    # Sanity: voms proxy must exist (unless --dry-run)
    if not args.dry_run:
        try:
            subprocess.run(["voms-proxy-info", "-exists"], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise SystemExit(
                "No valid VOMS proxy. Run `voms-proxy-init -voms cms -rfc` first, "
                "or pass --dry-run to emit a skeleton catalog without querying DAS."
            )

    for sample, spec in recipe.items():
        print(f"[{sample}] querying {spec['query']!r}", flush=True)
        if args.dry_run:
            datasets = ["/dry-run-placeholder/GEN-SIM-DIGI-RAW-MINIAOD"]
            files = []
            n_events = 0
        else:
            datasets = _das_query(f"dataset={spec['query']}")
            if not datasets:
                print(f"  ⚠ no datasets matched — sample omitted", flush=True)
                continue
            # Use most recent (DAS returns sorted by site; we take last)
            dataset = datasets[-1]
            files = _das_query(f"file dataset={dataset}")
            # `dasgoclient --query "file dataset=X | grep file.nevents"` returns
            # one line per file containing JUST the integer event count (with
            # trailing whitespace). The earlier `line.split()[1]` parser
            # assumed "<file> <nevents>" and crashed with IndexError on the
            # bare-int form.
            n_events = sum(
                int(line.strip())
                for line in _das_query(f"file dataset={dataset} | grep file.nevents")
                if line.strip().isdigit()
            ) if files else 0
            datasets = [dataset]
            print(f"  → {len(files)} files, {n_events:,} events", flush=True)
        catalog["samples"][sample] = {
            "dataset": datasets[0],
            "files": files,
            "n_events": n_events,
            "tier": spec["tier"],
            "target_events": spec["target_events"],
            "doc": spec["doc"],
            "campaign": args.campaign,
        }

    _save(catalog)


def cmd_list(args) -> None:
    if args.campaign:
        campaigns = [args.campaign]
    else:
        campaigns = [p.stem for p in CATALOG_DIR.glob("*.json")] if CATALOG_DIR.exists() else []
        if not campaigns:
            print("(no catalogs — run `refresh` first)", file=sys.stderr)
            return
    for c in campaigns:
        catalog = _load(c)
        print(f"━━━ {c} (refreshed {catalog.get('refreshed_at') or 'never'}) ━━━")
        if not catalog["samples"]:
            print("  (no samples)")
            continue
        for name, rec in sorted(catalog["samples"].items()):
            tier = rec.get("tier", "?")
            n = rec.get("n_events", 0)
            target = rec.get("target_events", 0)
            print(f"  T{tier}  {name:<28}  {n:>10,} ev   target {target:>8,}   {rec.get('doc', '')[:50]}")


def cmd_show(args) -> None:
    catalog = _load(args.campaign)
    rec = catalog["samples"].get(args.sample)
    if rec is None:
        raise SystemExit(f"Sample {args.sample!r} not in {args.campaign} catalog.")
    print(json.dumps(rec, indent=2))


def cmd_add_local(args) -> None:
    """Populate catalog entry from pre-existing ntuples on /ceph."""
    catalog = _load(args.campaign)
    sample_dir = Path(args.path)
    if not sample_dir.exists():
        raise SystemExit(f"Path does not exist: {sample_dir}")
    # Find .root files recursively
    root_files = sorted(sample_dir.rglob("*.root"))
    if not root_files:
        raise SystemExit(f"No .root files under {sample_dir}")
    # Count events with uproot
    n_events = 0
    try:
        import uproot
        for f in root_files:
            with uproot.open(f) as h:
                if "Events" in h:
                    n_events += int(h["Events"].num_entries)
    except ImportError:
        print("(uproot not in env — n_events left at 0)", file=sys.stderr)
    catalog["samples"][args.sample] = {
        "dataset": f"<local: {sample_dir}>",
        "files": [str(f) for f in root_files],
        "n_events": n_events,
        "tier": 1,
        "target_events": n_events,
        "doc": f"Locally staged in {sample_dir}",
        "campaign": args.campaign,
    }
    if catalog.get("refreshed_at") is None:
        catalog["refreshed_at"] = _now_iso()
    _save(catalog)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("refresh", help="Query DAS for every sample in the recipe")
    sp.add_argument("--campaign", default="Phase2Spring24")
    sp.add_argument("--dry-run", action="store_true",
                    help="Emit skeleton catalog with empty file lists (no DAS / no proxy needed)")
    sp.set_defaults(func=cmd_refresh)

    sp = sub.add_parser("list", help="List all catalogued samples")
    sp.add_argument("--campaign", default=None)
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("show", help="Print one sample's full record")
    sp.add_argument("sample")
    sp.add_argument("--campaign", default="Phase2Spring24")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("add-local", help="Populate a catalog entry from local files")
    sp.add_argument("sample")
    sp.add_argument("path")
    sp.add_argument("--campaign", default="Phase2Spring24")
    sp.set_defaults(func=cmd_add_local)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
