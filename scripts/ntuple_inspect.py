#!/usr/bin/env python3
"""
Provenance reader for L1DeepMET ntuple production outputs.

Given a sample directory produced by ``scripts/ntuple_produce.py``, prints:

  - tag, sample name, CMSSW campaign
  - the patch-applied state at production time
  - upstream FastPUPPI git sha (so the recipe is reproducible)
  - n_files, n_events, total bytes, success/fail counts
  - the full branch list from the first ROOT file
  - timestamp + who ran it

Usage
─────
  scripts/ntuple_inspect.py /ceph/cms/store/user/dprimosc/l1deepmet/<tag>/<sample>/
  scripts/ntuple_inspect.py /ceph/.../<tag>/             # all samples under a tag

  scripts/ntuple_inspect.py --json /ceph/.../<tag>/<sample>/   # machine-readable

Designed to be ingestible by a downstream agent: pipe to jq, etc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _inspect_sample(sample_dir: Path) -> dict:
    """Return a dict summarising one <tag>/<sample>/ directory."""
    out: dict = {"path": str(sample_dir), "ok": False}
    prov_path = sample_dir / "provenance.json"
    state_path = sample_dir / "state.json"

    if not sample_dir.exists():
        out["error"] = "directory does not exist"
        return out

    if prov_path.exists():
        with prov_path.open() as f:
            out["provenance"] = json.load(f)
    else:
        out["provenance"] = None

    if state_path.exists():
        with state_path.open() as f:
            state = json.load(f)
        jobs = state.get("jobs", {})
        out["jobs_total"] = len(jobs)
        out["jobs_ok"] = sum(1 for j in jobs.values() if j.get("status") == "ok")
        out["jobs_failed"] = sum(1 for j in jobs.values() if j.get("status") == "failed")
        # Wall-time aggregate
        total_s = sum(j.get("wall_s", 0) for j in jobs.values())
        out["wall_time_total_s"] = total_s
    else:
        out["jobs_total"] = 0
        out["jobs_ok"] = 0
        out["jobs_failed"] = 0

    # Walk the output dir for ROOT files (under <sample>/FP/<campaign>/)
    root_files = sorted(sample_dir.rglob("perfNano_*.root"))
    out["n_root_files"] = len(root_files)
    out["total_bytes"] = sum(f.stat().st_size for f in root_files) if root_files else 0
    if root_files:
        out["first_root"] = str(root_files[0])
        # Cheap event count from one file
        try:
            import uproot
            with uproot.open(root_files[0]) as h:
                if "Events" in h:
                    out["events_in_first_file"] = int(h["Events"].num_entries)
                    out["branches_in_first_file"] = sorted(h["Events"].keys())
        except Exception as e:
            out["branch_inspect_error"] = repr(e)

    out["ok"] = True
    return out


def _print_human(rec: dict) -> None:
    p = rec.get("path", "?")
    print(f"━━━ {p} ━━━")
    if not rec.get("ok"):
        print(f"  ✗ {rec.get('error', 'unknown error')}")
        return
    prov = rec.get("provenance") or {}
    print(f"  tag:                 {prov.get('tag', '<none>')}")
    print(f"  sample:              {prov.get('sample', '<none>')}")
    print(f"  campaign:            {prov.get('campaign', '<none>')}")
    print(f"  dataset:             {prov.get('dataset', '<none>')}")
    print(f"  produced_at:         {prov.get('produced_at', '<none>')}")
    print(f"  produced_by:         {prov.get('produced_by', '<none>')}")
    print(f"  cmssw_version:       {prov.get('cmssw_version', '<none>')}")
    print(f"  fastpuppi_sha:       {prov.get('fastpuppi_sha', '<none>')}")
    print(f"  repo_sha:            {prov.get('repo_sha', '<none>')}")
    print(f"  patch_applied:       {prov.get('patch_applied', '<unknown>')}")
    print(f"  extended_branches:   {prov.get('extended_branches_enabled', '<unknown>')}")
    print()
    print(f"  jobs:                {rec.get('jobs_ok', 0)}/{rec.get('jobs_total', 0)} ok, {rec.get('jobs_failed', 0)} failed")
    print(f"  wall time total:     {rec.get('wall_time_total_s', 0):.0f}s")
    print(f"  ROOT files:          {rec.get('n_root_files', 0)}")
    print(f"  total bytes:         {rec.get('total_bytes', 0) / 1e9:.2f} GB")
    if "events_in_first_file" in rec:
        print(f"  ev in first file:    {rec['events_in_first_file']:,}")
    if "branches_in_first_file" in rec:
        bs = rec["branches_in_first_file"]
        print(f"  branches in first:   {len(bs)} branches")
        # Show a sample of the more-interesting (extended) ones
        of_interest = [b for b in bs if any(k in b for k in
                       ("L1PuppiCands_", "L1Layer2Met", "ctl2MET", "L1Vertex"))]
        if of_interest:
            print(f"  L1Puppi/Layer2/Vtx:  {of_interest[:30]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="+",
                    help="One or more <tag>/, <tag>/<sample>/, or any sample dir")
    ap.add_argument("--json", action="store_true",
                    help="Print machine-readable JSON instead of human format")
    args = ap.parse_args()

    records = []
    for p_str in args.path:
        p = Path(p_str).resolve()
        # If it looks like a <tag>/ dir (has subdirs named like samples),
        # inspect each subdir.
        if (p / "provenance.json").exists():
            records.append(_inspect_sample(p))
        elif p.is_dir():
            # Try treating it as a tag root
            subdirs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if subdirs:
                for s in subdirs:
                    if (s / "provenance.json").exists() or (s / "state.json").exists():
                        records.append(_inspect_sample(s))
                    else:
                        # Maybe it's a sample dir without provenance yet
                        records.append(_inspect_sample(s))
            else:
                records.append({"path": str(p), "ok": False, "error": "empty directory"})
        else:
            records.append({"path": str(p), "ok": False, "error": "not a directory"})

    if args.json:
        json.dump(records, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        for r in records:
            _print_human(r)
            print()


if __name__ == "__main__":
    main()
