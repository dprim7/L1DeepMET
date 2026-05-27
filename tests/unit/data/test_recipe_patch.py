"""Tests for the FastPUPPI ntuple-recipe patch contents.

These are *content* tests on the patch files under ``patches/`` and on
``params.yaml::preprocess.feature_layout_extended``. They guard the contract:

- Every per-candidate feature listed in ``feature_layout_extended`` must
  have a corresponding accessor in ``patches/runPerformanceNTuple.patch``'s
  saveCands moreVariables block, OR be a derived feature added by the
  preprocessor (``px``, ``py``, ``encoded_pdgId``, ``encoded_charge``).
- ``dxyErr`` MUST NOT appear: it is uniformly zero even in the legacy
  25Jul8 ntuples because CMSSW's ``l1t::PFCandidate`` exposes
  ``float dxy()`` but has no ``dxyError()`` accessor. The feature carries
  no information; the actual physics observable is ``dxy``. We document
  this finding so future work doesn't re-add ``dxyErr`` chasing a
  non-existent uncertainty branch. (See
  ``reports/dxy_investigation/REPORT.md``.)

The integration test that actually runs ``cmsRun`` lives separately
under ``tests/integration/test_recipe_smoke.py`` because it needs the
CMSSW environment.
"""
from __future__ import annotations
from pathlib import Path

import re
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = REPO_ROOT / "patches" / "runPerformanceNTuple.patch"
PARAMS_PATH = REPO_ROOT / "params.yaml"

# Features that are computed by the preprocessor (in
# load_samples_to_numpy_extended or preprocess_data) rather than read
# directly from a ROOT branch. They must NOT appear in saveCands.
DERIVED_FEATURES = {"px", "py", "encoded_pdgId", "encoded_charge"}


def _patch_text() -> str:
    return PATCH_PATH.read_text()


def _saveCands_block() -> str:
    """Extract the `moreVariables = cms.PSet(...)` block from saveCands.

    The patch file is unified-diff format with leading `+`/` ` markers on
    every line. We strip those markers before parsing so we can read it
    as real Python. We then walk balanced parens from
    ``moreVariables = cms.PSet(`` to find the matching ``)`` — a regex
    wouldn't work because every ``cms.string("...")`` contains an
    embedded paren pair.
    """
    txt = _patch_text()
    # Strip the unified-diff leading `+`/` ` from each line. Lines
    # starting with `-` (deletions) and `@@` (hunk headers) are skipped —
    # we only care about the post-patch content.
    kept = []
    for line in txt.splitlines():
        if not line:
            kept.append(line)
            continue
        c = line[0]
        if c in ("+", " "):
            kept.append(line[1:])
        # silently drop "-", "@@", diff metadata, etc.
    stripped = "\n".join(kept)
    # Find the moreVariables PSet inside saveCands (not saveGenCands).
    sc = stripped.find("def saveCands():")
    assert sc != -1, "could not find `def saveCands():` in patch"
    mv = stripped.find("moreVariables = cms.PSet(", sc)
    assert mv != -1, "could not find moreVariables block in saveCands"
    # Walk balanced parens from the opening `(` of cms.PSet.
    open_paren = stripped.find("(", mv)
    depth = 0
    end = None
    for i in range(open_paren, len(stripped)):
        if stripped[i] == "(":
            depth += 1
        elif stripped[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end is not None, "unbalanced parens in saveCands moreVariables"
    return stripped[open_paren + 1 : end]


def _feature_layout() -> list[str]:
    with PARAMS_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    return cfg["preprocess"]["feature_layout_extended"]


# ─── feature_layout invariants ─────────────────────────────────────────────


def test_feature_layout_does_not_contain_dxyErr():
    """dxyErr is always zero in L1 ntuples — there is no dxyError accessor
    on l1t::PFCandidate or l1t::PFTrack. Verified empirically in
    /ceph/.../25Jul8_140X_v0 (all values 0.0 across 50 events × ~2700
    real candidates) and architecturally in CMSSW_14_2_0_pre2 headers.
    """
    layout = _feature_layout()
    assert "dxyErr" not in layout, (
        "dxyErr is uniformly zero in L1 ntuples (no dxyError accessor "
        "exists on l1t::PFCandidate). Use `dxy` for the impact-parameter "
        "value instead. See reports/dxy_investigation/REPORT.md."
    )


def test_feature_layout_contains_dxy():
    """The actual physics observable is `dxy` (float, set by L1 PF from
    the track POCA computation). This is what carries information about
    secondary vertices / non-prompt tracks."""
    assert "dxy" in _feature_layout(), \
        "feature_layout_extended must include `dxy` (the IP value)"


# ─── patch / layout consistency ────────────────────────────────────────────


def test_all_layout_features_have_savecands_accessor_or_are_derived():
    """Every per-candidate feature must either:
       (a) be added by the preprocessor (DERIVED_FEATURES), or
       (b) appear as a key in the patched saveCands moreVariables.

    The 4 base kinematic columns (pt, eta, phi, mass) are added
    unconditionally by L1PFCandTableProducer itself.
    """
    layout = _feature_layout()
    block = _saveCands_block()
    # Built-in columns of L1PFCandTableProducer (added unconditionally)
    builtin = {"pt", "eta", "phi", "mass", "puppi_weight"}
    # extract all `name = cms.string(...)` keys from the patch block
    keys = set(re.findall(r"^\s*(\w+)\s*=\s*cms\.string\(", block, re.M))
    missing = [
        f for f in layout
        if f not in keys
        and f not in DERIVED_FEATURES
        and f not in builtin
    ]
    assert not missing, (
        f"these feature_layout entries have no saveCands accessor: {missing}. "
        f"Either add `<name> = cms.string(\"<accessor>\")` to the patch, "
        f"or remove them from feature_layout_extended."
    )


def test_savecands_includes_dxy_accessor():
    """The patch MUST add a `dxy = cms.string("dxy")` accessor so the
    real impact-parameter value is emitted in the ntuple."""
    block = _saveCands_block()
    # accept either `dxy = cms.string("dxy")` or a guarded variant
    assert re.search(r'\bdxy\s*=\s*cms\.string\(', block), \
        "saveCands moreVariables must declare a `dxy` accessor"


def test_savecands_does_not_declare_dxyErr():
    """The patch must not declare a dxyErr accessor — there is none to
    call. Doing so would either (a) crash StringObjectFunction at parse
    time, or (b) silently emit zeros, polluting downstream features.
    """
    block = _saveCands_block()
    assert not re.search(r'\bdxyErr\s*=\s*cms\.string\(', block), \
        "saveCands must NOT declare dxyErr — no such accessor exists"
