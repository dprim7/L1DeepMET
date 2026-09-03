"""Extended-v1 preprocessing contract: 38-slot layout ↔ loader ↔ cleaning.

The 26Sep2 extended-v1 production added 12 per-candidate columns (see
reports/ntuple_extended_v1/CAMPAIGN.md): the four deployment-legal
hardware-word fields (hwZ0, hwDxy, hwTkQuality, hwEmID) and eight
privileged cluster/track features. This file pins:

- params.yaml ``feature_layout_extended`` grows 26 → 38, first 26 slots
  unchanged (existing H5 consumers keep their indices);
- every non-derived layout entry has a module-level branch mapping with
  the correct pad value (pads must match the recipe sentinel so cleaning
  treats padded and real-sentinel cells identically);
- cleaning policy from the 2026-09-02 audit of the 2k batch: clEta/clPhi
  carry -999 on no-cluster candidates (→ 0, as caloEta/caloPhi), clHoE
  has a long physical tail (p99.9 = 41, max = 688 measured) that is
  capped at 50 preserving hadronic-ness ordering; everything else is
  clean at source and stays raw;
- the loader fills the new slots from a real ROOT tree and pads missing
  branches with the declared pad.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

V1_NEW = ["hwZ0", "hwDxy", "hwTkQuality", "hwEmID",
          "clEta", "clPhi", "clSigmaRR", "clAbsZBary", "clHoE", "clPtError",
          "trackPtError", "trackHitPattern"]

V0_26 = [
    "pt", "eta", "phi", "puppi_weight", "px", "py", "encoded_pdgId",
    "encoded_charge", "dxy", "z0", "hwPt", "hwEta", "hwPhi", "hwPuppiWeight",
    "hwQual", "trackChi2RPhi", "trackChi2RZ", "trackChi2Bend", "trackNStubs",
    "trackMvaQual", "caloEta", "caloPhi", "clPuId", "clEmId", "clPt", "clEmEt",
]


def _layout() -> list[str]:
    with (REPO_ROOT / "params.yaml").open() as fh:
        return yaml.safe_load(fh)["preprocess"]["feature_layout_extended"]


class TestLayout:
    def test_38_slots(self):
        assert len(_layout()) == 38

    def test_first_26_slots_unchanged(self):
        """Existing extended-26 H5 consumers index by position — the v0
        prefix must stay frozen."""
        assert _layout()[:26] == V0_26

    def test_v1_columns_appended_in_order(self):
        assert _layout()[26:] == V1_NEW


class TestEventLayout:
    V0_13 = ["puppi_met_pt", "puppi_met_phi", "puppi_met_central_pt",
             "puppi_met_central_phi", "pf_met_pt", "pf_met_phi",
             "calo_met_pt", "calo_met_phi", "tk_met_pt", "tk_met_phi",
             "lead_vtx_z0", "lead_vtx_sumpt", "n_vtx"]

    def _event_layout(self) -> list[str]:
        with (REPO_ROOT / "params.yaml").open() as fh:
            return yaml.safe_load(fh)["preprocess"]["event_feature_layout"]

    def test_first_13_slots_unchanged(self):
        """event_features consumers index by position — prefix frozen."""
        assert self._event_layout()[:13] == self.V0_13

    def test_layer2_met_appended(self):
        """L1Layer2Met exists from the 26Sep2 production onward (hand-rolled
        monitorPerf — addCTL2Met never existed on 14_2_X). Appended at the
        END so existing indices survive."""
        layout = self._event_layout()
        assert layout[13:] == ["layer2_met_pt", "layer2_met_phi"]

    def test_layer2_met_mapped(self):
        from l1deepmet.data.preprocessing import EVENT_FEATURE_BRANCHES

        assert EVENT_FEATURE_BRANCHES["layer2_met_pt"] == ("L1Layer2Met_pt", "scalar")
        assert EVENT_FEATURE_BRANCHES["layer2_met_phi"] == ("L1Layer2Met_phi", "scalar")


class TestBranchMap:
    def test_every_non_derived_layout_entry_is_mapped(self):
        from l1deepmet.data.preprocessing import EXTENDED_DIRECT_BRANCHES

        derived = {"px", "py", "encoded_pdgId", "encoded_charge", "hcal_depth"}
        missing = [f for f in _layout()
                   if f not in derived and f not in EXTENDED_DIRECT_BRANCHES]
        assert not missing, f"layout entries without a branch mapping: {missing}"

    def test_v1_pad_values_match_recipe_sentinels(self):
        """Padded slots must be indistinguishable from the recipe's own
        'absent' sentinel so downstream cleaning handles both at once."""
        from l1deepmet.data.preprocessing import EXTENDED_DIRECT_BRANCHES

        expected_pads = {
            "hwZ0": 0.0, "hwDxy": 0.0, "hwTkQuality": 0.0, "hwEmID": 0.0,
            "clEta": -999.0, "clPhi": -999.0,
            "clSigmaRR": -1.0, "clAbsZBary": -1.0, "clHoE": -1.0,
            "clPtError": -1.0, "trackPtError": -1.0, "trackHitPattern": -1.0,
        }
        for name, pad in expected_pads.items():
            branch, actual_pad = EXTENDED_DIRECT_BRANCHES[name]
            assert branch == f"L1PuppiCands_{name}"
            assert actual_pad == pad, f"{name}: pad {actual_pad} != {pad}"


class TestCleaningV1:
    def _mini(self):
        layout = ["pt", "clEta", "clPhi", "clHoE", "clAbsZBary"]
        X = np.zeros((1, 6, len(layout)), dtype=np.float32)
        X[..., 0] = 3.0
        return layout, X

    def test_cleta_clphi_minus999_neutralized_to_zero(self):
        from l1deepmet.data.preprocessing import clean_extended_sentinels

        layout, X = self._mini()
        X[0, 0, 1] = -999.0
        X[0, 1, 1] = 2.1      # real HGCal eta — must survive
        X[0, 0, 2] = -999.0
        X[0, 1, 2] = -3.1
        Y, _ = clean_extended_sentinels(X, layout)
        assert Y[0, 0, 1] == 0.0 and Y[0, 0, 2] == 0.0
        assert Y[0, 1, 1] == np.float32(2.1) and Y[0, 1, 2] == np.float32(-3.1)

    def test_clhoe_tail_capped_at_50(self):
        """Audit (2k batch, 5 samples): clHoE p99.9 = 41, max = 688. Values
        beyond 50 are capped TO 50 (not sentinel'd) — 'very hadronic' keeps
        its ordering."""
        from l1deepmet.data.preprocessing import clean_extended_sentinels

        layout, X = self._mini()
        X[0, 0, 3] = 688.0
        X[0, 1, 3] = 51.0
        X[0, 2, 3] = 49.5     # below cap — survives
        X[0, 3, 3] = -1.0     # no-cluster / no-HCal sentinel — survives
        X[0, 4, 3] = 1.7
        Y, _ = clean_extended_sentinels(X, layout)
        assert Y[0, 0, 3] == 50.0 and Y[0, 1, 3] == 50.0
        assert Y[0, 2, 3] == np.float32(49.5)
        assert Y[0, 3, 3] == -1.0
        assert Y[0, 4, 3] == np.float32(1.7)

    def test_clabszbary_zero_is_preserved(self):
        """clAbsZBary == 0 means BARREL cluster (only HGCal fills a z
        barycenter) — it is a region flag and must never be cleaned."""
        from l1deepmet.data.preprocessing import clean_extended_sentinels

        layout, X = self._mini()
        X[0, 0, 4] = 0.0
        X[0, 1, 4] = 344.0
        X[0, 2, 4] = -1.0
        Y, _ = clean_extended_sentinels(X, layout)
        assert Y[0, 0, 4] == 0.0
        assert Y[0, 1, 4] == np.float32(344.0)
        assert Y[0, 2, 4] == -1.0

    def test_spec_covers_v1_problem_features(self):
        from l1deepmet.data.preprocessing import EXTENDED_CLEANING_SPEC

        assert EXTENDED_CLEANING_SPEC["clEta"] == ((-999.0,), None, 0.0)
        assert EXTENDED_CLEANING_SPEC["clPhi"] == ((-999.0,), None, 0.0)
        assert EXTENDED_CLEANING_SPEC["clHoE"] == ((), 50.0, 50.0)


class TestLoaderEndToEnd:
    @pytest.fixture()
    def sample_dir(self, tmp_path):
        uproot = pytest.importorskip("uproot")
        import awkward as ak

        d = tmp_path / "SampleA"
        d.mkdir()
        cands = {
            "pt": [[10.0, 3.0, 1.5], [7.0]],
            "phi": [[0.0, 1.0, -1.0], [0.5]],
            "eta": [[0.1, 2.0, -2.0], [1.0]],
            "puppiWeight": [[1.0, 1.0, 0.5], [1.0]],
            "pdgId": [[211.0, -211.0, 22.0], [22.0]],
            "charge": [[1.0, -1.0, 0.0], [0.0]],
            "z0": [[0.5, -1.2, 0.0], [0.0]],
            "hwZ0": [[10.0, -24.0, 0.0], [0.0]],
            "hwTkQuality": [[7.0, 3.0, 0.0], [0.0]],
            "hwEmID": [[0.0, 0.0, 5.0], [2.0]],
            "clEta": [[-999.0, -999.0, 2.1], [1.8]],
            "clHoE": [[-1.0, -1.0, 600.0], [1.5]],
            "trackHitPattern": [[100.0, 90.0, -1.0], [-1.0]],
        }
        with uproot.recreate(d / "mini.root") as f:
            f["Events"] = {
                **{f"L1PuppiCands_{k}": ak.Array(v) for k, v in cands.items()},
                "genMet_pt": np.array([50.0, 20.0]),
                "genMet_phi": np.array([0.0, np.pi / 2]),
            }
        return tmp_path

    def test_loader_fills_v1_slots_and_pads(self, sample_dir):
        from l1deepmet.data.preprocessing import load_samples_to_numpy_extended

        layout = _layout()
        idx = {n: i for i, n in enumerate(layout)}
        enc = {
            "L1PuppiCands_pdgId": {211.0: 1, -211.0: 1, 22.0: 3, -999.0: 0},
            "L1PuppiCands_charge": {1.0: 3, -1.0: 1, 0.0: 2, -999.0: 0},
        }
        out = load_samples_to_numpy_extended(
            sample_dir, ["SampleA"], layout,
            var_list_mc=["genMet_pt", "genMet_phi"],
            event_feature_layout=None, max_pf=8, encoding=enc,
        )
        X, EX, Y = out["SampleA"]
        assert X.shape == (2, 8, 38)
        assert EX.shape == (2, 0)
        # v1 slots carry the written values
        assert X[0, 0, idx["hwZ0"]] == 10.0
        assert X[0, 1, idx["hwTkQuality"]] == 3.0
        assert X[0, 2, idx["hwEmID"]] == 5.0
        assert X[0, 2, idx["clHoE"]] == 600.0          # raw here; capped by cleaning
        assert X[0, 0, idx["clEta"]] == -999.0
        # missing branches fall back to the declared pad everywhere
        assert np.all(X[..., idx["clPtError"]] == -1.0)
        assert np.all(X[..., idx["hwDxy"]] == 0.0)
        # padded candidate slots get the pad value too
        assert X[1, 1, idx["trackHitPattern"]] == -1.0
        assert X[1, 1, idx["clEta"]] == -999.0
        # targets: gen MET px/py
        np.testing.assert_allclose(Y[0], [50.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(Y[1], [0.0, 20.0], atol=1e-5)
