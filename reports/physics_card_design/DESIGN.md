# L1 physics card — open design questions

The CLAUDE.md "Evaluation metrics for architecture / recipe sweeps" standing
order specifies the full L1 card we expect from any architecture sweep. The
metric primitives that don't exist yet are stubbed in
``src/l1deepmet/metrics/physics.py`` (search for "STUBS — design pending")
and called by the scaffold orchestrator at ``scripts/evaluate_physics_card.py``.

This doc collects the open questions before any implementation lands. **Sign
off on each one before writing the unit tests** (tests in
``tests/unit/metrics/test_physics_card_stubs.py`` are `pytest.skip`'d until
the corresponding design line below is decided).

---

## Tier 1 — required for every sweep

### Q1.6 — `compute_rate_vs_threshold(reco_pt_minbias, …)`

The math is currently inline in ``src/l1deepmet/plotting.py::plot_trigger_rates``.
Extracting it lets ``compute_working_point`` reuse it without going through
matplotlib. Open:

- **Rate conversion factor**: how do "events passing threshold" become
  "rate in kHz"?
  - HL-LHC bunch crossing rate ~40 MHz nominal.
  - Phase2Spring24 sample is in-time PU200 → represents ~1 bunch crossing
    per event (no obvious pre-scale).
  - Likely factor: ``rate_khz = pass_fraction × 40_000 kHz / safety_factor``
    where safety_factor accounts for empty bunches / abort gap.
  - **Decision**: what value? Read off a CMS L1 TDR plot, or just make it
    a required CLI arg and let the user supply.
- **Threshold spacing**: ``np.arange(0, 501, 1)`` (matches
  ``compute_roc_curve``) or finer (0.5 GeV) near the working point?

### Q1.7 — `compute_working_point(rate_curve, turn_on_signal, target_rate_khz)`

The Tier-1 decision quantity. Open:

- **Target rate budget**: what's the L1 MET slice's kHz budget at HL-LHC?
  Phase-2 TDR usually cites 4–10 kHz for the MET trigger. **Pick a number**
  (or default to a list and emit a Pareto).
- **Multi-signal handling**: caller may want per-signal-sample efficiency
  at the same threshold T (VBF / TT semilep / SMS). Take a dict of
  ``{sample_name: turn_on_curve}`` or call N times?
- **Interpolation**: the rate curve drops fast at the working point.
  Log-linear in rate is probably right; straight linear can miss the
  threshold by a few GeV.

---

## Tier 2 — strongly recommended (often)

### Q2.10 — `compute_asymmetric_tails(gen_xy, reco_xy, fake_delta_gev, miss_delta_gev, pt_bins)`

Fakes (reco ≫ true; eats rate) vs misses (reco ≪ true; kills efficiency).

- **Fixed Δ_gev vs scaled** (e.g. ``Δ = max(20, 0.2 × gen_pt)``)?
  Fixed is simpler and easier to interpret in absolute terms. Scaled
  tracks fractional resolution at high pT but couples two effects.
- **Also report median tail distance** (useful when ``frac == 0`` but
  shapes differ)?
- **Per-bin sparsity guard** like ``compute_resolution_metrics`` —
  current test set has < 10 events in 200+ bins; need a min-N policy.

### Q2.9 — `compute_per_pu_card(gen_xy, reco_xy, n_vertex, pu_bin_edges)`

- **PU bin edges**: HL-LHC nominal PU 140–200, but reconstructed
  ``nL1Vtx`` from the patched recipe is typically 0–30 (it's emulator
  vertex count, not true PU). Calibrate from the actual nL1Vtx
  distribution? Quartiles by event count?
- **Per-bin AUC sparsity**: each bin needs both signal-class and
  background-class events; at low N this can be empty. Skip or return
  NaN with a count?
- **Per-bin working-point efficiency** — probably yes, but depends on
  Q1.7 being settled first.

### Q2.11 — `compute_per_eta_card(gen_xy, reco_xy, per_event_eta_summary, eta_split)`

**The big one — what does "per-event eta" mean for an algorithm whose
output is a 2-vector with no eta?** Options:

- (a) eta of the LEADING-pT candidate in the event.
- (b) eta of the highest ``|w_i × p_i|`` candidate (model-dependent —
      tells you "where the model's MET is coming from").
- (c) flag events as "barrel-MET" vs "endcap-MET" based on which η
      region dominates ``|Σ_{i in region} w_i × p_i|``.
- (d) skip event-level; compute per-CANDIDATE residuals on contribution
      to MET, partition by candidate eta. Requires per-candidate ground
      truth we don't have.

This summary stat would need to be computed UPSTREAM (preprocess.py)
or in the eval loop. Either way it adds a column to the H5.

### Q2.12 — `compute_puppi_ablation(eval_fn, features, gen_xy, puppi_slot_index)`

- **`eval_fn` signature**: ``features → predicted MET in GeV``.
  Caller passes a closure handling model, normalisation, input split.
  Reasonable but commits the eval API.
- **Slot vs name lookup**: hardcoded slot index (current default = 3,
  matches ``params.yaml::feature_layout_extended``) vs feature-name
  lookup against H5 ``feature_layout`` attr.
- **Random-feature variant**: also have ``compute_random_ablation`` that
  sets ``puppi_weight`` to U[0,1] random? Tests robustness vs reliance.

---

## Tier 3 — ship decisions only (out of scope of this doc)

Quantization gap and HLS4ML resource/latency tracking are noted in
CLAUDE.md but their design lives under
``src/l1deepmet/quantization/`` and ``src/l1deepmet/synthesis/``
(both currently stubs). Decide separately.

---

## Orchestration — `scripts/evaluate_physics_card.py`

The scaffold has 5 of its own open questions, all in its module docstring.
Most material:

- **Q-script-1**: model artifact format (`.keras` vs SavedModel vs
  checkpoint dir).
- **Q-script-3**: one-card-per-signal-sample (VBF / TT / SMS / WJets / DYToLL)
  vs one combined card. Affects the JSON schema downstream consumers
  (training script, ablation drivers) will read.
- **Q-script-2**: feature-scaling stats: pulled from training-time JSON
  vs stored in the model artifact vs re-derived. Current
  ``reports/event_count_justification/scripts/train_one.py`` derives
  per-cell stats from the training subset and stashes them in the result
  JSON — that's reproducible but not portable. For evaluation we need
  the stats that were used at TRAINING time of this specific model.

---

## How to use this doc

When you're ready to design one of these:

1. Pick the open question.
2. Write the decision (1-line or 1-paragraph) on a fresh PR/branch.
3. Update CLAUDE.md if the standing order needs sharpening.
4. Replace the corresponding ``@pytest.skip(...)`` in
   ``tests/unit/metrics/test_physics_card_stubs.py`` with a real
   expected-behaviour test.
5. Implement in ``src/l1deepmet/metrics/physics.py`` (replace the
   ``raise NotImplementedError``).
6. Wire into ``scripts/evaluate_physics_card.py``.
7. Run a card on a model and inspect — does the JSON give you what
   you'd want to cite in a paper?
