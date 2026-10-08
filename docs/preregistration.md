# Preregistration: evaluation of the frozen RAS-FR method on untouched test sets

Committed before `score_learning/induction/fresh_eval.py` is run on the data below.
No result on these sets has been computed or inspected.

## Why

In phase 1 of the study the cell vocabulary and calibration were designed while inspecting completeness
on hold-out A, hold-out B and the first 16 Palestrina works. Those sets are therefore not pristine with
respect to the representation. This evaluation uses pieces that were never an evaluation target at any
stage, so the cell vocabulary, nulls, calibration, metrics and baselines were all fixed without them.

## Frozen method

- Method code: frozen at evaluation commit `7776c40` (fields, nulls, residuals, calibration).
  Later commits changed evaluation code only (half-sampling, tie-robust precision@k, extra baseline).
- Training set and calibrated thresholds: read from `runs/induction/paper/final_eval.json`
  (same 16 training chorales, `z_cut`, `q`, `soft_z`); no recalibration.
- Replay seeds: hold-out C 2, Palestrina-2 3. Desynchronisation seed 99. Half-sampling seed 2026, 1000 draws.

## Test sets

- **Hold-out C**: every usable chorale (four complete voices) that is not in training, hold-out A,
  hold-out B or the development set, and whose corpus number is greater than 60
  (numbers 1–60 were loaded by pre-RAS-FR scripts and are excluded conservatively). n = 179.
- **Palestrina-2**: the next 40 four-voice Palestrina works in music21 corpus order after the 16 already
  used, with the same filter (`load_palestrina` criteria). n = 40, no overlap with the first 16.

## Compared nulls (identical cells and counts)

RAS-FR (voice-decomposition replay), independent bits (PMI), uniform (rarity), and productive-pattern
null (Webb 2007: a cell deviates only if its count lies outside the expectations of every binary
partition of its bits).

## Metrics (scope: fields excluding R, W, F)

Primary: mean atom AUC over atoms evaluable under all four nulls; specificity gap on vertical atoms
(real minus desynchronised). Secondary: mean AUC on prohibitions, precision@50/100 per polarity
(point values), per-atom AUC real and desynchronised, RAS-FR discovery-set completeness and lift.

## Hypotheses and decision rule

- H1 (hold-out C): RAS-FR mean atom AUC exceeds PMI, rarity and productive; supported if the 95%
  half-sampling interval of each difference lies above 0.
- H2 (hold-out C): RAS-FR specificity gap exceeds PMI, rarity and productive; same rule.
- H3 (Palestrina-2): same comparisons; exploratory, reported with intervals, no claim if they include 0.
- Expected from earlier sets and stated in advance: rarity will have the higher prohibition-only AUC.

All results are reported whether or not they support the hypotheses, and the manuscript will be revised
to match them.

## Amendment 1 (before any result on hold-out C or Palestrina-2 was inspected)

The first run of `fresh_eval.py` (commit `d2cbffa`) was stopped while still computing hold-out C; its
log contains no metric. The reason is a defect found by code inspection while building the expert
review pack, not by looking at any test-set result:

- In field D the real counts set the bit `ped` (the voice holds some pitch for at least four slices
  anywhere in the piece), but the null wrote `ped=0` for every replay. In the chorales `ped` is 1 for
  every voice, so real and null cells never overlapped: every real D cell looked over-represented and
  every null-only cell looked like an empty-cell prohibition.
- Reference atom D6 (`ped=1`, "pedal favoured") matched every real D cell. It left the other D atoms
  without negatives, which is why D never entered the common-atom AUC.

Changes:

- The null keeps the voice's real `ped` value (it is context, like the other voices' pitches).
- D6 is removed from the reference set: `ped` is a piece-level property of a voice, so no
  within-piece null can test it.

Everything downstream is recomputed with the corrected code: calibration, final evaluation on
hold-outs A/B and Palestrina, half-sampling intervals, baselines, ablations and sensitivity analyses.
The hypotheses, test sets, seeds, metrics and decision rule above are unchanged. The method is
re-frozen at the commit that contains this amendment.

## Amendment 2: confirmatory test of the simplified replay (revision, before Palestrina-3 is loaded)

Committed before `score_learning/induction/revision.py prereg3` is run. No piece of Palestrina-3 has
been parsed by any script of this project.

**Why.** The simplified replay (one pitch catalogue pooled over the four voices for fields H, M, S, G,
K, D, U; step catalogues of V and O unchanged; rule 7 dropped; recalibrated on desynchronised training
scores with the grid and target of Algorithm 2, giving `z_cut` 4.0, `q` 0.05, `z_soft` −2.0) was chosen
after its results on hold-outs A, B and Palestrina were seen, and was then evaluated on hold-out C and
Palestrina-2, whose frozen-method results were already known. A hybrid of the two simplest nulls
(rarity for prohibition atoms, PMI for preference atoms) was defined after all of these results were
seen. Neither has been tested on unseen data. No chorale is left that was not loaded before, so the
test set is a further sample of Palestrina.

**Frozen.** Method code at the commit containing this amendment; the 16 training chorales; the
simplified configuration above (calibrated on training scores only; recomputed by `calibrate_pooled`
and checked to equal the values above before evaluation); the frozen configuration in
`final_eval.json`. Replay seed 4, desynchronisation seed 99, half-sampling seed 2026, 1000 draws of
half the pieces.

**Test set: Palestrina-3.** The next 40 four-voice Palestrina works in music21 corpus order after the 56
already used (16 + Palestrina-2), with the `load_palestrina` filter (four parts, at least 8 slices).

**Compared.** Simplified replay; frozen RAS-FR; PMI; rarity; productive; hybrid. Same cells and counts;
metrics as above (fields excluding R, W, F; common atoms of all compared methods).

**Hypotheses** (simplified replay; decision rule: supported if the 95% half-sampling interval of every
difference in the hypothesis lies above 0):

- H4: mean atom AUC exceeds PMI, rarity and productive.
- H5: specificity gap exceeds PMI, rarity and productive.
- H6: specificity gap exceeds the hybrid.
- Reported without a directional claim: mean atom AUC against the hybrid and against the frozen replay
  (expected from the earlier sets: no clear difference to the hybrid).

The seven directional comparisons of H4–H6 form one family; one-sided half-sampling p-values are also
reported with Holm adjustment. Palestrina transfers a model trained on Bach chorales to another style;
earlier Palestrina samples gave weaker results than the chorales, and a failure here is a possible
outcome that will be reported as such.
