# Preregistration: Dependency-Specific Counterfactual Replay (DCRA)

Committed before any DCRA code is written. No DCRA result, no S5/S6 data and no \(E^*\) estimate exists
at the time of this commit. The frozen method (`rasfr-frozen`, `f7fb885`) and its baseline results
(`rasfr-frozen-baseline`, plus vocabulary robustness `6c681a6`) are not changed by anything below.
Evidence available before this commit, and which motivated the design, is listed in Section 11.

## 1. Question

Given a corpus of four-voice polyphony and a fixed pattern vocabulary, estimate for each pattern how much
of its observed frequency is due to cross-voice pitch coordination, as opposed to the structure of each
voice taken alone and the shared temporal context.

## 2. Dependency classes (fixed definitions)

- **D1, within-voice structure**: the law of each voice's pitch sequence on its own (steps, scale-degree
  usage, register, cadential behaviour, longer-range contour).
- **D2, shared temporal/structural context**:
  \(D_2 = \{\text{tonal context},\ \text{metre},\ \text{phrase boundaries},\ \text{joint onset/sustain structure}\}\).
  The joint onset/sustain structure is the four-voice matrix of attacks, holds and rests per slice,
  including cross-voice onset synchrony. D2 contains only shared, exogenous context; no pitch information
  of any voice is part of D2.
- **D3, cross-voice pitch coordination**: conditional dependence among simultaneous voice pitches given D2,
  i.e. the difference between
  \(P(X_S, X_A, X_T, X_B \mid D_2)\) and \(\prod_v P(X_v \mid D_2)\).

DCRA preserves D1 and D2 and removes D3.

## 3. Method

### 3.1 Primary intervention: sequence-level independent voice reconstruction (b)

For every piece and every voice \(v\) independently:

1. D2 of the piece is held fixed: tonal context, metre, phrase boundaries (fermatas for chorales; known
   by construction for synthetic data), and the complete four-voice attack/hold/rest matrix. These are
   copied, never regenerated. Tonal context is exactly the key information already present in the frozen
   data representation (synthetic: C major; real data: the piece-level tonic and mode of
   `ras.estimate_tonic`, unchanged). No modulation or tonicization detection is added, and the tonal
   representation is not redefined for any benchmark or in response to any result.

   **Hard invariant**: for every piece, every benchmark (S1–S6 and real data) and every reconstruction, the
   joint four-voice attack/hold/rest matrix of the counterfactual is identical, slice by slice, to that of
   the original piece. The code asserts this and aborts on any violation; it is an invariant of the method,
   not an acceptance metric.
2. The pitch sequence of \(v\) is regenerated as a whole, left to right. A new pitch is drawn only at
   slices where \(v\) attacks; at held slices the previous generated pitch continues; rests stay rests.
3. Model \(P_v\): first pitch of each phrase drawn from \(v\)'s phrase-initial (degree, octave) distribution;
   every later attack draws a semitone step from
   \(P_v(\text{step} \mid \text{scale degree of current pitch},\ \text{metric class of the attack},\ \text{phrase-position class})\),
   with metric class \(\in\{\text{strong}, \text{weak}\}\) (beat strength \(\ge 0.5\)) and phrase-position class
   \(\in\{\text{interior}, \text{penultimate attack}, \text{final attack}\}\). Non-diatonic current pitches use
   their own chromatic degree as state.
4. Backoff: if a state has fewer than 20 observations, drop phrase-position class, then metric class, then
   use \(v\)'s unconditional step distribution. Steps leaving \(v\)'s observed range (min–max pitch of \(v\) in
   the fitted data) are rejected and redrawn.
5. \(P_v\) is fitted by counting on the analysed collection itself, separately for each voice and pooled
   over its pieces. Leakage boundary: fitting and generation for voice \(v\) may use only \(v\)'s own pitches,
   \(v\)'s own steps, \(v\)'s own attacks/holds/rests, and the piece's D2 context. They never read any other
   voice's pitch, any vertical interval, any parallel or relative motion, any multi-voice configuration,
   or any pattern, cell or atom under test. Locked statement for the paper:
   *The marginal voice model is fitted separately for each analyzed corpus using only within-voice
   observations and the prespecified D2 context. No cross-voice pitch, interval, pattern, or target-cell
   information is used in fitting. Therefore, fitting on the analyzed corpus is not used to learn the
   dependency being tested.* \(P_v\) is thus an empirical marginal model used to construct a within-sample
   counterfactual, not a train/test prediction model.
6. R = 32 reconstructions per piece. Seeds: synthetic `7000 + 1000·g + r_dataset`; real sets use the
   frozen replay seed of the set + 500.

The model structure above is final. It is not changed after seeing any result, including S6 (Section 8).

### 3.2 Independent validation intervention: phrase-aligned cross-voice recombination (a)

The soprano of each piece is kept. Alto, tenor and bass of each phrase are taken from phrases of other pieces
in the same collection with the same metre, the same length in quarter beats and the same anacrusis offset,
transposed to the target piece's tonic (same mode only), each from a different donor piece.
Candidates producing voice crossing at more than 10% of slices or leaving the voice's observed range are
rejected. Phrases with no admissible donor are marked unmatched. R = 32 recombinations, seed
`8000 + set seed`.

(a) does not define ground truth, takes no part in model selection or parameter choice, and is applied only to
real data. It requires phrase boundaries, so it is applied to the Bach chorale sets only; Palestrina is
analysed with (b) only and this is reported as a limitation.

### 3.3 Output

For each pattern with observed count \(n\), DCRA expectation \(E_{\rm cf}\) (mean over R reconstructions) and
frozen-replay expectation \(E_0\):

- \(L_{\rm total} = \log(n / E_0)\): the frozen RAS-FR evidence scale;
- \(L_D = \log(n / E_{\rm cf})\): **dependency-specific evidence, the only formal output of DCRA**;
- \(L_M = \log(E_{\rm cf} / E_0)\): diagnostic;
- \(L_{\rm total} = L_D + L_M\).

\(L_M\) is the residual discrepancy relative to the frozen replay baseline; it is not evidence against
dependency. No Type I / Type II label is produced. Counts use a +0.5 continuity correction in numerator
and denominator.

**Unit of analysis.** Each pattern \(c\) (cell or atom) receives one corpus-level value
\(L_D(c) = \log\big(n(c) / E_{\rm cf}(c)\big)\), where \(n(c)\) is summed over all pieces of the analysed set and
\(E_{\rm cf}(c) = \sum_{\text{pieces } p} \bar e_p(c)\), with \(\bar e_p(c)\) the mean count of \(c\) over the 32
reconstructions of piece \(p\). Pieces are the bootstrap clusters. This matches the cell-level architecture of
frozen RAS-FR.

**Two uncertainties, reported separately and never merged.**

- *Sampling uncertainty* \(CI_{\rm cluster}\): for each pattern, 1000 resamples of pieces with replacement
  (seed 2026); in each resample \(n(c)\) and \(E_{\rm cf}(c)\) are recomputed from the resampled pieces' observed
  counts and their per-piece reconstruction means \(\bar e_p(c)\), giving a corpus-level \(L_D(c)\); the 2.5% and
  97.5% percentiles form the interval.
- *Monte Carlo uncertainty* \(SE_{\rm MC}\): from the 32 corpus-level reconstruction totals
  \(E^{(r)}(c) = \sum_p e_p^{(r)}(c)\), \(SE_{\rm MC}(c) = \mathrm{sd}_r(E^{(r)}(c)) / \big(\sqrt{32}\,E_{\rm cf}(c)\big)\) on
  the log scale. The R = 32 variation is not treated as bootstrap variation.

A pattern is *testable* if \(E_{\rm cf}(c) \ge 5\) (corpus level); untestable patterns are counted and reported,
never dropped silently. A testable pattern whose \(CI_{\rm cluster}\) excludes 0 is called *nominally
significant*. This is a per-pattern nominal 95% cluster-bootstrap interval, not a family-wise (FWER) or FDR
guarantee, and is never described as such. Counts of nominally significant patterns on real data are
descriptive/exploratory. The calibration gate (Section 6, gate 1) measures the empirical rate of nominal
significance where the truth is \(L_D = 0\); it is a calibration criterion, not a multiplicity-controlled
hypothesis test.

Patterns: the atoms of `fri.book_predicates()` (frozen vocabulary). Relational atoms are those of the
vertical fields V, O, H, S, U, K, J; within-voice atoms (fields M, G, D, P, Q) are negative controls.

## 4. Ground truth for synthetic data

Truth comes from the generating mechanism, never from (a) or (b).

- **Qualitative truth**: which atoms involve a planted cross-voice constraint and its direction.
- **Quantitative truth** \(E^* = E_{\prod_v P_v}\): from at least **10,000 independently generated samples**
  per generator (fresh seeds `9000 + g`, disjoint from all analysed data), each voice's sequence law is
  represented by its pool of sequences; \(E^*\) per analysed piece is the mean pattern count over at least
  10,000 combinations whose four voices come from four different samples (D2 is identical across samples in
  S1–S4 and S6; S5 has D3 = 0 by construction, so its truth is \(L_D^* = 0\) without estimation).
  True dependency evidence: \(L_D^* = \log(n / E^*)\) with the same \(n\).
- The Monte Carlo standard error of \(E^*\) is reported separately and is not merged into the DCRA bootstrap
  interval.

Phrase-aligned recombination does not constitute an independent validation of the synthetic ground truth,
because it is itself a finite-sample estimator of the same marginal-product counterfactual \(E^*\). Its role
is therefore restricted to intervention-robustness analysis on real musical data. (This sentence goes into
the paper.)

## 5. Synthetic benchmarks

All six are rebuilt; the existing `synthetic_truth.py` generators S1–S4 are reused unchanged for S1–S4.
Each benchmark: 20 independent replicate data sets of 40 pieces, seed `5000 + 1000·g + r`.

| Benchmark | Truth | Role |
|---|---|---|
| S1 | four independent diatonic random walks; D3 = 0 | calibration |
| S2 | S1 + register, per-voice degree and phrase-final preferences; D3 = 0 | calibration (D1 preferences) |
| S3 | S2 + no parallel perfect fifths/octaves between any pair; S–B similar motion rejected w.p. 0.6; pairwise D3 | power (planted: A1) |
| S4 | S3 + an upper voice a third above the bass in every slice; no doubled leading tone; pairwise + higher-order D3 | power (planted: A1, A11, A12) |
| S5 | S2 pitch process (each voice independent given its own attacks); voice-specific rhythms on a shared metre: strong beats attacked by all voices, weak-beat attacks, eighth-note passing attacks and ties drawn per voice; per-voice degree preference on strong beats; D3 = 0 given D2 | calibration: D2 must not be read as D3 |
| S6 | D3 = 0; D1 contains structure outside \(P_v\): an arch contour per phrase (rise in the first half, fall in the second), gap-fill after leaps (step back after a leap > 4 semitones w.p. 0.7), and repetition of an earlier phrase's step sequence w.p. 0.3 | misspecification stress test |

In S5 every counterfactual copies the original piece's joint attack/hold/rest matrix exactly (hard
invariant, Section 3.1), so S5 tests whether D2 alignment is mistaken for D3, not rhythm reconstruction.

S5 and S6 generator code is written after this commit to exactly this description; its parameters are
fixed in the code before the first DCRA run and recorded in the output.

## 6. Acceptance criteria (gates)

All on relational testable atoms unless stated; rates pooled over the 20 replicates.

1. **Calibration** (S1, S2, S5): proportion of nominally significant relational atoms \(\le 10\%\) in each
   benchmark (nominal level 5%).
2. **Power** (S3, S4): each planted atom has \(L_D < 0\) and nominally significant in \(\ge 80\%\) of replicates,
   evaluated per atom (A1 in S3; A1, A11, A12 in S4).
3. **Quantitative accuracy** (S1–S4): median over relational testable atoms of
   \(|L_D - L_D^*| = |\log E_{\rm cf} - \log E^*| \le 0.1\) in each benchmark. Higher-order atoms of S4
   (fields U, K, J and atoms A11, A12) are also reported separately.
4. **Structure preservation** (every analysed set, synthetic and real; the attack/hold/rest matrix is
   covered by the hard invariant of Section 3.1, not by this gate):
   per-voice total variation distance between original and reconstructed step distribution and scale-degree
   distribution \(\le 0.05\);
   within-voice negative-control atoms significant in \(\le 10\%\) (fields M, G, D, P, Q).
5. **Intervention robustness** (Bach hold-outs A, B, C; matched phrases only): Spearman correlation of
   \(L_D\) between (b) and (a) over relational testable atoms \(\ge 0.7\) on each set; sign agreement
   \(\ge 90\%\) among atoms significant under either intervention; Bland–Altman mean difference and 95%
   limits reported. Coverage (share of phrases matched) reported; if below 50% on a set, that set's
   robustness result is reported as inconclusive rather than passed.
6. **Real-data validation** (Bach hold-outs A, B, C): A1 has \(L_D < 0\) and significant under both (a) and (b)
   on every set.

**Reported without a gate**: S6 false-significance rate of (b), and whether (a)/(b) disagreement on S6-like
structure flags those atoms (computed on S6 with (a) used purely as a diagnostic there); A11/A12 and all
other atoms on real data; Palestrina-1/2/3 under (b); frozen RAS-FR side by side in every table; atom and
field-balanced AUC of \(L_D\) as a ranking score versus the frozen method, PMI, rarity and hybrid;
the expert study (raters H, M, S only) as qualitative validation.

## 7. Stopping rule

If any of gates 1–4 fails at the thresholds above, DCRA is not included in the DMKD submission, and the
method is not patched and re-tested. If gates 1–4 pass and gate 5 or 6 fails, DCRA may be submitted, but
the failing result is reported in full in the main text as a result and a principal limitation; it may not
be removed, moved out of view, or repaired by tuning. No unfavourable result of a preregistered analysis is
omitted from the paper.

## 8. Allowed and forbidden after this commit

Allowed: implementing exactly Sections 3–6; bug fixes that make the code match this text (each documented
in a dated amendment with the reason, before rerunning); additional analyses labelled exploratory.

Forbidden: changing \(P_v\)'s state variables, backoff, R, thresholds, benchmarks or seeds after seeing any
DCRA result; using S6 (or any real-data result) to modify \(P_v\) and then re-running S1–S5; using (a) for
ground truth, model selection or tuning; adding cross-voice information of any kind to \(P_v\); redefining the
tonal representation or adding modulation detection; merging \(SE_{\rm MC}\) into \(CI_{\rm cluster}\); reporting any subset of replicates or atoms selectively; editing
the frozen method or its outputs.

## 9. Outputs

New code only under `score_learning/dcra/`; results under `runs/dcra/`; this file and amendments under
`papers/rasfr/`. Pushed to Gitee `origin` only.

## 10. Order of work

1. S5, S6 generators and \(E^*\) estimator; record \(E^*\) Monte Carlo error.
2. (b) and the \(L_D\) bootstrap; structure-preservation checks.
3. Gates 1–4 on S1–S6, single run.
4. (a); gates 5–6 on Bach; Palestrina under (b).
5. Report all outcomes, including failures.

## 11. Evidence available before this commit (motivation, not tuning)

From the read-only null-ladder diagnostic (`c87a424`) on the old S1–S4 (3 replicates) and Bach hold-outs A, C:
the frozen V-field replay draws steps at held slices, overstating co-motion (Bach hold-out A: 17,734 vs 7,827
co-moving voice pairs; A1 expectation 836 → 314 with an onset mask); adding scale-degree state brought S1–S3
expectations within ±0.09 (log) of cross-piece recombination, while register and previous-step state added
nothing; unit-level replays conditioned on the real previous slice leaked coordination in S4 (log gap 0.39–0.83
for A1, A11, A12). Rotation halves co-motion exposure. These results fixed the choice of a sequence-level,
onset- and degree-aware reconstruction; no DCRA result was available.
