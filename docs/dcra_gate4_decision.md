# DCRA decision record: criterion 4 failed, validation stopped

Date: 2026-10-08. Preregistration: `dcra-prereg` (`e1c1669`). Generators frozen at `dcra-gen-freeze`,
\(E^*\) at `dcra-estar-freeze`, intervention (b) at `e4012e5`. No \(L_D\) has been computed; gates 1–3, 5 and
6 were not run.

## Compliance audit (preregistered text vs implementation)

1. **Phrase-initial sampling.** Preregistration §3.1 item 3: "first pitch of each phrase drawn from \(v\)'s
   phrase-initial (degree, octave) distribution". Implemented as written. The cross-phrase discontinuity
   that drives the failure is a direct consequence of this preregistered model structure.
2. **Object of the step TVD.** Gate 4: "per-voice total variation distance between original and
   reconstructed step distribution". No restriction to phrase-internal steps is stated; D1 is defined (§2)
   as the law of each voice's pitch sequence including its steps. Implemented as all consecutive sounding
   notes of the voice (as in the frozen representation `ras.horiz`). Phrase-internal steps are reported as an
   auxiliary quantity only and play no part in the decision.
3. **Comparison object.** Gate 4: "between original and reconstructed", for "every analysed set".
   Implemented as each analysed set's original pieces vs the pooled distribution of all R = 32
   reconstructions of that set; no external reference corpus. Pooling over the 32 reconstructions is the
   least strict reading (per-reconstruction comparison would add sampling noise).
4. **Seeds.** Preregistered reconstruction seeds (synthetic `7000 + 1000·g + r`; real sets: frozen replay
   seed + 500), all 20 replicate data sets per benchmark.

Implementation details not spelled out in the preregistration, none of which can explain the failure:
scale degree is taken within the mode and the phrase-initial distribution is split by mode with fallback to
the pooled distribution (all synthetic data are in C major, so this has no effect there); phrase-initial
pitches are also restricted to the voice's observed range (no effect when the tonic is constant, as in all
synthetic data); if no in-range candidate exists at a backoff level, the next level is used (at least 99% of
synthetic attacks use the finest level).

## Result (`runs/dcra/gate4_compliance.json`)

| Set | Sets passing | Max step TVD (range over sets) | Max degree TVD (range) |
|---|---|---|---|
| S1 | 0 / 20 | 0.051–0.056 | 0.053–0.062 |
| S2 | 0 / 20 | 0.051–0.060 | 0.037–0.044 |
| S3 | 0 / 20 | 0.052–0.061 | 0.036–0.050 |
| S4 | 0 / 20 | 0.055–0.061 | 0.041–0.054 |
| S5 | 0 / 20 | 0.051–0.061 | 0.038–0.052 |
| S6 | 0 / 20 | 0.063–0.078 | 0.039–0.051 |
| Bach hold-out A (40) | fail | 0.079 | 0.054 |
| Bach hold-out B | fail | 0.079 | 0.046 |
| Bach hold-out C (179) | pass | 0.045 | 0.021 |

The step criterion fails on all 120 synthetic sets and on Bach hold-outs A and B.

## Decision

Criterion 4 = **FAIL**. Under the stopping rule (§7) DCRA is not included in the DMKD submission and is not
patched and re-tested. The threshold, the step definition, the model \(P_v\) and the generators are unchanged.

## Diagnostic findings (recorded, not used to modify anything)

- What passed: zero cross-voice leakage (interface, import and perturbation-invariance audits); exact
  preservation of D2 (joint attack/hold/rest matrix, metre and bars, phrase boundaries, tonic and mode) and
  of voice ranges in every reconstruction.
- Principal cause: redrawing each phrase's first pitch independently destroys within-voice continuity
  across phrase boundaries, a genuine loss of D1. Phrase-internal step TVD is about 0.03–0.04 on synthetic
  sets, versus 0.05–0.08 for all steps.
- Contributing factor: the finite-sample TVD between a 40-piece set and its generating distribution is about
  0.033–0.045, so 0.05 leaves little margin; consistent with this, the 179-piece hold-out C passes. This
  explains why the criterion is strict; it is not a ground for changing it.
- Consequence for identification: since \(E_{\rm cf}\) does not preserve D1, a nonzero \(L_D\) could not be
  attributed to D3 rather than to D1 misspecification, which is exactly what criterion 4 was designed to
  guard against.
