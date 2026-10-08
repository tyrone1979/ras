# RAS-FR

Code and result data for **“Dependency-specific null models for statistically sound pattern
discovery: a case study in four-voice polyphony.”**

RAS-FR is an empirical voice-decomposition replay null for selecting candidate voice-leading
constraints and auditing whether their evidence is sensitive to disrupting voice alignment.
It does not use a language model and does not establish causal attribution.

The learner sees scores only. `score_learning/induction/oracle.py` contains the author-written
reference constraints used for evaluation; training code is forbidden to import it, which is
checked by `tests/test_induction_paper.py`.

## Repository contents

- `score_learning/induction/`: frozen learner, baselines and revision analyses.
- `score_learning/diagnostics/`: read-only replay diagnostic.
- `score_learning/dcra/`: preregistered DCRA validation code. Gate 4 failed; no dependency-specific
  `L_D` was computed.
- `docs/`: preregistrations, expert protocol, gate-4 decision and provenance.
- `results/`: JSON files used for the manuscript tables. Raw scores are loaded from the public
  `music21` corpus and are not redistributed.

No manuscript source or PDF is stored in this repository.

## Setup and tests

```bash
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python -m pytest tests
```

## Frozen pipeline

`final_eval` must run first. Evaluation scripts reject uncommitted changes to the induction
method. Commands write to `runs/induction/`.

```bash
venv/bin/python -m score_learning.induction.final_eval
venv/bin/python -m score_learning.induction.bootstrap 1000
venv/bin/python -m score_learning.induction.bootstrap_check 50
venv/bin/python -m score_learning.induction.ablation
venv/bin/python -m score_learning.induction.bit_substitution
venv/bin/python -m score_learning.induction.replay_seeds 10
venv/bin/python -m score_learning.induction.sensitivity
venv/bin/python -m score_learning.induction.paper
venv/bin/python -m score_learning.induction.knowledge_export
venv/bin/python -m score_learning.induction.fresh_eval 1000
venv/bin/python -m score_learning.induction.extra_baselines 200 hold_a hold_b palestrina hold_c palestrina2
```

## Revision and diagnostic analyses

`revision.py` exposes documented subcommands rather than one file per analysis:

```bash
venv/bin/python -m score_learning.induction.synthetic_truth 5
venv/bin/python -m score_learning.diagnostics.null_ladder
venv/bin/python -m score_learning.induction.revision attrib hold_a hold_b hold_c
venv/bin/python -m score_learning.induction.revision pvals hold_a hold_c 100
venv/bin/python -m score_learning.induction.revision tests
venv/bin/python -m score_learning.induction.revision nullstab
venv/bin/python -m score_learning.induction.revision vocab hold_a hold_c 200
venv/bin/python -m score_learning.induction.revision practical
venv/bin/python -m score_learning.induction.revision cost hold_a hold_c
venv/bin/python -m score_learning.induction.revision prereg3 1000
```

The remaining `simple`, `hybrid`, `balanced`, `calops`, `typeboot`, `attrbase`, `cells` and
expert-analysis commands are listed by:

```bash
venv/bin/python -m score_learning.induction.revision
```

## DCRA negative validation

The preregistration is `docs/preregistration_dcra.md`. The reconstruction and compliance checks
are reproducible with:

```bash
venv/bin/python -m score_learning.dcra.check_structure
venv/bin/python -m score_learning.dcra.audit_recon
venv/bin/python -m score_learning.dcra.gate4_compliance
```

Gate 4 failed because the reconstruction did not preserve within-voice step distributions on all
required sets. Per the stopping rule, the generator, threshold and step definition were not
changed and no `L_D` was computed. The full decision is in `docs/dcra_gate4_decision.md`.

## Data and provenance

`results/induction/paper/` contains frozen configurations, split identifiers, aggregate and
item-level anonymous expert results, and all manuscript analysis JSONs. The 66 MB parse cache and
raw identifiable rating sheets are intentionally excluded. See `docs/PROVENANCE.md` and
`docs/DATA_MANIFEST.md`.
