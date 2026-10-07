# RAS-FR

Voice-decomposition replay null for discovering voice-leading constraints in annotation-free
four-voice scores, with desynchronisation-based attribution (coordination vs register).

The learner sees scores only. `score_learning/induction/oracle.py` holds the reference
constraints used for evaluation; no training module imports it (checked by
`tests/test_induction_paper.py`).

## Setup

```bash
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python -m pytest tests
```

Scores come from the music21 corpus (371 Bach chorales, Palestrina). Results are written to
`runs/induction/`.

## Pipeline

Evaluation scripts refuse to run when `score_learning/induction/*.py` has uncommitted changes.

```bash
venv/bin/python -m score_learning.induction.final_eval          # frozen evaluation
venv/bin/python -m score_learning.induction.bootstrap 1000      # half-sampling intervals
venv/bin/python -m score_learning.induction.ablation            # catalogue ablation
venv/bin/python -m score_learning.induction.bit_substitution    # value substitution
venv/bin/python -m score_learning.induction.replay_seeds 10     # replay-seed sensitivity
venv/bin/python -m score_learning.induction.sensitivity         # training size, R, sparse rule
venv/bin/python -m score_learning.induction.fresh_eval          # hold-out C, Palestrina-2
venv/bin/python -m score_learning.induction.extra_baselines 200 # log-linear and time-shift nulls
venv/bin/python -m score_learning.induction.knowledge_export    # JSON artifact and checker
```

`final_eval` must run first; the other scripts read its frozen configuration.
