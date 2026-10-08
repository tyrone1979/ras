# Data manifest

## Public inputs

The analyses load Bach chorales and Palestrina works distributed with `music21`. No copyrighted
score files are redistributed here.

## Released result data

- `results/induction/paper/`: frozen configuration, corpus splits and identifiers, primary
  evaluation, baselines, half-sampling, preregistered tests, revision analyses, exported knowledge
  artifact, and anonymous aggregate/item-level expert results.
- `results/induction/devloop.json`: development-phase audit record.
- `results/induction/diagnostics/null_ladder.json`: read-only field-V replay diagnostic.
- `results/dcra/gate4_compliance.json`: preregistered DCRA gate-4 compliance result.

These are machine-readable JSON files consumed by the manuscript table-generation pipeline.

## Intentionally excluded

- `chorales.pkl`: a regenerable approximately 66 MB parse cache.
- Raw rating spreadsheets and consent records: retained by the authors to protect assessor
  confidentiality. Anonymous aggregates and item-level analysis outputs are released.
- Manuscript source, generated figures and PDFs: maintained in the authors' paper repository, not
  in this code/data repository.
