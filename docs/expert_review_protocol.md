# Blind expert assessment of acquired constraints: protocol and analysis plan

Fixed before any rating is collected. Materials: `约束盲评表.xlsx` (given to raters);
`runs/induction/paper/expert_key.json` (kept by the authors, never shown to raters).
Both are produced by `venv/bin/python -m score_learning.induction.expert_pack`.

## Purpose

The reference constraints used in the paper were written by the authors. This assessment asks
independent experts to judge statements produced by the compared methods, without a reference set,
so that (a) the methods can be compared on validity as judged by experts rather than by our oracle,
and (b) the oracle itself can be checked: how many expert-endorsed statements does it miss, and how
many statements it counts as hits do experts reject.

## Items

- Acquisition corpus: hold-out A (40 Bach chorales); training catalogues and thresholds as in the
  frozen evaluation.
- For each of the four nulls (RAS-FR replay, independent bits/PMI, uniform/rarity, productive
  pattern), the 20 most extreme prohibitions and the 20 most extreme preferences by signed z over the
  evaluated fields (R, W, F excluded). Ties are broken by a fixed random seed.
- Items with identical wording are merged; the key records every method and rank behind an item.
- Items are shuffled (seed 2026). Ten items are repeated at the end to measure rater consistency.
- Each item is a Chinese plain-language statement: direction (avoided / favoured), category, and a
  full description of the cell. Method names, counts, z values and oracle matches are not shown.

## Raters

At least three raters with formal training in tonal harmony and counterpoint (university teaching
staff or doctoral students in music theory or composition), not authors and not involved in the
method. Raters work independently and do not discuss items until all ratings are submitted.

## Rating scale

1. Valid: an accepted norm or textbook rule of the style.
2. Plausible: an interpretable stylistic tendency, not usually stated as a rule.
3. Style-dependent: holds only for some styles or periods.
4. Invalid: false, reversed, or musically meaningless.
0. Cannot judge: unclear or insufficient information.

"Acceptable" is defined as 1 or 2. "Cannot judge" is excluded from denominators.

## Analysis

1. **Reliability.** Krippendorff's alpha over raters, nominal on the four substantive categories and
   binary (acceptable vs not). Intra-rater agreement on the ten repeated items (proportion identical,
   Cohen's kappa). If binary alpha < 0.4, item-level results are reported only descriptively.
2. **Item score.** Proportion of raters judging the item acceptable; an item is endorsed if a strict
   majority does.
3. **Method validity (primary).** For each method and polarity, the mean item score over its top 20.
   RAS-FR minus each baseline, with 95% intervals from 2000 bootstrap resamples of raters and of
   items within method (paired where items are shared). A difference is claimed only if its
   interval excludes 0.
4. **Oracle check.** Cross-tabulate oracle match (from the key) against expert endorsement, per method
   and pooled: (i) endorsed but not in the oracle (constraints the reference set misses); (ii) in the
   oracle but not endorsed (questionable reference predicates); Cohen's kappa between the two.
   Re-compute the main paper's precision@20 with expert endorsement in place of oracle match.
5. **Provenance.** Mean item score by provenance class of the underlying cell (relational,
   persists without coordination, within-voice/positional; from the key's desynchronised z).

## Reporting

All ratings, the key and the analysis script are released. Results are reported whether or not
they favour RAS-FR. Comments in the free-text column are summarised for every item rated invalid by
a majority.
