# -*- coding: utf-8 -*-
"""冻结方法后的一次性最终评测：留出 A / 留出 B / Palestrina 上 RAS-FR vs PMI vs 随机。

  venv/bin/python -m score_learning.induction.final_eval

方法代码（score_learning/induction/*.py）有未提交改动时拒绝运行，结果绑定提交号。
"""
from __future__ import annotations

import json
import os
import random
import sys

from score_learning.induction.baselines import indep_fields, metrics, random_discovery
from score_learning.induction.fri import calibrate_from_desync, holdout_fields
from score_learning.induction.paper import _git_head, load_chorale_pool, load_palestrina

OUT = 'runs/induction/paper/final_eval.json'


def _row(m: dict) -> dict:
    e, pk = m['enrich'], m['p_at_k']
    return {
        'auc_mean': m['auc_mean'], 'auc_shared': m['auc_shared'], 'auc': m['auc'],
        'p_at_k': pk,
        'under': {k: e['under'][k] for k in ('n_disc', 'precision', 'base', 'lift')},
        'over': {k: e['over'][k] for k in ('n_disc', 'precision', 'base', 'lift')},
        'completeness': m['completeness'], 'miss': m['miss'],
    }


def _print(tag: str, r: dict):
    pk = r['p_at_k']
    print(f'  {tag:<4} AUC={r["auc_mean"]:.3f} shared={r["auc_shared"]:.3f}  '
          f'u@25/50/100={pk["under"][25]:.2f}/{pk["under"][50]:.2f}/{pk["under"][100]:.2f} '
          f'(base {pk["under"]["base"]:.2f})  '
          f'o@25/50/100={pk["over"][25]:.2f}/{pk["over"][50]:.2f}/{pk["over"][100]:.2f} '
          f'(base {pk["over"]["base"]:.2f})  '
          f'lift u={r["under"]["lift"]:.2f} o={r["over"]["lift"]:.2f}  Comp={r["completeness"]:.3f}',
          flush=True)


def main():
    commit = _git_head()
    if commit.endswith('-dirty') and '--allow-dirty' not in sys.argv:
        sys.exit(f'method code is dirty ({commit}); commit before the final evaluation')
    by, usable, _ = load_chorale_pool()
    hold_ids, train_ids = usable[-40:], usable[:-40]
    ids16 = random.Random(0).sample(train_ids, 16)
    rest = [i for i in train_ids if i not in set(ids16)]
    hold_b_ids = sorted(random.Random(7).sample(rest, 40))
    tr = [by[i] for i in ids16]
    sets = {
        'hold_a': ([by[i] for i in hold_ids], 0),
        'hold_b': ([by[i] for i in hold_b_ids], 1),
        'palestrina': (load_palestrina(16), 0),
    }

    cfg = calibrate_from_desync(tr, seed=0)
    cfg_i = calibrate_from_desync(tr, seed=0, transform=indep_fields)
    out = {'commit': commit, 'train_ids': ids16, 'hold_ids': hold_ids, 'hold_b_ids': hold_b_ids,
           'cfg': cfg, 'cfg_pmi': cfg_i, 'sets': {}}
    for name, (hold, seed) in sets.items():
        print(f'== {name} (n={len(hold)}) ==', flush=True)
        f = holdout_fields(tr, hold, seed=seed)
        ras, pmi = _row(metrics(f, cfg)), _row(metrics(indep_fields(f), cfg_i))
        rnd = random_discovery(f, cfg)
        _print('RAS', ras)
        _print('PMI', pmi)
        print(f'  RND Comp={rnd["comp_mean"]:.3f}±{rnd["comp_sd"]:.3f} (max {rnd["comp_max"]:.3f})',
              flush=True)
        out['sets'][name] = {
            'n': len(hold), 'ras': ras, 'pmi': pmi,
            'random': {k: rnd[k] for k in ('comp_mean', 'comp_sd', 'comp_max', 'atom_rate')},
        }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1, default=str)
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
