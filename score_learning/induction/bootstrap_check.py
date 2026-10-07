# -*- coding: utf-8 -*-
"""留出集 A 上比较三种重抽方案对 K 场终止式原子 AUC 差（RAS − PMI）的偏移。

  venv/bin/python -m score_learning.induction.bootstrap_check [B]

方案：全曲复制两份（不改变比例，只制造重复）、无放回半抽样、有放回重抽。
"""
from __future__ import annotations

import json
import random
import sys

from score_learning.induction import bootstrap as bs
from score_learning.induction.baselines import EVAL_ATOMS, atom_auc, indep_fields
from score_learning.induction.paper import _git_head, load_chorale_pool

OUT = 'runs/induction/paper/bootstrap_check.json'
K_ATOMS = sorted(n for n, f, _pol, _p in bs.fri.book_predicates() if f == 'K' and n in EVAL_ATOMS)


def _diff(per, idx):
    f = bs.fields_from(per, idx)
    r, p = atom_auc(f), atom_auc(indep_fields(f))
    return {a: r[a] - p[a] for a in K_ATOMS if r.get(a) is not None and p.get(a) is not None}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    with open(bs.FINAL) as fh:
        fin = json.load(fh)
    by, _usable, _ = load_chorale_pool()
    tr = [by[i] for i in fin['train_ids']]
    hold = [by[i] for i in fin['hold_ids']]
    per = bs.piece_counts(tr, hold, 0)
    m = len(hold)
    rng = random.Random(2026)
    schemes = {
        'half': lambda: rng.sample(range(m), m // 2),
        'with_replacement': lambda: [rng.randrange(m) for _ in range(m)],
    }
    out = {'commit': _git_head(), 'n': n, 'atoms': K_ATOMS,
           'point': _diff(per, list(range(m))), 'doubled': _diff(per, list(range(m)) * 2)}
    for name, gen in schemes.items():
        draws = [_diff(per, gen()) for _ in range(n)]
        out[name] = {a: sum(d[a] for d in draws if a in d) / max(1, sum(a in d for d in draws))
                     for a in K_ATOMS}
    for k in ('point', 'doubled', 'half', 'with_replacement'):
        print(f'{k:<17}', ' '.join(f'{a}:{out[k].get(a, float("nan")):+.2f}' for a in K_ATOMS))
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
