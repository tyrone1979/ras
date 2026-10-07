# -*- coding: utf-8 -*-
"""重放随机种子敏感度：同一留出集、同一冻结方法，只换零模型重放的种子。

  venv/bin/python -m score_learning.induction.replay_seeds [n_seed]
"""
from __future__ import annotations

import json
import random
import sys

from score_learning.induction.bootstrap import METHODS, draw_stats
from score_learning.induction.fri import holdout_fields

FINAL = 'runs/induction/paper/final_eval.json'
OUT = 'runs/induction/paper/replay_seeds.json'


def main():
    from score_learning.induction.paper import _desync_piece, load_chorale_pool, load_palestrina
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    with open(FINAL) as fh:
        fin = json.load(fh)
    by, _u, _ = load_chorale_pool()
    tr = [by[i] for i in fin['train_ids']]
    sets = {'hold_a': [by[i] for i in fin['hold_ids']], 'hold_b': [by[i] for i in fin['hold_b_ids']],
            'palestrina': load_palestrina(16)}
    out = {}
    for name, hold in sets.items():
        rng = random.Random(99)
        hold_d = [_desync_piece(p, rng) for p in hold]
        runs = []
        for s in range(n_seed):
            runs.append(draw_stats(holdout_fields(tr, hold, seed=100 + s),
                                   holdout_fields(tr, hold_d, seed=100 + s)))
        summ = {}
        for k in runs[0]['ras']:
            summ[k] = {}
            for m in METHODS:
                xs = [r[m][k] for r in runs]
                mu = sum(xs) / len(xs)
                summ[k][m] = {'mean': mu, 'sd': (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5,
                              'min': min(xs), 'max': max(xs)}
            print(f'{name:<11}{k:<11} ' + '  '.join(
                f'{m} {summ[k][m]["mean"]:.3f}±{summ[k][m]["sd"]:.3f}' for m in METHODS), flush=True)
        out[name] = {'n_seed': n_seed, 'summary': summ, 'runs': runs}
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
