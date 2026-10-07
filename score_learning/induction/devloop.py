# -*- coding: utf-8 -*-
"""方法开发只在开发集上看指标（不碰留出 A/B 与 Palestrina）。

  venv/bin/python -m score_learning.induction.devloop

开发集 = 训练池中除 16 首训练曲与留出 B 以外，再按固定种子抽的 40 首。
"""
from __future__ import annotations

import json
import random
import sys

from score_learning.induction.baselines import indep_fields, metrics
from score_learning.induction.fri import calibrate_from_desync, holdout_fields

OUT = 'runs/induction/devloop.json'


def dev_split():
    from score_learning.induction.paper import load_chorale_pool
    by, usable, _ = load_chorale_pool()
    train_ids = usable[:-40]
    ids16 = random.Random(0).sample(train_ids, 16)
    rest = [i for i in train_ids if i not in set(ids16)]
    hold_b = set(random.Random(7).sample(rest, 40))
    pool = [i for i in rest if i not in hold_b]
    dev_ids = sorted(random.Random(11).sample(pool, 40))
    return by, ids16, dev_ids


def _line(tag, m):
    e = m['enrich']
    pk = m['p_at_k']
    pks = '  '.join(
        f'{pol[0]}@k ' + ' '.join(f'{pk[pol][k]:.2f}' for k in (25, 50, 100, 200)) + f' (base {pk[pol]["base"]:.2f})'
        for pol in ('under', 'over'))
    return (f'{tag:<6} AUC={m["auc_mean"]:.3f} shared={m["auc_shared"]:.3f}  '
            f'under n={e["under"]["n_disc"]} lift={e["under"]["lift"]:.2f}  '
            f'over n={e["over"]["n_disc"]} lift={e["over"]["lift"]:.2f}  '
            f'Comp={m["completeness"]:.3f}\n       {pks}')


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'run'
    by, ids16, dev_ids = dev_split()
    tr = [by[i] for i in ids16]
    dev = [by[i] for i in dev_ids]
    f_dev = holdout_fields(tr, dev, seed=0)
    cfg = calibrate_from_desync(tr, seed=0)
    cfg_i = calibrate_from_desync(tr, seed=0, transform=indep_fields)
    m_r = metrics(f_dev, cfg)
    m_i = metrics(indep_fields(f_dev), cfg_i)
    print(_line('RAS', m_r))
    print(_line('PMI', m_i))
    print('under per field (n_disc, covered, n_cells):', m_r['enrich']['under']['per_field'])
    print('over  per field (n_disc, covered, n_cells):', m_r['enrich']['over']['per_field'])
    low = {k: round(v, 2) for k, v in m_r['auc'].items() if v is not None and v < 0.6}
    print('RAS atoms AUC<0.6:', low)
    try:
        with open(OUT) as f:
            log = json.load(f)
    except (OSError, ValueError):
        log = {}
    log[tag] = {'ras': {k: v for k, v in m_r.items()}, 'pmi': {k: v for k, v in m_i.items()},
                'cfg': cfg, 'cfg_pmi': cfg_i}
    with open(OUT, 'w') as f:
        json.dump(log, f, indent=1, default=str)


if __name__ == '__main__':
    main()
