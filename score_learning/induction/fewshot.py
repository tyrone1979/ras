# -*- coding: utf-8 -*-
"""k-shot：拟合 f_h/f_v/f_s 学习曲线；几何函数在未见曲上的检测 F1。"""
from __future__ import annotations

import json
import math
import os
from typing import List

from score_learning.induction.coverage import oracle, prf
from score_learning.induction.learn import fit, load_many, nll
from score_learning.induction.ras import FireParams, geometry_fire


def geom_table(pieces, params: FireParams) -> str:
    keys = ('A1', 'A2', 'A4', 'A5', 'A8')
    agg = {k: {'tp': 0, 'fp': 0, 'fn': 0, 'gold': 0} for k in keys}
    for ras in pieces:
        gold, pred = oracle(ras, params), geometry_fire(ras, params)
        for k in keys:
            m = prf(pred[k], gold[k])
            for x in ('tp', 'fp', 'fn', 'gold'):
                agg[k][x] += m[x]
    lines = []
    ok = True
    for k, a in agg.items():
        tp, fp, fn = a['tp'], a['fp'], a['fn']
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        f = 2 * p * r / (p + r) if p + r else 1.0
        if f < 0.9:
            ok = False
        lines.append(
            f'  {k}  P={p:.3f} R={r:.3f} F1={f:.3f}  '
            f'tp={tp} fp={fp} fn={fn} gold={a["gold"]}'
        )
    lines.append(f'  all>=0.9: {ok}')
    return '\n'.join(lines), ok, {k: dict(v) for k, v in agg.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ks', default='4,8,16')
    ap.add_argument('--test-start', type=int, default=31)
    ap.add_argument('--test-n', type=int, default=10)
    ap.add_argument('--hold-start', type=int, default=11)
    ap.add_argument('--hold-n', type=int, default=20)
    ap.add_argument('--out', default='runs/induction/fewshot.txt')
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(',') if x]
    test_ids = list(range(args.test_start, args.test_start + args.test_n))
    hold_ids = list(range(args.hold_start, args.hold_start + args.hold_n))
    params = FireParams()

    lines = ['# few-shot RAS 函数', f'test_ppl ids={test_ids}',
             f'hold_geom ids={hold_ids}  (调参用过 1–10，这里是未见曲)', '']

    print('load hold-out geometry set', hold_ids, flush=True)
    hold = load_many(hold_ids, t_max=None)
    gtxt, gok, gagg = geom_table(hold, params)
    lines += ['## 几何函数 f_geom(θ) 在未见 20 首上的检测', gtxt, '']

    print('load ppl test', test_ids, flush=True)
    te = load_many(test_ids, t_max=None)
    rows = []
    for k in ks:
        train_ids = list(range(1, k + 1))
        print(f'fit k={k} {train_ids}', flush=True)
        tr = load_many(train_ids, t_max=None)
        model = fit(tr)
        ev_tr, ev_te = nll(model, tr), nll(model, te)
        rows.append({
            'k': k, 'n_train': len(tr),
            'ppl_h_tr': ev_tr['ppl_h'], 'ppl_v_tr': ev_tr['ppl_v'], 'ppl_s_tr': ev_tr['ppl_s'],
            'ppl_h_te': ev_te['ppl_h'], 'ppl_v_te': ev_te['ppl_v'], 'ppl_s_te': ev_te['ppl_s'],
            'n_h': model['n_h'], 'n_pair': model['n_pair'],
        })
        lines.append(
            f'## k={k}  train={train_ids}\n'
            f'  events horiz={model["n_h"]} pair={model["n_pair"]} set={model["n_s"]}\n'
            f'  train ppl  h={ev_tr["ppl_h"]:.2f}  v={ev_tr["ppl_v"]:.2f}  s={ev_tr["ppl_s"]:.2f}\n'
            f'  test  ppl  h={ev_te["ppl_h"]:.2f}  v={ev_te["ppl_v"]:.2f}  s={ev_te["ppl_s"]:.2f}\n'
        )

    lines += ['## 学习曲线（留出困惑度，越低越好）',
              '  k   ppl_h   ppl_v   ppl_s']
    for r in rows:
        lines.append(
            f'  {r["k"]:<3} {r["ppl_h_te"]:7.2f} {r["ppl_v_te"]:7.2f} {r["ppl_s_te"]:7.2f}'
        )
    text = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(text)
    meta = {'hold_geom': gagg, 'hold_ok': gok, 'ppl': rows,
            'test_ids': test_ids, 'hold_ids': hold_ids}
    with open(args.out.replace('.txt', '.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(text)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
