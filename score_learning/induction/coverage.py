# -*- coding: utf-8 -*-
"""RAS 几何函数 vs checker。FireParams 与谓词逐项对齐。"""
from __future__ import annotations

import os
from typing import Dict, List, Set, Tuple

from score_learning.grammar.check import (
    bad_melodic, hidden_perfect_outer, overlap, parallel_perfect, spacing_ok,
)
from score_learning.induction.learn import load_many
from score_learning.induction.ras import VOICES, FireParams, ScoreRAS, geometry_fire


def _complete(ras: ScoreRAS):
    return [s for s in ras.slices if all(s.midis[v] is not None for v in VOICES)]


def oracle(ras: ScoreRAS, p: FireParams) -> Dict[str, Set[Tuple]]:
    full = _complete(ras)
    beats = [s for s in full if (not p.a4a5a8_beats_only) or s.metric != 'offbeat']
    a1, a2, a4, a5, a8 = set(), set(), set(), set(), set()

    def mid(s):
        return [s.midis[v] for v in VOICES]

    for a, b in zip(full, full[1:]):
        v1, v2 = mid(a), mid(b)
        t = round(b.t, 3)
        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            if parallel_perfect(v1[i], v1[j], v2[i], v2[j]):
                a1.add((t, i, j))
        if hidden_perfect_outer(v1[0], v1[3], v2[0], v2[3]):
            a2.add((t,))
    for a, b in zip(beats, beats[1:]):
        v1, v2 = mid(a), mid(b)
        t = round(b.t, 3)
        if getattr(p, 'a1a2_also_beats', True):
            for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
                if parallel_perfect(v1[i], v1[j], v2[i], v2[j]):
                    a1.add((t, i, j))
            if hidden_perfect_outer(v1[0], v1[3], v2[0], v2[3]):
                a2.add((t,))
        if overlap(v1, v2):
            a4.add((t,))
        if not spacing_ok(v2, tb_max=p.tb_max):
            a5.add((t,))
        for i in range(4):
            if bad_melodic(v1[i], v2[i]):
                a8.add((t, i))
    return {'A1': a1, 'A2': a2, 'A4': a4, 'A5': a5, 'A8': a8}


def prf(pred: Set, gold: Set) -> Dict[str, float]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if tp + fp else (1.0 if not gold else 0.0)
    r = tp / (tp + fn) if tp + fn else (1.0 if not pred else 0.0)
    f = 2 * p * r / (p + r) if p + r else (1.0 if not gold and not pred else 0.0)
    return {'p': p, 'r': r, 'f1': f, 'tp': tp, 'fp': fp, 'fn': fn, 'gold': len(gold)}


def run(ids: List[int], params: FireParams) -> str:
    pieces = load_many(ids, t_max=None)
    agg = {k: {'tp': 0, 'fp': 0, 'fn': 0, 'gold': 0} for k in ('A1', 'A2', 'A4', 'A5', 'A8')}
    lines = [f'# 几何函数 vs checker  n={len(ids)}  {params}', '']
    for ras in pieces:
        gold, pred = oracle(ras, params), geometry_fire(ras, params)
        lines.append(f'## {ras.title}')
        for k in agg:
            m = prf(pred[k], gold[k])
            for x in ('tp', 'fp', 'fn', 'gold'):
                agg[k][x] += m[x]
            lines.append(
                f'  {k}  P={m["p"]:.2f} R={m["r"]:.2f} F1={m["f1"]:.2f}  '
                f'tp={m["tp"]} fp={m["fp"]} fn={m["fn"]} gold={m["gold"]}'
            )
        lines.append('')
    lines.append('## 合计')
    ok = True
    for k, a in agg.items():
        tp, fp, fn = a['tp'], a['fp'], a['fn']
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        f = 2 * p * r / (p + r) if p + r else 1.0
        flag = 'OK' if f >= 0.9 else 'LOW'
        if f < 0.9:
            ok = False
        lines.append(
            f'  {k}  P={p:.3f} R={r:.3f} F1={f:.3f}  {flag}  '
            f'tp={tp} fp={fp} fn={fn} gold={a["gold"]}'
        )
    lines.append(f'\nall_f1>=0.9: {ok}')
    return '\n'.join(lines) + '\n'


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--out', default='runs/induction/coverage_rq0.txt')
    args = ap.parse_args()
    params = FireParams()
    text = run(list(range(1, args.n + 1)), params)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
