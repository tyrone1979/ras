# -*- coding: utf-8 -*-
"""RAS 几何函数当 checker：与 check_satb 的 R01/R02/R03/R04/R08 对齐。

不改 grammar/check.py（ESWA 复现口径不动）。新路径只读 FireParams。
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Tuple

from score_learning.grammar.analyze import analyze_chorale, chorale_paths
from score_learning.grammar.check import check_satb
from score_learning.induction.coverage import prf
from score_learning.induction.learn import load_many
from score_learning.induction.ras import FireParams, geometry_fire, load_chorale, midi_name

VOICE = 'SATB'
MAP = {'A1': 'R01', 'A2': 'R02', 'A4': 'R03', 'A5': 'R03/R04', 'A8': 'R08'}
PAIR_NAMES = {
    (0, 1): 'SA', (0, 2): 'ST', (0, 3): 'SB',
    (1, 2): 'AT', (1, 3): 'AB', (2, 3): 'TB',
}


def ras_violations(ras, params: FireParams | None = None) -> List[Dict]:
    params = params or FireParams()
    fired = geometry_fire(ras, params)
    out = []
    for key, evs in fired.items():
        rule = MAP[key]
        for e in sorted(evs):
            t = e[0]
            if key == 'A1':
                _, i, j = e
                out.append({'rule': rule, 't': t, 'msg': f'{PAIR_NAMES[(i, j)]} 平行纯音程',
                            'atom': key})
            elif key == 'A8':
                _, vi = e
                out.append({'rule': rule, 't': t, 'msg': f'{VOICE[vi]} 不良跳进',
                            'atom': key})
            else:
                out.append({'rule': rule, 't': t, 'msg': key, 'atom': key})
    out.sort(key=lambda x: (x['t'], x['rule']))
    return out


def analyze_violations(piece: Dict) -> List[Dict]:
    """analyze.py 拍网格 + 现有 check_satb（只收声部进行类规则）。"""
    grid, info = [], []
    tpc = piece['tonic_pc']
    for x in piece['beats']:
        midi = list(x['midi'])
        grid.append((x['t'], midi))
        half = [x['eighth'][v] if x['eighth'][v] is not None else midi[v]
                for v in range(4)]
        if half != midi:
            grid.append((x['t'] + 0.5, half))
        rel = [(m - tpc) % 12 for m in midi]
        info.append({
            't': x['t'], 'midi': midi, 'rel': rel,
            'members': {r: 'R' for r in rel},
            'lt_pc': None, 'tonic_chord': False, 'seventh_pc': None,
            'is_seventh': False, 'phrase_start': False,
        })
    ranges = {k: {'low': 0, 'high': 127} for k in 'SATB'}
    viol = check_satb(grid, info, ranges, tb_max=19)
    keep = {'R01', 'R02', 'R03', 'R03/R04', 'R08'}
    return [v for v in viol if v['rule'] in keep]


def _bucket(rule: str, t: float) -> Tuple[str, float]:
    return rule, round(float(t) * 2) / 2


def compare_piece(ras, analyzed) -> Dict[str, Dict]:
    ras_v = ras_violations(ras)
    ref_v = analyze_violations(analyzed) if analyzed else []
    pred = {_bucket(v['rule'], v['t']) for v in ras_v}
    gold = {_bucket(v['rule'], v['t']) for v in ref_v}
    by = {}
    for rule in ('R01', 'R02', 'R03', 'R03/R04', 'R08'):
        p = {x for x in pred if x[0] == rule}
        g = {x for x in gold if x[0] == rule}
        by[rule] = prf(p, g)
    by['ALL'] = prf(pred, gold)
    return {'by': by, 'n_ras': len(ras_v), 'n_ref': len(ref_v)}


def format_check(ras, viol: List[Dict]) -> str:
    L = [f'# RAS checker  {ras.title}',
         f'tonic={ras.tonic_pc}  meter={ras.meter}  n_viol={len(viol)}', '']
    if not viol:
        L.append('(no A1/A2/A4/A5/A8)')
        return '\n'.join(L) + '\n'
    L.append(f'{"t":>6}  rule     msg')
    for v in viol:
        L.append(f'{v["t"]:6.1f}  {v["rule"]:<8} {v["msg"]}')
    return '\n'.join(L) + '\n'


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--chorale', type=int, default=0,
                    help='print one piece; 0 = compare batch')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--out', default='runs/induction/check_compare.txt')
    args = ap.parse_args()
    params = FireParams()

    if args.chorale:
        ras = load_chorale(args.chorale, t_max=None)
        text = format_check(ras, ras_violations(ras, params))
        print(text)
        return

    paths = chorale_paths()
    ids = list(range(args.start, args.start + args.n))
    print('load RAS', ids, flush=True)
    pieces = load_many(ids, t_max=None)
    by_id = {}
    for ras in pieces:
        # title: Riemenschneider N  path
        n = int(ras.title.split()[1])
        by_id[n] = ras
    agg = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gold': 0})
    lines = [f'# RAS checker vs check_satb  ids={ids}', '']
    for n in ids:
        ras = by_id.get(n)
        if ras is None:
            lines.append(f'## {n} skip (RAS)')
            continue
        path = paths[n - 1]
        analyzed = analyze_chorale(path, pid=str(n))
        if not analyzed:
            lines.append(f'## {n} skip (analyze)')
            continue
        cmp_ = compare_piece(ras, analyzed)
        lines.append(f'## {n}  ras={cmp_["n_ras"]} ref={cmp_["n_ref"]}')
        for rule, m in cmp_['by'].items():
            if rule == 'ALL':
                continue
            for k in ('tp', 'fp', 'fn', 'gold'):
                agg[rule][k] += m[k]
            lines.append(
                f'  {rule:<8} P={m["p"]:.2f} R={m["r"]:.2f} F1={m["f1"]:.2f}  '
                f'tp={m["tp"]} fp={m["fp"]} fn={m["fn"]}'
            )
        m = cmp_['by']['ALL']
        for k in ('tp', 'fp', 'fn', 'gold'):
            agg['ALL'][k] += m[k]
        lines.append('')
    lines.append('## 合计')
    ok = True
    for rule in ('R01', 'R02', 'R03', 'R03/R04', 'R08', 'ALL'):
        a = agg[rule]
        tp, fp, fn = a['tp'], a['fp'], a['fn']
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        f = 2 * p * r / (p + r) if p + r else 1.0
        if rule != 'ALL' and f < 0.9:
            ok = False
        lines.append(
            f'  {rule:<8} P={p:.3f} R={r:.3f} F1={f:.3f}  '
            f'tp={tp} fp={fp} fn={fn} gold={a["gold"]}'
        )
    lines.append(f'\nper-rule F1>=0.9: {ok}')
    text = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
