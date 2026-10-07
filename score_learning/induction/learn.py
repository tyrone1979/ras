# -*- coding: utf-8 -*-
"""从多首 RAS 拟合一组函数（无 LLM、无教科书标签）。

四类条件分布，对应表示层：
  f_h  声部内：P(Δp | voice, metric, prev_Δp)
  f_v  声部对：P(Rel_mot, perfect | pair_role, metric, prev_perfect)
  f_s  音集：  P(degset_t | metric, degset_{t-1})
  f_a  几何原子质量（role, Rel, perfect, interval_hold）——可读的「规则候选」

回退：有上下文 → 只 metric → 均匀。留出 NLL 衡量拟合，不是听感。
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .ras import PAIRS, VOICES, ScoreRAS, load_chorale

DP_CLIP = 12


def _bin_dp(dp: int) -> int:
    if dp > DP_CLIP:
        return DP_CLIP
    if dp < -DP_CLIP:
        return -DP_CLIP
    return dp


def _set_key(slice_) -> Tuple[int, ...]:
    degs = sorted({d for d in slice_.deg.values() if d is not None})
    return tuple(degs)


class Cond:
    """P(y | x) with backoff to P(y) then uniform over seen y."""

    def __init__(self):
        self.joint: Dict[Any, Counter] = defaultdict(Counter)
        self.marg = Counter()

    def add(self, x, y, n: int = 1):
        self.joint[x][y] += n
        self.marg[y] += n

    def logp(self, x, y) -> float:
        c = self.joint.get(x)
        if c and sum(c.values()) >= 2:
            return math.log((c.get(y, 0) + 1) / (sum(c.values()) + len(self.marg) or 1))
        tot = sum(self.marg.values()) + len(self.marg) or 1
        return math.log((self.marg.get(y, 0) + 1) / tot)

    def top(self, x, k: int = 5) -> List[Tuple[Any, float, int]]:
        c = self.joint.get(x) or self.marg
        n = sum(c.values()) or 1
        return [(y, cnt / n, cnt) for y, cnt in c.most_common(k)]

    def n_ctx(self) -> int:
        return len(self.joint)


def fit(pieces: List[ScoreRAS]) -> Dict[str, Any]:
    f_h, f_v, f_s = Cond(), Cond(), Cond()
    atoms = Counter()
    n_pair = n_h = n_s = 0
    for ras in pieces:
        for v in VOICES:
            hs = ras.horiz[v]
            for i in range(1, len(hs)):
                met = 'downbeat' if abs(hs[i].t1 % ras.bar_len) < 1e-6 else 'other'
                x = (v, met, _bin_dp(hs[i - 1].dp))
                y = _bin_dp(hs[i].dp)
                f_h.add(x, y)
                n_h += 1
        for p in PAIRS:
            steps = ras.pairs[p]
            for i, st in enumerate(steps):
                prev_pf = steps[i - 1].perfect if i else '-'
                sl = next((s for s in ras.slices if abs(s.t - st.t) < 1e-6), None)
                met = sl.metric if sl else 'other'
                x = (st.role, met, prev_pf)
                y = (st.rel_mot, st.perfect)
                f_v.add(x, y)
                hold = int(st.d_interval == 0)
                atoms[(st.role, st.rel_mot, st.perfect, hold)] += 1
                n_pair += 1
        for a, b in zip(ras.slices, ras.slices[1:]):
            f_s.add((a.metric, _set_key(a)), _set_key(b))
            n_s += 1
    tot_a = sum(atoms.values()) or 1
    return {
        'f_h': f_h, 'f_v': f_v, 'f_s': f_s, 'atoms': atoms,
        'n_h': n_h, 'n_pair': n_pair, 'n_s': n_s, 'n_pieces': len(pieces),
        'atom_mass': {k: v / tot_a for k, v in atoms.items()},
    }


def nll(model: Dict[str, Any], pieces: List[ScoreRAS]) -> Dict[str, float]:
    f_h, f_v, f_s = model['f_h'], model['f_v'], model['f_s']
    lh = lv = ls = 0.0
    nh = nv = ns = 0
    for ras in pieces:
        for v in VOICES:
            hs = ras.horiz[v]
            for i in range(1, len(hs)):
                met = 'downbeat' if abs(hs[i].t1 % ras.bar_len) < 1e-6 else 'other'
                lh += -f_h.logp((v, met, _bin_dp(hs[i - 1].dp)), _bin_dp(hs[i].dp))
                nh += 1
        for p in PAIRS:
            steps = ras.pairs[p]
            for i, st in enumerate(steps):
                prev_pf = steps[i - 1].perfect if i else '-'
                sl = next((s for s in ras.slices if abs(s.t - st.t) < 1e-6), None)
                met = sl.metric if sl else 'other'
                lv += -f_v.logp((st.role, met, prev_pf), (st.rel_mot, st.perfect))
                nv += 1
        for a, b in zip(ras.slices, ras.slices[1:]):
            ls += -f_s.logp((a.metric, _set_key(a)), _set_key(b))
            ns += 1
    def mean(s, n):
        return s / n if n else float('nan')
    return {
        'nll_h': mean(lh, nh), 'nll_v': mean(lv, nv), 'nll_s': mean(ls, ns),
        'ppl_h': math.exp(mean(lh, nh)) if nh else float('nan'),
        'ppl_v': math.exp(mean(lv, nv)) if nv else float('nan'),
        'ppl_s': math.exp(mean(ls, ns)) if ns else float('nan'),
        'nh': nh, 'nv': nv, 'ns': ns,
    }


def format_functions(model: Dict[str, Any], eval_tr: Dict, eval_te: Dict,
                     train_ids: List[int], test_ids: List[int]) -> str:
    f_h, f_v, f_s, atoms = model['f_h'], model['f_v'], model['f_s'], model['atoms']
    L = [
        '# RAS 拟合函数（多首输入，无标签）',
        f'train={train_ids}  test={test_ids}  pieces_train={model["n_pieces"]}',
        f'events  horiz={model["n_h"]}  pair={model["n_pair"]}  set={model["n_s"]}',
        '',
        '## 留出拟合（NLL nats / 困惑度）',
        f'  train  f_h ppl={eval_tr["ppl_h"]:.2f}  f_v ppl={eval_tr["ppl_v"]:.2f}  '
        f'f_s ppl={eval_tr["ppl_s"]:.2f}',
        f'  test   f_h ppl={eval_te["ppl_h"]:.2f}  f_v ppl={eval_te["ppl_v"]:.2f}  '
        f'f_s ppl={eval_te["ppl_s"]:.2f}',
        '',
        '## f_h  声部内  P(Δp | voice, metric, prev_Δp)  若干上下文',
    ]
    show_h = [('S', 'other', 2), ('B', 'other', 0), ('S', 'downbeat', 0),
              ('T', 'other', 1), ('A', 'other', -2)]
    for x in show_h:
        if x not in f_h.joint:
            continue
        tops = ', '.join(f'Δp={y:+d}:{p:.2f}(n={n})' for y, p, n in f_h.top(x, 4))
        L.append(f'  {x}  →  {tops}')
    L += ['', '## f_v  声部对  P(Rel, perfect | role, metric, prev_perfect)']
    show_v = [('outer', 'downbeat', 'P8'), ('outer', 'other', '-'),
              ('adjacent', 'downbeat', '-'), ('adjacent', 'offbeat', '-'),
              ('diagonal', 'downbeat', 'P5')]
    for x in show_v:
        if x not in f_v.joint:
            # try any matching role
            cand = [k for k in f_v.joint if k[0] == x[0]]
            if not cand:
                continue
            x = cand[0]
        tops = ', '.join(f'{y[0]}/{y[1]}:{p:.2f}' for y, p, n in f_v.top(x, 4))
        L.append(f'  {x}  →  {tops}')
    L += ['', '## f_s  音集  P(set_t | metric, set_{t-1})  主音三和弦出发']
    tonic_ctx = [k for k in f_s.joint if k[1] == (0, 4, 7)]
    for x in tonic_ctx[:4]:
        tops = []
        for y, p, n in f_s.top(x, 4):
            tops.append(f'{y}:{p:.2f}')
        L.append(f'  {x}  →  {", ".join(tops)}')
    L += ['', '## f_a  几何原子（质量最高的 12 个）  hold=纵向音程不变']
    L.append('  mass   n     role      Rel        perfect  hold  读法')
    for (role, rel, pf, hold), n in atoms.most_common(12):
        mass = n / (sum(atoms.values()) or 1)
        read = []
        if rel == 'parallel' and pf.startswith('P') and pf != 'P1' and hold:
            read.append('平行纯音程保持')
        elif rel == 'static' and hold:
            read.append('两声部都不动')
        elif rel == 'oblique':
            read.append('一声部动、一声部留')
        elif rel == 'contrary' and pf.startswith('P'):
            read.append('反向进入/离开纯音程')
        elif rel == 'parallel' and not pf.startswith('P'):
            read.append('平行但不纯')
        L.append(f'  {mass:5.3f}  {n:5d}  {role:<9} {rel:<10} {pf:<7} {hold}     '
                 f'{read[0] if read else ""}')
    par_hold = sum(n for (role, rel, pf, hold), n in atoms.items()
                   if rel == 'parallel' and hold and pf in ('P5', 'P8', 'P5c', 'P8c'))
    par_any = sum(n for (role, rel, pf, hold), n in atoms.items() if rel == 'parallel')
    L += [
        '',
        f'  平行运动中「纯音程且保持」占比 {par_hold}/{par_any or 1} = '
        f'{par_hold / (par_any or 1):.3f}',
        '  （质量低 ⇒ 语料少用该几何；拟合的是函数，不是禁令标签）',
    ]
    return '\n'.join(L) + '\n'


def load_many(ids: List[int], t_max: Optional[float] = None, transpose: int = 0) -> List[ScoreRAS]:
    out = []
    for i, n in enumerate(ids):
        print(f'  RAS {i + 1}/{len(ids)}  chorale {n}  t={transpose}', flush=True)
        try:
            out.append(load_chorale(n, t_max=t_max, transpose=transpose))
        except Exception as e:
            print(f'    skip {n}: {e}', flush=True)
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=8, help='train chorales')
    p.add_argument('--test', type=int, default=2)
    p.add_argument('--tmax', type=float, default=0, help='0 = full piece')
    p.add_argument('--out', default='runs/induction/fitted_functions.txt')
    args = p.parse_args()
    t_max = None if args.tmax <= 0 else args.tmax
    train_ids = list(range(1, args.k + 1))
    test_ids = list(range(args.k + 1, args.k + 1 + args.test))
    print('load train', train_ids, flush=True)
    tr = load_many(train_ids, t_max)
    print('load test', test_ids, flush=True)
    te = load_many(test_ids, t_max)
    model = fit(tr)
    ev_tr, ev_te = nll(model, tr), nll(model, te)
    text = format_functions(model, ev_tr, ev_te, train_ids, test_ids)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(text)
    meta = {
        'train_ids': train_ids, 'test_ids': test_ids,
        'eval_train': ev_tr, 'eval_test': ev_te,
        'n_h': model['n_h'], 'n_pair': model['n_pair'], 'n_s': model['n_s'],
        'top_atoms': [
            {'role': r, 'rel': rel, 'perfect': pf, 'hold': h, 'n': n}
            for (r, rel, pf, h), n in model['atoms'].most_common(20)
        ],
    }
    with open(args.out.replace('.txt', '.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(text)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
