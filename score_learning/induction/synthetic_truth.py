# -*- coding: utf-8 -*-
"""合成真值基准（仅分析；方法代码与冻结配置不变）。

四个生成器，依赖结构已知：
  S1  四声部独立随机游走（无依赖）
  S2  独立声部 + 音区、各声部音级偏好、句末位置偏好（只有边际/位置依赖）
  S3  S2 + 成对协同：禁止任何两声部平行纯五/八度；S–B 同向进行以 0.6 概率被拒（偏好反向）
  S4  S3 + 高阶协同：每个切片须有上方声部在低音上方三度；禁止重复导音

所有曲目节奏与乐句结构相同，因此“跨曲重组声部”（每个声部取自另一首）恰好去掉声部间对齐、
保留每个声部的完整序列与拍位/句位。真值：大语料（N_TRUTH 首）上原谱与重组谱的格子计数之差。

  venv/bin/python -m score_learning.induction.synthetic_truth [n_rep]
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import replace

from score_learning.induction import fri
from score_learning.induction import ras as ras_mod
from score_learning.induction.extra_corpus import _satb_score
from score_learning.induction.fri import VOICES, annotate_z
from score_learning.induction.ras import from_music21
from score_learning.induction.revision import (
    PREDS, WITHIN, _dump, _git_head, classify, desync_metric, desync_rot, eval_disc,
)

DIA = [m for m in range(36, 90) if m % 12 in (0, 2, 4, 5, 7, 9, 11)]
RANGE = {'S': (60, 79), 'A': (53, 74), 'T': (48, 69), 'B': (40, 62)}
START = {'S': 72, 'A': 67, 'T': 60, 'B': 48}
STEP_P = {0: .20, 1: .22, -1: .22, 2: .10, -2: .10, 3: .05, -3: .05, 4: .03, -4: .03}
N_PHRASE, PHRASE_Q = 8, 6
N_TRAIN, N_HOLD, N_TRUTH = 16, 40, 600
GENS = ('S1', 'S2', 'S3', 'S4')
PLANTED = {'S1': (), 'S2': (), 'S3': ('A1',), 'S4': ('A1', 'A11', 'A12')}
VERT = ('V', 'O', 'H', 'S', 'U', 'K', 'J')
SGN = {'under': -1.0, 'over': 1.0}


def _deg_w(gen, v, pc, end):
    if gen == 'S1':
        return 1.0
    if end:
        if v == 'B':
            return {0: 6.0, 7: 2.0}.get(pc, 0.05)
        return {0: 3.0, 4: 2.0, 7: 2.0}.get(pc, 0.1)
    if v == 'B':
        return {0: 3.0, 7: 3.0, 5: 1.5, 9: 1.5}.get(pc, 1.0)
    if v == 'S':
        return {0: 2.0, 4: 2.0, 7: 2.0}.get(pc, 1.0)
    return 1.0


def _propose(gen, v, cur, end, rng):
    i = DIA.index(cur)
    lo, hi = RANGE[v]
    mid = (lo + hi) / 2
    cand, w = [], []
    for k, p in STEP_P.items():
        j = i + k
        if not 0 <= j < len(DIA) or not lo <= DIA[j] <= hi:
            continue
        m = DIA[j]
        if (cur - mid > 4 and k > 0) or (mid - cur > 4 and k < 0):
            p *= 0.5
        cand.append(m)
        w.append(p * _deg_w(gen, v, m % 12, end))
    return rng.choices(cand, w)[0]


def _parallel(a0, b0, a1, b1):
    da, db = a1 - a0, b1 - b0
    if da == 0 or db == 0 or (da > 0) != (db > 0):
        return False
    return (a0 - b0) % 12 in (0, 7) and (a1 - b1) % 12 == (a0 - b0) % 12


def _ok(gen, prev, nxt, rng):
    if gen in ('S1', 'S2'):
        return True
    for x in range(4):
        for y in range(x + 1, 4):
            vx, vy = VOICES[x], VOICES[y]
            if _parallel(prev[vx], prev[vy], nxt[vx], nxt[vy]):
                return False
    ds, db = nxt['S'] - prev['S'], nxt['B'] - prev['B']
    if ds and db and (ds > 0) == (db > 0) and rng.random() < 0.6:
        return False
    if gen == 'S4':
        b = nxt['B'] % 12
        if not any((nxt[v] - b) % 12 in (3, 4) for v in ('S', 'A', 'T')):
            return False
        if sum(nxt[v] % 12 == 11 for v in VOICES) > 1:
            return False
    return True


def gen_events(gen, rng):
    cur = {v: min(RANGE[v][1], max(RANGE[v][0], START[v] + 2 * rng.randint(-2, 2))) for v in VOICES}
    cur = {v: min(DIA, key=lambda m: abs(m - cur[v])) for v in VOICES}
    ev = []
    for _ph in range(N_PHRASE):
        for q in range(PHRASE_Q + 1):
            end = q == PHRASE_Q
            nxt = None
            for _ in range(500):
                nxt = {v: _propose(gen, v, cur[v], end, rng) for v in VOICES}
                if _ok(gen, cur, nxt, rng):
                    break
            cur = nxt
            ev.append((cur['S'], cur['A'], cur['T'], cur['B'], 2.0 if end else 1.0))
    return ev


def make_pieces(gen, n, seed):
    """合成谱都在 C 自然音阶上写成，主音已知：解析时固定为 C 大调，跨曲重组不改变音级。"""
    rng = random.Random(seed)
    est = ras_mod.estimate_tonic
    ras_mod.estimate_tonic = lambda _m: (0, 'major')
    try:
        out = []
        for i in range(n):
            sc = _satb_score(gen_events(gen, rng), title=f'{gen}-{seed}-{i}')
            out.append(from_music21(sc, title=f'{gen}-{seed}-{i}', t_max=None))
    finally:
        ras_mod.estimate_tonic = est
    return out


def recombine(pieces, rng):
    """新曲 i 的 S 取自曲 i，A/T/B 各取自独立随机排列下的另一首（同长度、同节奏）。"""
    n = len(pieces)
    perm = {}
    for v in ('A', 'T', 'B'):
        while True:
            p = list(range(n))
            rng.shuffle(p)
            if all(p[i] != i for i in range(n)):
                break
        perm[v] = p
    out = []
    for i, pc in enumerate(pieces):
        src = {'S': pc, **{v: pieces[perm[v][i]] for v in ('A', 'T', 'B')}}
        L = min(len(pc.slices), *(len(src[v].slices) for v in VOICES))
        new = []
        for k in range(L):
            s = pc.slices[k]
            midis = {v: src[v].slices[k].midis[v] for v in VOICES}
            deg = {v: None if midis[v] is None else (midis[v] - pc.tonic_pc) % 12 for v in VOICES}
            b = midis['B']
            new.append(replace(s, midis=midis, deg=deg, bass_deg=None if b is None else (b - pc.tonic_pc) % 12))
        out.append(replace(pc, slices=new))
    return out


def _nr(fields):
    return {(fld, r[0]): r[2] for fld, rows in fields.items() if not fld.startswith('_') for r in rows}


def truth_labels(tr, big, seed):
    """dep：原谱与重组谱计数显著不同且比值超出 [2/3, 3/2]；indep：|z|<2 或比值在 [0.8, 1.25] 内。"""
    a = _nr(fri.holdout_fields(tr, big, seed=seed))
    b = _nr(fri.holdout_fields(tr, recombine(big, random.Random(seed + 7)), seed=seed))
    lab = {}
    for k in set(a) | set(b):
        x, y = a.get(k, 0), b.get(k, 0)
        z = (x - y) / math.sqrt(x + y + 1)
        r = (x + 1) / (y + 1)
        if abs(z) > 3 and not 2 / 3 <= r <= 1.5:
            lab[k] = 'avoid' if x < y else 'prefer'
        elif abs(z) < 2 or 0.8 <= r <= 1.25:
            lab[k] = 'indep'
        else:
            lab[k] = 'grey'
    return lab


def evaluate(gen, tr, hold, truth, cfg, seed):
    f = fri.holdout_fields(tr, hold, seed=seed)
    und, ov = eval_disc(f, cfg)
    disc = [('under', fl, c) for fl, c in sorted(und)] + [('over', fl, c) for fl, c in sorted(ov)]
    zr = annotate_z(f)
    ops = {'rot': lambda: [desync_rot(p, random.Random(99 + i)) for i, p in enumerate(hold)],
           'metric': lambda: [desync_metric(p, random.Random(99 + i)) for i, p in enumerate(hold)],
           'recombine': lambda: recombine(hold, random.Random(99))}
    res = {'n_under': len(und), 'n_over': len(ov), 'by_op': {}}
    for op, mk in ops.items():
        zd = annotate_z(fri.holdout_fields(tr, mk(), seed=seed))
        lab = {d: classify(d[1], SGN[d[0]] * zr[d[1]].get(d[2], 0.0), SGN[d[0]] * zd.get(d[1], {}).get(d[2], 0.0))
               for d in disc}
        r = {}
        for pol in ('under', 'over'):
            want = 'avoid' if pol == 'under' else 'prefer'
            vert = [d for d in disc if d[0] == pol and lab[d] in ('typeI', 'typeII')]
            t1 = [d for d in vert if lab[d] == 'typeI']
            t2 = [d for d in vert if lab[d] == 'typeII']
            tl = lambda d: truth.get((d[1], d[2]), 'indep')
            dep = [d for d in vert if tl(d) == want]
            r[pol] = {
                'n_vertical': len(vert), 'n_typeI': len(t1), 'n_typeII': len(t2),
                'n_true_dep': len(dep),
                'typeI_true': sum(tl(d) == want for d in t1), 'typeI_false': sum(tl(d) == 'indep' for d in t1),
                'typeII_true': sum(tl(d) == 'indep' for d in t2), 'typeII_false': sum(tl(d) == want for d in t2),
                'dep_found_typeI': sum(lab[d] == 'typeI' for d in dep),
            }
        planted = {}
        for name in PLANTED[gen]:
            _n, fld, pol, pred = next(x for x in PREDS if x[0] == name)
            hits = [d for d in disc if d[1] == fld and d[0] == pol and pred(d[2], fld)]
            planted[name] = {'n_disc': len(hits), 'classes': dict(Counter(lab[d] for d in hits))}
        r['planted'] = planted
        res['by_op'][op] = r
    # 真值集合本身：大语料上依赖格子有多少被发现
    vert_dep = {k: v for k, v in truth.items() if k[0] in VERT and v in ('avoid', 'prefer')}
    found = {(fl, c) for _p, fl, c in disc}
    res['truth_dep_cells'] = len(vert_dep)
    res['truth_dep_found'] = sum(k in found for k in vert_dep)
    res['within_dep_cells'] = sum(1 for k, v in truth.items() if k[0] in WITHIN and v in ('avoid', 'prefer'))
    return res


def main(n_rep=5):
    with open('runs/induction/paper/final_eval.json') as fh:
        cfg = json.load(fh)['cfg']
    out = {'commit': _git_head(), 'cfg': cfg, 'n_train': N_TRAIN, 'n_hold': N_HOLD, 'n_truth': N_TRUTH,
           'n_rep': n_rep, 'gens': {}}
    for g, gen in enumerate(GENS):
        t0 = time.time()
        tr0 = make_pieces(gen, N_TRAIN, 1000 + g)
        big = make_pieces(gen, N_TRUTH, 2000 + g)
        truth = truth_labels(tr0, big, seed=0)
        tc = Counter(v for k, v in truth.items() if k[0] in VERT)
        print(f'{gen} truth {dict(tc)} within-dep '
              f'{sum(1 for k, v in truth.items() if k[0] in WITHIN and v in ("avoid", "prefer"))} '
              f'{time.time() - t0:.0f}s', flush=True)
        reps = []
        for r in range(n_rep):
            tr = make_pieces(gen, N_TRAIN, 3000 + 100 * g + r)
            hold = make_pieces(gen, N_HOLD, 4000 + 100 * g + r)
            x = evaluate(gen, tr, hold, truth, cfg, seed=r)
            reps.append(x)
            rr = x['by_op']['rot']
            print(f'  {gen} rep {r} u/o {x["n_under"]}/{x["n_over"]} '
                  f'rot u {rr["under"]} o {rr["over"]} planted {rr["planted"]} {time.time() - t0:.0f}s', flush=True)
        out['gens'][gen] = {'truth_counts': dict(tc), 'reps': reps}
    _dump('synthetic', out)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
