# -*- coding: utf-8 -*-
"""零模型逐层恢复诊断（只读、只诊断；不改 rasfr-frozen 的任何方法代码，不实现新方法）。

问题：冻结 replay 的期望计数与“只破坏声部间音高协调”的反事实相差多少，差在哪一层？

单位：相邻两个完整切片 (a, b)。L1–L4 只在 b 处重抽声部，a 取真实值。
  L0  冻结 null（fri.null_*_counts 原样调用）
  L1  + 起音掩码：b 处无起音的声部保持 a 的音高；有起音的声部按该场原有方式抽
      （V：步进目录；H/S/U：音高目录）
  L2  起音处一律“a 的真实音高 + 步进目录独立抽”（V 场与 L1 相同）
  L3  + 状态：步进按 a 的音高在该声部训练音域中的五分位条件抽
  L3D 状态改为 a 的音级（相对主音的音高级），检验步进大小是否由音阶位置决定
  L4  + 转移：音区五分位 + 音级 + 前一步类别（样本 < MIN_COND 时逐级退回）
参照（反事实 / 对照）：
  CF   跨曲重组（仅合成数据：所有曲目共享节奏与乐句网格，保留 D1+D2，只破坏 D3）
  ROT  冻结的独立旋转；MET 保拍旋转

  venv/bin/python -m score_learning.diagnostics.null_ladder [n_rep]
"""
from __future__ import annotations

import bisect
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict

from score_learning.induction import fri
from score_learning.induction import synthetic_truth as st
from score_learning.induction.fri import PAIRS, VOICES
from score_learning.induction.paper import _git_head
from score_learning.induction.revision import PREDS, SEEDS, desync_metric, desync_rot, load_sets

R = 16
MIN_COND = 20
TARGETS = ('A1', 'A2', 'A5', 'A11', 'A12', 'E2')
FIELDS = ('V', 'H', 'S', 'U')
LEVELS = ('L0', 'L1', 'L2', 'L3', 'L3D', 'L4')
H_PAIRS = (('S', 'A'), ('A', 'T'), ('T', 'B'))
OUT = 'runs/induction/diagnostics/null_ladder.json'


# ---------------------------------------------------------------------------- cells of one unit

def _complete(s):
    return all(s.midis[v] is not None for v in VOICES)


def unit_cells(a_m, b_m, tonic):
    """一个单位 (a, b) 在四个场里的格子；V 用 a→b，H/S/U 用 b。"""
    out = []
    for va, vb in PAIRS:
        dpa, dpb = b_m[va] - a_m[va], b_m[vb] - a_m[vb]
        out.append(('V', fri.v_cell(dpa, dpb, a_m[va] - a_m[vb], b_m[va] - b_m[vb], (va, vb))))
    for va, vb in H_PAIRS:
        out.append(('H', fri.h_cell(va, vb, b_m[va], b_m[vb])))
    out.append(('S', fri._set_from_midis(b_m, tonic)))
    out.append(('U', fri.u_cell(b_m, tonic)))
    return out


def units(p):
    sl = p.slices
    return [(sl[i - 1], sl[i]) for i in range(1, len(sl)) if _complete(sl[i - 1]) and _complete(sl[i])]


def _prev_class(dp):
    if dp is None:
        return 'none'
    if dp == 0:
        return 'rep'
    if abs(dp) <= 2:
        return 'su' if dp > 0 else 'sd'
    return 'lu' if dp > 0 else 'ld'


def prev_steps(p):
    """每个切片处、每个声部此前最后一次起音的步进（不含当前切片）。"""
    out, last, cur = [], {v: None for v in VOICES}, {v: None for v in VOICES}
    for s in p.slices:
        out.append(dict(last))
        for v in VOICES:
            m = s.midis[v]
            if m is None:
                cur[v] = None
                continue
            if s.attack.get(v) and cur[v] is not None:
                last[v] = m - cur[v]
            cur[v] = m
    return out


# ---------------------------------------------------------------------------- catalogues from training

class Cats:
    def __init__(self, tr):
        self.dp, self.pitch = fri.catalogs(tr)
        self.cut = {}
        for v in VOICES:
            xs = sorted(self.pitch[v])
            self.cut[v] = [xs[int(len(xs) * q / 5)] for q in range(1, 5)]
        self.by_state = defaultdict(list)
        self.by_deg = defaultdict(list)
        self.by_state_prev = defaultdict(list)
        for p in tr:
            pv = prev_steps(p)
            idx = {id(s): i for i, s in enumerate(p.slices)}
            for a, b in units(p):
                for v in VOICES:
                    if not b.attack.get(v):
                        continue
                    dp = b.midis[v] - a.midis[v]
                    st_ = self.state(v, a.midis[v])
                    dg = (a.midis[v] - p.tonic_pc) % 12
                    self.by_state[(v, st_)].append(dp)
                    self.by_deg[(v, dg)].append(dp)
                    self.by_state_prev[(v, st_, dg, _prev_class(pv[idx[id(b)]][v]))].append(dp)

    def state(self, v, m):
        return bisect.bisect_right(self.cut[v], m)

    def draw(self, level, v, a_m, prev_dp, rng, tonic=0):
        if level in ('L1', 'L2'):
            return rng.choice(self.dp[v])
        s, dg = self.state(v, a_m), (a_m - tonic) % 12
        if level == 'L4':
            c = self.by_state_prev.get((v, s, dg, _prev_class(prev_dp)), [])
            if len(c) >= MIN_COND:
                return rng.choice(c)
        if level in ('L3D', 'L4'):
            c = self.by_deg.get((v, dg), [])
            if len(c) >= MIN_COND:
                return rng.choice(c)
        c = self.by_state.get((v, s), [])
        return rng.choice(c) if len(c) >= MIN_COND else rng.choice(self.dp[v])


# ---------------------------------------------------------------------------- counting

def real_counts(pieces):
    c, n = Counter(), 0
    for p in pieces:
        for a, b in units(p):
            n += 1
            c.update(unit_cells(a.midis, b.midis, p.tonic_pc))
    return c, n


def ladder_counts(pieces, cats, level, rng):
    """返回 R 次重放各自的计数，以及单位数。"""
    per = [Counter() for _ in range(R)]
    n = 0
    for p in pieces:
        pv = prev_steps(p)
        idx = {id(s): i for i, s in enumerate(p.slices)}
        for a, b in units(p):
            n += 1
            prev = pv[idx[id(b)]]
            for r in range(R):
                bm = {}
                for v in VOICES:
                    if not b.attack.get(v):
                        bm[v] = a.midis[v]
                    elif level == 'L1':
                        bm[v] = None
                    else:
                        bm[v] = a.midis[v] + cats.draw(level, v, a.midis[v], prev[v], rng, p.tonic_pc)
                cells = []
                if level == 'L1':
                    vm = {v: (bm[v] if bm[v] is not None else a.midis[v] + rng.choice(cats.dp[v])) for v in VOICES}
                    pm = {v: (bm[v] if bm[v] is not None else rng.choice(cats.pitch[v])) for v in VOICES}
                    cells += [x for x in unit_cells(a.midis, vm, p.tonic_pc) if x[0] == 'V']
                    cells += [x for x in unit_cells(a.midis, pm, p.tonic_pc) if x[0] != 'V']
                else:
                    cells = unit_cells(a.midis, bm, p.tonic_pc)
                per[r].update(cells)
    return per, n


def l0_counts(pieces, cats, rng):
    """冻结 null：每个场按 fri 原函数，返回 R 次的平均计数（原函数一次给出 R 次之和）与各场单位数。"""
    tot = Counter()
    nv = fri.null_v_counts(pieces, cats.dp, rng)
    tot.update({('V', k): v for k, v in nv.items()})
    for fld, fn in (('H', fri.null_h_counts), ('S', fri.null_s_counts), ('U', fri.null_u_counts)):
        tot.update({(fld, k): v for k, v in fn(pieces, cats.pitch, rng).items()})
    return tot


def field_units(cnt):
    n = Counter()
    for (fld, _c), v in cnt.items():
        n[fld] += v
    return n


# ---------------------------------------------------------------------------- targets

PRED = {name: (fld, pol, pred) for name, fld, pol, pred in PREDS}


def target_count(cnt, name):
    fld, _pol, pred = PRED[name]
    return sum(v for (f, c), v in cnt.items() if f == fld and pred(c, fld))


def n_target_cells(cnt, name):
    fld, _pol, pred = PRED[name]
    return sum(1 for (f, c), v in cnt.items() if f == fld and v > 0 and pred(c, fld))


def motion_exposure(cnt):
    """V 场里两声部同时移动的 (单位, 声部对) 数。"""
    return sum(v for (f, c), v in cnt.items() if f == 'V' and c.startswith('both=1'))


def _lr(x, y):
    return math.log((x + 0.5) / (y + 0.5))


def summarise(real, n_real, nulls, refs):
    """nulls：level -> (每次重放计数列表 或 平均计数, 各场单位数)；refs：name -> (计数, 各场单位数)。
    所有期望值换算到真实数据的各场单位数。"""
    fu_real = field_units(real)
    out = {}
    for t in TARGETS:
        fld = PRED[t][0]
        nr = target_count(real, t)
        row = {'field': fld, 'polarity': PRED[t][1], 'n_real': nr, 'exposure_real': fu_real[fld],
               'cells_real': n_target_cells(real, t), 'nulls': {}, 'refs': {}}
        for name, (cnt, fu) in refs.items():
            lam = target_count(cnt, t) * fu_real[fld] / max(fu[fld], 1)
            row['refs'][name] = {'count': lam, 'cells': n_target_cells(cnt, t), 'log_real_over': _lr(nr, lam),
                                 'motion_exposure': motion_exposure(cnt) * fu_real['V'] / max(fu['V'], 1)}
        cf = row['refs'].get('CF', {}).get('count')
        for lev, (per, fu) in nulls.items():
            if isinstance(per, list):
                xs = [target_count(c, t) * fu_real[fld] / max(fu[fld], 1) for c in per]
                lam, sd = statistics.mean(xs), statistics.stdev(xs)
                mot = statistics.mean(motion_exposure(c) for c in per) * fu_real['V'] / max(fu['V'], 1)
                cells = statistics.mean(n_target_cells(c, t) for c in per)
            else:
                lam = target_count(per, t) * fu_real[fld] / max(fu[fld], 1)
                sd, cells = None, n_target_cells(per, t)
                mot = motion_exposure(per) * fu_real['V'] / max(fu['V'], 1)
            z = (nr - (lam + 0.5)) / math.sqrt(lam + 1.0)
            d = {'null_mean': lam, 'null_sd': sd, 'z': z, 'log_real_over_null': _lr(nr, lam),
                 'cells': cells, 'motion_exposure': mot}
            if cf is not None:
                d['log_cf_over_null'] = _lr(cf, lam)
                d['log_real_over_cf'] = _lr(nr, cf)
            row['nulls'][lev] = d
        out[t] = row
    return out


def run_set(tr, hold, seed, cf_pieces=None):
    cats = Cats(tr)
    rng = random.Random(seed)
    real, n_real = real_counts(hold)
    nulls = {}
    l0 = l0_counts(hold, cats, rng)
    nulls['L0'] = (l0, field_units(l0))
    for lev in LEVELS[1:]:
        per, _n = ladder_counts(hold, cats, lev, rng)
        nulls[lev] = (per, field_units(per[0]))
    refs = {}
    for name, mk in (('ROT', lambda: [desync_rot(p, random.Random(99 + i)) for i, p in enumerate(hold)]),
                     ('MET', lambda: [desync_metric(p, random.Random(99 + i)) for i, p in enumerate(hold)])):
        c, _ = real_counts(mk())
        refs[name] = (c, field_units(c))
    if cf_pieces is not None:
        c, _ = real_counts(cf_pieces)
        refs['CF'] = (c, field_units(c))
    return summarise(real, n_real, nulls, refs)


def _print(tag, res):
    for t, row in res.items():
        cf = row['refs'].get('CF', {}).get('count')
        s = ' '.join(f'{lev} {d["null_mean"]:.1f}' for lev, d in row['nulls'].items())
        r = ' '.join(f'{k} {v["count"]:.1f}' for k, v in row['refs'].items())
        print(f'  {tag:<10} {t:<4} real {row["n_real"]:>5} | {s} | {r}', flush=True)


def main(n_rep=3):
    t0 = time.time()
    out = {'commit': _git_head(), 'R': R, 'levels': LEVELS, 'targets': TARGETS, 'synthetic': {}, 'bach': {}}
    for g, gen in enumerate(st.GENS):
        reps = []
        for r in range(n_rep):
            tr = st.make_pieces(gen, st.N_TRAIN, 3000 + 100 * g + r)
            hold = st.make_pieces(gen, st.N_HOLD, 4000 + 100 * g + r)
            cf = st.recombine(hold, random.Random(99 + r))
            res = run_set(tr, hold, seed=r, cf_pieces=cf)
            _print(f'{gen} r{r}', res)
            reps.append(res)
        out['synthetic'][gen] = reps
        print(f'{gen} done {time.time() - t0:.0f}s', flush=True)
    fin, tr, sets = load_sets(['hold_a', 'hold_c'])
    for name, hold in sets.items():
        res = run_set(tr, hold, seed=SEEDS[name])
        _print(name, res)
        out['bach'][name] = res
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'wrote {OUT} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
