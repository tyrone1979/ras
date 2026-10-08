# -*- coding: utf-8 -*-
"""合成真值 E* = E_{∏_v P_v}（预注册 `dcra-prereg` 第 4 节）。评测代码，不属于 DCRA 方法。

对每个生成器：用全新种子 9000 + g 生成 N_POOL ≥ 10,000 个样本（与任何被分析数据不相交）；
每个组合的四个声部取自四个不同样本，组成 N_COMBO ≥ 10,000 个跨样本组合；
E*(c) = 组合上格子/原子计数的均值（每首），SE_MC = sd / sqrt(N_COMBO)。
S1–S4、S6 中所有样本的 D2 相同（组合前逐一断言）；S5 的 D3 = 0，真值 L_D* = 0，不估计。

另报告对照量（不是真值）：前 N_CHECK 个原样本的真实计数均值，用于核对 S1/S2/S6 中 E* 与真实均值一致。

  venv/bin/python -m score_learning.dcra.estar [S1 S2 ...]
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

from score_learning.dcra import canon, synth
from score_learning.induction import fri
from score_learning.induction.fri import VOICES
from score_learning.induction.paper import _git_head
from score_learning.induction.revision import PREDS

N_POOL = 10_000
N_COMBO = 10_000
N_CHECK = 2_000
GENS_ESTAR = ('S1', 'S2', 'S3', 'S4', 'S6')
FIELDS = ('V', 'O', 'H', 'S', 'U', 'K', 'J', 'M', 'G', 'D', 'P', 'Q')
RELATIONAL = ('V', 'O', 'H', 'S', 'U', 'K', 'J')
COUNTERS = {f: getattr(fri, f'real_{f.lower()}_counts') for f in FIELDS}
ATOMS = [(n, f, pol, pred) for n, f, pol, pred in PREDS if f in FIELDS]
OUT_DIR = 'runs/dcra/estar'
WORKERS = 7


def pool_seed(gen: str) -> int:
    return 9000 + synth.GENS.index(gen)


def piece_counts(p: canon.Piece):
    """一首曲子的格子计数 {(场, 格子): n} 与原子计数 {原子: n}。"""
    sr = canon.to_ras(p)
    cells: Dict[tuple, int] = {}
    for f, fn in COUNTERS.items():
        for c, n in fn([sr]).items():
            cells[(f, c)] = n
    atoms = {}
    for name, f, _pol, pred in ATOMS:
        atoms[name] = sum(n for (ff, c), n in cells.items() if ff == f and pred(c, f))
    return cells, atoms


def combine(donors: List[canon.Piece], title: str) -> canon.Piece:
    base = donors[0]
    for d in donors[1:]:
        canon.assert_same_d2(base, d)
    p = canon.Piece(title=title, voices={v: d.voices[v] for v, d in zip(VOICES, donors)},
                    bar_starts=base.bar_starts, bar_len=base.bar_len, phrase_ends=base.phrase_ends,
                    tonic_pc=base.tonic_pc, mode=base.mode)
    canon.assert_same_d2(base, p)
    return p


class Acc:
    def __init__(self):
        self.n = 0
        self.s = defaultdict(float)
        self.ss = defaultdict(float)

    def add(self, d: Dict):
        self.n += 1
        for k, x in d.items():
            self.s[k] += x
            self.ss[k] += x * x

    def merge(self, o: 'Acc'):
        self.n += o.n
        for k, x in o.s.items():
            self.s[k] += x
        for k, x in o.ss.items():
            self.ss[k] += x

    def summary(self, keys) -> Dict:
        out = {}
        for k in keys:
            m = self.s.get(k, 0.0) / self.n
            var = max(self.ss.get(k, 0.0) / self.n - m * m, 0.0) * self.n / max(self.n - 1, 1)
            out[k] = {'mean': m, 'sd': math.sqrt(var), 'se_mc': math.sqrt(var / self.n)}
        return out


def _work(args):
    items = args
    a_cells, a_atoms = Acc(), Acc()
    for p in items:
        cells, atoms = piece_counts(p)
        a_cells.add(cells)
        a_atoms.add(atoms)
    return a_cells, a_atoms


def _chunks(xs, k):
    n = math.ceil(len(xs) / k)
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def _parallel(pieces):
    a_cells, a_atoms = Acc(), Acc()
    with ProcessPoolExecutor(WORKERS) as ex:
        for c, a in ex.map(_work, _chunks(pieces, WORKERS * 4)):
            a_cells.merge(c)
            a_atoms.merge(a)
    return a_cells, a_atoms


def estimate(gen: str) -> Dict:
    t0 = time.time()
    seed = pool_seed(gen)
    pool = synth.make_pieces(gen, N_POOL, seed)
    t_gen = time.time() - t0
    rng = random.Random(seed + 100_000)
    combos = []
    for i in range(N_COMBO):
        idx = rng.sample(range(N_POOL), 4)
        combos.append(combine([pool[j] for j in idx], f'{gen}-estar-{i}'))
    c_cells, c_atoms = _parallel(combos)
    r_cells, r_atoms = _parallel(pool[:N_CHECK])
    keys_cells = sorted(set(c_cells.s) | set(r_cells.s))
    atom_names = [n for n, *_ in ATOMS]
    ce, re_ = c_atoms.summary(atom_names), r_atoms.summary(atom_names)
    atoms = {}
    for name, f, pol, _ in ATOMS:
        e, r = ce[name], re_[name]
        atoms[name] = {
            'field': f, 'polarity': pol, 'relational': f in RELATIONAL,
            'estar_per_piece': e['mean'], 'estar_sd': e['sd'], 'estar_se_mc': e['se_mc'],
            'estar_log_se_mc': e['se_mc'] / e['mean'] if e['mean'] > 0 else None,
            'check_real_per_piece': r['mean'], 'check_real_se': r['se_mc'],
            'check_log_real_over_estar': (math.log((r['mean'] + 1e-9) / e['mean'])
                                          if e['mean'] > 0 else None),
        }
    cs, rs = c_cells.summary(keys_cells), r_cells.summary(keys_cells)
    cells = {f'{f}|{c}': {'estar_per_piece': cs[(f, c)]['mean'], 'estar_se_mc': cs[(f, c)]['se_mc'],
                          'check_real_per_piece': rs[(f, c)]['mean']}
             for f, c in keys_cells}
    return {
        'gen': gen, 'pool_seed': seed, 'combo_seed': seed + 100_000, 'n_pool': N_POOL,
        'n_combo': c_atoms.n, 'n_check': r_atoms.n, 'd2': repr(canon.structure(pool[0])[:6]),
        'atoms': atoms, 'cells': cells, 'git': _git_head(),
        'sec_generate': round(t_gen, 1), 'sec_total': round(time.time() - t0, 1),
        'note': 'E* = mean per-piece count over cross-sample voice combinations; SE_MC is Monte '
                'Carlo error of E* only and is never merged into DCRA bootstrap intervals. '
                'check_* is the mean real count of the first N_CHECK pool samples (not truth).',
    }


def load(gen: str) -> Dict:
    with gzip.open(f'{OUT_DIR}/{gen}.json.gz', 'rt', encoding='utf-8') as fh:
        return json.load(fh)


def main(argv):
    gens = argv or list(GENS_ESTAR)
    os.makedirs(OUT_DIR, exist_ok=True)
    for g in gens:
        if g not in GENS_ESTAR:
            raise SystemExit(f'{g}: no E* (S5 truth is L_D* = 0 by construction)')
        res = estimate(g)
        with gzip.open(f'{OUT_DIR}/{g}.json.gz', 'wt', encoding='utf-8') as fh:
            json.dump(res, fh, separators=(',', ':'), ensure_ascii=False)
        print(g, f"n_combo={res['n_combo']} gen={res['sec_generate']}s total={res['sec_total']}s",
              flush=True)
        for name, a in res['atoms'].items():
            if a['relational']:
                print(f"  {name:4s} {a['field']} E*={a['estar_per_piece']:8.3f} "
                      f"±{a['estar_se_mc']:.3f}  real={a['check_real_per_piece']:8.3f}  "
                      f"log(real/E*)={a['check_log_real_over_estar']}", flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
