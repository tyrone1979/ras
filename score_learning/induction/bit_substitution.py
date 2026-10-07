# -*- coding: utf-8 -*-
"""格子消融：把带理论色彩的特征位换成同族同大小的其它取值，看数据是否把教科书取值排在最欠密。

  venv/bin/python -m score_learning.induction.bit_substitution

- V/A1：pc07 = 音程类 ∈ {0,7}；替换为全部 66 个二元音程类集合；
- M/A6：外声部从 11 级出发不上行半音；替换为从 d 级出发（d=0..11）；
- M/A7：从 10 级出发不下行级进；替换为从 d 级出发；
- S/A11：11 级重复；替换为 d 级重复。
统计量与主方法相同（Poisson z，零模型 = 训练目录上的声部分解重放），z 越负越欠密。
"""
from __future__ import annotations

import json
import random
from collections import Counter
from itertools import combinations
from typing import Dict, List

from score_learning.induction.fri import (
    REPLAY, VOICES, _complete_pairs, _voice_changes, catalogs, cell_z,
)
from score_learning.induction.ras import PAIRS

OUT = 'runs/induction/paper/bit_substitution.json'


def _parallel_pcs(pieces, dp_cat, rng) -> Dict[str, Counter]:
    """平行同音程（都动、等距、同向、音程保持）时的音程类计数；'_n' 为全部声部对步数。"""
    real, null = Counter(), Counter()
    for ras in pieces:
        for a, b in _complete_pairs(ras):
            for va, vb in PAIRS:
                dpa, dpb = b.midis[va] - a.midis[va], b.midis[vb] - a.midis[vb]
                real['_n'] += 1
                if dpa != 0 and dpa == dpb:
                    real[(b.midis[va] - b.midis[vb]) % 12] += 1
            for _ in range(REPLAY):
                draw = {v: rng.choice(dp_cat[v]) for v in VOICES}
                for va, vb in PAIRS:
                    null['_n'] += 1
                    if draw[va] != 0 and draw[va] == draw[vb]:
                        null[(a.midis[va] - a.midis[vb]) % 12] += 1
    return {'real': real, 'null': null}


def _moves(pieces, pitch_cat, rng, outer_only: bool) -> Dict[str, Counter]:
    """(出发级, 音程) 计数；零模型同 F_M：下一音从该声部音域库独立抽。"""
    real, null = Counter(), Counter()
    for ras in pieces:
        for v in VOICES:
            if outer_only and v not in ('S', 'B'):
                continue
            for m0, d0, m1, _d1 in _voice_changes(ras, v):
                real[(d0, m1 - m0)] += 1
                for _ in range(REPLAY):
                    x = rng.choice(pitch_cat[v])
                    if len(set(pitch_cat[v])) > 1:
                        while x == m0:
                            x = rng.choice(pitch_cat[v])
                    null[(d0, x - m0)] += 1
    return {'real': real, 'null': null}


def _doubled(pieces, pitch_cat, rng) -> Dict[str, Counter]:
    """每个切片里出现 ≥2 次的音级；零模型同 F_S：四声部独立抽。'_n' 为切片数。"""
    real, null = Counter(), Counter()

    def add(c, midis, tonic):
        c['_n'] += 1
        degs = Counter((midis[v] - tonic) % 12 for v in VOICES)
        for d, n in degs.items():
            if n >= 2:
                c[d] += 1

    for ras in pieces:
        full = [s for s in ras.slices if all(s.midis[v] is not None for v in VOICES)]
        for s in full:
            add(real, s.midis, ras.tonic_pc)
        for _ in range(max(len(full), 1) * REPLAY):
            add(null, {v: rng.choice(pitch_cat[v]) for v in VOICES}, ras.tonic_pc)
    return {'real': real, 'null': null}


def _z(real: Counter, null: Counter, match) -> float:
    nr = sum(n for k, n in real.items() if k != '_n' and match(k))
    n0 = sum(n for k, n in null.items() if k != '_n' and match(k))
    return cell_z(nr, n0, real['_n'] or sum(real.values()), null['_n'] or sum(null.values()))


def _rank(scores: Dict, target) -> dict:
    order = sorted(scores, key=lambda k: scores[k])
    return {'rank': order.index(target) + 1, 'n': len(order), 'z_target': scores[target],
            'top3': [(str(k), round(scores[k], 2)) for k in order[:3]]}


def family_tests(train, hold, seed: int = 0) -> dict:
    dp_cat, pitch_cat = catalogs(train)
    rng = random.Random(seed)
    par = _parallel_pcs(hold, dp_cat, rng)
    mo_out = _moves(hold, pitch_cat, rng, outer_only=True)
    mo_all = _moves(hold, pitch_cat, rng, outer_only=False)
    dbl = _doubled(hold, pitch_cat, rng)
    for c in (mo_out, mo_all):
        for side in ('real', 'null'):
            c[side]['_n'] = sum(c[side].values())

    a1 = {S: _z(par['real'], par['null'], lambda k, S=S: k in S)
          for S in (frozenset(x) for x in combinations(range(12), 2))}
    a6 = {d: _z(mo_out['real'], mo_out['null'], lambda k, d=d: k[0] == d and k[1] != 1)
          for d in range(12)}
    a7 = {d: _z(mo_all['real'], mo_all['null'], lambda k, d=d: k[0] == d and k[1] not in (-1, -2))
          for d in range(12)}
    a11 = {d: _z(dbl['real'], dbl['null'], lambda k, d=d: k == d) for d in range(12)}
    return {
        'A1_pc_pair': _rank(a1, frozenset({0, 7})),
        'A6_from_degree_not_up_semitone': _rank(a6, 11),
        'A7_from_degree_not_down_step': _rank(a7, 10),
        'A11_doubled_degree': _rank(a11, 11),
    }


def main():
    from score_learning.induction.paper import load_chorale_pool, load_palestrina
    by, usable, _ = load_chorale_pool()
    hold_ids, train_ids = usable[-40:], usable[:-40]
    ids16 = random.Random(0).sample(train_ids, 16)
    rest = [i for i in train_ids if i not in set(ids16)]
    hold_b_ids = sorted(random.Random(7).sample(rest, 40))
    tr = [by[i] for i in ids16]
    sets = {'hold_a': [by[i] for i in hold_ids], 'hold_b': [by[i] for i in hold_b_ids],
            'palestrina': load_palestrina(16)}
    out = {}
    for name, hold in sets.items():
        res = family_tests(tr, hold)
        out[name] = res
        for k, v in res.items():
            print(f'{name:<11}{k:<32} rank {v["rank"]}/{v["n"]}  z={v["z_target"]:.2f}  top3 {v["top3"]}',
                  flush=True)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1, default=str)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
