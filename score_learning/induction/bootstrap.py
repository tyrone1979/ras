# -*- coding: utf-8 -*-
"""冻结方法下按曲目半抽样留出集，给 RAS-FR 与 PMI / 稀有度基线差值的置信区间。

  venv/bin/python -m score_learning.induction.bootstrap [B]

每首曲的真实/零模型计数只算一次（计数按曲可加），子样本 = 无放回抽 n/2 首曲的计数和。
有放回重抽会复制整曲，稀疏的 K 场终止式格子残差排序因此系统偏移（点估计落在区间外），
故用无放回半抽样：有限总体修正下 n/2 子样本均值方差 ≈ 全样本方差，无需再缩放。
时移版留出谱与原谱用同一组重抽下标，得到纵向原子 AUC 的 real − desync 差（特异度）。
AUC 均值只在本次重抽中三种方法都有定义的共同原子上取。只用与阈值无关的指标。
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

from score_learning.induction import fri
from score_learning.induction.ablation import VERT_FIELDS, uniform_fields
from score_learning.induction.baselines import (EVAL_ATOMS, EVAL_FIELDS, atom_auc, indep_fields, precision_at_k,
                                                productive_fields)
from score_learning.induction.fri import catalogs, residual
from score_learning.induction.paper import _desync_piece, _git_head, load_chorale_pool, load_palestrina

FINAL = 'runs/induction/paper/final_eval.json'
OUT = 'runs/induction/paper/bootstrap.json'

_NULL = {
    'V': lambda d, dp, pc, rng: fri.null_v_counts(d, dp, rng),
    'H': lambda d, dp, pc, rng: fri.null_h_counts(d, pc, rng),
    'M': lambda d, dp, pc, rng: fri.null_m_counts(d, pc, rng),
    'S': lambda d, dp, pc, rng: fri.null_s_counts(d, pc, rng),
    'P': lambda d, dp, pc, rng: fri.null_p_counts(d, rng),
    'O': lambda d, dp, pc, rng: fri.null_o_counts(d, dp, rng),
    'G': lambda d, dp, pc, rng: fri.null_g_counts(d, pc, rng),
    'Q': lambda d, dp, pc, rng: fri.null_q_counts(d, rng),
    'K': lambda d, dp, pc, rng: fri.null_k_counts(d, pc, rng),
    'J': lambda d, dp, pc, rng: fri.null_j_counts(d, rng),
    'D': lambda d, dp, pc, rng: fri.null_d_counts(d, pc, rng),
    'R': lambda d, dp, pc, rng: fri.null_r_counts(d, pc, rng),
    'U': lambda d, dp, pc, rng: fri.null_u_counts(d, pc, rng),
    'W': lambda d, dp, pc, rng: fri.null_w_counts(d, dp, rng),
    'F': lambda d, dp, pc, rng: fri.null_f_counts(d, rng),
}
METHODS = {'ras': lambda f: f, 'pmi': indep_fields, 'rarity': uniform_fields, 'productive': productive_fields}
BASES = [m for m in METHODS if m != 'ras']
# 固定 k 的 precision@k 在半样本里对应不同比例的格子，与全样本不是同一估计量，只报点值。
CI_KEYS = ('auc_common', 'auc_under', 'vert_gap')
_POL = {n: pol for n, _f, pol, _p in fri.book_predicates()}
_VERT = {n for n, f, _pol, _p in fri.book_predicates() if f in VERT_FIELDS}


def piece_counts(train, hold, seed: int = 0) -> List[Dict[str, Tuple[Counter, Counter]]]:
    """场在外层、曲在内层，随机数消耗顺序与 holdout_fields 相同，b=0 复现冻结评测。"""
    dp, pc = catalogs(train)
    rng = random.Random(seed)
    out = [{} for _ in hold]
    for f in fri.ALL_FIELDS:
        for row, p in zip(out, hold):
            row[f] = (getattr(fri, f'real_{f.lower()}_counts')([p]), _NULL[f]([p], dp, pc, rng))
    return out


def fields_from(per: List[Dict[str, Tuple[Counter, Counter]]], idx: List[int]) -> Dict[str, list]:
    out = {}
    for f in fri.ALL_FIELDS:
        real, null = Counter(), Counter()
        for i in idx:
            real.update(per[i][f][0])
            null.update(per[i][f][1])
        out[f] = residual(real, null)
    return out


def _mean(auc: dict, atoms) -> float:
    xs = [auc[a] for a in atoms]
    return sum(xs) / len(xs) if xs else float('nan')


def draw_stats(f: dict, fd: dict, methods: dict = METHODS) -> Dict[str, Dict[str, float]]:
    tf = {m: t(f) for m, t in methods.items()}
    auc = {m: atom_auc(x) for m, x in tf.items()}
    auc_d = {m: atom_auc(t(fd)) for m, t in methods.items()}
    common = EVAL_ATOMS & set.intersection(*[{a for a, x in v.items() if x is not None} for v in auc.values()])
    vert = common & _VERT & set.intersection(
        *[{a for a, x in v.items() if x is not None} for v in auc_d.values()])
    under = {a for a in common if _POL[a] == 'under'}
    out = {}
    for m in methods:
        pk = precision_at_k(tf[m], flds=EVAL_FIELDS)
        out[m] = {
            'auc_common': _mean(auc[m], common),
            'auc_under': _mean(auc[m], under),
            'vert_gap': _mean(auc[m], vert) - _mean(auc_d[m], vert),
            'p50_under': pk['under'][50], 'p100_under': pk['under'][100],
            'p50_over': pk['over'][50], 'p100_over': pk['over'][100],
        }
    return out


def _ci(xs: List[float]) -> Tuple[float, float]:
    s = sorted(xs)
    return s[int(0.025 * (len(s) - 1))], s[int(0.975 * (len(s) - 1))]


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    with open(FINAL) as fh:
        fin = json.load(fh)
    by, _usable, _ = load_chorale_pool()
    tr = [by[i] for i in fin['train_ids']]
    sets = {
        'hold_a': ([by[i] for i in fin['hold_ids']], 0),
        'hold_b': ([by[i] for i in fin['hold_b_ids']], 1),
        'palestrina': (load_palestrina(16), 0),
    }
    out = {'commit': _git_head(), 'final_commit': fin['commit'], 'n_boot': n_boot,
           'scheme': 'half-sampling without replacement', 'sets': {}}
    for name, (hold, seed) in sets.items():
        t0 = time.time()
        rng_d = random.Random(99)
        per = piece_counts(tr, hold, seed)
        per_d = piece_counts(tr, [_desync_piece(p, rng_d) for p in hold], seed)
        rng = random.Random(2026)
        draws = []
        for b in range(n_boot + 1):
            idx = list(range(len(hold))) if b == 0 else rng.sample(range(len(hold)), len(hold) // 2)
            draws.append(draw_stats(fields_from(per, idx), fields_from(per_d, idx)))
            if b % 50 == 0:
                print(f'  {name} b={b} {time.time() - t0:.0f}s', flush=True)
        point, boot = draws[0], draws[1:]
        summary = {}
        for k in point['ras']:
            summary[k] = {m: point[m][k] for m in METHODS}
            for base in BASES:
                summary[k][f'diff_{base}'] = point['ras'][k] - point[base][k]
            s = summary[k]
            line = f'  {name:<11}{k:<11} ' + ' '.join(f'{m} {s[m]:.3f}' for m in METHODS)
            if k in CI_KEYS:
                s.update({f'{m}_ci': _ci([x[m][k] for x in boot]) for m in METHODS})
                for base in BASES:
                    d = [x['ras'][k] - x[base][k] for x in boot]
                    s[f'diff_{base}_ci'] = _ci(d)
                    s[f'p_le0_{base}'] = sum(v <= 0 for v in d) / len(d)
                line += ''.join(f'  -{b} {s["diff_" + b]:+.3f} [{s["diff_" + b + "_ci"][0]:+.3f},'
                                f'{s["diff_" + b + "_ci"][1]:+.3f}]' for b in BASES)
            print(line, flush=True)
        out['sets'][name] = {'n': len(hold), 'summary': summary}
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
