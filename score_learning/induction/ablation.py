# -*- coding: utf-8 -*-
"""零模型消融（同一格子、同一冻结划分），只报与阈值无关的 AUC 与 precision@k。

  venv/bin/python -m score_learning.induction.ablation

- replay：RAS-FR 主方法；
- indep：格内各位独立（PMI 推广）；
- uniform：每场零模型为均匀分布（= 纯稀有度/频度排序）；
- pooled：四声部共用一个音高目录；
- swapped：声部目录互换（S↔B、A↔T），检验零模型是否依赖声部身份。

AUC 只量敏感度：零模型越离谱（例如大量越音域、交叉），违例格越显得「被回避」。
特异度用同一留出谱的声部时移版：纵向协同已被毁掉，纵向原子的 AUC 应回落到 0.5；
real − desync 的差才是可归因于声部协同的信号。均值只在各变体都有定义的共同原子上取，
并排除零模型按教科书对比构造的场（baselines.DESIGNED_NULL_FIELDS）；含这些场的数另存 *_all。
"""
from __future__ import annotations

import json
import random
from collections import Counter

from score_learning.induction.baselines import EVAL_ATOMS, EVAL_FIELDS, atom_auc, indep_fields, precision_at_k
from score_learning.induction.fri import (
    ALL_FIELDS, book_predicates, catalogs, compute_fields, holdout_fields, residual,
)

OUT = 'runs/induction/paper/ablation.json'
VERT_FIELDS = ('V', 'O', 'H', 'S', 'U', 'W', 'K', 'J')
SWAP = {'S': 'B', 'B': 'S', 'A': 'T', 'T': 'A'}


def uniform_fields(fields):
    out = {}
    for f, rows in fields.items():
        if f not in ALL_FIELDS:
            out[f] = rows
            continue
        real = Counter({r[0]: r[2] for r in rows})
        n0 = sum(r[3] for r in rows)
        out[f] = residual(real, Counter({r[0]: n0 / max(len(rows), 1) for r in rows}))
    return out


def swapped_fields(train, hold, seed: int = 0):
    dp, pc = catalogs(train)
    return compute_fields(hold, {v: dp[SWAP[v]] for v in dp}, {v: pc[SWAP[v]] for v in pc},
                          random.Random(seed))


def _variants(tr, hold, seed):
    f = holdout_fields(tr, hold, seed=seed)
    return {
        'replay': f,
        'indep': indep_fields(f),
        'uniform': uniform_fields(f),
        'pooled': holdout_fields(tr, hold, seed=seed, pooled=True),
        'swapped': swapped_fields(tr, hold, seed=seed),
    }


def _mean(auc: dict, atoms) -> float:
    xs = [auc[a] for a in atoms]
    return sum(xs) / len(xs) if xs else float('nan')


def main():
    from score_learning.induction.paper import _desync_piece, load_chorale_pool, load_palestrina
    by, usable, _ = load_chorale_pool()
    hold_ids, train_ids = usable[-40:], usable[:-40]
    ids16 = random.Random(0).sample(train_ids, 16)
    rest = [i for i in train_ids if i not in set(ids16)]
    hold_b_ids = sorted(random.Random(7).sample(rest, 40))
    tr = [by[i] for i in ids16]
    sets = {'hold_a': ([by[i] for i in hold_ids], 0), 'hold_b': ([by[i] for i in hold_b_ids], 1),
            'palestrina': (load_palestrina(16), 0)}
    vert = {n for n, f, _p, _q in book_predicates() if f in VERT_FIELDS}
    out = {}
    for name, (hold, seed) in sets.items():
        rng = random.Random(99)
        hold_d = [_desync_piece(p, rng) for p in hold]
        real = {v: (atom_auc(fv), precision_at_k(fv, flds=EVAL_FIELDS), precision_at_k(fv))
                for v, fv in _variants(tr, hold, seed).items()}
        des = {v: atom_auc(fv) for v, fv in _variants(tr, hold_d, seed).items()}
        common_all = set.intersection(*[{a for a, x in r[0].items() if x is not None} for r in real.values()])
        common = common_all & EVAL_ATOMS
        common_v = set.intersection(
            *[{a for a, x in d.items() if x is not None} for d in des.values()]) & common & vert
        out[name] = {'common_atoms': sorted(common), 'common_atoms_all': sorted(common_all),
                     'vertical_atoms': sorted(common_v), 'variants': {}}
        for v in real:
            auc, pk, pk_all = real[v]
            row = {
                'auc_common': _mean(auc, common),
                'auc_common_all': _mean(auc, common_all),
                'auc_vert_real': _mean(auc, common_v),
                'auc_vert_desync': _mean(des[v], common_v),
                'auc': auc, 'auc_desync': des[v], 'p_at_k': pk, 'p_at_k_all': pk_all,
            }
            row['vert_gap'] = row['auc_vert_real'] - row['auc_vert_desync']
            out[name]['variants'][v] = row
            print(f'{name:<11}{v:<8} AUC(common {len(common)})={row["auc_common"]:.3f}  '
                  f'vert({len(common_v)}) real={row["auc_vert_real"]:.3f} '
                  f'desync={row["auc_vert_desync"]:.3f} gap={row["vert_gap"]:+.3f}  '
                  f'u@50={pk["under"][50]:.2f} o@50={pk["over"][50]:.2f}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1, default=str)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
