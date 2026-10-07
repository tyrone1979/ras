# -*- coding: utf-8 -*-
"""冻结方法的敏感性分析（评测集：留出 A；不改方法代码）。

  venv/bin/python -m score_learning.induction.sensitivity [ksize|replay|sparse ...]

ksize   训练众赞歌数 k 的学习曲线：每个训练集各自在时移谱上校准，报告平均原子 AUC、特异度差、发现集规模。
replay  重放次数 R 的影响（训练集同 final_eval.json）。
sparse  去掉第一阶段的稀疏过密规则（z>=1.15）后的发现集；排序指标不受任何阈值规则影响。
"""
from __future__ import annotations

import json
import random
import sys
import time

from score_learning.induction import bootstrap as bs
from score_learning.induction import fri
from score_learning.induction.baselines import EVAL_FIELDS
from score_learning.induction.fri import SPARSE_FIELDS, annotate_z, calibrate_from_desync, discoveries
from score_learning.induction.paper import _desync_piece, _git_head, load_chorale_pool, load_palestrina

OUT = 'runs/induction/paper/sensitivity.json'
KS = (1, 4, 8, 16, 32, 64, 128, 'all')
METHODS3 = {m: bs.METHODS[m] for m in ('ras', 'pmi', 'rarity')}


def _null_k(f: str, k: int):
    g = getattr(fri, f'null_{f.lower()}_counts')
    if f in ('V', 'O', 'W'):
        return lambda d, dp, pc, rng: g(d, dp, rng, k=k)
    if f in ('P', 'Q', 'J', 'F'):
        return lambda d, dp, pc, rng: g(d, rng, k=k)
    return lambda d, dp, pc, rng: g(d, pc, rng, k=k)


def piece_counts_r(train, hold, seed: int, k: int):
    dp, pc = fri.catalogs(train)
    rng = random.Random(seed)
    out = [{} for _ in hold]
    for f in fri.ALL_FIELDS:
        nf = _null_k(f, k)
        for row, p in zip(out, hold):
            row[f] = (getattr(fri, f'real_{f.lower()}_counts')([p]), nf([p], dp, pc, rng))
    return out


def _disc_summary(f, cfg):
    und, ov = discoveries(f, cfg['z_cut'], cfg['q'], soft=True, soft_z=cfg.get('soft_z'))
    keep = lambda s: sum(1 for fld, _c in s if fld in EVAL_FIELDS)
    return {'n_under': keep(und), 'n_over': keep(ov)}


def run_ksize(by, usable, fin, hold, hold_d):
    pool = usable[:-40]
    res = []
    for k in KS:
        kk = len(pool) if k == 'all' else k
        n_seed = 5 if kk <= 64 else (3 if kk <= 128 else 1)
        for s in range(n_seed):
            t0 = time.time()
            ids = random.Random(1000 + 17 * kk + s).sample(pool, kk)
            tr = [by[i] for i in ids]
            cfg = calibrate_from_desync(tr, seed=0)
            per, per_d = bs.piece_counts(tr, hold, 0), bs.piece_counts(tr, hold_d, 0)
            full = list(range(len(hold)))
            f, fd = bs.fields_from(per, full), bs.fields_from(per_d, full)
            st = bs.draw_stats(f, fd, METHODS3)
            row = {'k': kk, 'seed': s, 'z_cut': cfg['z_cut'], 'q': cfg['q'],
                   **{f'{m}_{x}': st[m][x] for m in METHODS3 for x in ('auc_common', 'vert_gap')},
                   **_disc_summary(f, cfg)}
            res.append(row)
            print(f'  k={kk:<4} s={s} AUC ras {row["ras_auc_common"]:.3f} pmi {row["pmi_auc_common"]:.3f} '
                  f'gap ras {row["ras_vert_gap"]:+.3f}  disc u/o {row["n_under"]}/{row["n_over"]}  '
                  f'{time.time() - t0:.0f}s', flush=True)
    return res


def run_replay(by, fin, hold, hold_d):
    tr = [by[i] for i in fin['train_ids']]
    res = []
    for r in (4, 8, 16, 32, 64):
        for s in range(3):
            per, per_d = piece_counts_r(tr, hold, 100 + s, r), piece_counts_r(tr, hold_d, 100 + s, r)
            full = list(range(len(hold)))
            st = bs.draw_stats(bs.fields_from(per, full), bs.fields_from(per_d, full), METHODS3)
            row = {'R': r, 'seed': s, **{f'{m}_{x}': st[m][x] for m in METHODS3
                                         for x in ('auc_common', 'vert_gap', 'p50_under')}}
            res.append(row)
            print(f'  R={r:<3} s={s} AUC ras {row["ras_auc_common"]:.3f} pmi {row["pmi_auc_common"]:.3f} '
                  f'gap ras {row["ras_vert_gap"]:+.3f}', flush=True)
    return res


def _over_without_sparse(f, cfg):
    """去掉“稀疏场 z>=1.15、n_r>=3、rho>0”规则单独加入的过密格。"""
    _und, ov = discoveries(f, cfg['z_cut'], cfg['q'], soft=True, soft_z=cfg.get('soft_z'))
    _uh, ov_hard = discoveries(f, cfg['z_cut'], cfg['q'], soft=False)
    zmap = annotate_z(f)
    out = set()
    for fld, cell in ov:
        row = next(r for r in f[fld] if r[0] == cell)
        sparse_only = (fld in SPARSE_FIELDS and zmap[fld][cell] >= 1.15 and row[2] >= 3 and row[1] > 0
                       and (fld, cell) not in ov_hard and not (row[3] == 0 and row[2] >= 3 and row[1] >= 0.8))
        if not sparse_only:
            out.add((fld, cell))
    return out


def run_sparse(by, fin, sets):
    tr = [by[i] for i in fin['train_ids']]
    cfg = fin['cfg']
    preds = fri.book_predicates()
    res = {}
    for name, (hold, seed) in sets.items():
        f = fri.holdout_fields(tr, hold, seed=seed)
        und, ov = discoveries(f, cfg['z_cut'], cfg['q'], soft=True, soft_z=cfg.get('soft_z'))
        ov_ns = _over_without_sparse(f, cfg)
        row = {}
        for tag, over in (('with', ov), ('without', ov_ns)):
            hits = {n: any(p(c, fld) for ff, c in (und if pol == 'under' else over) if ff == fld)
                    for n, fld, pol, p in preds if fld in EVAL_FIELDS}
            cov = [any(p(c, ff) for _n, fld, pol, p in preds if fld == ff and pol == 'over')
                   for ff, c in over if ff in EVAL_FIELDS]
            row[tag] = {'n_over': len(cov), 'precision_over': sum(cov) / max(len(cov), 1),
                        'completeness_eval': sum(hits.values()) / len(hits),
                        'miss': sorted(n for n, v in hits.items() if not v)}
        res[name] = row
        print(f'  {name:<11} over with {row["with"]["n_over"]} prec {row["with"]["precision_over"]:.2f} '
              f'comp {row["with"]["completeness_eval"]:.3f} | without {row["without"]["n_over"]} '
              f'prec {row["without"]["precision_over"]:.2f} comp {row["without"]["completeness_eval"]:.3f} '
              f'miss+ {sorted(set(row["without"]["miss"]) - set(row["with"]["miss"]))}', flush=True)
    return res


def main():
    parts = [a for a in sys.argv[1:] if a in ('ksize', 'replay', 'sparse')] or ['ksize', 'replay', 'sparse']
    with open(bs.FINAL) as fh:
        fin = json.load(fh)
    by, usable, _ = load_chorale_pool()
    hold = [by[i] for i in fin['hold_ids']]
    rng_d = random.Random(99)
    hold_d = [_desync_piece(p, rng_d) for p in hold]
    try:
        with open(OUT) as fh:
            out = json.load(fh)
    except FileNotFoundError:
        out = {}
    out['commit'] = _git_head()
    if 'ksize' in parts:
        print('== ksize', flush=True)
        out['ksize'] = run_ksize(by, usable, fin, hold, hold_d)
    if 'replay' in parts:
        print('== replay', flush=True)
        out['replay'] = run_replay(by, fin, hold, hold_d)
    if 'sparse' in parts:
        print('== sparse', flush=True)
        sets = {'hold_a': (hold, 0), 'hold_b': ([by[i] for i in fin['hold_b_ids']], 1),
                'palestrina': (load_palestrina(16), 0)}
        out['sparse'] = run_sparse(by, fin, sets)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1, default=str)
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
