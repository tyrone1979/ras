# -*- coding: utf-8 -*-
"""第二轮审稿意见的补充分析（冻结方法代码不变；全部为探索性分析）。

  venv/bin/python -m score_learning.induction.revision attrib [hold_a hold_b hold_c]
  venv/bin/python -m score_learning.induction.revision pvals  [hold_a hold_c]
  venv/bin/python -m score_learning.induction.revision simple <set> [n_half]
  venv/bin/python -m score_learning.induction.revision expert
  venv/bin/python -m score_learning.induction.revision cpack

attrib  归因的稳健性：Type I 阈值 θ（冻结值 1/2）扫描；四种去同步方案（冻结的独立旋转、10 次旋转平均、
        保持拍位的旋转、乐句内旋转）；各方案保留了什么（逐场真实计数的总变差距离、拍位一致率）；
        参考约束的 Type 判定随方案与 AUC 降幅阈值的变化。
pvals   多重检验：在重放零模型下模拟“真实”计数（每单元重放 1 次），零假设对每个格子都成立，
        冻结规则给出的发现全为假发现；由此估计 BH 部分的 FDR，并用模拟 z 的场内经验分布直接构造 p 值、
        重做 BH；(z_cut, q)、z_soft 与校准目标的敏感性；禁止与偏好分别报告。
simple  简化版：合并音高目录 + 去掉规则 7，在合并目录下重新校准；五个评测集上与冻结版和三个基线比较
        （半抽样区间），并给出分层（协同/旋律/位置 × 极性）与逐场 AUC、逐约束配对检验所需的原子 AUC。
expert  专家盲评的逐条结果、留一评审人、按参考集/专家认可/二者并集计的 P@20。
cpack   生成留出 C 上的盲评材料（同一格式，供评审人评分）。
"""
from __future__ import annotations

import bisect
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace

from score_learning.induction import bootstrap as bs
from score_learning.induction import fri
from score_learning.induction.ablation import VERT_FIELDS, uniform_fields
from score_learning.induction.baselines import (
    EVAL_ATOMS, EVAL_FIELDS, atom_auc, indep_fields, precision_at_k, productive_fields,
)
from score_learning.induction.fri import (
    MIN_NULL, SPARSE_FIELDS, VOICES, _norm_cdf, _phrase_spans, annotate_z, discoveries, residual,
)
from score_learning.induction.paper import _desync_piece, _git_head, load_chorale_pool, load_palestrina
from score_learning.induction.sensitivity import _null_k, _over_without_sparse

OUT = 'runs/induction/paper/revision_{}.json'
SEEDS = {'hold_a': 0, 'hold_b': 1, 'palestrina': 0, 'hold_c': 2, 'palestrina2': 3, 'palestrina3': 4}
THETAS = (0.25, 1 / 3, 0.5, 2 / 3, 0.75)
DROPS = (0.05, 0.10, 0.15, 0.20)
WITHIN = ('M', 'G', 'D', 'P', 'Q')
CATEGORY = {**{f: 'coordination' for f in ('V', 'O', 'H', 'S', 'U', 'K', 'J')},
            **{f: 'melodic' for f in ('M', 'G', 'D')}, 'P': 'positional', 'Q': 'positional'}
PREDS = fri.book_predicates()
ATOM_FIELD = {n: f for n, f, _p, _q in PREDS}
ATOM_POL = {n: p for n, _f, p, _q in PREDS}
VERT_ATOMS = {n for n, f, _p, _q in PREDS if f in VERT_FIELDS}
METHODS = {'ras': lambda f: f, 'pmi': indep_fields, 'rarity': uniform_fields, 'productive': productive_fields}


# ---------------------------------------------------------------------------- data

def load_sets(names):
    with open(bs.FINAL) as fh:
        fin = json.load(fh)
    by, usable, _ = load_chorale_pool()
    tr = [by[i] for i in fin['train_ids']]
    out = {}
    for n in names:
        if n == 'hold_a':
            out[n] = [by[i] for i in fin['hold_ids']]
        elif n == 'hold_b':
            out[n] = [by[i] for i in fin['hold_b_ids']]
        elif n == 'palestrina':
            out[n] = load_palestrina(16)
        elif n == 'hold_c':
            from score_learning.induction.fresh_eval import fresh_ids
            out[n] = [by[i] for i in fresh_ids(fin, usable)]
        elif n == 'palestrina2':
            from score_learning.induction.fresh_eval import load_palestrina_fresh
            out[n] = load_palestrina_fresh()
        elif n == 'palestrina3':
            from score_learning.induction.fresh_eval import load_palestrina_fresh
            out[n] = load_palestrina_fresh(skip=56, limit=40)
    return fin, tr, out


def eval_disc(f, cfg, rule7=True):
    und, ov = discoveries(f, cfg['z_cut'], cfg['q'], soft=True, soft_z=cfg.get('soft_z'))
    if not rule7:
        ov = _over_without_sparse(f, cfg)
    keep = lambda s: {(fld, c) for fld, c in s if fld in EVAL_FIELDS}
    return keep(und), keep(ov)


def matches(fld, cell, pol):
    return any(f == fld and p == pol and pred(cell, fld) for _n, f, p, pred in PREDS)


def disc_summary(und, ov):
    prec = lambda s, pol: sum(matches(f, c, pol) for f, c in s) / max(len(s), 1)
    hit = {n: any(pred(c, fld) for ff, c in (und if pol == 'under' else ov) if ff == fld)
           for n, fld, pol, pred in PREDS if fld in EVAL_FIELDS}
    return {'n_under': len(und), 'n_over': len(ov), 'prec_under': prec(und, 'under'),
            'prec_over': prec(ov, 'over'), 'completeness': sum(hit.values()) / len(hit),
            'miss': sorted(n for n, v in hit.items() if not v)}


# ---------------------------------------------------------------------------- desynchronisation schemes

def _rotate(p, idx_map):
    """idx_map[v][i] = 源切片下标；只移动音高与音级（与冻结的 desynchronize 相同）。"""
    sl = p.slices
    new = []
    for i, s in enumerate(sl):
        midis, deg = {}, {}
        for v in VOICES:
            m = sl[idx_map[v][i]].midis[v]
            midis[v] = m
            deg[v] = None if m is None else (m - p.tonic_pc) % 12
        b = midis.get('B')
        new.append(replace(s, midis=midis, deg=deg, bass_deg=None if b is None else (b - p.tonic_pc) % 12))
    return replace(p, slices=new)


def desync_rot(p, rng):
    return _desync_piece(p, rng)


def _metric_agree(sl, o):
    n = len(sl)
    return sum(sl[i].pos_in_bar == sl[(i + o) % n].pos_in_bar for i in range(n)) / n


def desync_metric(p, rng, min_agree=0.9):
    """独立旋转，但只用使拍位基本对齐的位移（拍位一致率 >= min_agree；不足时取一致率最高的 5 个位移）。"""
    sl = p.slices
    n = len(sl)
    if n < 4:
        return p
    ag = {o: _metric_agree(sl, o) for o in range(1, n)}
    cand = [o for o, a in ag.items() if a >= min_agree] or sorted(ag, key=lambda o: -ag[o])[:5]
    shifts = {'S': 0, **{v: rng.choice(cand) for v in ('A', 'T', 'B')}}
    return _rotate(p, {v: [(i + shifts[v]) % n for i in range(n)] for v in VOICES})


def desync_phrase(p, rng):
    """乐句内旋转：每个乐句内 A/T/B 各自独立循环位移，声部材料不离开所在乐句。"""
    sl = p.slices
    n = len(sl)
    if n < 4:
        return p
    idx = {v: list(range(n)) for v in VOICES}
    spans = _phrase_spans(p)
    for t0, t1, _fin in spans:
        ii = [i for i, s in enumerate(sl) if t0 - 1e-6 <= s.t < t1]
        m = len(ii)
        if m < 3:
            continue
        for v in ('A', 'T', 'B'):
            o = rng.randint(1, m - 1)
            for k, i in enumerate(ii):
                idx[v][i] = ii[(k + o) % m]
    return _rotate(p, idx)


SCHEMES = {'rot': desync_rot, 'metric': desync_metric, 'phrase': desync_phrase}
N_MULTI = 10


def tvd(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()) or 1, sum(b.values()) or 1
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


# ---------------------------------------------------------------------------- attrib

def classify(fld, zr, zd, theta=0.5):
    if fld in WITHIN:
        return 'within'
    if zr <= 0:
        return 'none'
    return 'typeI' if zd < theta * zr else 'typeII'


def run_attrib(names):
    fin, tr, sets = load_sets(names)
    cfg = fin['cfg']
    out = {'commit': _git_head(), 'thetas': THETAS, 'schemes': list(SCHEMES) + ['rot_multi'], 'sets': {}}
    for name, hold in sets.items():
        t0 = time.time()
        seed = SEEDS[name]
        f = fri.holdout_fields(tr, hold, seed=seed)
        und, ov = eval_disc(f, cfg)
        zr = annotate_z(f)
        disc = [('under', fld, c) for fld, c in sorted(und)] + [('over', fld, c) for fld, c in sorted(ov)]
        sgn = {'under': -1.0, 'over': 1.0}
        zeta_r = {d: sgn[d[0]] * zr[d[1]][d[2]] for d in disc}
        zeta_d, fields_d, holds_d = {}, {}, {}
        for sc, fn in SCHEMES.items():
            rng = random.Random(99)
            hd = [fn(p, rng) for p in hold]
            holds_d[sc] = hd
            fields_d[sc] = fri.holdout_fields(tr, hd, seed=seed)
            zd = annotate_z(fields_d[sc])
            zeta_d[sc] = {d: sgn[d[0]] * zd.get(d[1], {}).get(d[2], 0.0) for d in disc}
        acc = defaultdict(float)
        for j in range(N_MULTI):
            rng = random.Random(1000 + j)
            zd = annotate_z(fri.holdout_fields(tr, [desync_rot(p, rng) for p in hold], seed=seed))
            for d in disc:
                acc[d] += sgn[d[0]] * zd.get(d[1], {}).get(d[2], 0.0) / N_MULTI
        zeta_d['rot_multi'] = dict(acc)

        dist = {}
        for sc in zeta_d:
            for th in THETAS:
                for pol in ('under', 'over'):
                    c = Counter(classify(d[1], zeta_r[d], zeta_d[sc][d], th) for d in disc if d[0] == pol)
                    dist[f'{sc}|{th:.3f}|{pol}'] = dict(c)
        base = {d: classify(d[1], zeta_r[d], zeta_d['rot'][d]) for d in disc}
        agree = {}
        for sc in zeta_d:
            for th in THETAS:
                same = [classify(d[1], zeta_r[d], zeta_d[sc][d], th) == base[d] for d in disc if base[d] in ('typeI', 'typeII')]
                agree[f'{sc}|{th:.3f}'] = sum(same) / max(len(same), 1)
        ratios = {sc: sorted(zeta_d[sc][d] / zeta_r[d] for d in disc if d[1] not in WITHIN and zeta_r[d] > 0)
                  for sc in zeta_d}
        ambig = {sc: sum(0.25 <= r < 0.75 for r in rs) / max(len(rs), 1) for sc, rs in ratios.items()}
        by_field = {}
        for pol in ('under', 'over'):
            for fld in EVAL_FIELDS:
                c = Counter(base[d] for d in disc if d[0] == pol and d[1] == fld)
                if c:
                    by_field[f'{pol}|{fld}'] = dict(c)
        preserve = {}
        for sc, hd in holds_d.items():
            preserve[sc] = {fld: tvd(getattr(fri, f'real_{fld.lower()}_counts')(hold),
                                     getattr(fri, f'real_{fld.lower()}_counts')(hd)) for fld in EVAL_FIELDS}
        # 每个方案的位移在拍位上的一致率（声部起音落在同一拍位的比例）
        ag_m = []
        for p in hold:
            sl = p.slices
            n = len(sl)
            if n < 4:
                continue
            ag = {o: _metric_agree(sl, o) for o in range(1, n)}
            cand = [o for o, a in ag.items() if a >= 0.9] or sorted(ag, key=lambda o: -ag[o])[:5]
            ag_m.append((sum(ag[o] for o in cand) / len(cand), sum(ag.values()) / len(ag)))
        preserve['metric_agree'] = {'metric': sum(a for a, _ in ag_m) / len(ag_m),
                                    'rot': sum(b for _, b in ag_m) / len(ag_m)}
        # 参考约束的 Type（RAS-FR 原子 AUC，真实 vs 各方案）
        auc_r = atom_auc(f)
        ref = {}
        for sc in SCHEMES:
            auc_d = atom_auc(fields_d[sc])
            atoms = sorted(a for a in EVAL_ATOMS & VERT_ATOMS if auc_r.get(a) is not None and auc_d.get(a) is not None)
            for dr in DROPS:
                lab = {}
                for a in atoms:
                    if auc_r[a] < 0.6:
                        lab[a] = 'weak'
                    else:
                        lab[a] = 'typeI' if auc_r[a] - auc_d[a] >= dr else 'typeII'
                ref[f'{sc}|{dr:.2f}'] = {'counts': dict(Counter(lab.values())), 'labels': lab}
            ref[f'{sc}|auc'] = {a: (auc_r[a], auc_d[a]) for a in atoms}
        out['sets'][name] = {'n_disc': {'under': len(und), 'over': len(ov)}, 'dist': dist, 'agree': agree,
                             'ambiguous_frac': ambig, 'ratio_quantiles': {
                                 sc: [rs[int(q * (len(rs) - 1))] for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
                                 for sc, rs in ratios.items()},
                             'ratios': ratios, 'by_field': by_field, 'preserve': preserve, 'reference': ref}
        print(f'== {name} n={len(hold)} disc u/o {len(und)}/{len(ov)}  {time.time() - t0:.0f}s', flush=True)
        for sc in zeta_d:
            for pol in ('under', 'over'):
                print(f'  {sc:<9}{pol:<6}' + '  '.join(
                    f'θ={th:.2f} ' + ','.join(f'{k}:{v}' for k, v in sorted(dist[f"{sc}|{th:.3f}|{pol}"].items()))
                    for th in (1 / 3, 0.5, 2 / 3)), flush=True)
            print(f'  {sc:<9}agree θ=1/3 {agree[f"{sc}|{1/3:.3f}"]:.2f} 1/2 {agree[f"{sc}|0.500"]:.2f} '
                  f'2/3 {agree[f"{sc}|{2/3:.3f}"]:.2f}  ambiguous {ambig[sc]:.2f}', flush=True)
        for sc in SCHEMES:
            print(f'  ref {sc:<7}' + '  '.join(f'drop {dr:.2f} {ref[f"{sc}|{dr:.2f}"]["counts"]}' for dr in DROPS))
            print(f'  tvd {sc:<7}' + ' '.join(f'{k}:{v:.2f}' for k, v in preserve[sc].items()))
        print(f'  metric agreement of shifts: {preserve["metric_agree"]}')
    _dump('attrib', out)


# ---------------------------------------------------------------------------- pvals

Z_GRID = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
Q_GRID = (0.05, 0.08, 0.12, 0.2)
TARGETS = (0.01, 0.02, 0.03, 0.05, 0.08)


def bh_part(fields, z_cut, q, pmap=None):
    """规则 1 与 3（BH 部分），可替换 p 值。pmap[(fld, cell)] = (p_lo, p_hi)。"""
    zmap = annotate_z(fields)
    items = {'under': [], 'over': []}
    for fld, rows in fields.items():
        if fld not in fri.ALL_FIELDS:
            continue
        min_n = 3 if fld in SPARSE_FIELDS else MIN_NULL
        zn = z_cut + max(0.0, math.log10(max(len(rows), 1) / 40.0))
        for cell, _rho, nr, n0, *_ in rows:
            if n0 < min_n and nr < min_n:
                continue
            z = zmap[fld][cell]
            if pmap is None:
                plo, phi = max(_norm_cdf(z), 1e-15), max(1 - _norm_cdf(z), 1e-15)
            else:
                plo, phi = pmap[(fld, cell)]
            if z <= -zn * 0.5:
                items['under'].append(((fld, cell), plo))
            if z >= zn * 0.5:
                items['over'].append(((fld, cell), phi))
    out = {}
    for pol, its in items.items():
        keep = fri._bh_mask([p for _k, p in its], q)
        out[pol] = {k for (k, _p), kp in zip(its, keep) if kp}
    return out


def tested_z(fields):
    zmap = annotate_z(fields)
    out = {}
    for fld, rows in fields.items():
        if fld not in fri.ALL_FIELDS:
            continue
        min_n = 3 if fld in SPARSE_FIELDS else MIN_NULL
        out[fld] = [zmap[fld][r[0]] for r in rows if not (r[3] < min_n and r[2] < min_n)]
    return out


def sim_fields(per, hold, dp, pc, rng):
    out = {}
    for f in fri.ALL_FIELDS:
        nf = _null_k(f, 1)
        real, null = Counter(), Counter()
        for i, p in enumerate(hold):
            real.update(nf([p], dp, pc, rng))
            null.update(per[i][f][1])
        out[f] = residual(real, null)
    return out


def ev(s):
    return {k for k in s if k[0] in EVAL_FIELDS}


def run_pvals(names, n_sim=100):
    fin, tr, sets = load_sets(names)
    cfg = fin['cfg']
    dp, pc = fri.catalogs(tr)
    out = {'commit': _git_head(), 'n_sim': n_sim, 'sets': {}}
    # 校准目标的敏感性（训练集时移谱，与冻结校准相同的场）
    fd0 = fri._desync_fields(tr, 0)
    coup = ('V', 'H', 'O', 'M', 'G', 'S')
    denom = max(sum(len(fd0[f]) for f in coup), 1)
    n0m = {(fld, r[0]): r[3] for fld, rows in fd0.items() for r in rows}
    nrm = {(fld, r[0]): r[2] for fld, rows in fd0.items() for r in rows}
    fp = {}
    for zc in Z_GRID:
        for q in Q_GRID:
            und, _ = discoveries(fd0, zc, q, soft=False)
            fp[(zc, q)] = len({(f, c) for f, c in und if f in coup and nrm.get((f, c), 1) == 0
                               and n0m.get((f, c), 0) >= 12}) / denom
    chosen = {}
    for tg in TARGETS:
        best = min(((abs(v - tg) + 0.001 * zc, zc, q) for (zc, q), v in fp.items() if v <= 0.12))
        chosen[tg] = {'z_cut': best[1], 'q': best[2], 'fp': fp[(best[1], best[2])]}
    out['calib_fp'] = {f'{zc}|{q}': v for (zc, q), v in fp.items()}
    out['target'] = {str(k): v for k, v in chosen.items()}
    print('calibration target ->', {k: (v['z_cut'], v['q'], round(v['fp'], 3)) for k, v in chosen.items()}, flush=True)

    for name, hold in sets.items():
        t0 = time.time()
        seed = SEEDS[name]
        per = bs.piece_counts(tr, hold, seed)
        f = bs.fields_from(per, list(range(len(hold))))
        rng = random.Random(4242)
        null_z = defaultdict(list)
        grid_v = defaultdict(list)
        soft_v = defaultdict(list)
        full_v = []
        for s in range(n_sim):
            fs = sim_fields(per, hold, dp, pc, rng)
            for fld, zs in tested_z(fs).items():
                null_z[fld].extend(zs)
            for zc in Z_GRID:
                for q in Q_GRID:
                    b = bh_part(fs, zc, q)
                    hu, ho = discoveries(fs, zc, q, soft=False)
                    grid_v[(zc, q)].append((len(ev(b['under'])), len(ev(b['over'])), len(ev(hu)), len(ev(ho))))
            hard, _ = discoveries(fs, cfg['z_cut'], cfg['q'], soft=False)
            for sz in fri.SOFT_Z_GRID:
                _el, adds = fri.bass_path_soft_adds(fs, sz, hard)
                soft_v[sz].append(len(adds))
            u, o = eval_disc(fs, cfg)
            _u7, o7 = eval_disc(fs, cfg, rule7=False)
            full_v.append((len(u), len(o), len(o7)))
            if (s + 1) % 10 == 0:
                print(f'  {name} sim {s + 1} {time.time() - t0:.0f}s', flush=True)
        mean = lambda xs: sum(xs) / len(xs)
        grid = {}
        for zc in Z_GRID:
            for q in Q_GRID:
                b = bh_part(f, zc, q)
                hu, ho = discoveries(f, zc, q, soft=False)
                ru, ro = len(ev(b['under'])), len(ev(b['over']))
                vs = grid_v[(zc, q)]
                vu, vo = mean([v[0] for v in vs]), mean([v[1] for v in vs])
                hu_n, ho_n = len(ev(hu)), len(ev(ho))
                vhu, vho = mean([v[2] for v in vs]), mean([v[3] for v in vs])
                grid[f'{zc}|{q}'] = {
                    'bh_under': ru, 'bh_over': ro, 'null_bh_under': vu, 'null_bh_over': vo,
                    'fdr_bh_under': vu / ru if ru else None, 'fdr_bh_over': vo / ro if ro else None,
                    'p_any_bh_under': mean([v[0] > 0 for v in vs]), 'p_any_bh_over': mean([v[1] > 0 for v in vs]),
                    'hard_under': hu_n, 'hard_over': ho_n, 'null_hard_under': vhu, 'null_hard_over': vho,
                    'fdr_hard_under': vhu / hu_n if hu_n else None, 'fdr_hard_over': vho / ho_n if ho_n else None,
                    'prec_hard_under': sum(matches(a, c, 'under') for a, c in ev(hu)) / max(hu_n, 1),
                    'prec_hard_over': sum(matches(a, c, 'over') for a, c in ev(ho)) / max(ho_n, 1),
                }
        soft = {}
        hard, _ = discoveries(f, cfg['z_cut'], cfg['q'], soft=False)
        for sz in fri.SOFT_Z_GRID:
            el, adds = fri.bass_path_soft_adds(f, sz, hard)
            soft[str(sz)] = {'real_adds': len(adds), 'eligible': el, 'null_adds': mean(soft_v[sz]),
                             'prec': sum(matches(a, c, 'under') for a, c in adds) / max(len(adds), 1)}
        # 经验 p 值：模拟 z 的场内经验分布
        srt = {fld: sorted(zs) for fld, zs in null_z.items()}
        import bisect
        zmap = annotate_z(f)
        pmap = {}
        for fld, rows in f.items():
            zs = srt.get(fld, [])
            m = len(zs)
            for r in rows:
                z = zmap[fld][r[0]]
                lo = (1 + bisect.bisect_right(zs, z)) / (1 + m)
                hi = (1 + m - bisect.bisect_left(zs, z)) / (1 + m)
                pmap[(fld, r[0])] = (lo, hi)
        b_apx = bh_part(f, cfg['z_cut'], cfg['q'])
        b_emp = bh_part(f, cfg['z_cut'], cfg['q'], pmap)
        jac = lambda a, b: len(a & b) / max(len(a | b), 1)
        emp = {}
        for pol in ('under', 'over'):
            a, e = ev(b_apx[pol]), ev(b_emp[pol])
            emp[pol] = {'approx': len(a), 'empirical': len(e), 'both': len(a & e), 'jaccard': jac(a, e),
                        'prec_approx': sum(matches(x, c, pol) for x, c in a) / max(len(a), 1),
                        'prec_empirical': sum(matches(x, c, pol) for x, c in e) / max(len(e), 1)}
        u_full, o_full = eval_disc(f, cfg)
        # 经验 p 值替代规则 1/3 后的完整发现集
        und_e = (u_full - ev(b_apx['under'])) | ev(b_emp['under'])
        ov_e = (o_full - ev(b_apx['over'])) | ev(b_emp['over'])
        # 名义 p 值的校准：模拟 z 的分位数相对标准正态
        allz = sorted(z for zs in null_z.values() for z in zs)
        calib = {str(a): sum(z <= -_inv_norm(1 - a) for z in allz) / len(allz) for a in (0.001, 0.01, 0.05)}
        calib_hi = {str(a): sum(z >= _inv_norm(1 - a) for z in allz) / len(allz) for a in (0.001, 0.01, 0.05)}
        out['sets'][name] = {
            'grid': grid, 'soft': soft, 'empirical_bh': emp,
            'full_real': {'under': len(u_full), 'over': len(o_full),
                          'over_no_rule7': len(eval_disc(f, cfg, rule7=False)[1])},
            'full_null_mean': {'under': mean([v[0] for v in full_v]), 'over': mean([v[1] for v in full_v]),
                               'over_no_rule7': mean([v[2] for v in full_v])},
            'full_empirical': {'under': len(und_e), 'over': len(ov_e), 'jac_under': jac(und_e, u_full),
                               'jac_over': jac(ov_e, o_full)},
            'nominal_tail_lo': calib, 'nominal_tail_hi': calib_hi, 'n_null_z': len(allz),
        }
        g = grid[f'{cfg["z_cut"]}|{cfg["q"]}']
        print(f'== {name}: frozen BH under {g["bh_under"]} (null {g["null_bh_under"]:.2f}, FDR^ {g["fdr_bh_under"]}) '
              f'over {g["bh_over"]} (null {g["null_bh_over"]:.2f}, FDR^ {g["fdr_bh_over"]})', flush=True)
        print(f'   full rules real u/o {len(u_full)}/{len(o_full)}  null mean {out["sets"][name]["full_null_mean"]}')
        print(f'   empirical-p BH {emp}', flush=True)
        print(f'   nominal tails lo {calib} hi {calib_hi}', flush=True)
    _dump('pvals_' + '_'.join(names), out)


def _inv_norm(p):
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------- simple

def pooled_cats(tr):
    dp, pc = fri.catalogs(tr)
    bag = [x for v in VOICES for x in pc[v]] or [60]
    return dp, {v: bag for v in VOICES}


def piece_counts_cats(dp, pc, hold, seed):
    rng = random.Random(seed)
    out = [{} for _ in hold]
    for f in fri.ALL_FIELDS:
        for row, p in zip(out, hold):
            row[f] = (getattr(fri, f'real_{f.lower()}_counts')([p]), bs._NULL[f]([p], dp, pc, rng))
    return out


def calibrate_pooled(tr, seed=0):
    """与 calibrate_from_desync 相同的网格与目标，零模型换成合并音高目录。"""
    def dfields(s):
        rng = random.Random(s)
        shifted = []
        for p in tr:
            n = max(len(p.slices), 2)
            shifted.append(fri.desynchronize(p, {'S': 0, 'A': rng.randint(1, n - 1),
                                                 'T': rng.randint(1, n - 1), 'B': rng.randint(1, n - 1)}))
        fl = fri.fit_fields(shifted, seed=s + 1, pooled=True)
        fl.pop('_dp', None)
        fl.pop('_pitch', None)
        return fl
    fields = dfields(seed)
    coup = ('V', 'H', 'O', 'M', 'G', 'S')
    denom = max(sum(len(fields[f]) for f in coup), 1)
    n0m = {(fld, r[0]): r[3] for fld, rows in fields.items() for r in rows}
    nrm = {(fld, r[0]): r[2] for fld, rows in fields.items() for r in rows}
    best = {'z_cut': fri.Z_CUT, 'q': fri.BH_Q, 'false_frac': 1.0, 'score': 99.0}
    for zc in Z_GRID:
        for q in Q_GRID:
            und, _ = discoveries(fields, zc, q, soft=False)
            frac = len({(f, c) for f, c in und if f in coup and nrm.get((f, c), 1) == 0
                        and n0m.get((f, c), 0) >= 12}) / denom
            score = abs(frac - 0.03) + 0.001 * zc
            if frac <= 0.12 and score < best['score']:
                best = {'z_cut': zc, 'q': q, 'false_frac': frac, 'score': score}
    per_seed = [(fields, discoveries(fields, best['z_cut'], best['q'], soft=False)[0])]
    for s in (seed + 1, seed + 2):
        fl = dfields(s)
        per_seed.append((fl, discoveries(fl, best['z_cut'], best['q'], soft=False)[0]))
    chosen = None
    for sz in fri.SOFT_Z_GRID:
        fr = sum(len(fri.bass_path_soft_adds(fl, sz, h)[1]) / max(fri.bass_path_soft_adds(fl, sz, h)[0], 1)
                 for fl, h in per_seed) / len(per_seed)
        if chosen is None and fr <= fri.SOFT_FALSE_MAX:
            chosen, chosen_fr = sz, fr
    best.update({'soft_z': chosen, 'soft_false_frac': chosen_fr if chosen is not None else None})
    return best


def stats(fm, fmd):
    """fm / fmd：方法 -> 真实 / 去同步的场。共同原子在所有方法上取交集。"""
    auc = {m: atom_auc(x) for m, x in fm.items()}
    auc_d = {m: atom_auc(x) for m, x in fmd.items()}
    common = EVAL_ATOMS & set.intersection(*[{a for a, v in d.items() if v is not None} for d in auc.values()])
    vert = common & VERT_ATOMS & set.intersection(*[{a for a, v in d.items() if v is not None} for d in auc_d.values()])
    mean = lambda d, s: (sum(d[a] for a in s) / len(s)) if s else float('nan')
    groups = {'under': {a for a in common if ATOM_POL[a] == 'under'},
              'over': {a for a in common if ATOM_POL[a] == 'over'}}
    for cat in ('coordination', 'melodic', 'positional'):
        for pol in ('under', 'over'):
            groups[f'{cat}/{pol}'] = {a for a in common if CATEGORY[ATOM_FIELD[a]] == cat and ATOM_POL[a] == pol}
        groups[cat] = {a for a in common if CATEGORY[ATOM_FIELD[a]] == cat}
    out = {}
    for m in fm:
        pk = precision_at_k(fm[m], flds=EVAL_FIELDS)
        row = {'auc_common': mean(auc[m], common), 'vert_gap': mean(auc[m], vert) - mean(auc_d[m], vert),
               'p50_under': pk['under'][50], 'p100_under': pk['under'][100],
               'p50_over': pk['over'][50], 'p100_over': pk['over'][100]}
        for g, s in groups.items():
            row[f'auc_{g}'] = mean(auc[m], s)
        out[m] = row
    return out, auc, auc_d, common, vert, groups


def run_simple(name, n_half=300):
    fin, tr, sets = load_sets([name])
    hold = sets[name]
    seed = SEEDS[name]
    cfg = fin['cfg']
    t0 = time.time()
    cfg_s = calibrate_pooled(tr)
    print(f'pooled calibration {cfg_s} ({time.time() - t0:.0f}s)', flush=True)
    rng_d = random.Random(99)
    hold_d = [_desync_piece(p, rng_d) for p in hold]
    dp, pc = fri.catalogs(tr)
    dpp, pcp = pooled_cats(tr)
    per = piece_counts_cats(dp, pc, hold, seed)
    per_d = piece_counts_cats(dp, pc, hold_d, seed)
    pers = piece_counts_cats(dpp, pcp, hold, seed)
    pers_d = piece_counts_cats(dpp, pcp, hold_d, seed)

    def build(idx):
        f, fd = bs.fields_from(per, idx), bs.fields_from(per_d, idx)
        fs, fsd = bs.fields_from(pers, idx), bs.fields_from(pers_d, idx)
        fm = {m: t(f) for m, t in METHODS.items()}
        fmd = {m: t(fd) for m, t in METHODS.items()}
        fm['simple'], fmd['simple'] = fs, fsd
        return fm, fmd, f, fs, fd, fsd

    full = list(range(len(hold)))
    fm, fmd, f, fs, fd, fsd = build(full)
    point, auc, auc_d, common, vert, groups = stats(fm, fmd)
    disc = {}
    for tag, ff, ffd, c, r7 in (('frozen', f, fd, cfg, True), ('simple', fs, fsd, cfg_s, False),
                                ('frozen_no_rule7', f, fd, cfg, False), ('pooled_with_rule7', fs, fsd, cfg_s, True)):
        und, ov = eval_disc(ff, c, rule7=r7)
        s = disc_summary(und, ov)
        zr, zd = annotate_z(ff), annotate_z(ffd)
        cls = Counter()
        for pol, st in (('under', und), ('over', ov)):
            sg = -1.0 if pol == 'under' else 1.0
            for fld, cell in st:
                cls[f'{pol}|{classify(fld, sg * zr[fld][cell], sg * zd.get(fld, {}).get(cell, 0.0))}'] += 1
        s['classes'] = dict(cls)
        disc[tag] = s
        print(f'  disc {tag:<18} u/o {s["n_under"]}/{s["n_over"]} prec {s["prec_under"]:.2f}/{s["prec_over"]:.2f} '
              f'comp {s["completeness"]:.3f} {dict(cls)}', flush=True)
    rng = random.Random(2026)
    boot = []
    for b in range(n_half):
        idx = rng.sample(full, len(hold) // 2)
        fmb, fmdb, *_ = build(idx)
        boot.append(stats(fmb, fmdb)[0])
        if (b + 1) % 25 == 0:
            print(f'  {name} half {b + 1} {time.time() - t0:.0f}s', flush=True)
    keys = list(point['ras'])
    summ = {}
    for k in keys:
        s = {m: point[m][k] for m in point}
        for m in point:
            xs = [x[m][k] for x in boot if not math.isnan(x[m][k])]
            s[f'{m}_ci'] = bs._ci(xs) if len(xs) > 10 else None
        for a, b2 in (('simple', 'ras'), ('simple', 'pmi'), ('simple', 'rarity'), ('simple', 'productive'),
                      ('ras', 'pmi'), ('ras', 'rarity'), ('ras', 'productive')):
            d = [x[a][k] - x[b2][k] for x in boot if not (math.isnan(x[a][k]) or math.isnan(x[b2][k]))]
            s[f'diff_{a}_{b2}'] = point[a][k] - point[b2][k]
            if len(d) > 10:
                s[f'diff_{a}_{b2}_ci'] = bs._ci(d)
                s[f'p_le0_{a}_{b2}'] = sum(v <= 0 for v in d) / len(d)
        summ[k] = s
    out = {'commit': _git_head(), 'set': name, 'n': len(hold), 'n_half': n_half, 'cfg_frozen': cfg,
           'cfg_simple': cfg_s, 'summary': summ, 'discovery': disc,
           'atoms': {m: {'real': auc[m], 'desync': auc_d[m]} for m in auc},
           'common': sorted(common), 'vertical': sorted(vert), 'groups': {g: sorted(s) for g, s in groups.items()}}
    for k in ('auc_common', 'vert_gap', 'auc_under', 'auc_over'):
        s = summ[k]
        print(f'  {k:<11}' + ' '.join(f'{m} {s[m]:.3f}' for m in point) +
              '  ' + ' '.join(f'{a}-{b2} {s.get(f"diff_{a}_{b2}", float("nan")):+.3f}{s.get(f"diff_{a}_{b2}_ci")}'
                              for a, b2 in (('simple', 'ras'), ('simple', 'pmi'), ('simple', 'rarity'),
                                            ('simple', 'productive'))), flush=True)
    _dump(f'simple_{name}', out)


# ---------------------------------------------------------------------------- expert

def run_expert(which='a'):
    from score_learning.induction import expert_analyse as ea
    key_path, sheet_re, _ = ea.SETS[which]
    key = json.load(open(key_path))['items']
    first = [it for it in key if it['repeat_of'] is None]
    ratings = ea.load_ratings(sheet_re)
    raters = sorted(ratings)
    alias = {r: f'R{i + 1}' for i, r in enumerate(raters)}
    acc = lambda v: v in ea.ACCEPT

    def score(it, rs):
        v = [ratings[r][it['item']] for r in rs if ratings[r][it['item']] in ea.CATS]
        return sum(acc(x) for x in v) / len(v) if v else None

    items = []
    for it in first:
        items.append({'item': it['item'], 'polarity': it['polarity'], 'statement': it['statement'],
                      'cells': it['cells'], 'methods': it['methods'], 'reference': bool(it['matches_reference']),
                      'ratings': {alias[r]: ratings[r][it['item']] for r in raters},
                      'score': score(it, raters), 'class': ea._cls(it)})
    for i in items:
        i['endorsed'] = i['score'] is not None and i['score'] > 0.5

    def validity(rs):
        out = {}
        for m in ea.METHODS:
            sc = [score(it, rs) for it in first if m in it['methods']]
            sc = [x for x in sc if x is not None]
            out[m] = sum(sc) / len(sc) if sc else None
        return out

    loo = {alias[r]: validity([x for x in raters if x != r]) for r in raters}
    # 每位评审人单独
    single = {alias[r]: validity([r]) for r in raters}
    # P@20：按参考集、按专家认可、按二者并集
    pk = {}
    for m in ea.METHODS:
        for pol in ('prohibition', 'preference'):
            sel = [i for i in items if m in i['methods'] and i['polarity'] == pol and i['score'] is not None]
            n = len(sel)
            pk[f'{m}/{pol}'] = {'n': n, 'reference': sum(i['reference'] for i in sel) / n,
                                'endorsed': sum(i['endorsed'] for i in sel) / n,
                                'union': sum(i['reference'] or i['endorsed'] for i in sel) / n}

    def wilson(k, n, z=1.96):
        if n == 0:
            return None
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return ((c - h) / d, (c + h) / d)

    endorse_ci = {}
    for m in ea.METHODS:
        sel = [i for i in items if m in i['methods'] and i['score'] is not None]
        k = sum(i['endorsed'] for i in sel)
        endorse_ci[m] = {'k': k, 'n': len(sel), 'rate': k / len(sel), 'wilson': wilson(k, len(sel))}
    ref_end = [i for i in items if i['reference'] and i['score'] is not None]
    noref = [i for i in items if not i['reference'] and i['score'] is not None]
    out = {'raters': [alias[r] for r in raters], 'items': items, 'leave_one_out': loo, 'single_rater': single,
           'p_at_20': pk, 'endorsed_rate': endorse_ci,
           'reference_endorsed': {'k': sum(i['endorsed'] for i in ref_end), 'n': len(ref_end),
                                  'wilson': wilson(sum(i['endorsed'] for i in ref_end), len(ref_end))},
           'nonreference_endorsed': {'k': sum(i['endorsed'] for i in noref), 'n': len(noref),
                                     'wilson': wilson(sum(i['endorsed'] for i in noref), len(noref))}}
    print('leave-one-out', json.dumps(loo, indent=0))
    print('single', json.dumps(single, indent=0))
    print('P@20', json.dumps(pk, indent=0))
    print('endorsed', json.dumps(endorse_ci, indent=0))
    print(out['reference_endorsed'], out['nonreference_endorsed'])
    _dump('expert' if which == 'a' else f'expert_{which}', out)


# ---------------------------------------------------------------------------- C pack

def run_cpack():
    from score_learning.induction import expert_pack as ep
    fin, tr, sets = load_sets(['hold_c'])
    hold = sets['hold_c']
    rng_d = random.Random(99)
    hold_d = [_desync_piece(p, rng_d) for p in hold]
    f, fd = fri.holdout_fields(tr, hold, seed=2), fri.holdout_fields(tr, hold_d, seed=2)
    zr, zd = annotate_z(f), annotate_z(fd)
    merged = {}
    for m, t in METHODS.items():
        for pol, rows in ep.top_items(t(f)).items():
            for rank, (fld, cell, _s) in enumerate(rows, 1):
                stmt = ep.describe(fld, cell)
                e = merged.setdefault((pol, stmt), {'polarity': pol, 'statement': stmt, 'category': ep.CATEGORY[fld],
                                                    'cells': set(), 'ranks': {}})
                e['cells'].add((fld, cell))
                e['ranks'][m] = min(rank, e['ranks'].get(m, rank))
    rng = random.Random(ep.SEED + 1)
    items = list(merged.values())
    rng.shuffle(items)
    repeats = rng.sample(range(len(items)), ep.N_REPEAT)
    tail = [items[i] for i in repeats]
    rng.shuffle(tail)
    order = items + tail
    sheet, key = [], []
    for i, it in enumerate(order, 1):
        first = items.index(it) + 1
        sgn = -1.0 if it['polarity'] == 'prohibition' else 1.0
        sheet.append({'item': f'X{i:03d}', 'direction': '回避' if sgn < 0 else '偏好',
                      'category': it['category'], 'statement': it['statement']})
        key.append({'item': f'X{i:03d}', 'repeat_of': None if i == first else f'X{first:03d}',
                    'polarity': it['polarity'], 'statement': it['statement'],
                    'cells': sorted(f'{a}:{b}' for a, b in it['cells']), 'methods': it['ranks'],
                    'matches_reference': sorted({n for fld, cell in it['cells'] for n, fp, p, pred in PREDS
                                                 if fp == fld and p == ('under' if sgn < 0 else 'over')
                                                 and pred(cell, fld)}),
                    'signed_z_real': {f'{a}:{b}': round(sgn * zr[a][b], 2) for a, b in it['cells']},
                    'signed_z_desync': {f'{a}:{b}': round(sgn * zd.get(a, {}).get(b, 0.0), 2) for a, b in it['cells']}})
    path = 'papers/rasfr/expert_review/约束盲评表_留出C.xlsx'
    ep.write_xlsx(sheet, path)
    with open('runs/induction/paper/expert_key_c.json', 'w') as fh:
        json.dump({'commit': _git_head(), 'corpus': 'hold-out C', 'top_n': ep.TOP_N, 'n_unique': len(items),
                   'n_repeats': ep.N_REPEAT, 'items': key}, fh, indent=1, ensure_ascii=False)
    print(f'{len(items)} unique items (+{ep.N_REPEAT} repeats); wrote {path}')


# ---------------------------------------------------------------------------- paired tests

SETS5 = ('hold_a', 'hold_b', 'palestrina', 'hold_c', 'palestrina2')


def sign_test(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def wilcoxon_exact(d):
    """双侧精确 Wilcoxon 符号秩检验（零差去掉，并列取平均秩，秩和乘 2 取整后做 DP）。"""
    d = [x for x in d if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0, 0.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[order[j + 1]]) - abs(d[order[i]])) < 1e-12:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    r2 = [int(round(2 * r)) for r in ranks]
    w_plus = sum(r for r, x in zip(r2, d) if x > 0)
    tot = sum(r2)
    dist = Counter({0: 1})
    for r in r2:
        nxt = Counter()
        for s, c in dist.items():
            nxt[s] += c
            nxt[s + r] += c
        dist = nxt
    total = 2 ** n
    lo = sum(c for s, c in dist.items() if s <= min(w_plus, tot - w_plus)) / total
    rb = (2 * w_plus - tot) / tot
    return min(1.0, 2 * lo), rb


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def run_tests():
    res = {}
    for s in SETS5:
        try:
            d = json.load(open(OUT.format(f'simple_{s}')))
        except FileNotFoundError:
            continue
        res[s] = d
    rows = []
    for s, d in res.items():
        common = d['common']
        for a in ('ras', 'simple'):
            for b in ('pmi', 'rarity', 'productive') + (('ras',) if a == 'simple' else ()):
                x = [d['atoms'][a]['real'][t] - d['atoms'][b]['real'][t] for t in common]
                w, l = sum(v > 1e-12 for v in x), sum(v < -1e-12 for v in x)
                pw, rb = wilcoxon_exact(x)
                rows.append({'set': s, 'a': a, 'b': b, 'n': len(x), 'wins': w, 'losses': l, 'ties': len(x) - w - l,
                             'mean_diff': sum(x) / len(x), 'median_diff': sorted(x)[len(x) // 2],
                             'p_superiority': (w + 0.5 * (len(x) - w - l)) / len(x),
                             'p_sign': sign_test(w, l), 'p_wilcoxon': pw, 'rank_biserial': rb})
    for a in ('ras', 'simple'):
        fam = [r for r in rows if r['a'] == a and r['b'] != 'ras']
        for key in ('p_sign', 'p_wilcoxon'):
            for r, p in zip(fam, holm([r[key] for r in fam])):
                r[key + '_holm'] = p
    # 汇总指标的半抽样单侧 p（diff <= 0 的比例）做 Holm：预注册的 H1/H2 在留出 C 上 6 个比较
    fresh = json.load(open('runs/induction/paper/fresh_eval.json'))['sets']['hold_c']['summary']
    prim = [(k, b, fresh[k][f'p_le0_{b}']) for k in ('auc_common', 'vert_gap') for b in ('pmi', 'rarity', 'productive')]
    adj = holm([max(p, 1 / 1000) for _k, _b, p in prim])
    out = {'paired': rows, 'prereg_holm': [{'metric': k, 'base': b, 'p_half': p, 'p_holm': q}
                                           for (k, b, p), q in zip(prim, adj)]}
    # 分层与逐场
    strata = {}
    for s, d in res.items():
        sm = d['summary']
        strata[s] = {k: {m: sm[k][m] for m in ('ras', 'simple', 'pmi', 'rarity', 'productive')}
                     for k in sm if k.startswith('auc_')}
        strata[s]['_n'] = {g: len(v) for g, v in d['groups'].items()}
        strata[s]['_n']['common'] = len(d['common'])
        strata[s]['_n']['vertical'] = len(d['vertical'])
        pf = {}
        for fld in EVAL_FIELDS:
            atoms = [t for t in d['common'] if ATOM_FIELD[t] == fld]
            if atoms:
                pf[fld] = {'n': len(atoms), **{m: sum(d['atoms'][m]['real'][t] for t in atoms) / len(atoms)
                                               for m in ('ras', 'simple', 'pmi', 'rarity', 'productive')}}
        strata[s]['_per_field'] = pf
    out['strata'] = strata
    for r in rows:
        print(f'{r["set"]:<12}{r["a"]:>7} vs {r["b"]:<10} n={r["n"]:<3} {r["wins"]}/{r["losses"]}/{r["ties"]} '
              f'mean {r["mean_diff"]:+.3f} PS {r["p_superiority"]:.2f} rb {r["rank_biserial"]:+.2f} '
              f'sign {r["p_sign"]:.3f}({r.get("p_sign_holm", float("nan")):.3f}) '
              f'wil {r["p_wilcoxon"]:.3f}({r.get("p_wilcoxon_holm", float("nan")):.3f})')
    print(out['prereg_holm'])
    _dump('tests', out)


# ---------------------------------------------------------------------------- hybrid baseline

def hybrid_stats(fm, fmd):
    """冻结重放、简化重放与混合基线（禁止原子用稀有度、偏好原子用 PMI）的汇总指标。"""
    auc = {m: atom_auc(fm[m]) for m in ('ras', 'simple', 'pmi', 'rarity')}
    auc_d = {m: atom_auc(fmd[m]) for m in ('ras', 'simple', 'pmi', 'rarity')}
    pick = lambda d: {a: (d['rarity'] if ATOM_POL[a] == 'under' else d['pmi']).get(a) for a in d['pmi']}
    auc['hybrid'], auc_d['hybrid'] = pick(auc), pick(auc_d)
    common = EVAL_ATOMS & set.intersection(*[{a for a, v in d.items() if v is not None} for d in auc.values()])
    vert = common & VERT_ATOMS & set.intersection(*[{a for a, v in d.items() if v is not None} for d in auc_d.values()])
    mean = lambda d, s: (sum(d[a] for a in s) / len(s)) if s else float('nan')
    under = {a for a in common if ATOM_POL[a] == 'under'}
    over = common - under
    pk = {m: precision_at_k(fm[m], flds=EVAL_FIELDS) for m in ('ras', 'simple', 'pmi', 'rarity')}
    pk['hybrid'] = {'under': pk['rarity']['under'], 'over': pk['pmi']['over']}
    out = {}
    for m in auc:
        out[m] = {'auc_common': mean(auc[m], common), 'vert_gap': mean(auc[m], vert) - mean(auc_d[m], vert),
                  'auc_under': mean(auc[m], under), 'auc_over': mean(auc[m], over),
                  'p50_under': pk[m]['under'][50], 'p50_over': pk[m]['over'][50]}
    return out, common, vert


def run_hybrid(name, n_half=300):
    fin, tr, sets = load_sets([name])
    hold = sets[name]
    seed = SEEDS[name]
    t0 = time.time()
    rng_d = random.Random(99)
    hold_d = [_desync_piece(p, rng_d) for p in hold]
    dp, pc = fri.catalogs(tr)
    dpp, pcp = pooled_cats(tr)
    per, per_d = piece_counts_cats(dp, pc, hold, seed), piece_counts_cats(dp, pc, hold_d, seed)
    pers, pers_d = piece_counts_cats(dpp, pcp, hold, seed), piece_counts_cats(dpp, pcp, hold_d, seed)
    meths = {'ras': METHODS['ras'], 'pmi': METHODS['pmi'], 'rarity': METHODS['rarity']}

    def build(idx):
        f, fd = bs.fields_from(per, idx), bs.fields_from(per_d, idx)
        fm = {m: t(f) for m, t in meths.items()}
        fmd = {m: t(fd) for m, t in meths.items()}
        fm['simple'], fmd['simple'] = bs.fields_from(pers, idx), bs.fields_from(pers_d, idx)
        return hybrid_stats(fm, fmd)

    full = list(range(len(hold)))
    point, common, vert = build(full)
    rng = random.Random(2026)
    boot = []
    for b in range(n_half):
        boot.append(build(rng.sample(full, len(hold) // 2))[0])
        if (b + 1) % 50 == 0:
            print(f'  {name} half {b + 1} {time.time() - t0:.0f}s', flush=True)
    summ = {}
    for k in point['ras']:
        s = {m: point[m][k] for m in point}
        for m in point:
            xs = [x[m][k] for x in boot if not math.isnan(x[m][k])]
            s[f'{m}_ci'] = bs._ci(xs) if len(xs) > 10 else None
        for a in ('ras', 'simple'):
            d = [x[a][k] - x['hybrid'][k] for x in boot if not (math.isnan(x[a][k]) or math.isnan(x['hybrid'][k]))]
            s[f'diff_{a}_hybrid'] = point[a][k] - point['hybrid'][k]
            s[f'diff_{a}_hybrid_ci'] = bs._ci(d) if len(d) > 10 else None
            s[f'p_le0_{a}_hybrid'] = sum(v <= 0 for v in d) / len(d) if d else None
            s[f'p_ge0_{a}_hybrid'] = sum(v >= 0 for v in d) / len(d) if d else None
        summ[k] = s
        print(f'  {k:<11}' + ' '.join(f'{m} {s[m]:.3f}' for m in point) +
              f"  ras-hyb {s['diff_ras_hybrid']:+.3f}{s['diff_ras_hybrid_ci']} "
              f"simple-hyb {s['diff_simple_hybrid']:+.3f}{s['diff_simple_hybrid_ci']}", flush=True)
    _dump(f'hybrid_{name}', {'commit': _git_head(), 'set': name, 'n': len(hold), 'n_half': n_half,
                              'summary': summ, 'common': sorted(common), 'vertical': sorted(vert)})


# ---------------------------------------------------------------------------- M3: calibration operator

COUP = ('V', 'H', 'O', 'M', 'G', 'S')
CAL_RUNS = (('rot', 0), ('rot', 1), ('rot', 2), ('rot', 3), ('rot', 4), ('metric', 0), ('phrase', 0))


def _desync_train_fields(tr, op, seed):
    rng = random.Random(seed)
    f = fri.fit_fields([SCHEMES[op](p, rng) for p in tr], seed=seed + 1)
    f.pop('_dp', None)
    f.pop('_pitch', None)
    return f


def _false_cells(fields, z_cut, q):
    """算法 2 的假禁止：耦合场中真实计数为 0 且零模型计数 >= 12 的禁止。"""
    und, _ = discoveries(fields, z_cut, q, soft=False)
    nr = {(fld, r[0]): r[2] for fld, rows in fields.items() for r in rows}
    n0 = {(fld, r[0]): r[3] for fld, rows in fields.items() for r in rows}
    return {(f, c) for f, c in und if f in COUP and nr.get((f, c), 1) == 0 and n0.get((f, c), 0) >= 12}


def calib_with(fields):
    denom = max(sum(len(fields[f]) for f in COUP), 1)
    best = {'z_cut': fri.Z_CUT, 'q': fri.BH_Q, 'false_frac': 1.0, 'score': 99.0}
    for zc in Z_GRID:
        for q in Q_GRID:
            frac = len(_false_cells(fields, zc, q)) / denom
            score = abs(frac - 0.03) + 0.001 * zc
            if frac <= 0.12 and score < best['score']:
                best = {'z_cut': zc, 'q': q, 'false_frac': frac, 'score': score}
    return best


def per_field_false(fields, z_cut, q):
    fc = _false_cells(fields, z_cut, q)
    out = {}
    for f in COUP:
        rows = fields[f]
        out[f] = {'cells': len(rows), 'false': sum(1 for ff, _ in fc if ff == f),
                  'sparse_frac': sum(r[3] < 12 for r in rows) / max(len(rows), 1)}
        out[f]['false_frac'] = out[f]['false'] / max(len(rows), 1)
    return out


def run_calops(names):
    fin, tr, sets = load_sets(names)
    cfg = fin['cfg']
    cal = {}
    for op, sd in CAL_RUNS:
        fd = _desync_train_fields(tr, op, sd)
        b = calib_with(fd)
        cal[f'{op}|{sd}'] = {**b, 'per_field_frozen': per_field_false(fd, cfg['z_cut'], cfg['q'])}
        print(f'calibration {op} seed {sd}: z_cut {b["z_cut"]} q {b["q"]} false {b["false_frac"]:.3f}', flush=True)
    out = {'commit': _git_head(), 'calibration': cal, 'sets': {}}
    sgn = {'under': -1.0, 'over': 1.0}
    for name, hold in sets.items():
        seed = SEEDS[name]
        f = fri.holdout_fields(tr, hold, seed=seed)
        zr = annotate_z(f)
        zd = {}
        for op in SCHEMES:
            rng = random.Random(99)
            fd_h = fri.holdout_fields(tr, [SCHEMES[op](p, rng) for p in hold], seed=seed)
            zd[op] = annotate_z(fd_h)
            if op == 'rot':
                out_field = per_field_false(fd_h, cfg['z_cut'], cfg['q'])
        res = {'per_field_heldout_desync': out_field, 'runs': {}}
        base = None
        for key in ['rot|0'] + [k for k in cal if k != 'rot|0']:
            c = {**cfg, 'z_cut': cal[key]['z_cut'], 'q': cal[key]['q']}
            und, ov = eval_disc(f, c)
            disc = [('under', fl, ce) for fl, ce in und] + [('over', fl, ce) for fl, ce in ov]
            cls = {op: {d: classify(d[1], sgn[d[0]] * zr[d[1]][d[2]], sgn[d[0]] * zd[op].get(d[1], {}).get(d[2], 0.0))
                        for d in disc} for op in SCHEMES}
            if base is None:
                base = cls['rot']
            r = {'z_cut': c['z_cut'], 'q': c['q'], 'n_under': len(und), 'n_over': len(ov), 'by_op': {}}
            for op in SCHEMES:
                att = [d for d in disc if d in base and base[d] in ('typeI', 'typeII')]
                r['by_op'][op] = {
                    'dist': {pol: dict(Counter(cls[op][d] for d in disc if d[0] == pol)) for pol in ('under', 'over')},
                    'agree_with_frozen': sum(cls[op][d] == base[d] for d in att) / max(len(att), 1),
                    'n_common_attributable': len(att)}
            res['runs'][key] = r
            print(f'  {name} cal {key:<9} ({c["z_cut"]},{c["q"]}) u/o {len(und)}/{len(ov)} ' +
                  ' '.join(f'{op}:{r["by_op"][op]["agree_with_frozen"]:.2f}' for op in SCHEMES), flush=True)
        out['sets'][name] = res
    _dump('calops', out)


# ---------------------------------------------------------------------------- M4: uncertainty of the Type

N_TYPEBOOT = 200
DESYNC_SEEDS = (99, 100, 101, 102, 103)


def run_typeboot(names, n_boot=N_TYPEBOOT):
    fin, tr, sets = load_sets(names)
    cfg = fin['cfg']
    dp, pc = fri.catalogs(tr)
    sgn = {'under': -1.0, 'over': 1.0}
    out = {'commit': _git_head(), 'n_boot': n_boot, 'desync_seeds': DESYNC_SEEDS, 'sets': {}}
    for name, hold in sets.items():
        t0 = time.time()
        seed = SEEDS[name]
        f = fri.holdout_fields(tr, hold, seed=seed)
        und, ov = eval_disc(f, cfg)
        disc = [('under', fl, c) for fl, c in sorted(und)] + [('over', fl, c) for fl, c in sorted(ov)]
        zr0 = annotate_z(f)
        rng0 = random.Random(99)
        zd0 = annotate_z(fri.holdout_fields(tr, [desync_rot(p, rng0) for p in hold], seed=seed))
        lab0 = {d: classify(d[1], sgn[d[0]] * zr0[d[1]][d[2]], sgn[d[0]] * zd0.get(d[1], {}).get(d[2], 0.0))
                for d in disc}
        per = piece_counts_cats(dp, pc, hold, seed)
        per_d = []
        for ds in DESYNC_SEEDS:
            rng = random.Random(ds)
            per_d.append(piece_counts_cats(dp, pc, [desync_rot(p, rng) for p in hold], seed))
        rng = random.Random(2026)
        cnt = {d: Counter() for d in disc}
        n = len(hold)
        for b in range(n_boot):
            idx = [rng.randrange(n) for _ in range(n)]
            zr = annotate_z(bs.fields_from(per, idx))
            zd = annotate_z(bs.fields_from(per_d[rng.randrange(len(per_d))], idx))
            for d in disc:
                cnt[d][classify(d[1], sgn[d[0]] * zr[d[1]].get(d[2], 0.0), sgn[d[0]] * zd.get(d[1], {}).get(d[2], 0.0))] += 1
            if (b + 1) % 50 == 0:
                print(f'  {name} boot {b + 1} {time.time() - t0:.0f}s', flush=True)
        items = []
        for d in disc:
            p = {k: v / n_boot for k, v in cnt[d].items()}
            items.append({'polarity': d[0], 'field': d[1], 'cell': d[2], 'label': lab0[d], 'p': p,
                          'p_label': p.get(lab0[d], 0.0), 'ref': matches(d[1], d[2], d[0])})
        att = [i for i in items if i['label'] in ('typeI', 'typeII', 'none')]
        summ = {}
        for pol in ('under', 'over'):
            for lab in ('typeI', 'typeII', 'none'):
                xs = [i['p_label'] for i in att if i['polarity'] == pol and i['label'] == lab]
                summ[f'{pol}|{lab}'] = {'n': len(xs), 'mean_p': sum(xs) / len(xs) if xs else None,
                                        'ge90': sum(x >= 0.9 for x in xs), 'ge75': sum(x >= 0.75 for x in xs),
                                        'lt60': sum(x < 0.6 for x in xs)}
            # 期望类别计数：按自助概率加权
            summ[f'{pol}|expected'] = {lab: sum(i['p'].get(lab, 0.0) for i in att if i['polarity'] == pol)
                                       for lab in ('typeI', 'typeII', 'none')}
        # 无协同的“真值”数据：把留出乐谱错位（种子 98）后当作真实数据，再用种子 99 错位归因
        truth = {}
        for op in ('rot', 'metric'):
            rng_t = random.Random(98)
            h0 = [SCHEMES[op](p, rng_t) for p in hold]
            f0 = fri.holdout_fields(tr, h0, seed=seed)
            u0, o0 = eval_disc(f0, cfg)
            z0 = annotate_z(f0)
            rng_a = random.Random(99)
            z1 = annotate_z(fri.holdout_fields(tr, [desync_rot(p, rng_a) for p in h0], seed=seed))
            d0 = [('under', fl, c) for fl, c in u0] + [('over', fl, c) for fl, c in o0]
            lab = Counter((d[0], classify(d[1], sgn[d[0]] * z0[d[1]][d[2]], sgn[d[0]] * z1.get(d[1], {}).get(d[2], 0.0)))
                          for d in d0)
            truth[op] = {f'{pol}|{l}': v for (pol, l), v in lab.items()}
        out['sets'][name] = {'items': items, 'summary': summ, 'no_coordination_truth': truth}
        print(f'== {name} {time.time() - t0:.0f}s', json.dumps(summ), json.dumps(truth), flush=True)
    _dump('typeboot', out)


# ---------------------------------------------------------------------------- M6: attribution of baseline discoveries

def _topk(fields, k_u, k_o):
    z = annotate_z(fields)
    rows = []
    for fld in EVAL_FIELDS:
        for r in fri._ranked(fields[fld]):
            rows.append((fld, r[0], z[fld][r[0]]))
    und = sorted((x for x in rows if x[2] < 0), key=lambda x: x[2])[:k_u]
    ov = sorted((x for x in rows if x[2] > 0), key=lambda x: -x[2])[:k_o]
    return [('under', a, b) for a, b, _ in und] + [('over', a, b) for a, b, _ in ov]


def run_attrbase(names):
    fin, tr, sets = load_sets(names)
    cfg = fin['cfg']
    sgn = {'under': -1.0, 'over': 1.0}
    out = {'commit': _git_head(), 'sets': {}}
    for name, hold in sets.items():
        seed = SEEDS[name]
        f = fri.holdout_fields(tr, hold, seed=seed)
        rng = random.Random(99)
        fd = fri.holdout_fields(tr, [desync_rot(p, rng) for p in hold], seed=seed)
        und, ov = eval_disc(f, cfg)
        k_u, k_o = len(und), len(ov)
        zr_ras, zd_ras = annotate_z(f), annotate_z(fd)
        res = {'k_under': k_u, 'k_over': k_o, 'methods': {}}
        for m, t in METHODS.items():
            fm, fmd = t(f), t(fd)
            zr, zd = annotate_z(fm), annotate_z(fmd)
            disc = _topk(fm, k_u, k_o)
            r = {}
            for judge, (a, b) in (('own', (zr, zd)), ('replay', (zr_ras, zd_ras))):
                lab = {d: classify(d[1], sgn[d[0]] * a[d[1]].get(d[2], 0.0), sgn[d[0]] * b.get(d[1], {}).get(d[2], 0.0))
                       for d in disc}
                for pol in ('under', 'over'):
                    sel = [d for d in disc if d[0] == pol]
                    c = Counter(lab[d] for d in sel)
                    vert = [d for d in sel if lab[d] in ('typeI', 'typeII', 'none')]
                    rel = [d for d in vert if lab[d] == 'typeI']
                    r[f'{judge}|{pol}'] = {'dist': dict(c), 'n_vertical': len(vert),
                                           'rel_frac': len(rel) / max(len(vert), 1),
                                           'rel_ref_prec': sum(matches(d[1], d[2], pol) for d in rel) / max(len(rel), 1),
                                           'ref_prec_all': sum(matches(d[1], d[2], pol) for d in sel) / max(len(sel), 1)}
            res['methods'][m] = r
            print(f'  {name} {m:<10} own u {r["own|under"]["dist"]} o {r["own|over"]["dist"]} | '
                  f'replay-judged u {r["replay|under"]["dist"]} o {r["replay|over"]["dist"]}', flush=True)
        out['sets'][name] = res
    _dump('attrbase', out)


# ---------------------------------------------------------------------------- preregistered test of the simplified replay

PREREG3_CFG = {'z_cut': 4.0, 'q': 0.05, 'soft_z': -2.0}
PREREG3_DIRECTIONAL = (('auc_common', 'pmi'), ('auc_common', 'rarity'), ('auc_common', 'productive'),
                       ('vert_gap', 'pmi'), ('vert_gap', 'rarity'), ('vert_gap', 'productive'),
                       ('vert_gap', 'hybrid'))
SIX = ('simple', 'ras', 'pmi', 'rarity', 'productive', 'hybrid')


def six_stats(fm, fmd):
    auc = {m: atom_auc(fm[m]) for m in SIX[:5]}
    auc_d = {m: atom_auc(fmd[m]) for m in SIX[:5]}
    pick = lambda d: {a: (d['rarity'] if ATOM_POL[a] == 'under' else d['pmi']).get(a) for a in d['pmi']}
    auc['hybrid'], auc_d['hybrid'] = pick(auc), pick(auc_d)
    common = EVAL_ATOMS & set.intersection(*[{a for a, v in d.items() if v is not None} for d in auc.values()])
    vert = common & VERT_ATOMS & set.intersection(*[{a for a, v in d.items() if v is not None} for d in auc_d.values()])
    mean = lambda d, s: (sum(d[a] for a in s) / len(s)) if s else float('nan')
    under = {a for a in common if ATOM_POL[a] == 'under'}
    over = common - under
    out = {m: {'auc_common': mean(auc[m], common), 'vert_gap': mean(auc[m], vert) - mean(auc_d[m], vert),
               'auc_under': mean(auc[m], under), 'auc_over': mean(auc[m], over)} for m in SIX}
    return out, auc, auc_d, common, vert


def run_prereg3(n_half=1000):
    fin, tr, _ = load_sets([])
    cfg_s = calibrate_pooled(tr)
    for k, v in PREREG3_CFG.items():
        if abs(cfg_s[k] - v) > 1e-9:
            raise SystemExit(f'simplified calibration {cfg_s} differs from the preregistered {PREREG3_CFG}')
    print('simplified calibration matches the preregistration', cfg_s, flush=True)
    _, _, sets = load_sets(['palestrina3'])
    hold = sets['palestrina3']
    seed = SEEDS['palestrina3']
    t0 = time.time()
    rng_d = random.Random(99)
    hold_d = [_desync_piece(p, rng_d) for p in hold]
    dp, pc = fri.catalogs(tr)
    dpp, pcp = pooled_cats(tr)
    per, per_d = piece_counts_cats(dp, pc, hold, seed), piece_counts_cats(dp, pc, hold_d, seed)
    pers, pers_d = piece_counts_cats(dpp, pcp, hold, seed), piece_counts_cats(dpp, pcp, hold_d, seed)

    def build(idx):
        f, fd = bs.fields_from(per, idx), bs.fields_from(per_d, idx)
        fm = {m: t(f) for m, t in METHODS.items()}
        fmd = {m: t(fd) for m, t in METHODS.items()}
        fm['simple'], fmd['simple'] = bs.fields_from(pers, idx), bs.fields_from(pers_d, idx)
        return fm, fmd, f, fd

    full = list(range(len(hold)))
    fm, fmd, f, fd = build(full)
    point, auc, auc_d, common, vert = six_stats(fm, fmd)
    disc = {}
    for tag, ff, c, r7 in (('simple', fm['simple'], cfg_s, False), ('frozen', f, fin['cfg'], True)):
        und, ov = eval_disc(ff, c, rule7=r7)
        disc[tag] = disc_summary(und, ov)
    rng = random.Random(2026)
    boot = []
    for b in range(n_half):
        boot.append(six_stats(*build(rng.sample(full, len(hold) // 2))[:2])[0])
        if (b + 1) % 100 == 0:
            print(f'  palestrina3 half {b + 1} {time.time() - t0:.0f}s', flush=True)
    summ = {}
    for k in point['simple']:
        s = {m: point[m][k] for m in SIX}
        for m in SIX:
            xs = [x[m][k] for x in boot if not math.isnan(x[m][k])]
            s[f'{m}_ci'] = bs._ci(xs)
        for b2 in SIX[1:]:
            d = [x['simple'][k] - x[b2][k] for x in boot if not (math.isnan(x['simple'][k]) or math.isnan(x[b2][k]))]
            s[f'diff_simple_{b2}'] = point['simple'][k] - point[b2][k]
            s[f'diff_simple_{b2}_ci'] = bs._ci(d)
            s[f'p_le0_simple_{b2}'] = sum(v <= 0 for v in d) / len(d)
        summ[k] = s
    fam = [{'metric': k, 'base': b2, 'diff': summ[k][f'diff_simple_{b2}'], 'ci': summ[k][f'diff_simple_{b2}_ci'],
            'p_half': summ[k][f'p_le0_simple_{b2}']} for k, b2 in PREREG3_DIRECTIONAL]
    for r, q in zip(fam, holm([max(r['p_half'], 1 / n_half) for r in fam])):
        r['p_holm'] = q
        r['supported'] = r['ci'][0] > 0
    hyp = {'H4': all(r['supported'] for r in fam[:3]), 'H5': all(r['supported'] for r in fam[3:6]),
           'H6': fam[6]['supported']}
    out = {'commit': _git_head(), 'set': 'palestrina3', 'n': len(hold), 'titles': [p.title for p in hold],
           'n_half': n_half, 'cfg_simple': cfg_s, 'cfg_frozen': fin['cfg'], 'summary': summ, 'family': fam,
           'hypotheses': hyp, 'discovery': disc, 'atoms': {m: {'real': auc[m], 'desync': auc_d[m]} for m in auc},
           'common': sorted(common), 'vertical': sorted(vert)}
    for k in ('auc_common', 'vert_gap', 'auc_under', 'auc_over'):
        print(f'  {k:<11}' + ' '.join(f'{m} {summ[k][m]:.3f}' for m in SIX), flush=True)
    for r in fam:
        print(f"  {r['metric']:<10} simple-{r['base']:<10} {r['diff']:+.3f} {r['ci']} p {r['p_half']:.3f} "
              f"holm {r['p_holm']:.3f} {'yes' if r['supported'] else 'no'}", flush=True)
    print('hypotheses', hyp)
    _dump('prereg3', out)


# ---------------------------------------------------------------------------- practical significance

DELTAS = (0.05, 0.10)
N_ATOM_BOOT = 10000


def _hybrid(atoms, t, kind='real'):
    """稀有度管禁止、PMI 管偏好：两个最简基线各取其长。"""
    return atoms['rarity' if ATOM_POL[t] == 'under' else 'pmi'][kind][t]


def _hl(x):
    """Hodges–Lehmann 位置估计：全部 Walsh 平均的中位数。"""
    w = sorted((x[i] + x[j]) / 2 for i in range(len(x)) for j in range(i, len(x)))
    n = len(w)
    return (w[n // 2] + w[(n - 1) // 2]) / 2


def run_practical():
    rng = random.Random(2026)
    out = {}
    for s in SETS5:
        try:
            d = json.load(open(OUT.format(f'simple_{s}')))
        except FileNotFoundError:
            continue
        A, common, vert = d['atoms'], d['common'], d['vertical']
        val = {m: {t: A[m]['real'][t] for t in common} for m in ('ras', 'simple', 'pmi', 'rarity', 'productive')}
        val['hybrid'] = {t: _hybrid(A, t) for t in common}
        gap = {m: sum(A[m]['real'][t] - A[m]['desync'][t] for t in vert) / len(vert)
               for m in ('ras', 'simple', 'pmi', 'rarity', 'productive')}
        gap['hybrid'] = sum(_hybrid(A, t) - _hybrid(A, t, 'desync') for t in vert) / len(vert)
        res = {'n': len(common), 'mean': {m: sum(v.values()) / len(v) for m, v in val.items()}, 'gap': gap,
               'cmp': {}}
        for pol in ('under', 'over'):
            sel = [t for t in common if ATOM_POL[t] == pol]
            res[f'mean_{pol}'] = {m: sum(val[m][t] for t in sel) / len(sel) for m in val}
        for a in ('ras', 'simple'):
            for b in ('pmi', 'rarity', 'productive', 'hybrid'):
                x = sorted(((val[a][t] - val[b][t], t) for t in common), reverse=True)
                dx = [v for v, _ in x]
                n = len(dx)
                # 去掉增益最大的 k 条原子后平均差降到 0 以下所需的 k
                k0 = next((k for k in range(n + 1) if sum(dx[k:]) <= 0), n)
                pos = sum(v for v in dx if v > 0)
                boot = []
                for _ in range(N_ATOM_BOOT):
                    smp = [dx[rng.randrange(n)] for _ in range(n)]
                    boot.append(sum(smp) / n)
                boot.sort()
                res['cmp'][f'{a}-{b}'] = {
                    'mean': sum(dx) / n, 'median': sorted(dx)[n // 2] if n % 2 else (sorted(dx)[n // 2 - 1] + sorted(dx)[n // 2]) / 2,
                    'hl': _hl(dx),
                    'band': {str(dl): {'better': sum(v > dl for v in dx), 'equiv': sum(abs(v) <= dl for v in dx),
                                       'worse': sum(v < -dl for v in dx)} for dl in DELTAS},
                    'k_to_zero': k0,
                    'top3_share': sum(v for v in dx[:3] if v > 0) / pos if pos > 0 else None,
                    'top3': [(t, round(v, 3)) for v, t in x[:3]],
                    'bottom3': [(t, round(v, 3)) for v, t in x[-3:]],
                    'atom_boot_ci': (boot[int(0.025 * N_ATOM_BOOT)], boot[int(0.975 * N_ATOM_BOOT) - 1]),
                }
        out[s] = res
        c = res['cmp']
        print(f"{s:<12} mean ras {res['mean']['ras']:.3f} simple {res['mean']['simple']:.3f} "
              f"pmi {res['mean']['pmi']:.3f} rar {res['mean']['rarity']:.3f} hyb {res['mean']['hybrid']:.3f} | "
              f"gap ras {gap['ras']:.3f} hyb {gap['hybrid']:.3f}")
        for k in ('ras-pmi', 'ras-rarity', 'ras-hybrid', 'simple-hybrid'):
            v = c[k]
            print(f"   {k:<14} mean {v['mean']:+.3f} HL {v['hl']:+.3f} med {v['median']:+.3f} "
                  f"atomCI [{v['atom_boot_ci'][0]:+.3f},{v['atom_boot_ci'][1]:+.3f}] k0 {v['k_to_zero']} "
                  f"band.05 {v['band']['0.05']} top3 {v['top3']}")
    _dump('practical', out)


# ---------------------------------------------------------------------------- cost

def _loc(mod, names):
    """函数体的非空、非注释、非文档字符串行数。"""
    import ast
    import inspect
    src = inspect.getsource(mod)
    tree = ast.parse(src)
    lines = src.splitlines()
    tot = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant):
                body = body[1:]
            if not body:
                continue
            seg = lines[body[0].lineno - 1: node.end_lineno]
            tot += sum(1 for ln in seg if ln.strip() and not ln.strip().startswith('#'))
    return tot


def run_cost(names, reps=3):
    from score_learning.induction import ablation, baselines
    fin, tr, sets = load_sets(names)
    real_fns = {f: getattr(fri, f'real_{f.lower()}_counts') for f in fri.ALL_FIELDS}
    null_fns = {f: getattr(fri, f'null_{f.lower()}_counts') for f in fri.ALL_FIELDS}
    dp_only = {'V', 'O', 'W'}
    no_cat = {'P', 'Q', 'J', 'F'}

    def med(fn):
        ts = []
        r = None
        for _ in range(reps):
            t0 = time.perf_counter()
            r = fn()
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2], r

    out = {'reps': reps, 'replay': fri.REPLAY, 'sets': {}}
    t_cat, (dp_cat, pitch_cat) = med(lambda: fri.catalogs(tr))
    t_cal, cal = med(lambda: fri.calibrate_from_desync(tr, seed=0))
    t_soft, _ = med(lambda: fri.calibrate_soft_z(tr, cal['z_cut'], cal['q']))
    out['train'] = {'n_pieces': len(tr), 'catalogs_s': t_cat, 'calibrate_s': t_cal, 'soft_z_s': t_soft}
    print('train', out['train'], flush=True)
    for s, hold in sets.items():
        n_sl = sum(len(p.slices) for p in hold)

        def reals():
            return {f: real_fns[f](hold) for f in fri.ALL_FIELDS}

        def nulls():
            rng = random.Random(SEEDS[s])
            o = {}
            for f in fri.ALL_FIELDS:
                if f in no_cat:
                    o[f] = null_fns[f](hold, rng)
                elif f in dp_only:
                    o[f] = null_fns[f](hold, dp_cat, rng)
                else:
                    o[f] = null_fns[f](hold, pitch_cat, rng)
            return o

        t_real, rc = med(reals)
        t_null, nc = med(nulls)
        t_res, f = med(lambda: {k: fri.residual(rc[k], nc[k]) for k in fri.ALL_FIELDS})
        t_pmi, _ = med(lambda: baselines.indep_fields(f))
        t_rar, _ = med(lambda: ablation.uniform_fields(f))
        t_prod, _ = med(lambda: baselines.productive_fields(f))
        out['sets'][s] = {'n_pieces': len(hold), 'n_slices': n_sl, 'real_s': t_real, 'replay_s': t_null,
                          'residual_s': t_res, 'pmi_s': t_pmi, 'rarity_s': t_rar, 'productive_s': t_prod}
        print(s, out['sets'][s], flush=True)
    null_names = [f'null_{f.lower()}_counts' for f in fri.ALL_FIELDS] + ['catalogs', '_complete_pairs']
    cal_names = ['calibrate_from_desync', 'calibrate_soft_z', '_desync_fields']
    out['loc'] = {
        'real_counting': _loc(fri, [f'real_{f.lower()}_counts' for f in fri.ALL_FIELDS]),
        'replay_null': _loc(fri, null_names),
        'calibration': _loc(fri, cal_names),
        'pmi': _loc(baselines, ['indep_rows', 'indep_fields']),
        'rarity': _loc(ablation, ['uniform_fields']),
        'productive': _loc(baselines, ['productive_rows', 'productive_fields', '_sig', '_bits', '_cell_ids']),
    }
    print('loc', out['loc'])
    _dump('cost', out)


# ---------------------------------------------------------------------------- balanced AUC, leave-one-field-out

BAL_METHODS = ('ras', 'simple', 'pmi', 'rarity', 'hybrid')
BAL_PAIRS = (('ras', 'pmi'), ('ras', 'rarity'), ('ras', 'hybrid'),
             ('simple', 'pmi'), ('simple', 'rarity'), ('simple', 'hybrid'))


def _group_mean(d, atoms, key):
    g = defaultdict(list)
    for a in atoms:
        g[key(a)].append(d[a])
    return sum(sum(v) / len(v) for v in g.values()) / len(g) if g else float('nan')


def balanced_stats(fm):
    auc = {m: atom_auc(fm[m]) for m in ('ras', 'simple', 'pmi', 'rarity')}
    auc['hybrid'] = {a: (auc['rarity'] if ATOM_POL[a] == 'under' else auc['pmi']).get(a) for a in auc['pmi']}
    common = EVAL_ATOMS & set.intersection(*[{a for a, v in d.items() if v is not None} for d in auc.values()])
    out = {}
    for m in BAL_METHODS:
        d = auc[m]
        out[m] = {'atom': sum(d[a] for a in common) / len(common),
                  'field': _group_mean(d, common, lambda a: ATOM_FIELD[a]),
                  'category': _group_mean(d, common, lambda a: CATEGORY[ATOM_FIELD[a]]),
                  'family': _group_mean(d, common, lambda a: a[0])}
    return out, auc, common


def run_balanced(name, n_half=300):
    fin, tr, sets = load_sets([name])
    hold = sets[name]
    seed = SEEDS[name]
    t0 = time.time()
    dp, pc = fri.catalogs(tr)
    dpp, pcp = pooled_cats(tr)
    per, pers = piece_counts_cats(dp, pc, hold, seed), piece_counts_cats(dpp, pcp, hold, seed)

    def build(idx):
        f = bs.fields_from(per, idx)
        fm = {'ras': f, 'pmi': indep_fields(f), 'rarity': uniform_fields(f), 'simple': bs.fields_from(pers, idx)}
        return balanced_stats(fm)

    full = list(range(len(hold)))
    point, auc, common = build(full)
    # 留一场 / 留一类：点估计
    lofo = {}
    for grp, key in (('field', lambda a: ATOM_FIELD[a]), ('category', lambda a: CATEGORY[ATOM_FIELD[a]])):
        for g in sorted({key(a) for a in common}):
            keep = [a for a in common if key(a) != g]
            mean = {m: sum(auc[m][a] for a in keep) / len(keep) for m in BAL_METHODS}
            lofo[f'{grp}|{g}'] = {'n_dropped': len(common) - len(keep),
                                 **{f'{a}-{b}': mean[a] - mean[b] for a, b in BAL_PAIRS}}
    rng = random.Random(2026)
    boot = []
    for b in range(n_half):
        boot.append(build(rng.sample(full, len(hold) // 2))[0])
        if (b + 1) % 50 == 0:
            print(f'  {name} half {b + 1} {time.time() - t0:.0f}s', flush=True)
    summ = {}
    for k in point['ras']:
        s = {m: point[m][k] for m in BAL_METHODS}
        for a, b2 in BAL_PAIRS:
            d = [x[a][k] - x[b2][k] for x in boot]
            s[f'{a}-{b2}'] = point[a][k] - point[b2][k]
            s[f'{a}-{b2}_ci'] = bs._ci(d)
            s[f'{a}-{b2}_p_le0'] = sum(v <= 0 for v in d) / len(d)
            s[f'{a}-{b2}_p_ge0'] = sum(v >= 0 for v in d) / len(d)
        summ[k] = s
        print(f'  {k:<9}' + ' '.join(f'{m} {s[m]:.3f}' for m in BAL_METHODS) + '  ' +
              ' '.join(f'{a}-{b2} {s[f"{a}-{b2}"]:+.3f}{tuple(round(v, 3) for v in s[f"{a}-{b2}_ci"])}'
                       for a, b2 in BAL_PAIRS), flush=True)
    fields = Counter(ATOM_FIELD[a] for a in common)
    _dump(f'balanced_{name}', {'commit': _git_head(), 'set': name, 'n': len(hold), 'n_half': n_half,
                               'summary': summ, 'lofo': lofo, 'atoms_per_field': dict(fields),
                               'common': sorted(common)})


# ---------------------------------------------------------------------------- vocabulary robustness (coarsened cells)

def _drop(keys):
    def f(cell):
        parts = []
        for seg in cell.split('/to/'):
            kept = [x for x in seg.split('|') if not ('=' in x and x.split('=', 1)[0] in keys)]
            parts.append('|'.join(kept) or '*')
        return '/to/'.join(parts)
    return f


def _h_merge(cell):
    pair, gap = cell.split('|gap=')
    return f'{pair}|gap=' + {'le8': 'le12', '8to12': 'le12', '12to19': 'gt12', 'gt19': 'gt12'}.get(gap, gap)


VOCABS = {
    'full': {},
    'no_degree': {'M': _drop({'lt', 'b7'}), 'S': _drop({'nlt', 'n7', 'missTon'}), 'P': None, 'U': None,
                  'K': _drop({'b', 'set', 'ch'}), 'J': _drop({'b', 'set', 'ch'}), 'Q': _drop({'b', 'sop'})},
    'no_position': {'K': _drop({'slot'}), 'J': _drop({'slot'}), 'P': None, 'Q': None},
    'coarse_geometry': {'V': _drop({'role', 'leapU'}), 'H': _h_merge, 'D': _drop({'hov', 'holdO'})},
}


def _coarse_per(per, vocab):
    out = []
    for row in per:
        new = {}
        for f, (real, null) in row.items():
            fn = vocab.get(f, 'keep')
            if fn is None:
                new[f] = (Counter(), Counter())
            elif fn == 'keep':
                new[f] = (real, null)
            else:
                r2, n2 = Counter(), Counter()
                for c, v in real.items():
                    r2[fn(c)] += v
                for c, v in null.items():
                    n2[fn(c)] += v
                new[f] = (r2, n2)
        out.append(new)
    return out


def _coarse_groups(per, vocab):
    g = defaultdict(lambda: defaultdict(set))
    for row in per:
        for f, (real, null) in row.items():
            fn = vocab.get(f, 'keep')
            if fn is None:
                continue
            for c in set(real) | set(null):
                g[f][c if fn == 'keep' else fn(c)].add(c)
    return g


def coarse_atom_auc(fields, groups):
    """粗化词汇上的原子 AUC：原子只有在每个粗格都“纯”（其细格全部满足或全部不满足谓词）时才可表达。"""
    z = annotate_z(fields)
    out = {}
    for name, fld, pol, pred in PREDS:
        if fld not in groups:
            out[name] = None
            continue
        cells = [r[0] for r in fri._ranked(fields.get(fld, []), 3 if fld in SPARSE_FIELDS else MIN_NULL)]
        same = [p for _n, f, pl, p in PREDS if f == fld and pl == pol]
        sgn = -1.0 if pol == 'under' else 1.0
        pos, neg, ok = [], [], True
        for c in cells:
            fine = groups[fld].get(c, set())
            flags = [pred(x, fld) for x in fine]
            if any(flags) and not all(flags):
                ok = False
                break
            if flags and all(flags):
                pos.append(sgn * z[fld][c])
            elif not any(p(x, fld) for x in fine for p in same):
                neg.append(sgn * z[fld][c])
        if not ok or not pos or not neg:
            out[name] = None
            continue
        neg.sort()
        s = sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p))
                for p in pos)
        out[name] = s / (len(pos) * len(neg))
    return out


def run_vocab(names, n_half=200):
    fin, tr, sets = load_sets(names)
    cfg = fin['cfg']
    dp, pc = fri.catalogs(tr)
    dpp, pcp = pooled_cats(tr)
    out = {'commit': _git_head(), 'n_half': n_half, 'sets': {}}
    for name, hold in sets.items():
        t0 = time.time()
        seed = SEEDS[name]
        rng_d = random.Random(99)
        hold_d = [_desync_piece(p, rng_d) for p in hold]
        base = {'ras': (piece_counts_cats(dp, pc, hold, seed), piece_counts_cats(dp, pc, hold_d, seed)),
                'simple': (piece_counts_cats(dpp, pcp, hold, seed), piece_counts_cats(dpp, pcp, hold_d, seed))}
        res = {}
        full_auc = None
        for vn, vocab in VOCABS.items():
            groups = _coarse_groups(base['ras'][0] + base['ras'][1], vocab)
            cp = {m: (_coarse_per(a, vocab), _coarse_per(b, vocab)) for m, (a, b) in base.items()}

            def stats(idx):
                f = bs.fields_from(cp['ras'][0], idx)
                fd = bs.fields_from(cp['ras'][1], idx)
                fm = {'ras': f, 'pmi': indep_fields(f), 'rarity': uniform_fields(f),
                      'simple': bs.fields_from(cp['simple'][0], idx)}
                fmd = {'ras': fd, 'pmi': indep_fields(fd), 'rarity': uniform_fields(fd),
                       'simple': bs.fields_from(cp['simple'][1], idx)}
                auc = {m: coarse_atom_auc(x, groups) for m, x in fm.items()}
                auc_d = {m: coarse_atom_auc(x, groups) for m, x in fmd.items()}
                for d in (auc, auc_d):
                    d['hybrid'] = {a: (d['rarity'] if ATOM_POL[a] == 'under' else d['pmi']).get(a) for a in d['pmi']}
                return auc, auc_d, f, fd

            full_idx = list(range(len(hold)))
            auc, auc_d, f, fd = stats(full_idx)
            common = EVAL_ATOMS & set.intersection(*[{a for a, v in d.items() if v is not None} for d in auc.values()])
            vert = common & VERT_ATOMS & set.intersection(*[{a for a, v in d.items() if v is not None}
                                                            for d in auc_d.values()])
            if vn == 'full':
                full_auc = auc
            def mean(d, s):
                v = [d[a] for a in s if d.get(a) is not None]
                return sum(v) / len(v) if v else float('nan')
            ms = ('ras', 'simple', 'pmi', 'rarity', 'hybrid')
            point = {m: {'auc': mean(auc[m], common), 'gap': mean(auc[m], vert) - mean(auc_d[m], vert),
                         'auc_full_vocab_same_atoms': mean(full_auc[m], common)} for m in ms}
            und, ov = eval_disc(f, cfg)
            zr, zd = annotate_z(f), annotate_z(fd)
            disc = [('under', fl, c) for fl, c in und] + [('over', fl, c) for fl, c in ov]
            cls = Counter((d[0], classify(d[1], SGN_POL[d[0]] * zr[d[1]][d[2]],
                                          SGN_POL[d[0]] * zd.get(d[1], {}).get(d[2], 0.0))) for d in disc)
            rng = random.Random(2026)
            boot = []
            for _b in range(n_half):
                idx = rng.sample(full_idx, len(hold) // 2)
                a2, ad2, _f, _fd = stats(idx)
                boot.append({m: (mean(a2[m], common), mean(a2[m], vert) - mean(ad2[m], vert)) for m in ms})
            diffs = {}
            for a, b2 in BAL_PAIRS:
                for j, k in ((0, 'auc'), (1, 'gap')):
                    d = [x[a][j] - x[b2][j] for x in boot if not (math.isnan(x[a][j]) or math.isnan(x[b2][j]))]
                    diffs[f'{k}|{a}-{b2}'] = {'d': point[a][k] - point[b2][k], 'ci': bs._ci(d) if len(d) > 10 else None}
            res[vn] = {'n_common': len(common), 'n_vert': len(vert), 'atoms': sorted(common), 'point': point,
                       'diffs': diffs, 'n_under': len(und), 'n_over': len(ov),
                       'classes': {f'{p}|{c}': v for (p, c), v in cls.items()}}
            print(f'  {name} {vn:<16} atoms {len(common)} vert {len(vert)} ' +
                  ' '.join(f'{m} {point[m]["auc"]:.3f}/{point[m]["gap"]:+.3f}' for m in ms) +
                  f' | u/o {len(und)}/{len(ov)} {dict(cls)} {time.time() - t0:.0f}s', flush=True)
        out['sets'][name] = res
    _dump('vocab', out)


SGN_POL = {'under': -1.0, 'over': 1.0}


# ---------------------------------------------------------------------------- stability of the null vs training size

STAB_KS = (16, 32, 64, 128)
STAB_DRAWS = 5


def _spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float('nan')


def _cat_hist(cat):
    return {v: Counter(cat[v]) for v in VOICES}


def run_nullstab():
    fin, _tr, sets = load_sets(['hold_a'])
    by, usable, _ = load_chorale_pool()
    hold = sets['hold_a']
    excl = set(fin['hold_ids']) | set(fin.get('hold_b_ids', [])) | set(fin.get('dev_ids', []))
    pool = [i for i in usable[:-40] if i not in excl]
    cfg = fin['cfg']
    out = {'commit': _git_head(), 'pool': len(pool), 'ks': {}}
    for k in STAB_KS:
        t0 = time.time()
        draws = []
        for s in range(STAB_DRAWS):
            ids = random.Random(5000 + 31 * k + s).sample(pool, k)
            tr = [by[i] for i in ids]
            dp, pcat = fri.catalogs(tr)
            f = fri.holdout_fields(tr, hold, seed=0)
            z = annotate_z(f)
            und, ov = eval_disc(f, cfg)
            cal = fri.calibrate_from_desync(tr, seed=0)
            und_c, ov_c = eval_disc(f, {**cfg, 'z_cut': cal['z_cut'], 'q': cal['q']})
            n0 = {(fld, r[0]): r[3] for fld, rows in f.items() if fld in EVAL_FIELDS for r in rows}
            N0 = {fld: sum(r[3] for r in rows) or 1 for fld, rows in f.items() if fld in EVAL_FIELDS}
            draws.append({'pitch': _cat_hist(pcat), 'step': _cat_hist(dp),
                          'share': {c: v / N0[c[0]] for c, v in n0.items()}, 'n0': n0,
                          'z': {(fld, c): v for fld, d in z.items() if fld in EVAL_FIELDS for c, v in d.items()},
                          'disc': und | ov, 'disc_cal': und_c | ov_c, 'cal': (cal['z_cut'], cal['q']),
                          'auc': sum(v for a, v in atom_auc(f).items() if a in EVAL_ATOMS and v is not None) /
                                 max(sum(1 for a, v in atom_auc(f).items() if a in EVAL_ATOMS and v is not None), 1)})
        pairs = [(a, b) for a in range(STAB_DRAWS) for b in range(a + 1, STAB_DRAWS)]
        tv = lambda key: sum(sum(tvd(draws[a][key][v], draws[b][key][v]) for v in VOICES) / 4
                             for a, b in pairs) / len(pairs)
        cells = set.intersection(*[set(d['z']) for d in draws])
        rho = sum(_spearman([draws[a]['z'][c] for c in sorted(cells)], [draws[b]['z'][c] for c in sorted(cells)])
                  for a, b in pairs) / len(pairs)
        big = [c for c in set.intersection(*[set(d['n0']) for d in draws])
               if sum(d['n0'][c] for d in draws) / STAB_DRAWS >= MIN_NULL]
        cv = []
        for c in big:
            xs = [d['share'][c] for d in draws]
            m = sum(xs) / len(xs)
            cv.append(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / m if m else 0.0)
        cv.sort()
        jac = lambda key: sum(len(draws[a][key] & draws[b][key]) / max(len(draws[a][key] | draws[b][key]), 1)
                              for a, b in pairs) / len(pairs)
        aucs = [d['auc'] for d in draws]
        row = {'tvd_pitch': tv('pitch'), 'tvd_step': tv('step'), 'z_spearman': rho, 'n_cells': len(cells),
               'cv_median': cv[len(cv) // 2], 'cv_p90': cv[int(0.9 * (len(cv) - 1))], 'n_cv_cells': len(cv),
               'jaccard_frozen': jac('disc'), 'jaccard_recal': jac('disc_cal'),
               'cal': [d['cal'] for d in draws], 'auc_mean': sum(aucs) / len(aucs),
               'auc_sd': math.sqrt(sum((x - sum(aucs) / len(aucs)) ** 2 for x in aucs) / (len(aucs) - 1)),
               'n_disc': [len(d['disc']) for d in draws]}
        out['ks'][str(k)] = row
        print(f'k={k} ' + ' '.join(f'{a} {v:.3f}' if isinstance(v, float) else f'{a} {v}' for a, v in row.items())
              + f' {time.time() - t0:.0f}s', flush=True)
    _dump('nullstab', out)


def _dump(tag, obj):
    path = OUT.format(tag)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o),
                  ensure_ascii=False)
    print('wrote', path, flush=True)


def run_cells(names):
    """每场候选格子数：出现过的格子、可检验格子、可评估参考约束、冻结发现。"""
    fin, tr, sets = load_sets(names)
    cfg = fin['cfg']
    dp, pc = fri.catalogs(tr)
    common = None
    out = {'commit': _git_head(), 'sets': {}}
    for name in names:
        hold = sets[name]
        per = piece_counts_cats(dp, pc, hold, SEEDS[name])
        f = bs.fields_from(per, list(range(len(hold))))
        und, ov = eval_disc(f, cfg)
        auc = atom_auc(f)
        rows = {}
        for fld in EVAL_FIELDS:
            rs = f.get(fld, [])
            min_n = 3 if fld in fri.SPARSE_FIELDS else fri.MIN_NULL
            rows[fld] = {'cells': len(rs),
                         'testable': sum(1 for _c, _r, nr, n0, *_ in rs if not (n0 < min_n and nr < min_n)),
                         'atoms': sum(1 for a in EVAL_ATOMS if ATOM_FIELD[a] == fld),
                         'atoms_eval': sum(1 for a in EVAL_ATOMS if ATOM_FIELD[a] == fld and auc.get(a) is not None),
                         'under': sum(1 for x, _ in und if x == fld), 'over': sum(1 for x, _ in ov if x == fld),
                         'sparse': fld in fri.SPARSE_FIELDS}
            print(name, fld, rows[fld], flush=True)
        out['sets'][name] = {'n': len(hold), 'fields': rows}
    _dump('cells', out)


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'cells':
        run_cells(args or ['hold_a', 'hold_c'])
        return
    if cmd == 'attrib':
        run_attrib(args or ['hold_a', 'hold_b', 'hold_c'])
    elif cmd == 'pvals':
        n = int(args[-1]) if args and args[-1].isdigit() else 100
        run_pvals([a for a in args if not a.isdigit()] or ['hold_a', 'hold_c'], n)
    elif cmd == 'simple':
        run_simple(args[0], int(args[1]) if len(args) > 1 else 300)
    elif cmd == 'expert':
        run_expert(*sys.argv[2:3])
    elif cmd == 'cpack':
        run_cpack()
    elif cmd == 'tests':
        run_tests()
    elif cmd == 'hybrid':
        run_hybrid(args[0], int(args[1]) if len(args) > 1 else 300)
    elif cmd == 'calops':
        run_calops(args or ['hold_a', 'hold_c'])
    elif cmd == 'typeboot':
        n = int(args[-1]) if args and args[-1].isdigit() else N_TYPEBOOT
        run_typeboot([a for a in args if not a.isdigit()] or ['hold_a', 'hold_c'], n)
    elif cmd == 'attrbase':
        run_attrbase(args or ['hold_a', 'hold_c'])
    elif cmd == 'prereg3':
        run_prereg3(int(args[0]) if args else 1000)
    elif cmd == 'practical':
        run_practical()
    elif cmd == 'cost':
        run_cost(args or ['hold_a', 'hold_c'])
    elif cmd == 'balanced':
        run_balanced(args[0], int(args[1]) if len(args) > 1 else 300)
    elif cmd == 'nullstab':
        run_nullstab()
    elif cmd == 'vocab':
        n = int(args[-1]) if args and args[-1].isdigit() else 200
        run_vocab([a for a in args if not a.isdigit()] or ['hold_a', 'hold_c'], n)


if __name__ == '__main__':
    main()
