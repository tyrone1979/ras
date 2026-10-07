# -*- coding: utf-8 -*-
"""零模型对照与机会水平（与主表同一训练/留出划分）。

  venv/bin/python -m score_learning.induction.baselines

- indep：同一格子，零模型换成「格内各特征位独立」（PMI 的多位推广），同样在时移谱上校准；
- random：每场随机抽与 RAS-FR 同样数量的欠密/过密格，给出 Completeness 的机会水平；
- noise：每格再劈一个无关随机位，看恢复是否稳定。
"""
from __future__ import annotations

import json
import random
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from score_learning.induction.fri import (
    ALL_FIELDS, MIN_NULL, REPLAY, SPARSE_FIELDS, Row, _ranked, annotate_z, book_predicates,
    calibrate_from_desync, discoveries, geom_recovery, holdout_fields, residual,
)

OUT = 'runs/induction/paper/baselines.json'

# 这些场的零模型按教科书对比构造（F 不抽 2/4 小节句、句末排除主低音；W 强制同向 55%；
# R 混合音库再移八度造越界），对应原子在 RAS-FR 下必然显著，比较时一律排除。
DESIGNED_NULL_FIELDS = ('R', 'W', 'F')
EVAL_FIELDS = tuple(f for f in ALL_FIELDS if f not in DESIGNED_NULL_FIELDS)
EVAL_ATOMS = frozenset(n for n, f, _p, _q in book_predicates() if f in EVAL_FIELDS)


def _bits(cell: str) -> List[str]:
    body, slot = cell, None
    if '/to/' in body and '|slot=' in body:
        body, slot = body.rsplit('|slot=', 1)
    if '/to/' in body:
        a, b = body.split('/to/', 1)
        bits = ['a:' + x for x in a.split('|')] + ['b:' + x for x in b.split('|')]
    else:
        bits = body.split('|')
    if slot is not None:
        bits.append('slot=' + slot)
    return bits


def _sig(bits: List[str]) -> Tuple[str, ...]:
    return tuple(b.split('=', 1)[0] if '=' in b else f'#{i}' for i, b in enumerate(bits))


def indep_rows(rows: List[Row]) -> List[Row]:
    """零模型 = 同签名格内各位边缘分布之积（支撑集沿用原场的格）。"""
    real = Counter({r[0]: r[2] for r in rows})
    groups: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for cell in real:
        groups[_sig(_bits(cell))].append(cell)
    null = Counter()
    for sig, cells in groups.items():
        n_g = sum(real[c] for c in cells)
        if n_g == 0:
            for c in cells:
                null[c] = 0.0
            continue
        marg = [Counter() for _ in sig]
        for c in cells:
            for i, b in enumerate(_bits(c)):
                marg[i][b] += real[c]
        for c in cells:
            p = 1.0
            for i, b in enumerate(_bits(c)):
                p *= marg[i][b] / n_g
            null[c] = REPLAY * n_g * p
    return residual(real, null)


def indep_fields(fields: Dict[str, List[Row]]) -> Dict[str, List[Row]]:
    return {f: (indep_rows(r) if f in ALL_FIELDS else r) for f, r in fields.items()}


def productive_rows(rows: List[Row]) -> List[Row]:
    """Webb (2007) 式 productive 检验的零模型：格 c 的位集合按每一种二分 {X, Y} 给出期望
    n_g·P(X)·P(Y)；零模型取这些期望中离观测最近的那个，故只有观测高于（低于）全部二分期望时
    才有正（负）偏差。支撑集与 indep_rows 相同。"""
    real = Counter({r[0]: r[2] for r in rows})
    groups: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for cell in real:
        groups[_sig(_bits(cell))].append(cell)
    null = Counter()
    for sig, cells in groups.items():
        n_g = sum(real[c] for c in cells)
        if n_g == 0 or len(sig) < 2:
            for c in cells:
                null[c] = REPLAY * real[c]
            continue
        ids = np.stack([_cell_ids(sig, c) for c in cells], axis=1)
        enc, px, py = _SIG_PARTS[sig]
        w = np.array([real[c] for c in cells], dtype=float)
        proj = np.empty(ids.shape)
        for j in range(ids.shape[0]):
            proj[j] = np.bincount(ids[j], weights=w, minlength=len(enc[j]))[ids[j]]
        exps = proj[px] * proj[py] / n_g
        val = np.clip(w, exps.min(axis=0), exps.max(axis=0))
        for c, v in zip(cells, val):
            null[c] = REPLAY * float(v)
    return residual(real, null)


# 与计数无关的缓存：每个签名的子集编码器与二分下标对，每个格在各子集上的投影编号。
_SIG_PARTS: Dict[Tuple[str, ...], tuple] = {}
_CELL_IDS: Dict[str, np.ndarray] = {}


def _cell_ids(sig: Tuple[str, ...], cell: str) -> np.ndarray:
    hit = _CELL_IDS.get(cell)
    if hit is not None:
        return hit
    m = len(sig)
    full = (1 << m) - 1
    if sig not in _SIG_PARTS:
        subs = range(1, full)
        pairs = [(s - 1, (full ^ s) - 1) for s in subs if s < (full ^ s)]
        _SIG_PARTS[sig] = ([{} for _ in subs], np.array([a for a, _ in pairs]),
                           np.array([b for _, b in pairs]))
    enc = _SIG_PARTS[sig][0]
    b = _bits(cell)
    out = np.array([enc[s - 1].setdefault(tuple(b[i] for i in range(m) if s >> i & 1), len(enc[s - 1]))
                    for s in range(1, full)], dtype=np.int64)
    _CELL_IDS[cell] = out
    return out


def productive_fields(fields: Dict[str, List[Row]]) -> Dict[str, List[Row]]:
    return {f: (productive_rows(r) if f in ALL_FIELDS else r) for f, r in fields.items()}


LOGLIN_ITERS = 40
_PAIR_IDS: Dict[str, np.ndarray] = {}
_PAIR_ENC: Dict[Tuple[str, ...], List[dict]] = {}


def _pair_ids(sig: Tuple[str, ...], cell: str) -> np.ndarray:
    hit = _PAIR_IDS.get(cell)
    if hit is not None:
        return hit
    m = len(sig)
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    enc = _PAIR_ENC.setdefault(sig, [{} for _ in pairs])
    b = _bits(cell)
    out = np.array([enc[k].setdefault((b[i], b[j]), len(enc[k])) for k, (i, j) in enumerate(pairs)],
                   dtype=np.int64)
    _PAIR_IDS[cell] = out
    return out


def loglin_rows(rows: List[Row]) -> List[Row]:
    """二阶对数线性零模型：同签名格上用 IPF 拟合与真实计数全部两两位边缘一致的期望，
    偏差只来自三阶及以上的交互。支撑集与 indep_rows 相同。"""
    real = Counter({r[0]: r[2] for r in rows})
    groups: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for cell in real:
        groups[_sig(_bits(cell))].append(cell)
    null = Counter()
    for sig, cells in groups.items():
        n_g = sum(real[c] for c in cells)
        if n_g == 0 or len(sig) < 3:
            for c in cells:
                null[c] = REPLAY * real[c]
            continue
        ids = np.stack([_pair_ids(sig, c) for c in cells], axis=1)
        w = np.array([real[c] for c in cells], dtype=float)
        target = [np.bincount(ids[k], weights=w) for k in range(ids.shape[0])]
        mu = np.full(len(cells), n_g / len(cells))
        for _ in range(LOGLIN_ITERS):
            for k in range(ids.shape[0]):
                cur = np.bincount(ids[k], weights=mu, minlength=len(target[k]))
                ratio = np.divide(target[k], cur, out=np.zeros_like(cur), where=cur > 0)
                mu = mu * ratio[ids[k]]
        for c, v in zip(cells, mu):
            null[c] = REPLAY * float(v)
    return residual(real, null)


def loglin_fields(fields: Dict[str, List[Row]]) -> Dict[str, List[Row]]:
    return {f: (loglin_rows(r) if f in ALL_FIELDS else r) for f, r in fields.items()}


def random_discovery(fields: Dict[str, List[Row]], cfg: dict, n_draw: int = 500,
                     seed: int = 0) -> dict:
    """每场每极性随机抽与 RAS-FR 同数的格，统计各原子命中概率与 Completeness。"""
    und, ov = discoveries(fields, cfg['z_cut'], cfg['q'], soft=bool(cfg.get('soft', True)),
                          soft_z=cfg.get('soft_z', -1.5))
    ranked = {f: [r[0] for r in _ranked(rows, 3 if f in SPARSE_FIELDS else MIN_NULL)]
              for f, rows in fields.items() if f in ALL_FIELDS}
    n_u = Counter(f for f, _ in und)
    n_o = Counter(f for f, _ in ov)
    rng = random.Random(seed)
    preds = book_predicates()
    hit_n = Counter()
    comps = []
    for _ in range(n_draw):
        pick = {}
        for f, cells in ranked.items():
            pick[(f, 'under')] = set(rng.sample(cells, min(n_u[f], len(cells))))
            pick[(f, 'over')] = set(rng.sample(cells, min(n_o[f], len(cells))))
        ok = 0
        for name, fld, pol, pred in preds:
            if any(pred(c, fld) for c in pick.get((fld, pol), ())):
                hit_n[name] += 1
                ok += 1
        comps.append(ok / len(preds))
    mu = sum(comps) / len(comps)
    sd = (sum((x - mu) ** 2 for x in comps) / len(comps)) ** 0.5
    return {'comp_mean': mu, 'comp_sd': sd, 'comp_max': max(comps),
            'atom_rate': {n: hit_n[n] / n_draw for n, *_ in preds},
            'n_under': dict(n_u), 'n_over': dict(n_o)}


def _strip_noise(cell: str) -> str:
    return cell[5:] if cell.startswith('nz=') else cell


def noise_fields(fields: Dict[str, List[Row]], seed: int = 0) -> Dict[str, List[Row]]:
    """每格按二项分布劈成 nz=0/1 两格（与音乐无关的特征位）。"""
    rng = random.Random(seed)

    def split(n):
        k = sum(1 for _ in range(int(round(n))) if rng.random() < 0.5)
        return k, n - k

    out = {}
    for f, rows in fields.items():
        if f not in ALL_FIELDS:
            out[f] = rows
            continue
        real, null = Counter(), Counter()
        for cell, _rho, nr, n0, *_ in rows:
            r0, r1 = split(nr)
            z0, z1 = split(n0)
            real[f'nz=0|{cell}'], real[f'nz=1|{cell}'] = r0, r1
            null[f'nz=0|{cell}'], null[f'nz=1|{cell}'] = z0, z1
        out[f] = residual(real, null)
    return out


def recover_stripped(fields: Dict[str, List[Row]], cfg: dict) -> dict:
    und, ov = discoveries(fields, cfg['z_cut'], cfg['q'], soft=bool(cfg.get('soft', True)),
                          soft_z=cfg.get('soft_z', -1.5))
    hits = {}
    for name, fld, pol, pred in book_predicates():
        found = und if pol == 'under' else ov
        hits[name] = any(pred(_strip_noise(c), fld) for f, c in found if f == fld)
    return {'completeness': sum(hits.values()) / len(hits), 'hits': hits,
            'miss': [n for n, v in hits.items() if not v]}


def atom_auc(fields: Dict[str, List[Row]]) -> Dict[str, float]:
    """原子格在同场内按带极性 z 排序的 AUC（机会水平 0.5，与发现集大小无关）。"""
    z = annotate_z(fields)
    preds = book_predicates()
    out = {}
    for name, fld, pol, pred in preds:
        cells = [r[0] for r in _ranked(fields.get(fld, []), 3 if fld in SPARSE_FIELDS else MIN_NULL)]
        same = [p for _n, f, pl, p in preds if f == fld and pl == pol]
        sgn = -1.0 if pol == 'under' else 1.0
        pos = [sgn * z[fld][c] for c in cells if pred(c, fld)]
        neg = [sgn * z[fld][c] for c in cells if not any(p(c, fld) for p in same)]
        if not pos or not neg:
            out[name] = None
            continue
        neg.sort()
        s = sum(bisect_left(neg, p) + 0.5 * (bisect_right(neg, p) - bisect_left(neg, p)) for p in pos)
        out[name] = s / (len(pos) * len(neg))
    return out


def enrichment(fields: Dict[str, List[Row]], cfg: dict) -> dict:
    """发现集被同极性原子覆盖的比例，相对该极性可评测场全部格的覆盖率。"""
    und, ov = discoveries(fields, cfg['z_cut'], cfg['q'], soft=bool(cfg.get('soft', True)),
                          soft_z=cfg.get('soft_z', -1.5))
    preds = book_predicates()
    res = {}
    for pol, found in (('under', und), ('over', ov)):
        cov_d = cov_all = n_d = n_all = 0
        per_field = {}
        for fld in ALL_FIELDS:
            ap = [p for _n, f, pl, p in preds if f == fld and pl == pol]
            if not ap:
                continue
            cells = [r[0] for r in _ranked(fields.get(fld, []), 3 if fld in SPARSE_FIELDS else MIN_NULL)]
            fd = {c for f, c in found if f == fld}
            fd_n = fd_cov = 0
            for c in cells:
                cov = any(p(c, fld) for p in ap)
                n_all += 1
                cov_all += cov
                if c in fd:
                    n_d += 1
                    cov_d += cov
                    fd_n += 1
                    fd_cov += cov
            per_field[fld] = (fd_n, fd_cov, len(cells))
        prec = cov_d / max(n_d, 1)
        base = cov_all / max(n_all, 1)
        res[pol] = {'n_disc': n_d, 'precision': prec, 'base': base,
                    'lift': prec / base if base else None, 'per_field': per_field}
    return res


def precision_at_k(fields: Dict[str, List[Row]], ks=(25, 50, 100, 200), flds=ALL_FIELDS) -> dict:
    """全部可评测格按带极性 z 排序取前 k，被同极性原子覆盖的比例（与发现集大小无关）。"""
    z = annotate_z(fields)
    preds = book_predicates()
    out = {}
    for pol in ('under', 'over'):
        sgn = -1.0 if pol == 'under' else 1.0
        scored = []
        for fld in flds:
            ap = [p for _n, f, pl, p in preds if f == fld and pl == pol]
            if not ap:
                continue
            for c in (r[0] for r in _ranked(fields.get(fld, []), 3 if fld in SPARSE_FIELDS else MIN_NULL)):
                scored.append((sgn * z[fld][c], any(p(c, fld) for p in ap)))
        base = sum(cov for _s, cov in scored) / max(len(scored), 1)
        out[pol] = {'base': base}
        for k in ks:
            out[pol][k] = _tied_precision(scored, k)
    return out


def _tied_precision(scored: List[Tuple[float, bool]], k: int) -> float:
    """前 k 的覆盖率；跨越第 k 名的并列组按随机打破并列的期望计入，与集合遍历顺序无关。"""
    k = min(k, len(scored))
    if k == 0:
        return 0.0
    groups: Dict[float, List[int]] = {}
    for s, cov in scored:
        g = groups.setdefault(s, [0, 0])
        g[0] += 1
        g[1] += cov
    hit, left = 0.0, k
    for s in sorted(groups, reverse=True):
        n, c = groups[s]
        take = min(n, left)
        hit += c * take / n
        left -= take
        if left == 0:
            break
    return hit / k


def metrics(fields: Dict[str, List[Row]], cfg: dict) -> dict:
    auc = atom_auc(fields)
    vals = [v for v in auc.values() if v is not None]
    shared = [v for n, v in auc.items() if not n.startswith('C') and v is not None]
    _, meta = geom_recovery(fields, cfg)
    return {
        'auc_mean': sum(vals) / len(vals),
        'auc_shared': sum(shared) / len(shared),
        'auc': auc,
        'enrich': enrichment(fields, cfg),
        'p_at_k': precision_at_k(fields),
        'completeness': meta['completeness'],
        'miss': [n for n, v in meta['hits'].items() if not v],
    }


def _summ(meta: dict) -> dict:
    return {'completeness': meta['completeness'], 'fdr': meta.get('fdr'),
            'miss': [n for n, v in meta['hits'].items() if not v]}


def main():
    from score_learning.induction.paper import load_chorale_pool, load_palestrina
    by, usable, _ = load_chorale_pool()
    hold_ids, train_ids = usable[-40:], usable[:-40]
    ids16 = random.Random(0).sample(train_ids, 16)
    ch16 = [by[i] for i in ids16]
    hold_ch = [by[i] for i in hold_ids]
    pal = load_palestrina(16)

    print('== RAS-FR (replay null) ==', flush=True)
    cfg = calibrate_from_desync(ch16, seed=0)
    ho = holdout_fields(ch16, hold_ch, seed=0)
    ho_p = holdout_fields(ch16, pal, seed=0)
    _, m_ras = geom_recovery(ho, cfg)
    _, m_ras_p = geom_recovery(ho_p, cfg)
    print('RAS', _summ(m_ras), flush=True)

    print('== independent-bit (PMI) null ==', flush=True)
    cfg_i = calibrate_from_desync(ch16, seed=0, transform=indep_fields)
    _, m_ind = geom_recovery(indep_fields(ho), cfg_i)
    _, m_ind_p = geom_recovery(indep_fields(ho_p), cfg_i)
    _, m_ind_rascfg = geom_recovery(indep_fields(ho), cfg)
    print('indep cfg', {k: v for k, v in cfg_i.items() if k != 'soft_trace'}, flush=True)
    print('indep', _summ(m_ind), flush=True)

    print('== random discovery (chance) ==', flush=True)
    rnd = random_discovery(ho, cfg)
    print('random comp', rnd['comp_mean'], '+-', rnd['comp_sd'], 'max', rnd['comp_max'], flush=True)

    print('== irrelevant noise bit ==', flush=True)
    noise = [recover_stripped(noise_fields(ho, seed=s), cfg) for s in range(5)]
    noise_mu = sum(n['completeness'] for n in noise) / len(noise)
    print('noise comp mean', noise_mu, [n['miss'] for n in noise], flush=True)

    out = {
        'train_ids': ids16,
        'ras': {'cfg': cfg, 'chorale': _summ(m_ras), 'palestrina': _summ(m_ras_p)},
        'indep': {'cfg': cfg_i, 'chorale': _summ(m_ind), 'palestrina': _summ(m_ind_p),
                  'chorale_with_ras_cfg': _summ(m_ind_rascfg)},
        'random_discovery': rnd,
        'noise_bit': {'comp_mean': noise_mu, 'runs': noise},
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
