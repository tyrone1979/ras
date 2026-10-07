# -*- coding: utf-8 -*-
"""投稿一键表：冻结本体、留出 ρ、消融、k-shot、Palestrina 外推。

  venv/bin/python -m score_learning.induction.paper
"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
import subprocess
from typing import Dict, List, Tuple

from score_learning.induction.extra_corpus import load_extra_split, write_extra_musicxml
from score_learning.induction.fri import (
    SOFT_FALSE_MAX, SOFT_Z_GRID, bass_path_soft_adds, calibrate_from_desync,
    desynchronize, discoveries, equivariance_report, fit_fields, geom_recovery,
    holdout_fields, nopair_fields, permute_labels, structure_ablation,
)
from score_learning.induction.learn import load_many
from score_learning.induction.oracle import main_predicates
from score_learning.induction.ras import from_music21, load_chorale

OUT_DIR = 'runs/induction/paper'
N_CHORALE = 371
SEEDS_MAIN = 20
SEEDS_TAIL = 5   # k=64/314 较重


def _strip(fields: dict) -> dict:
    fields = dict(fields)
    fields.pop('_dp', None)
    fields.pop('_pitch', None)
    return fields


def _clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    return x


def _hits(meta: dict) -> dict:
    return _clean({
        'completeness': meta['completeness'],
        'completeness_count': meta.get('completeness_count'),
        'fdr': meta['fdr'],
        'hits': meta['hits'],
        'a1_nullmass_rank': meta.get('a1_nullmass_rank'),
        'atom_star': meta.get('atom_star'),
        'atom_rank': meta.get('atom_rank'),
    })


def _shared_atoms():
    """跨风格共享原子：去掉终止族 C*（体裁相关）。"""
    return [n for n, *_ in main_predicates() if not n.startswith('C')]


def _shared_comp(meta: dict) -> dict:
    atoms = _shared_atoms()
    hits = meta['hits']
    ok = sum(1 for a in atoms if hits.get(a))
    miss = [a for a in atoms if not hits.get(a)]
    return {'completeness': ok / len(atoms), 'n': ok, 'n_atom': len(atoms), 'miss': miss}


def _miss(meta: dict) -> List[str]:
    return [n for n, *_ in main_predicates() if not meta['hits'].get(n)]


def _git_head() -> str:
    try:
        head = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
        dirty = subprocess.call(['git', 'diff', '--quiet', '--', 'score_learning/induction/*.py']) != 0
        return head + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _desync_piece(p, rng: random.Random):
    n = max(len(p.slices), 2)
    return desynchronize(p, {'S': 0, 'A': rng.randint(1, n - 1),
                             'T': rng.randint(1, n - 1), 'B': rng.randint(1, n - 1)})


def load_chorale_pool(n_max: int = N_CHORALE) -> Tuple[Dict[int, object], List[int], dict]:
    os.makedirs(OUT_DIR, exist_ok=True)
    cache_path = os.path.join(OUT_DIR, 'chorales.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            by, usable, skip = pickle.load(f)
        if len(by) + len(skip) >= n_max:
            with open(os.path.join(OUT_DIR, 'skip.json'), 'w') as f:
                json.dump({'skip': skip, 'usable': usable, 'n_usable': len(usable)}, f, indent=2)
            print(f'usable {len(usable)}  skip {len(skip)}  (pkl)', flush=True)
            return by, usable, skip
    skip = {}
    by = {}
    for n in range(1, n_max + 1):
        print(f'  cache {n}/{n_max}', flush=True)
        try:
            by[n] = load_chorale(n, t_max=None)
        except Exception as e:
            skip[n] = str(e)
    usable = sorted(by)
    with open(os.path.join(OUT_DIR, 'skip.json'), 'w') as f:
        json.dump({'skip': skip, 'usable': usable, 'n_usable': len(usable)}, f, indent=2)
    with open(cache_path, 'wb') as f:
        pickle.dump((by, usable, skip), f)
    print(f'usable {len(usable)}  skip {len(skip)}', flush=True)
    return by, usable, skip


def load_palestrina(limit: int = 16):
    from music21 import corpus
    out, tried = [], 0
    for p in corpus.getComposer('palestrina'):
        tried += 1
        if tried > 200 or len(out) >= limit:
            break
        try:
            sc = corpus.parse(p)
            if len(sc.parts) != 4:
                continue
            ras = from_music21(sc, title=str(p), t_max=None)
            if len(ras.slices) < 8:
                continue
            out.append(ras)
        except Exception:
            continue
    return out


def kshot_holdout(by, train_ids, hold_ids, ks, n_seed, tail_seed=SEEDS_TAIL, cfg=None,
                  train_extra=None, hold_extra=None, chorale_only: bool = True) -> dict:
    """k-shot 学习曲线。

    默认 chorale_only=True：训练/留出都只用众赞歌，不拼固定 extra，
    才能看出 Comp 随 k 的斜率。主表「+extra」完整度另报，不进本曲线。
    """
    hold_ch = [by[i] for i in hold_ids if i in by]
    if chorale_only:
        hold = hold_ch
        train_extra = None
        hold_extra = None
    else:
        hold = list(hold_extra or []) + hold_ch[:8]
    pool = [i for i in train_ids if i in by]
    atoms = [n for n, *_ in main_predicates()]
    table = []
    for k in ks:
        n_s = n_seed if k <= 32 else tail_seed
        comps, cnts, fdrs = [], [], []
        atom_hit = {a: [] for a in atoms}
        a1_ranks = []
        for s in range(n_s):
            rng = random.Random(1000 + 17 * k + s)
            kk = min(k, len(pool))
            ids = rng.sample(pool, kk)
            tr = [by[i] for i in ids]
            if train_extra:
                tr = tr + list(train_extra)
            fields = holdout_fields(tr, hold, seed=s)
            _, meta = geom_recovery(fields, cfg)
            comps.append(meta['completeness'])
            cnts.append(meta['completeness_count'])
            fdrs.append(meta['fdr'])
            for a, ok in meta['hits'].items():
                atom_hit[a].append(int(ok))
            if meta.get('a1_nullmass_rank') is not None:
                a1_ranks.append(meta['a1_nullmass_rank'])
        mu = sum(comps) / len(comps)
        sd = (sum((x - mu) ** 2 for x in comps) / len(comps)) ** 0.5
        row = {
            'k': k, 'n_seed': n_s, 'n_train_actual': min(k, len(pool)),
            'comp_mean': mu, 'comp_sd': sd,
            'count_mean': sum(cnts) / len(cnts),
            'fdr_mean': sum(fdrs) / len(fdrs),
            'atom': {a: sum(atom_hit[a]) / len(atom_hit[a]) for a in atoms},
            'a1_n0_rank_mean': (sum(a1_ranks) / len(a1_ranks)) if a1_ranks else None,
        }
        table.append(row)
        print(
            f'k={k}  {mu:.3f}±{sd:.3f}  FDR={row["fdr_mean"]:.3f}  '
            f'A1={row["atom"].get("A1", 0):.2f} A2={row["atom"].get("A2", 0):.2f}  '
            f'seeds={n_s}',
            flush=True,
        )
    proto = (
        'k chorales only → chorale hold (no fixed extra; slope-readable)'
        if chorale_only else
        'k chorales + fixed extra → extra hold + chorale subset'
    )
    return {'protocol': proto, 'chorale_only': chorale_only, 'rows': table}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('== cache chorales ==', flush=True)
    by, usable, skip = load_chorale_pool()
    hold_ids = usable[-40:]
    train_ids = usable[:-40]
    print(f'train_pool {len(train_ids)} hold {len(hold_ids)}', flush=True)

    rng = random.Random(0)
    ids16 = rng.sample(train_ids, 16)
    ch16 = [by[i] for i in ids16]
    hold_ch = [by[i] for i in hold_ids]
    rest = [i for i in train_ids if i not in set(ids16)]
    hold_b_ids = sorted(random.Random(7).sample(rest, 40))
    hold_b = [by[i] for i in hold_b_ids]
    pal = load_palestrina(16)
    print('palestrina n=', len(pal), flush=True)

    # ---------- 主表：只用真实语料（16 首众赞歌训练，无合成谱）----------
    print('== MAIN calibrate on desync chorales ==', flush=True)
    cfg = calibrate_from_desync(ch16, seed=0)
    print('cfg', {k: v for k, v in cfg.items() if k != 'soft_trace'}, flush=True)
    frozen = {
        'commit': _git_head(),
        'cfg': cfg,
        'soft_z_grid': list(SOFT_Z_GRID),
        'soft_false_max': SOFT_FALSE_MAX,
        'train_ids': ids16,
        'hold_ids': hold_ids,
        'hold_b_ids': hold_b_ids,
    }
    with open(os.path.join(OUT_DIR, 'frozen_cfg.json'), 'w') as f:
        json.dump(_clean(frozen), f, indent=2)

    ins = _strip(fit_fields(ch16, seed=0))
    _, m_in = geom_recovery(ins, cfg)
    ho_ch = holdout_fields(ch16, hold_ch, seed=0)
    _, m_ch = geom_recovery(ho_ch, cfg)
    _, m_chb = geom_recovery(holdout_fields(ch16, hold_b, seed=1), cfg)
    m_pal = None
    pal_ho = None
    if len(pal) >= 8:
        pal_ho = holdout_fields(ch16, pal, seed=0)
        _, m_pal = geom_recovery(pal_ho, cfg)
    sh_ch, sh_chb = _shared_comp(m_ch), _shared_comp(m_chb)
    sh_pal = _shared_comp(m_pal) if m_pal else None
    print('chorale hold', m_ch['completeness'], 'shared', sh_ch, flush=True)
    print('chorale hold B', m_chb['completeness'], 'shared', sh_chb, flush=True)
    if m_pal:
        print('palestrina', m_pal['completeness'], 'shared', sh_pal, flush=True)

    # soft 规则开/关
    soft_modes = (('calibrated', cfg), ('bass_path_off', {**cfg, 'soft_z': None}),
                  ('all_soft_off', {**cfg, 'soft': False}))
    soft_abl = {}
    for label, fields in (('chorale', ho_ch), ('palestrina', pal_ho)):
        if fields is None:
            continue
        for mode, c in soft_modes:
            _, m = geom_recovery(fields, c)
            soft_abl[f'{label}/{mode}'] = {
                'completeness': m['completeness'], 'fdr': m['fdr'],
                'shared': _shared_comp(m)['completeness'], 'miss': _miss(m),
            }

    # 负对照：低音粗路径软欠密在真谱 vs 时移谱上的新增率
    soft_neg = {'desync_frac_at_chosen': cfg.get('soft_false_frac'), 'trace': cfg.get('soft_trace')}
    if cfg.get('soft_z') is not None:
        hard, _ = discoveries(ins, cfg['z_cut'], cfg['q'], soft=False)
        elig, adds = bass_path_soft_adds(ins, cfg['soft_z'], hard)
        soft_neg['real_frac_at_chosen'] = len(adds) / max(elig, 1)
        soft_neg['real_adds'] = adds

    _, m_np = geom_recovery(nopair_fields(ho_ch), cfg)
    _, m_po = geom_recovery(holdout_fields(ch16, hold_ch, seed=0, pooled=True), cfg)
    rngp = random.Random(20)
    perm = [permute_labels(p, rngp) for p in ch16]
    _, m_perm = geom_recovery(_strip(fit_fields(perm, seed=3)), cfg)

    print('== k-shot hold-out (chorale-only) ==', flush=True)
    ktab = kshot_holdout(
        by, train_ids, hold_ids, [1, 4, 8, 16, 32, 64, 314], SEEDS_MAIN,
        cfg=cfg, chorale_only=True,
    )

    print('== structure ablation ==', flush=True)
    ablate_txt = structure_ablation(ch16)
    print('== equivariance ==', flush=True)
    eq_txt = equivariance_report(ids16)

    # ---------- 附录：合成补充谱（与 oracle 同源，只作可检测性演示）----------
    print('== APPENDIX synthetic extra ==', flush=True)
    write_extra_musicxml()
    train_extra, hold_extra = load_extra_split(seed=42, hold_frac=0.45)
    trx = ch16 + list(train_extra)
    cfg_x = calibrate_from_desync(trx, seed=0)
    _, m_ho = geom_recovery(holdout_fields(trx, list(hold_extra) + hold_ch[:8], seed=0), cfg_x)
    _, m_ho_x = geom_recovery(holdout_fields(trx, hold_extra, seed=0), cfg_x)
    # 负对照：把留出补充谱做声部时移；若 C 族仍被“发现”，说明读到的是生成脚本而非声部关系
    rngd = random.Random(99)
    hold_extra_d = [_desync_piece(p, rngd) for p in hold_extra]
    _, m_ho_xd = geom_recovery(holdout_fields(trx, hold_extra_d, seed=0), cfg_x)
    c_atoms = [n for n, *_ in main_predicates() if n.startswith('C')]
    rare = ['B2', 'B5', 'C2', 'C3', 'C5', 'C8']

    def _ach_lines(title, meta):
        L = ['', f'## {title} 达成/未达成']
        for n, *_ in main_predicates():
            ok = meta['hits'].get(n, False)
            L.append(f'{n:<8}{"达成" if ok else "未达成"}')
        n_hit = sum(1 for v in meta['hits'].values() if v)
        L.append(f'Completeness={n_hit}/{len(meta["hits"])}={meta["completeness"]:.3f} FDR={meta["fdr"]:.3f}')
        return L

    payload = {
        'skip_n': len(skip),
        'usable_n': len(usable),
        'hold_ids': hold_ids,
        'hold_b_ids': hold_b_ids,
        'main_atoms': [n for n, *_ in main_predicates()],
        'shared_atoms': _shared_atoms(),
        'frozen_commit': frozen['commit'],
        'calibrate': cfg,
        'main': {
            'in_sample_k16': _hits(m_in),
            'holdout_chorale': _hits(m_ch),
            'holdout_chorale_shared': sh_ch,
            'holdout_chorale_b': _hits(m_chb),
            'holdout_chorale_b_shared': sh_chb,
            'palestrina_n': len(pal),
            'palestrina': _hits(m_pal) if m_pal else None,
            'palestrina_shared': sh_pal,
            'soft_ablation': soft_abl,
            'soft_negative_control': soft_neg,
            'nopair_holdout': _hits(m_np),
            'pooled_pitch_holdout': _hits(m_po),
            'ngram_rarity_holdout_comp_count': m_ch.get('completeness_count'),
            'voice_shuffle_in_sample': _hits(m_perm),
            'kshot': ktab,
        },
        'appendix_synthetic': {
            'calibrate': cfg_x,
            'extra_train_n': len(train_extra),
            'extra_hold_n': len(hold_extra),
            'holdout_extra_plus_chorale': _hits(m_ho),
            'holdout_extra_only': _hits(m_ho_x),
            'holdout_extra_desync_negctrl': _hits(m_ho_xd),
            'rare_atoms_extra_only': {a: m_ho_x['hits'].get(a) for a in rare},
            'c_atoms_extra_only': {a: m_ho_x['hits'].get(a) for a in c_atoms},
            'c_atoms_extra_desync': {a: m_ho_xd['hits'].get(a) for a in c_atoms},
        },
    }
    txt_path = os.path.join(OUT_DIR, 'tables.txt')
    json_path = os.path.join(OUT_DIR, 'tables.json')
    soft_cfg = {k: v for k, v in cfg.items() if k != 'soft_trace'}
    lines = [
        '# RAS-FR paper tables',
        f'usable={len(usable)} skip={len(skip)} hold={hold_ids[:3]}...{hold_ids[-1:]}',
        f'frozen commit={frozen["commit"]}  (runs/induction/paper/frozen_cfg.json)',
        '',
        '## MAIN — real corpora only (train = 16 chorales, no synthetic)',
        f'calibrate {soft_cfg}',
        f'soft_z trace (desync false frac) { {k: round(v, 3) for k, v in cfg["soft_trace"].items()} }',
        f'in-sample k=16    Comp={m_in["completeness"]:.3f} FDR={m_in["fdr"]:.3f}',
        f'chorale hold A    Comp={m_ch["completeness"]:.3f} FDR={m_ch["fdr"]:.3f}  '
        f'shared={sh_ch["completeness"]:.3f} miss={_miss(m_ch)}',
        f'chorale hold B    Comp={m_chb["completeness"]:.3f} FDR={m_chb["fdr"]:.3f}  '
        f'shared={sh_chb["completeness"]:.3f} miss={_miss(m_chb)}',
    ]
    if m_pal:
        lines.append(
            f'Palestrina n={len(pal)} Comp={m_pal["completeness"]:.3f} FDR={m_pal["fdr"]:.3f}  '
            f'shared={sh_pal["completeness"]:.3f} miss={_miss(m_pal)}'
        )
    lines += ['', '## soft rules on/off']
    for k, v in soft_abl.items():
        lines.append(f'{k:<28} Comp={v["completeness"]:.3f} shared={v["shared"]:.3f} '
                     f'FDR={v["fdr"]:.3f} miss={v["miss"]}')
    lines += [
        '',
        '## negative control: bass-path soft adds (frac of eligible K/J cells)',
        f'desync at chosen soft_z = {soft_neg.get("desync_frac_at_chosen")}',
        f'real   at chosen soft_z = {soft_neg.get("real_frac_at_chosen")}',
        '',
        '## baselines / ablations (chorale hold A)',
        f'no pair tapes  Comp={m_np["completeness"]:.3f} A1={m_np["hits"].get("A1")}',
        f'pooled pitch   Comp={m_po["completeness"]:.3f}',
        f'n-gram rarity (count rank) Comp={m_ch["completeness_count"]:.3f}',
        f'voice shuffle  A1={m_perm["hits"].get("A1")} A3={m_perm["hits"].get("A3")}',
        '',
        '## k-shot hold-out (chorale-only train+hold)',
        f'protocol: {ktab.get("protocol")}',
    ]
    for r in ktab['rows']:
        lines.append(
            f'k={r["k"]:<3}  {r["comp_mean"]:.3f}±{r["comp_sd"]:.3f}  '
            f'count={r["count_mean"]:.3f}  FDR={r["fdr_mean"]:.3f}  '
            f'A1={r["atom"]["A1"]:.2f} A2={r["atom"]["A2"]:.2f}'
        )
    lines += _ach_lines('chorale hold A（主表）', m_ch)
    lines += ['', ablate_txt, eq_txt]
    lines += [
        '',
        '## APPENDIX — synthetic extra (generated alongside oracle; detectability only)',
        f'calibrate {({k: v for k, v in cfg_x.items() if k != "soft_trace"})}',
        f'extra train={len(train_extra)} hold={len(hold_extra)}',
        f'hold extra+8 chorales Comp={m_ho["completeness"]:.3f} FDR={m_ho["fdr"]:.3f} miss={_miss(m_ho)}',
        f'hold extra only       Comp={m_ho_x["completeness"]:.3f} FDR={m_ho_x["fdr"]:.3f} miss={_miss(m_ho_x)}',
        f'neg ctrl desync extra Comp={m_ho_xd["completeness"]:.3f} FDR={m_ho_xd["fdr"]:.3f}',
        f'C atoms extra   { {a: m_ho_x["hits"].get(a) for a in c_atoms} }',
        f'C atoms desync  { {a: m_ho_xd["hits"].get(a) for a in c_atoms} }',
    ]
    text = '\n'.join(lines) + '\n'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(text)
    print('wrote', txt_path, json_path, flush=True)


if __name__ == '__main__':
    main()
