# -*- coding: utf-8 -*-
"""把 RAS-FR 的发现集导出为可审计的知识制品，并作为检查器在乐谱上执行。

  venv/bin/python -m score_learning.induction.knowledge_export

每条约束：条件（格子各位的取值）、极性、统计证据（真实计数、期望、rho、z、收录规则）、
依赖溯源（错位副本上的带符号 z 与分类）、可执行谓词。参考约束（oracle）的匹配单独存放，
只用于评测，不参与导出。获取语料：留出 A（训练音库与阈值同 final_eval.json）。
"""
from __future__ import annotations

import dataclasses
import json
import random
from collections import Counter
from typing import Dict, List

from score_learning.induction import bootstrap as bs
from score_learning.induction import fri
from score_learning.induction.baselines import EVAL_FIELDS
from score_learning.induction.fri import EMPTY_LAM, annotate_z, discoveries, field_totals
from score_learning.induction.paper import _desync_piece, _git_head, load_chorale_pool

OUT = 'runs/induction/paper/knowledge_artifact.json'
VERTICAL = {'V', 'O', 'H', 'S', 'U', 'K', 'J'}
WITHIN_VOICE = {'M', 'G', 'D'}
LOCAL = {'V': 2, 'O': 2, 'H': 1, 'S': 1, 'U': 1, 'M': 2, 'G': 2, 'D': 3}
NULL_REMOVES = {
    'V': 'joint choice of the two voices\' steps', 'O': 'joint choice of adjacent voices\' steps',
    'H': 'joint choice of simultaneous pitches', 'S': 'joint choice of the four pitches of a sonority',
    'U': 'joint choice of the four pitches of a sonority', 'M': 'dependence of the next pitch on the current one',
    'G': 'dependence of the next pitch on the current one', 'D': 'dependence of a voice\'s middle note on its neighbours',
    'K': 'dependence between consecutive sonorities', 'J': 'dependence between succession and phrase position',
    'P': 'dependence of the final sonority on its position', 'Q': 'dependence of the cadence pair on phrase end',
}


def condition(cell: str) -> Dict[str, str]:
    out = {}
    for i, part in enumerate(cell.replace('/to/', '|to|').split('|')):
        k, _, v = part.partition('=')
        out[k if v else f'_{i}'] = v if v else k
    return out


def admitted_by(fld, cell, row, pol, hard, zmap, cfg, totals):
    nr, n0, rho = row[2], row[3], row[1]
    if (fld, cell) in hard:
        return 'BH / z threshold'
    n_r, n_0 = totals
    if pol == 'under' and nr == 0 and n0 * n_r / n_0 >= EMPTY_LAM:
        return 'empty-cell Poisson rule'
    if pol == 'over' and n0 == 0 and nr >= 3 and rho >= 0.8:
        return 'absent-from-null rule'
    if pol == 'under':
        return 'calibrated bass-path rule'
    return 'sparse-field rule (phase 1)'


def provenance(fld, s_real, s_desync):
    if fld in WITHIN_VOICE or fld in ('P', 'Q'):
        return 'within-voice or positional (not tested by desynchronisation)'
    if s_real <= 0:
        return 'no evidence'
    return 'relational' if s_desync < 0.5 * s_real else 'persists without coordination (register/marginal)'


def export(fields, fields_d, cfg, oracle_preds) -> List[dict]:
    und, ov = discoveries(fields, cfg['z_cut'], cfg['q'], soft=True, soft_z=cfg.get('soft_z'))
    _hu, _ho = discoveries(fields, cfg['z_cut'], cfg['q'], soft=False)
    zr, zd = annotate_z(fields), annotate_z(fields_d)
    out = []
    for pol, found, hard in (('prohibition', und, _hu), ('preference', ov, _ho)):
        sgn = -1.0 if pol == 'prohibition' else 1.0
        for fld, cell in sorted(found):
            if fld not in EVAL_FIELDS:
                continue
            row = next(r for r in fields[fld] if r[0] == cell)
            totals = field_totals(fields[fld])
            lam = (row[3] + fri.ALPHA) * totals[0] / max(totals[1], 1)
            s_real, s_des = sgn * zr[fld][cell], sgn * zd.get(fld, {}).get(cell, 0.0)
            ref = [n for n, f, p, pred in oracle_preds
                   if f == fld and p == ('under' if pol == 'prohibition' else 'over') and pred(cell, fld)]
            out.append({
                'id': f'{fld}-{"P" if pol == "prohibition" else "F"}{len(out) + 1:04d}',
                'field': fld, 'polarity': pol, 'cell': cell, 'condition': condition(cell),
                'evidence': {'n_real': row[2], 'expected': round(lam, 2),
                             'observed_over_expected': round(row[2] / lam, 3) if lam > 0 else None,
                             'rho': round(row[1], 3),
                             'z': round(zr[fld][cell], 2),
                             'admitted_by': admitted_by(fld, cell, row, 'under' if pol == 'prohibition' else 'over',
                                                        hard, zr, cfg, totals)},
                'provenance': {'null': 'voice-decomposition replay', 'removes': NULL_REMOVES.get(fld, ''),
                               'signed_z_real': round(s_real, 2), 'signed_z_desync': round(s_des, 2),
                               'class': provenance(fld, s_real, s_des)},
                'executable': {'type': 'cell_match', 'field': fld, 'cell': cell,
                               'locatable': fld in LOCAL},
                'evaluation_only': {'matches_reference': ref},
            })
    return out


def check(piece, artifact: List[dict], polarity: str = 'prohibition', max_ratio: float = 0.25) -> List[dict]:
    """在一首乐谱上执行导出的禁止：只用强回避的（观测/期望 <= max_ratio）。
    局部场给出位置（小节、拍），其余场给出整曲计数。"""
    rules = {}
    for k in artifact:
        r = k['evidence']['observed_over_expected']
        if k['polarity'] == polarity and r is not None and r <= max_ratio:
            rules.setdefault(k['field'], {})[k['cell']] = k['id']
    hits = []
    for fld, cells in rules.items():
        real = getattr(fri, f'real_{fld.lower()}_counts')
        w = LOCAL.get(fld)
        if w is None:
            for cell, n in real([piece]).items():
                if cell in cells:
                    hits.append({'id': cells[cell], 'field': fld, 'count': n})
            continue
        for i in range(len(piece.slices) - w + 1):
            win = dataclasses.replace(piece, slices=piece.slices[i:i + w])
            for cell in real([win]):
                if cell in cells:
                    s = piece.slices[i + w - 1]
                    hits.append({'id': cells[cell], 'field': fld, 'bar': s.bar, 'beat': s.pos_in_bar})
    return hits


def main():
    with open(bs.FINAL) as fh:
        fin = json.load(fh)
    by, _usable, _ = load_chorale_pool()
    tr = [by[i] for i in fin['train_ids']]
    hold = [by[i] for i in fin['hold_ids']]
    rng_d = random.Random(99)
    hold_d = [_desync_piece(p, rng_d) for p in hold]
    f, fd = fri.holdout_fields(tr, hold, seed=0), fri.holdout_fields(tr, hold_d, seed=0)
    art = export(f, fd, fin['cfg'], fri.book_predicates())
    cls = Counter((k['polarity'], k['provenance']['class']) for k in art)
    print(f'{len(art)} constraints;', dict(cls))
    strong = [k for k in art if k['polarity'] == 'prohibition'
              and (k['evidence']['observed_over_expected'] or 1) <= 0.25]
    print(f'strong prohibitions (obs/exp <= 0.25): {len(strong)}; matching a reference prohibition: '
          f'{sum(bool(k["evaluation_only"]["matches_reference"]) for k in strong)}')
    demo = []
    for p, pd in zip(by_ids(by, fin['hold_b_ids'][:10]), desync_all(by, fin['hold_b_ids'][:10])):
        n_ctx = max(len(p.slices), 1)
        demo.append({'title': p.title, 'flags_real_per_100_slices': 100 * len(check(p, art)) / n_ctx,
                     'flags_desync_per_100_slices': 100 * len(check(pd, art)) / n_ctx})
    for d in demo:
        print(f'  {d["title"][-30:]:<30} real {d["flags_real_per_100_slices"]:.1f}  '
              f'desync {d["flags_desync_per_100_slices"]:.1f}')
    with open(OUT, 'w') as fh:
        json.dump({'commit': _git_head(), 'acquisition_corpus': 'hold-out A', 'cfg': fin['cfg'],
                   'constraints': art, 'checker_demo_hold_b': demo}, fh, indent=1, ensure_ascii=False)
    print('wrote', OUT)


def by_ids(by, ids):
    return [by[i] for i in ids]


def desync_all(by, ids):
    rng = random.Random(7)
    return [_desync_piece(by[i], rng) for i in ids]


if __name__ == '__main__':
    main()
