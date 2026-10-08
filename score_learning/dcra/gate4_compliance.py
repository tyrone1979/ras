# -*- coding: utf-8 -*-
"""门槛 4 合规审计：只按预注册种子与定义计算 TVD 部分；不计算 L_D，不改 (b)。

预注册第 6 节门槛 4："per-voice total variation distance between original and reconstructed step
distribution and scale-degree distribution <= 0.05"，适用于 every analysed set。
主定义（与预注册一致）：被分析集合的原曲 vs 该集合全部 R = 32 次重建的合并分布；
步进 = 声部相邻发声音之间的半音差（含跨乐句、跨休止，与冻结表示 ras.horiz 一致）；
音级 = 每个发声音的调式内音级。辅助量（不用于判定）：乐句内部步进、按时值加权的音级。
重建种子：合成 7000 + 1000·g + r；真实集合 = 冻结 replay 种子 + 500。

  venv/bin/python -m score_learning.dcra.gate4_compliance
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from score_learning.dcra import canon, recon, synth
from score_learning.dcra.audit_recon import _marg, _tvd
from score_learning.induction.fri import VOICES

OUT = 'runs/dcra/gate4_compliance.json'
THRESH = 0.05
N_REP = 20


def _deg_dur(pieces):
    out = {v: Counter() for v in VOICES}
    for p in pieces:
        for v in VOICES:
            w = recon.voice_view(p, v)
            for e in p.voices[v]:
                if e.midi is not None:
                    out[v][recon.degree(w, e.midi)] += e.dur
    return out


def evaluate(pieces, base_seed: int) -> dict:
    models = recon.fit(pieces)
    cfs = [recon.reconstruct(p, models, base_seed, i, r)
           for i, p in enumerate(pieces) for r in range(recon.R_DEFAULT)]
    so, sio, do = _marg(pieces)
    sc, sic, dc = _marg(cfs)
    ddo, ddc = _deg_dur(pieces), _deg_dur(cfs)
    per = {v: {'step': _tvd(so[v], sc[v]), 'degree': _tvd(do[v], dc[v]),
               'aux_step_within_phrase': _tvd(sio[v], sic[v]),
               'aux_degree_duration': _tvd(ddo[v], ddc[v])} for v in VOICES}
    mx_s = max(x['step'] for x in per.values())
    mx_d = max(x['degree'] for x in per.values())
    return {'per_voice': per, 'max_step': mx_s, 'max_degree': mx_d,
            'pass': mx_s <= THRESH and mx_d <= THRESH}


def _synth_job(args):
    g, r = args
    ps = synth.make_pieces(g, 40, synth.dataset_seed(g, r))
    return g, r, evaluate(ps, 7000 + 1000 * synth.GENS.index(g) + r)


def main():
    out = {'threshold': THRESH, 'synthetic': {}, 'real': {}}
    jobs = [(g, r) for g in synth.GENS for r in range(N_REP)]
    with ProcessPoolExecutor(7) as ex:
        for g, r, res in ex.map(_synth_job, jobs):
            out['synthetic'].setdefault(g, {})[r] = res
    from score_learning.induction.revision import SEEDS, load_sets
    names = ['hold_a', 'hold_b', 'hold_c']
    _f, _t, sets = load_sets(names)
    for n in names:
        ps = [canon.from_ras(sr) for sr in sets[n]]
        out['real'][n] = evaluate(ps, SEEDS[n] + 500)
    summ = {}
    for g, reps in out['synthetic'].items():
        ms = [x['max_step'] for x in reps.values()]
        md = [x['max_degree'] for x in reps.values()]
        summ[g] = {'sets_passing': sum(x['pass'] for x in reps.values()), 'n_sets': len(reps),
                   'max_step_range': [min(ms), max(ms)], 'max_degree_range': [min(md), max(md)],
                   'step_sets_over': sum(m > THRESH for m in ms),
                   'degree_sets_over': sum(m > THRESH for m in md)}
    for n, x in out['real'].items():
        summ[n] = {'pass': x['pass'], 'max_step': x['max_step'], 'max_degree': x['max_degree']}
    out['summary'] = summ
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    sys.exit(main())
