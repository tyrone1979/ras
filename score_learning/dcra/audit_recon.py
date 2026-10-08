# -*- coding: utf-8 -*-
"""(b) 的方法级审计：泄漏边界、结构不变量、边际保留（不计算 L_D，不跑验收门槛）。

1. 接口：VoiceView 字段集合 = 允许集合；拟合/生成函数只接收 VoiceView。
2. 依赖：recon.py 不导入任何模式/格子/原子定义、评测或真值模块。
3. 扰动不变性（功能性泄漏检验）：把其他声部的音高全部随机替换（节奏不变），
   声部 v 的拟合模型与重建序列必须逐位相同。
4. 结构：每次重建后硬断言 D2 完全一致、音域合法（recon.check_reconstruction）。
5. 边际：每声部步进分布、调式内音级分布的 TVD（原曲 vs 全部重建）。

  venv/bin/python -m score_learning.dcra.audit_recon
"""
from __future__ import annotations

import ast
import inspect
import json
import random
import sys
import time
from collections import Counter
from dataclasses import fields, replace

from score_learning.dcra import canon, recon, synth
from score_learning.dcra.canon import Event
from score_learning.induction.fri import VOICES

OUT = 'runs/dcra/audit_recon.json'
R = recon.R_DEFAULT
AUDIT_SEED = 424242
ALLOWED_IMPORTS = {'__future__', 'random', 'collections', 'dataclasses', 'functools', 'typing',
                   'music21', 'score_learning.dcra', 'score_learning.dcra.canon',
                   'score_learning.induction.fri'}
FORBIDDEN_NAMES = {'oracle', 'revision', 'PREDS', 'book_predicates', 'estar', 'synth',
                   'synthetic_truth', 'main_predicates', 'real_v_counts', 'v_cell', 'h_cell',
                   'u_cell', '_set_from_midis', 'PAIRS'}


# ---------------------------------------------------------------- 1–2 static

def audit_static() -> dict:
    src = inspect.getsource(recon)
    tree = ast.parse(src)
    imports, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
            for a in node.names:
                names.add(a.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    fri_names = {a.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                 and node.module == 'score_learning.induction.fri' for a in node.names}
    sig_fit = inspect.signature(recon.fit_voice)
    sig_gen = inspect.signature(recon.generate_voice)
    return {
        'voice_view_fields': [f.name for f in fields(recon.VoiceView)],
        'voice_view_fields_ok': tuple(f.name for f in fields(recon.VoiceView)) == recon.VOICE_VIEW_FIELDS
        and set(recon.VOICE_VIEW_FIELDS) == {'voice', 'events', 'bar_starts', 'bar_len',
                                             'phrase_ends', 'tonic_pc', 'mode'},
        'imports': sorted(i for i in imports if i),
        'imports_ok': all(any(i == a or i.startswith(a + '.') for a in ALLOWED_IMPORTS)
                          for i in imports if i),
        'fri_names_imported': sorted(fri_names),
        'fri_names_ok': fri_names <= {'VOICES'},
        'forbidden_names_found': sorted(names & FORBIDDEN_NAMES),
        'fit_voice_signature': str(sig_fit),
        'generate_voice_signature': str(sig_gen),
        'signatures_ok': list(sig_fit.parameters) == ['views']
        and list(sig_gen.parameters) == ['model', 'view', 'rng'],
    }


# ---------------------------------------------------------------- 3 perturbation

def _scramble_others(p: canon.Piece, keep: str, rng: random.Random) -> canon.Piece:
    voices = {}
    for v in VOICES:
        if v == keep:
            voices[v] = p.voices[v]
        else:
            voices[v] = tuple(Event(e.onset, e.dur, None if e.midi is None else rng.randint(30, 90))
                              for e in p.voices[v])
    return replace(p, voices=voices)


def audit_perturbation(pieces, label: str) -> dict:
    rng = random.Random(AUDIT_SEED)
    models = recon.fit(pieces)
    res = {}
    for v in VOICES:
        pert = [_scramble_others(p, v, rng) for p in pieces]
        m2 = recon.fit(pert)
        same_model = m2[v] == models[v]
        same_seq = True
        for i, (p, q) in enumerate(zip(pieces, pert)):
            for r in (0, 1):
                a = recon.generate_voice(models[v], recon.voice_view(p, v),
                                         random.Random(recon.voice_seed(AUDIT_SEED, i, v, r)))
                b = recon.generate_voice(m2[v], recon.voice_view(q, v),
                                         random.Random(recon.voice_seed(AUDIT_SEED, i, v, r)))
                same_seq &= a == b
        res[v] = {'model_identical': same_model, 'sequence_identical': same_seq}
    return {'set': label, 'voices': res,
            'ok': all(x['model_identical'] and x['sequence_identical'] for x in res.values())}


# ---------------------------------------------------------------- 4–5 structure and marginals

def _marg(pieces):
    step = {v: Counter() for v in VOICES}
    step_in = {v: Counter() for v in VOICES}
    deg = {v: Counter() for v in VOICES}
    for p in pieces:
        for v in VOICES:
            w = recon.voice_view(p, v)
            first = {i for slots in recon.phrase_slots(w) for i in slots[:1]}
            prev = None
            for i, e in enumerate(p.voices[v]):
                if e.midi is None:
                    continue
                deg[v][recon.degree(w, e.midi)] += 1
                if prev is not None:
                    step[v][e.midi - prev] += 1
                    if i not in first:
                        step_in[v][e.midi - prev] += 1
                prev = e.midi
    return step, step_in, deg


def _tvd(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


def audit_structure(pieces, label: str, base_seed: int) -> dict:
    t0 = time.time()
    models = recon.fit(pieces)
    cfs, fails = [], []
    for i, p in enumerate(pieces):
        for r in range(R):
            try:
                cfs.append(recon.reconstruct(p, models, base_seed, i, r))
            except (canon.StructureViolation, RuntimeError) as e:
                fails.append(f'{type(e).__name__}: {e}'[:160])
    so, sio, do = _marg(pieces)
    sc, sic, dc = _marg(cfs)
    tv = {v: {'step': round(_tvd(so[v], sc[v]), 4), 'step_within_phrase': round(_tvd(sio[v], sic[v]), 4),
              'degree': round(_tvd(do[v], dc[v]), 4),
              'range': [models[v].lo, models[v].hi],
              'cf_range': [min(e.midi for q in cfs for e in q.voices[v] if e.midi is not None),
                           max(e.midi for q in cfs for e in q.voices[v] if e.midi is not None)]}
          for v in VOICES}
    backoff = Counter()
    for v in VOICES:
        m = models[v]
        for w in (recon.voice_view(p, v) for p in pieces):
            prev = None
            for i, first, t, pp in recon.attack_states(w):
                if not first:
                    for k in recon._keys(recon.degree(w, prev), recon.metric_class(w, t), pp):
                        if sum(m.step.get(k, {}).values()) >= recon.MIN_STATE or k[0] == 'L3':
                            backoff[k[0]] += 1
                            break
                prev = w.events[i].midi
    tot = sum(backoff.values())
    backoff = {k: round(n / tot, 3) for k, n in sorted(backoff.items())}
    return {'set': label, 'pieces': len(pieces), 'reconstructions': len(cfs),
            'structure_failures': len(fails), 'fail_examples': fails[:5],
            'tvd': tv, 'backoff_level_share': backoff,
            'max_tvd_step': max(x['step'] for x in tv.values()),
            'max_tvd_degree': max(x['degree'] for x in tv.values()),
            'sec': round(time.time() - t0, 1)}


def main():
    out = {'static': audit_static()}
    print('static', json.dumps(out['static'], ensure_ascii=False), flush=True)
    sets = {g: synth.make_pieces(g, 40, synth.dataset_seed(g, 0)) for g in synth.GENS}
    from score_learning.induction.revision import load_sets
    _f, _t, bach = load_sets(['hold_a'])
    sets['bach_hold_a'] = [canon.from_ras(sr) for sr in bach['hold_a']]
    out['perturbation'] = {}
    out['structure'] = {}
    for k, ps in sets.items():
        out['perturbation'][k] = pr = audit_perturbation(ps[:10], k)
        out['structure'][k] = st = audit_structure(ps, k, AUDIT_SEED)
        print(k, 'perturbation_ok', pr['ok'], 'recon', st['reconstructions'],
              'fail', st['structure_failures'], 'maxTVD step', st['max_tvd_step'],
              'degree', st['max_tvd_degree'], f"{st['sec']}s", flush=True)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print('wrote', OUT)


if __name__ == '__main__':
    sys.exit(main())
