# -*- coding: utf-8 -*-
"""第一批检查：构谱器往返、D2 硬不变量、S1–S6 的结构不变量（不跑 DCRA，不算 E*）。

  venv/bin/python -m score_learning.dcra.check_structure
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from dataclasses import replace

from score_learning.dcra import canon, synth
from score_learning.dcra.canon import Event, StructureViolation
from score_learning.induction.fri import VOICES

OUT = 'runs/dcra/check_structure.json'
N = 40


def _expect_violation(fn) -> bool:
    try:
        fn()
    except StructureViolation:
        return True
    return False


def invariant_selftest(p: canon.Piece) -> dict:
    """assert_same_d2 必须接受只改音高的版本，拒绝改节奏/乐句/调性的版本。"""
    ok_same = not _expect_violation(lambda: canon.assert_same_d2(p, p))
    s = p.voices['S']
    pitch_only = replace(p, voices={**p.voices, 'S': tuple(
        Event(e.onset, e.dur, None if e.midi is None else e.midi + 2) for e in s)})
    ok_pitch = not _expect_violation(lambda: canon.assert_same_d2(p, pitch_only))
    a = p.voices['A']
    merged = (Event(a[0].onset, a[0].dur + a[1].dur, a[0].midi),) + a[2:]
    rhythm = replace(p, voices={**p.voices, 'A': merged})
    rest = replace(p, voices={**p.voices, 'T': (Event(0.0, p.voices['T'][0].dur, None),)
                              + p.voices['T'][1:]})
    phrase = replace(p, phrase_ends=p.phrase_ends[1:])
    tonal = replace(p, tonic_pc=(p.tonic_pc + 7) % 12)
    return {
        'accepts_identical': ok_same,
        'accepts_pitch_only_change': ok_pitch,
        'rejects_rhythm_change': _expect_violation(lambda: canon.assert_same_d2(p, rhythm)),
        'rejects_attack_to_rest': _expect_violation(lambda: canon.assert_same_d2(p, rest)),
        'rejects_phrase_change': _expect_violation(lambda: canon.assert_same_d2(p, phrase)),
        'rejects_tonal_change': _expect_violation(lambda: canon.assert_same_d2(p, tonal)),
    }


def rhythm_stats(pieces) -> dict:
    n_sl = n_any = n_all = n_strong = n_strong_all = 0
    att = Counter()
    for p in pieces:
        for i, row in enumerate(canon.onset_matrix(p)):
            t = i * canon.SLICE_STEP
            n_sl += 1
            for v, x in zip(VOICES, row):
                att[v] += x == 'A'
            if 'A' in row:
                n_any += 1
                n_all += all(x == 'A' for x in row)
            if (t % synth.BAR_LEN) in (0.0, 2.0):
                n_strong += 1
                n_strong_all += all(x == 'A' for x in row)
    return {
        'slices': n_sl,
        'attack_rate': {v: round(att[v] / n_sl, 3) for v in VOICES},
        'all_attack_given_any': round(n_all / n_any, 3),
        'all_attack_on_strong_beats': round(n_strong_all / n_strong, 3),
    }


def comotion(pieces) -> dict:
    """声部对同向率 vs 独立乘积（只作说明；S5/S6 的独立性由构造保证）。"""
    out = {}
    for x in range(4):
        for y in range(x + 1, 4):
            vx, vy = VOICES[x], VOICES[y]
            both = same = upx = upy = 0
            for p in pieces:
                mx = {e.onset: e.midi for e in p.voices[vx]}
                my = {e.onset: e.midi for e in p.voices[vy]}
                common = sorted(set(mx) & set(my))
                for t0, t1 in zip(common, common[1:]):
                    dx, dy = mx[t1] - mx[t0], my[t1] - my[t0]
                    if dx and dy:
                        both += 1
                        same += (dx > 0) == (dy > 0)
                        upx += dx > 0
                        upy += dy > 0
            if both:
                px, py = upx / both, upy / both
                out[vx + vy] = {'n': both, 'same': round(same / both, 3),
                                'indep': round(px * py + (1 - px) * (1 - py), 3)}
    return out


def s6_stats(pieces) -> dict:
    up1 = n1 = up2 = n2 = gap = gap_fill = rep = nph = 0
    for p in pieces:
        for v in VOICES:
            ev = p.voices[v]
            per = len(ev) // synth.N_PHRASE
            phr = [[ev[k * per + q].midi - ev[k * per + q - 1].midi for q in range(1, per)]
                   for k in range(synth.N_PHRASE)]
            dia = [[synth.DIA.index(ev[k * per + q].midi) - synth.DIA.index(ev[k * per + q - 1].midi)
                    for q in range(1, per)] for k in range(synth.N_PHRASE)]
            for k, steps in enumerate(phr):
                half = len(steps) / 2
                for q, d in enumerate(steps):
                    if d == 0:
                        continue
                    if q < half - 0.5:
                        n1 += 1
                        up1 += d > 0
                    else:
                        n2 += 1
                        up2 += d > 0
                if k > 0:
                    nph += 1
                    rep += any(dia[k] == dia[j] for j in range(k))
            for i in range(2, len(ev)):
                d0, d1 = ev[i - 1].midi - ev[i - 2].midi, ev[i].midi - ev[i - 1].midi
                if abs(d0) > synth.S6_GAP_SEMIS:
                    gap += 1
                    gap_fill += d1 != 0 and (d1 > 0) != (d0 > 0) and abs(d1) <= 2
    return {'up_share_first_half': round(up1 / n1, 3), 'up_share_second_half': round(up2 / n2, 3),
            'step_back_after_leap': round(gap_fill / max(gap, 1), 3),
            'phrases_repeating_earlier_contour': round(rep / max(nph, 1), 3)}


def check_gen(gen: str) -> dict:
    t0 = time.time()
    pieces = synth.make_pieces(gen, N, synth.dataset_seed(gen, 0))
    fails = []
    for p in pieces:
        try:
            canon.check_roundtrip(p, canon.to_ras(p))
        except StructureViolation as e:
            fails.append(str(e))
    d2 = {canon.structure(p) for p in pieces}
    res = {'n': N, 'roundtrip_fail': len(fails), 'fail_examples': fails[:3],
           'distinct_d2': len(d2), 'rhythm': rhythm_stats(pieces), 'comotion': comotion(pieces),
           'invariant_selftest': invariant_selftest(pieces[0]), 'sec': round(time.time() - t0, 1)}
    if gen == 'S6':
        res['s6'] = s6_stats(pieces)
        res['s2_reference'] = s6_stats(synth.make_pieces('S2', N, synth.dataset_seed('S2', 0)))
    return res


def check_bach(name: str = 'hold_a') -> dict:
    from score_learning.induction.revision import load_sets
    _fin, _tr, sets = load_sets([name])
    stats = Counter()
    fails = []
    for sr in sets[name]:
        try:
            p = canon.from_ras(sr)
            canon.validate(p)
            if canon.ras_onset_matrix(sr, drop_trailing_rest=True) != canon.onset_matrix(p):
                raise StructureViolation(f'{sr.title}: original vs canonical matrix')
            for s in sr.slices[:len(canon.onset_matrix(p))]:
                for v in VOICES:
                    e = canon._sounding(p.voices[v], s.t)
                    if (e.midi if e else None) != s.midis[v]:
                        raise StructureViolation(f'{sr.title}/{v}: original pitch differs at {s.t}')
            canon.check_roundtrip(p, canon.to_ras(p))
            stats['ok'] += 1
        except StructureViolation as e:
            stats['fail'] += 1
            fails.append(str(e)[:160])
    return {'set': name, **stats, 'fail_examples': fails[:5]}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = {'gens': {}}
    for g in synth.GENS:
        out['gens'][g] = r = check_gen(g)
        print(g, json.dumps({k: r[k] for k in ('roundtrip_fail', 'distinct_d2', 'rhythm', 'sec')}),
              flush=True)
    for name in ('hold_a', 'hold_c'):
        out[name] = check_bach(name)
        print(name, out[name], flush=True)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print('wrote', OUT)


if __name__ == '__main__':
    sys.exit(main())
