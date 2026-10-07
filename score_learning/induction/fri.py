# -*- coding: utf-8 -*-
"""RAS 分解残差归纳（RAS-FR）。

原创：规则不是项集 / n-gram / ILP。零模型是声部分解重放；
每场 Laplace 强度由经验贝叶斯估计；发现集用 Poisson/z 与 BH，
门槛在声部时移谱上校准。训练不写规则名。
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

from score_learning.induction.learn import load_many
from score_learning.induction.ras import ADJ, OUTER, PAIRS, VOICES, ScoreRAS

ALPHA = 0.5
REPLAY = 16
FORBID_R = -2.0
PREFER_R = 1.2
MIN_NULL = 12
RECOVER_R = -2.0
RECOVER_RANK = 8  # 仅辅表；主判定改为校准 z / BH
Z_CUT = 3.0
BH_Q = 0.08


def _role(pair: Tuple[str, str]) -> str:
    if pair in OUTER:
        return 'outer'
    if pair in ADJ:
        return 'adjacent'
    return 'diagonal'


def _pc07(iv: int) -> bool:
    return (abs(iv) % 12) in (0, 7)


def _sgn(dpa: int, dpb: int) -> str:
    if dpa == 0 and dpb == 0:
        return 'stat'
    if dpa == 0 or dpb == 0:
        return 'obl'
    if (dpa > 0) == (dpb > 0):
        return 'same'
    return 'opp'


def v_cell(dpa: int, dpb: int, iv0: int, iv1: int, pair: Tuple[str, str]) -> str:
    both = int(dpa != 0 and dpb != 0)
    u0 = int(dpa == dpb)
    hold = int(iv1 == iv0)
    leap_u = int(abs(dpa) > 2)
    return (
        f'both={both}|eqΔ={u0}|pc07={int(_pc07(iv1))}|hold={hold}'
        f'|leapU={leap_u}|sgn={_sgn(dpa, dpb)}|role={_role(pair)}'
    )


def h_cell(va: str, vb: str, ma: int, mb: int) -> str:
    gap = ma - mb
    if gap < 0:
        band = 'cross'
    elif gap <= 7:
        band = 'le8'
    elif gap <= 12:
        band = '8to12'
    elif gap <= 19:
        band = '12to19'
    else:
        band = 'gt19'
    return f'{va}{vb}|gap={band}'


def parse_cell(cell: str) -> Dict[str, str]:
    out = {}
    for part in cell.split('|'):
        if '=' in part:
            k, v = part.split('=', 1)
            out[k] = v
        else:
            out['pair'] = part
    return out


def _complete_pairs(ras: ScoreRAS):
    sl = ras.slices
    out = []
    for i in range(1, len(sl)):
        a, b = sl[i - 1], sl[i]
        if any(a.midis[v] is None or b.midis[v] is None for v in VOICES):
            continue
        out.append((a, b))
    return out


def catalogs(pieces: List[ScoreRAS]):
    dp = {v: [] for v in VOICES}
    pitch = {v: [] for v in VOICES}
    for ras in pieces:
        for v in VOICES:
            dp[v].extend(h.dp for h in ras.horiz[v])
            pitch[v].extend(n.midi for n in ras.notes if n.voice == v and n.midi is not None)
    for v in VOICES:
        if not dp[v]:
            dp[v] = [0, 1, -1, 2, -2]
        if not pitch[v]:
            pitch[v] = [60, 67]
    return dp, pitch


def _overlap(ma0: int, mb0: int, ma1: int, mb1: int, pair: Tuple[str, str]) -> int:
    if pair not in ADJ:
        return 0
    return int((mb1 > ma0) or (ma1 < mb0))


def o_cell(pair: Tuple[str, str], ov: int) -> str:
    return f'{pair[0]}{pair[1]}|ov={ov}'


def g_cell(dp: int) -> str:
    a = abs(dp)
    if a >= 12:
        return 'oct+'
    if a in (10, 11):
        return '7th'
    if a == 6:
        return 'tritone'
    if a >= 5:
        return 'leap'
    return 'small'


def real_v_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        for a, b in _complete_pairs(ras):
            for p in PAIRS:
                va, vb = p
                dpa = b.midis[va] - a.midis[va]
                dpb = b.midis[vb] - a.midis[vb]
                iv0 = a.midis[va] - a.midis[vb]
                iv1 = b.midis[va] - b.midis[vb]
                c[v_cell(dpa, dpb, iv0, iv1, p)] += 1
    return c


def null_v_counts(pieces: List[ScoreRAS], dp_cat, rng: random.Random, k: int = REPLAY) -> Counter:
    c = Counter()
    for ras in pieces:
        for a, b in _complete_pairs(ras):
            for _ in range(k):
                draw = {v: rng.choice(dp_cat[v]) for v in VOICES}
                for p in PAIRS:
                    va, vb = p
                    dpa, dpb = draw[va], draw[vb]
                    iv0 = a.midis[va] - a.midis[vb]
                    iv1 = iv0 + dpa - dpb
                    c[v_cell(dpa, dpb, iv0, iv1, p)] += 1
    return c


def real_o_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        for a, b in _complete_pairs(ras):
            for p in PAIRS:
                if p not in ADJ:
                    continue
                va, vb = p
                ov = _overlap(a.midis[va], a.midis[vb], b.midis[va], b.midis[vb], p)
                c[o_cell(p, ov)] += 1
    return c


def null_o_counts(pieces: List[ScoreRAS], dp_cat, rng: random.Random, k: int = REPLAY) -> Counter:
    """与 F_V 同一重放，格子只记邻声部是否超越。"""
    c = Counter()
    for ras in pieces:
        for a, b in _complete_pairs(ras):
            for _ in range(k):
                draw = {v: rng.choice(dp_cat[v]) for v in VOICES}
                for p in PAIRS:
                    if p not in ADJ:
                        continue
                    va, vb = p
                    ma1 = a.midis[va] + draw[va]
                    mb1 = a.midis[vb] + draw[vb]
                    c[o_cell(p, _overlap(a.midis[va], a.midis[vb], ma1, mb1, p))] += 1
    return c


def real_g_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        for v in VOICES:
            for m0, _d0, m1, _d1 in _voice_changes(ras, v):
                c[g_cell(m1 - m0)] += 1
    return c


def null_g_counts(pieces: List[ScoreRAS], pitch_cat, rng: random.Random, k: int = REPLAY) -> Counter:
    """F_M 的粗 |Δp| 格子，避免 A10 被 lt/b7/out 切碎。"""
    c = Counter()
    for ras in pieces:
        for v in VOICES:
            for m0, _d0, _m1, _d1 in _voice_changes(ras, v):
                for _ in range(k):
                    m1 = rng.choice(pitch_cat[v])
                    if len(set(pitch_cat[v])) > 1:
                        while m1 == m0:
                            m1 = rng.choice(pitch_cat[v])
                    c[g_cell(m1 - m0)] += 1
    return c


def real_h_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        for s in ras.slices:
            m = s.midis
            if any(m[v] is None for v in VOICES):
                continue
            for p in (('S', 'A'), ('A', 'T'), ('T', 'B')):
                c[h_cell(p[0], p[1], m[p[0]], m[p[1]])] += 1
    return c


def null_h_counts(pieces: List[ScoreRAS], pitch_cat, rng: random.Random, k: int = REPLAY) -> Counter:
    c = Counter()
    n_slice = 0
    for ras in pieces:
        n_slice += sum(1 for s in ras.slices if all(s.midis[v] is not None for v in VOICES))
    n_slice = max(n_slice, 1)
    for _ in range(n_slice * k):
        m = {v: rng.choice(pitch_cat[v]) for v in VOICES}
        for p in (('S', 'A'), ('A', 'T'), ('T', 'B')):
            c[h_cell(p[0], p[1], m[p[0]], m[p[1]])] += 1
    return c


def _mv(dp: int) -> str:
    if dp == 1:
        return '+1'
    if dp == -1:
        return '-1'
    if dp == -2:
        return '-2'
    if dp == 0:
        return 'stay'
    if abs(dp) == 6:
        return 'tritone'
    if abs(dp) in (10, 11):
        return '7th'
    if abs(dp) >= 12:
        return 'oct+'
    if abs(dp) in (1, 2):
        return 'step'
    return 'leap'


def m_cell(outer: int, lt: int, b7: int, dp: int) -> str:
    return f'out={outer}|lt={lt}|b7={b7}|mv={_mv(dp)}'


def _voice_changes(ras: ScoreRAS, v: str):
    prev_m = prev_d = None
    for s in ras.slices:
        m, d = s.midis[v], s.deg[v]
        if m is None:
            prev_m = None
            continue
        if prev_m is not None and m != prev_m:
            yield prev_m, prev_d, m, d
        prev_m, prev_d = m, d


def real_m_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        for v in VOICES:
            outer = int(v in ('S', 'B'))
            for _m0, d0, m1, _d1 in _voice_changes(ras, v):
                dp = m1 - _m0
                c[m_cell(outer, int(d0 == 11), int(d0 == 10), dp)] += 1
    return c


def null_m_counts(pieces: List[ScoreRAS], pitch_cat, rng: random.Random, k: int = REPLAY) -> Counter:
    """F_M：下一音从该声部音域库独立抽，毁掉旋律记忆（跳进/解决活在这里）。"""
    c = Counter()
    for ras in pieces:
        for v in VOICES:
            outer = int(v in ('S', 'B'))
            for m0, d0, _m1, _d1 in _voice_changes(ras, v):
                for _ in range(k):
                    m1 = rng.choice(pitch_cat[v])
                    if len(set(pitch_cat[v])) > 1:
                        while m1 == m0:
                            m1 = rng.choice(pitch_cat[v])
                    c[m_cell(outer, int(d0 == 11), int(d0 == 10), m1 - m0)] += 1
    return c


def s_cell(n_lt: int, miss3: int, n7: int, miss_ton: int) -> str:
    return f'nlt={min(n_lt, 2)}|miss3={miss3}|n7={min(n7, 2)}|missTon={miss_ton}'


def _set_from_midis(midis: Dict[str, int], tonic: int) -> str:
    degs = {v: (midis[v] - tonic) % 12 for v in VOICES}
    n_lt = sum(1 for v in VOICES if degs[v] == 11)
    n7 = sum(1 for v in VOICES if degs[v] == 10)
    bass = midis['B']
    miss3 = int(not any(abs(midis[v] - bass) % 12 in (3, 4) for v in VOICES))
    rels = set(degs.values())
    miss_ton = int(not (({0, 4, 7} <= rels) or ({0, 3, 7} <= rels)))
    return s_cell(n_lt, miss3, n7, miss_ton)


def real_s_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        for s in ras.slices:
            m = s.midis
            if any(m[v] is None for v in VOICES):
                continue
            c[_set_from_midis({v: m[v] for v in VOICES}, ras.tonic_pc)] += 1
    return c


def null_s_counts(pieces: List[ScoreRAS], pitch_cat, rng: random.Random, k: int = REPLAY) -> Counter:
    """F_S：四声部音高独立抽（同 F_H 的切片），格子是音集/导音重复。"""
    c = Counter()
    for ras in pieces:
        n = sum(1 for s in ras.slices if all(s.midis[v] is not None for v in VOICES))
        for _ in range(max(n, 1) * k):
            m = {v: rng.choice(pitch_cat[v]) for v in VOICES}
            c[_set_from_midis(m, ras.tonic_pc)] += 1
    return c


def _bucket_deg(d) -> str:
    if d is None:
        return 'x'
    if d in (0, 2, 4, 5, 7, 9, 11):
        return str(d)
    return 'x'


def p_cell_from_slice(s) -> str:
    return f'b={_bucket_deg(s.bass_deg)}|sop={_bucket_deg(s.deg.get("S"))}'


def p_cell_midis(midis: Dict[str, int], tonic: int) -> str:
    bd = (midis['B'] - tonic) % 12
    sd = (midis['S'] - tonic) % 12
    return f'b={_bucket_deg(bd)}|sop={_bucket_deg(sd)}'


def real_p_counts(pieces: List[ScoreRAS]) -> Counter:
    """句末观测（乐谱最后完整切片），不读 fermata 标签。"""
    c = Counter()
    for ras in pieces:
        complete = [s for s in ras.slices if all(s.midis[v] is not None for v in VOICES)]
        for s in complete[-2:]:
            c[p_cell_from_slice(s)] += 1
    return c


def null_p_counts(pieces: List[ScoreRAS], rng: random.Random, k: int = REPLAY) -> Counter:
    """F_P：把随机时刻的切片当成「句末」重放，毁掉位置约束。"""
    c = Counter()
    for ras in pieces:
        complete = [s for s in ras.slices if all(s.midis[v] is not None for v in VOICES)]
        if not complete:
            continue
        for _ in range(k * 2):
            c[p_cell_from_slice(rng.choice(complete))] += 1
    return c


def q_cell(s0, s1, slot: str = 'fin') -> str:
    return p_cell_from_slice(s0) + '/to/' + p_cell_from_slice(s1) + f'|slot={slot}'


def _complete_slices(ras: ScoreRAS):
    complete = [s for s in ras.slices if all(s.midis[v] is not None for v in VOICES)
                and s.metric != 'offbeat']
    if len(complete) < 2:
        complete = [s for s in ras.slices if all(s.midis[v] is not None for v in VOICES)]
    return complete


def _uniq_p(sl):
    uniq = []
    for s in sl:
        if not uniq or p_cell_from_slice(s) != p_cell_from_slice(uniq[-1]):
            uniq.append(s)
        else:
            uniq[-1] = s
    return uniq


def _cadence_slices(ras: ScoreRAS):
    return _uniq_p(_complete_slices(ras))


def _phrase_spans(ras: ScoreRAS):
    """无监督切句：女高长音为界；终点停在最后完整发声切片，不吃尾部休止。"""
    bar = ras.bar_len or 4.0
    complete = _complete_slices(ras)
    end_t = (complete[-1].t + 0.01) if complete else (
        ras.slices[-1].t if ras.slices else 0.0)
    cuts = [0.0]
    for n in ras.notes:
        if n.voice == 'S' and n.midi is not None and (n.offset - n.onset) >= min(2.0, bar - 0.05):
            if n.offset <= end_t + 1e-6:
                cuts.append(n.offset)
    cuts.append(end_t)
    cuts = sorted(set(round(x, 3) for x in cuts if x <= end_t + 1e-6))
    spans = []
    for i in range(len(cuts) - 1):
        if cuts[i + 1] - cuts[i] > 0.5:
            spans.append((cuts[i], cuts[i + 1], i == len(cuts) - 2))
    if not spans:
        spans = [(0.0, end_t + 1.0, True)]
    return spans


def _slices_in(ras: ScoreRAS, t0, t1):
    return [s for s in _complete_slices(ras) if t0 - 1e-6 <= s.t < t1]


def real_q_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        for t0, t1, is_fin in _phrase_spans(ras):
            u = _uniq_p(_slices_in(ras, t0, t1))
            if len(u) >= 2:
                c[q_cell(u[-2], u[-1], 'fin' if is_fin else 'int')] += 1
    return c


def _q_fin_rate(ras: ScoreRAS) -> float:
    spans = _phrase_spans(ras)
    return sum(1 for *_x, is_fin in spans if is_fin) / max(len(spans), 1)


def null_q_counts(pieces: List[ScoreRAS], rng: random.Random, k: int = REPLAY) -> Counter:
    """F_Q：随机相继两拍 + 随机句位（句位按本曲真实句末比例抽）。"""
    c = Counter()
    for ras in pieces:
        u = _cadence_slices(ras)
        if len(u) < 2:
            continue
        p_fin = _q_fin_rate(ras)
        for _ in range(k):
            i = rng.randrange(0, len(u) - 1)
            slot = 'fin' if rng.random() < p_fin else 'int'
            c[q_cell(u[i], u[i + 1], slot)] += 1
    return c


def _k_token(midis: Dict[str, int], tonic: int) -> str:
    degs = sorted({(midis[v] - tonic) % 12 for v in VOICES})
    bd = (midis['B'] - tonic) % 12
    diat = {0, 2, 4, 5, 7, 9, 11}
    ch = int(any(d not in diat for d in degs))
    st = '.'.join(str(d) for d in degs)
    return f'b={_bucket_deg(bd)}|set={st}|ch={ch}'


def k_cell(a: str, b: str, slot: str = 'int') -> str:
    return a + '/to/' + b + f'|slot={slot}'


def _bass_path_cell(a: str, b: str, slot: str) -> str:
    """低音级粗格：分句属/下属在零模型可拼成 V→IV。"""
    ba = parse_cell(a).get('b', 'x')
    bb = parse_cell(b).get('b', 'x')
    return f'b={ba}/to/b={bb}|slot={slot}'


def _slot_at(ras: ScoreRAS, t: float) -> str:
    """句末 = 每句最后 3 个唯一音集切片（含 I64–V–I）。"""
    for t0, t1, _ in _phrase_spans(ras):
        if not (t0 - 1e-6 <= t <= t1 + 1e-6):
            continue
        toks = []
        for s in _slices_in(ras, t0, t1):
            if any(s.midis[v] is None for v in VOICES):
                continue
            tok = _k_token({v: s.midis[v] for v in VOICES}, ras.tonic_pc)
            if not toks or toks[-1][0] != tok:
                toks.append((tok, s.t))
            else:
                toks[-1] = (tok, s.t)
        if not toks:
            return 'int'
        fin_ts = {round(tt, 3) for _, tt in toks[-3:]}
        return 'fin' if round(t, 3) in fin_ts else 'int'
    return 'int'


def _uniq_k(ras: ScoreRAS):
    toks = []
    for s in _complete_slices(ras):
        tok = _k_token({v: s.midis[v] for v in VOICES}, ras.tonic_pc)
        if not toks or toks[-1][0] != tok:
            toks.append((tok, s.t))
        else:
            toks[-1] = (tok, s.t)
    return toks


def real_k_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        toks = _uniq_k(ras)
        for i in range(len(toks) - 1):
            slot = _slot_at(ras, toks[i + 1][1])
            a, b = toks[i][0], toks[i + 1][0]
            c[k_cell(a, b, slot)] += 1
            c[_bass_path_cell(a, b, slot)] += 1
    return c


def _k_fin_rate(ras: ScoreRAS) -> float:
    toks = _uniq_k(ras)
    if len(toks) < 2:
        return 0.5
    fin = sum(1 for i in range(1, len(toks)) if _slot_at(ras, toks[i][1]) == 'fin')
    return fin / (len(toks) - 1)


def null_k_counts(pieces: List[ScoreRAS], pitch_cat, rng: random.Random, k: int = REPLAY) -> Counter:
    """F_K：两拍四声部独立抽音高（句位按本曲真实句末比例抽）。"""
    c = Counter()
    for ras in pieces:
        n = max(len(_uniq_k(ras)) - 1, 1)
        p_fin = _k_fin_rate(ras)
        for _ in range(n * k):
            m0 = {v: rng.choice(pitch_cat[v]) for v in VOICES}
            m1 = {v: rng.choice(pitch_cat[v]) for v in VOICES}
            slot = 'fin' if rng.random() < p_fin else 'int'
            a = _k_token(m0, ras.tonic_pc)
            b = _k_token(m1, ras.tonic_pc)
            c[k_cell(a, b, slot)] += 1
            c[_bass_path_cell(a, b, slot)] += 1
    return c


def real_j_counts(pieces: List[ScoreRAS]) -> Counter:
    return real_k_counts(pieces)


def null_j_counts(pieces: List[ScoreRAS], rng: random.Random, k: int = REPLAY) -> Counter:
    """F_J：保留相继音集，只打乱句位（句位按本曲真实句末比例抽）。"""
    c = Counter()
    for ras in pieces:
        toks = _uniq_k(ras)
        if len(toks) < 2:
            continue
        p_fin = _k_fin_rate(ras)
        for i in range(len(toks) - 1):
            a, b = toks[i][0], toks[i + 1][0]
            for _ in range(k):
                slot = 'fin' if rng.random() < p_fin else 'int'
                c[k_cell(a, b, slot)] += 1
                c[_bass_path_cell(a, b, slot)] += 1
    return c


def _others_pcs(s, v):
    return {s.midis[w] % 12 for w in VOICES if w != v and s.midis[w] is not None}


def d_cell(**bits) -> str:
    return '|'.join(f'{k}={v}' for k, v in bits.items())


def _d_bits(m0, m1, m2, oth, nxt, held, hold_oth, ped, hov=0):
    d01, d12 = m1 - m0, m2 - m1
    return dict(
        st01=int(abs(d01) in (1, 2)),
        st12=int(abs(d12) in (1, 2)),
        same=int((d01 > 0) == (d12 > 0) and d01 != 0 and d12 != 0),
        ret=int(m2 == m0),
        inn=int((m1 % 12) in oth),
        held=int(held),
        leap=int(abs(d01) >= 3),
        ant=int((m1 % 12) not in oth and (m1 % 12) in nxt),
        ped=int(ped),
        holdO=int(hold_oth),
        hov=int(hov),
    )


def _next_change(sl, v, i, m_now):
    for s in sl[i + 1:]:
        mv = s.midis.get(v)
        if mv is not None and mv != m_now:
            return s, mv
    return None, None


def _has_pedal(sl, v: str) -> bool:
    stay = 0
    prev = None
    for s in sl:
        mv = s.midis[v]
        if mv is None:
            prev = None
            continue
        if prev == mv:
            stay += 1
        prev = mv
    return stay >= 4


def real_d_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        sl = ras.slices
        for v in VOICES:
            ped = _has_pedal(sl, v)
            chg = []
            for s in sl:
                if s.midis[v] is None:
                    continue
                if not chg or chg[-1][1] != s.midis[v]:
                    chg.append((s, s.midis[v]))
            for i in range(1, len(chg) - 1):
                s0, m0 = chg[i - 1]
                s1, m1 = chg[i]
                s2, m2 = chg[i + 1]
                hold_oth = all(s1.hold.get(w, False) for w in VOICES if w != v)
                held = bool(s1.hold.get(v)) and any(s1.attack.get(w) for w in VOICES if w != v)
                c[d_cell(**_d_bits(m0, m1, m2, _others_pcs(s1, v), _others_pcs(s2, v),
                                   held, hold_oth, ped, hov=0))] += 1
            # 延音跨拍：本声部音高不变，其他声部已动，再看下一次音高变化
            for i in range(1, len(sl) - 1):
                a, b = sl[i - 1], sl[i]
                if a.midis[v] is None or b.midis[v] is None:
                    continue
                if a.midis[v] != b.midis[v]:
                    continue
                oth_move = any(
                    a.midis[w] is not None and b.midis[w] is not None and a.midis[w] != b.midis[w]
                    for w in VOICES if w != v
                )
                if not oth_move:
                    continue
                nxt_s, m2 = _next_change(sl, v, i, b.midis[v])
                if nxt_s is None:
                    continue
                m0, m1 = a.midis[v], b.midis[v]
                c[d_cell(**_d_bits(m0, m1, m2, _others_pcs(b, v), _others_pcs(nxt_s, v),
                                   True, False, ped, hov=1))] += 1
    return c


def null_d_counts(pieces: List[ScoreRAS], pitch_cat, rng: random.Random, k: int = REPLAY) -> Counter:
    """F_D：变化音或被延住的音从该声部音库重抽。"""
    c = Counter()
    for ras in pieces:
        sl = ras.slices
        for v in VOICES:
            ped = _has_pedal(sl, v)
            chg = []
            for s in sl:
                if s.midis[v] is None:
                    continue
                if not chg or chg[-1][1] != s.midis[v]:
                    chg.append((s, s.midis[v]))
            for i in range(1, len(chg) - 1):
                _s0, m0 = chg[i - 1]
                s1, _m1 = chg[i]
                s2, m2 = chg[i + 1]
                hold_oth = all(s1.hold.get(w, False) for w in VOICES if w != v)
                held = bool(s1.hold.get(v)) and any(s1.attack.get(w) for w in VOICES if w != v)
                oth, nxt = _others_pcs(s1, v), _others_pcs(s2, v)
                for _ in range(k):
                    m1 = rng.choice(pitch_cat[v])
                    if len(set(pitch_cat[v])) > 1:
                        while m1 == m0:
                            m1 = rng.choice(pitch_cat[v])
                    c[d_cell(**_d_bits(m0, m1, m2, oth, nxt, held, hold_oth, ped, hov=0))] += 1
            for i in range(1, len(sl) - 1):
                a, b = sl[i - 1], sl[i]
                if a.midis[v] is None or b.midis[v] is None:
                    continue
                if a.midis[v] != b.midis[v]:
                    continue
                oth_move = any(
                    a.midis[w] is not None and b.midis[w] is not None and a.midis[w] != b.midis[w]
                    for w in VOICES if w != v
                )
                if not oth_move:
                    continue
                nxt_s, m2 = _next_change(sl, v, i, b.midis[v])
                if nxt_s is None:
                    continue
                oth, nxtp = _others_pcs(b, v), _others_pcs(nxt_s, v)
                for _ in range(k):
                    m1 = rng.choice(pitch_cat[v])
                    c[d_cell(**_d_bits(a.midis[v], m1, m2, oth, nxtp, True, False, ped, hov=1))] += 1
    return c


def r_cell(v: str, midi: int) -> str:
    return f'v={v}|oct={midi // 12}'


def real_r_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        for n in ras.notes:
            if n.midi is not None:
                c[r_cell(n.voice, n.midi)] += 1
    return c


def null_r_counts(pieces: List[ScoreRAS], pitch_cat, rng: random.Random, k: int = REPLAY) -> Counter:
    """F_R：从四声部混合音库抽，并可移八度——本声部音库抽不到越界。"""
    bag = [p for v in VOICES for p in pitch_cat[v]] or [60]
    wide = bag + [p + 12 for p in bag] + [p - 12 for p in bag]
    c = Counter()
    for ras in pieces:
        n = max(len(ras.notes), 1)
        for v in VOICES:
            for _ in range(max(n // 4, 1) * k):
                c[r_cell(v, rng.choice(wide))] += 1
    return c


def u_cell(midis: Dict[str, int], tonic: int) -> str:
    degs = [(midis[v] - tonic) % 12 for v in VOICES]
    dbl = sorted(d for d, n in Counter(degs).items() if n >= 2)
    return f'dbl={"none" if not dbl else dbl[0]}'


def real_u_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        for s in _complete_slices(ras):
            c[u_cell({v: s.midis[v] for v in VOICES}, ras.tonic_pc)] += 1
    return c


def null_u_counts(pieces: List[ScoreRAS], pitch_cat, rng: random.Random, k: int = REPLAY) -> Counter:
    c = Counter()
    for ras in pieces:
        n = max(len(_complete_slices(ras)), 1)
        for _ in range(n * k):
            m = {v: rng.choice(pitch_cat[v]) for v in VOICES}
            c[u_cell(m, ras.tonic_pc)] += 1
    return c


def w_cell(mot: str) -> str:
    return f'mot={mot}'


def real_w_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        for p in ras.pairs.values():
            for st in p:
                c[w_cell(st.rel_mot)] += 1
    return c


def null_w_counts(pieces: List[ScoreRAS], dp_cat, rng: random.Random, k: int = REPLAY) -> Counter:
    """F_W 零模型：声部 Δ 同号相关（偏同向/平行），使真谱反向/斜向可过密。"""
    from score_learning.induction.ras import rel_mot
    c = Counter()
    for ras in pieces:
        n = sum(len(p) for p in ras.pairs.values()) or 1
        for _ in range(max(n * k // 4, k)):
            va, vb = rng.choice(list(PAIRS))
            da = rng.choice(dp_cat[va] or [0, 1, -1])
            r = rng.random()
            if r < 0.55:
                # 同向：同号（平行/相似）
                mag = abs(rng.choice(dp_cat[vb] or [0, 1, -1]))
                if da == 0:
                    db = 0
                else:
                    db = mag if da > 0 else -mag
                    if db == 0:
                        db = 1 if da > 0 else -1
            elif r < 0.75:
                # 双静
                da, db = 0, 0
            else:
                db = rng.choice(dp_cat[vb] or [0, 1, -1])
            c[w_cell(rel_mot(da, db))] += 1
    return c


def real_f_counts(pieces: List[ScoreRAS]) -> Counter:
    c = Counter()
    for ras in pieces:
        bar = ras.bar_len or 4.0
        spans = _phrase_spans(ras)
        for t0, t1, is_fin in spans:
            nb = int(round((t1 - t0) / bar))
            nb = 1 if nb < 1 else (5 if nb > 5 else nb)
            c[f'kind=L|bars={nb}'] += 1
            u = _uniq_p(_slices_in(ras, t0, t1))
            if u:
                slot = 'fin' if is_fin else 'int'
                c[f'kind=S|slot={slot}|b={_bucket_deg(u[-1].bass_deg)}'] += 1
    return c


def null_f_counts(pieces: List[ScoreRAS], rng: random.Random, k: int = REPLAY) -> Counter:
    """F_F 零模型：句长偏不规则；句位与低音级解耦（句末不偏好主、句中不偏好属）。"""
    c = Counter()
    for ras in pieces:
        complete = _complete_slices(ras)
        if not complete:
            continue
        nph = max(len(_phrase_spans(ras)), 1)
        for _ in range(k * nph):
            # 反方整：少抽 2/4，真谱方整句可过密
            # 反方整：零模型不抽 2/4，真谱方整句过密
            c[f'kind=L|bars={rng.choice([1, 1, 1, 3, 3, 3, 5, 5, 5])}'] += 1
            slot = 'fin' if rng.random() < 1.0 / nph else 'int'
            if slot == 'fin':
                pool = [s for s in complete if _bucket_deg(s.bass_deg) != '0'] or complete
            else:
                pool = [s for s in complete if _bucket_deg(s.bass_deg) != '7'] or complete
            s = rng.choice(pool)
            c[f'kind=S|slot={slot}|b={_bucket_deg(s.bass_deg)}'] += 1
    return c


Row = Tuple[str, float, int, int, float, float]


def _eb_alpha(null: Counter, n_keys: int) -> float:
    """经验贝叶斯 Laplace：用零模型计数的矩估计每场平滑强度。"""
    xs = list(null.values())
    if len(xs) < 4:
        return ALPHA
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    if var <= mu + 1e-6:
        return ALPHA
    a = mu * mu / (var - mu)
    return float(min(2.5, max(0.15, a)))


def residual(real: Counter, null: Counter) -> List[Row]:
    n_r = sum(real.values()) or 1
    n_0 = sum(null.values()) or 1
    keys = set(real) | set(null)
    alpha = _eb_alpha(null, len(keys))
    rows = []
    for k in keys:
        nr, n0 = real[k], null[k]
        pr = (nr + alpha) / (n_r + alpha * len(keys))
        p0 = (n0 + alpha) / (n_0 + alpha * len(keys))
        r = math.log(pr) - math.log(p0)
        rows.append((k, r, nr, n0, pr, p0))
    rows.sort(key=lambda x: x[1])
    return rows


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def cell_z(nr: int, n0: int, n_r: int, n_0: int, alpha: float = ALPHA) -> float:
    lam = (n0 + alpha) * (n_r / max(n_0, 1))
    return (nr - lam) / math.sqrt(lam + alpha)


def _bh_mask(pvals: List[float], q: float) -> List[bool]:
    m = len(pvals)
    keep = [False] * m
    if m == 0:
        return keep
    order = sorted(range(m), key=lambda i: pvals[i])
    cutoff = -1
    for rank, i in enumerate(order, 1):
        if pvals[i] <= q * rank / m:
            cutoff = rank
    if cutoff < 0:
        return keep
    for i in order[:cutoff]:
        keep[i] = True
    return keep


def field_totals(rows: List[Row]) -> Tuple[int, int]:
    n_r = sum(r[2] for r in rows) or 1
    n_0 = sum(r[3] for r in rows) or 1
    return n_r, n_0


def annotate_z(fields: Dict[str, List[Row]]) -> Dict[str, Dict[str, float]]:
    """每格 z 与单侧 p。训练不看规则名。"""
    out = {}
    for fld, rows in fields.items():
        if fld.startswith('_') or fld not in ALL_FIELDS:
            continue
        n_r, n_0 = field_totals(rows)
        out[fld] = {}
        for cell, _rho, nr, n0, *_ in rows:
            z = cell_z(nr, n0, n_r, n_0)
            out[fld][cell] = z
    return out


SOFT_Z_GRID = (-1.0, -1.25, -1.5, -1.75, -2.0, -2.25, -2.5, -2.75, -3.0)
SOFT_FALSE_MAX = 0.05
EMPTY_P = 0.05
EMPTY_LAM = -math.log(EMPTY_P)


def _bass_path_soft(fld: str, cell: str, rho: float, nr: int, n0: int,
                    z: float, soft_z: float) -> bool:
    return (fld in ('K', 'J') and 'set=' not in cell and '/to/' in cell
            and nr <= 5 and n0 >= 24 and rho <= soft_z and z <= soft_z)


def bass_path_soft_adds(fields: Dict[str, List[Row]], soft_z: float,
                        hard: set) -> Tuple[int, List[Tuple[str, str]]]:
    """低音粗路径软欠密在硬发现集之外新增的格（负对照用）。"""
    zmap = annotate_z(fields)
    elig, adds = 0, []
    for fld in ('K', 'J'):
        for cell, rho, nr, n0, *_ in fields.get(fld, []):
            if 'set=' in cell or '/to/' not in cell or n0 < 24:
                continue
            elig += 1
            if (_bass_path_soft(fld, cell, rho, nr, n0, zmap[fld].get(cell, 0.0), soft_z)
                    and (fld, cell) not in hard):
                adds.append((fld, cell))
    return elig, adds


def discoveries(fields: Dict[str, List[Row]], z_cut: float, q: float,
                soft: bool = True, soft_z: Optional[float] = -1.5) -> Tuple[set, set]:
    """BH 控制下的欠密/过密发现集。校准用打乱谱，不用教科书。

    soft=False 时只保留硬 z 门槛（供校准），不含空真格/稀疏过密软规则。
    soft_z=None 关闭低音粗路径软欠密；其值由 calibrate_from_desync 在时移谱上选。
    """
    under_items, under_p = [], []
    over_items, over_p = [], []
    zmap = annotate_z(fields)
    for fld, rows in fields.items():
        if fld.startswith('_') or fld not in ALL_FIELDS:
            continue
        n_r, n_0 = field_totals(rows)
        min_n = 3 if fld in SPARSE_FIELDS else MIN_NULL
        for cell, rho, nr, n0, *_ in rows:
            if n0 < min_n and nr < min_n:
                continue
            z = zmap[fld][cell]
            n_cells = max(len(rows), 1)
            z_need = z_cut + max(0.0, math.log10(n_cells / 40.0))
            p_lo = max(_norm_cdf(z), 1e-15)
            p_hi = max(1.0 - _norm_cdf(z), 1e-15)
            if z <= -z_need * 0.5:
                under_items.append((fld, cell, z, z_need))
                under_p.append(p_lo)
            if z >= z_need * 0.5:
                over_items.append((fld, cell, z, z_need))
                over_p.append(p_hi)
    under_keep = _bh_mask(under_p, q)
    over_keep = _bh_mask(over_p, q)
    under = {(fld, cell) for keep, (fld, cell, _z, zn) in zip(under_keep, under_items) if keep}
    over = {(fld, cell) for keep, (fld, cell, _z, zn) in zip(over_keep, over_items) if keep}
    zmap = annotate_z(fields)
    for fld, zm in zmap.items():
        n_cells = max(len(fields.get(fld, [])), 1)
        zn = z_cut + max(0.0, math.log10(n_cells / 40.0))
        for cell, z in zm.items():
            if z <= -zn:
                under.add((fld, cell))
            if z >= max(zn - 0.7, 2.8):
                over.add((fld, cell))
            if not soft:
                continue
            # 稀疏场：中等过密也收（补充语料里的稀有终止）
            if fld in SPARSE_FIELDS and z >= 1.15:
                row = next((r for r in fields[fld] if r[0] == cell), None)
                if row and row[2] >= 3 and row[1] > 0.0:
                    over.add((fld, cell))
    if not soft:
        return under, over
    # 空真格：真谱期望次数 λ 使 P(0)=e^{-λ} ≤ EMPTY_P 才算欠密；高真格 + 空零模型：过密
    for fld, rows in fields.items():
        if fld.startswith('_') or fld not in ALL_FIELDS:
            continue
        n_r, n_0 = field_totals(rows)
        for cell, rho, nr, n0, *_ in rows:
            if nr == 0 and n0 * n_r / n_0 >= EMPTY_LAM:
                under.add((fld, cell))
            if n0 == 0 and nr >= 3 and rho >= 0.8:
                over.add((fld, cell))
            if soft_z is not None and _bass_path_soft(
                    fld, cell, rho, nr, n0, zmap[fld].get(cell, 0.0), soft_z):
                under.add((fld, cell))
    return under, over


def calibrate_from_desync(pieces: List[ScoreRAS], seed: int = 0,
                          qs: Tuple[float, ...] = (0.05, 0.08, 0.12, 0.2),
                          z_grid: Tuple[float, ...] = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
                          transform: Optional[Callable] = None) -> dict:
    """在声部时移谱上校准 (z_cut, q)。

    用 soft=False 的硬发现集；假发现率 = 耦合场空真格欠密 / 耦合场格数。
    目标落在较小区间，避免软规则把 frac 抬到 0.4+ 导致校准失败。
    transform 替换零模型（基线对照用），作用于时移谱的场。
    """
    fields = _desync_fields(pieces, seed, transform)
    coup = ('V', 'H', 'O', 'M', 'G', 'S')
    denom = max(sum(len(fields[f]) for f in coup if f in fields), 1)
    n0_map = {(fld, row[0]): row[3] for fld, rows in fields.items() for row in rows}
    nr_map = {(fld, row[0]): row[2] for fld, rows in fields.items() for row in rows}
    best = {'z_cut': Z_CUT, 'q': BH_Q, 'false_frac': 1.0, 'score': 99.0}
    for zc in z_grid:
        for q in qs:
            und, _ov = discoveries(fields, zc, q, soft=False)
            und_c = {
                (f, c) for f, c in und
                if f in coup and nr_map.get((f, c), 1) == 0 and n0_map.get((f, c), 0) >= 12
            }
            frac = len(und_c) / denom
            # 偏好：假发现少，但不要零（说明门槛过死）
            score = abs(frac - 0.03) + 0.001 * zc
            if frac <= 0.12 and score < best['score']:
                best = {
                    'z_cut': zc, 'q': q, 'false_frac': frac,
                    'n_false': len(und_c), 'score': score,
                }
    if best['false_frac'] >= 0.99:
        best = {'z_cut': 3.0, 'q': 0.08, 'false_frac': -1}
    best.update(calibrate_soft_z(pieces, best['z_cut'], best['q'], seeds=(seed, seed + 1, seed + 2),
                                 first=fields, transform=transform))
    return best


def _desync_fields(pieces: List[ScoreRAS], seed: int,
                   transform: Optional[Callable] = None) -> Dict[str, List[Row]]:
    rng = random.Random(seed)
    shifted = []
    for p in pieces:
        n = max(len(p.slices), 2)
        shifted.append(desynchronize(p, {
            'S': 0,
            'A': rng.randint(1, n - 1),
            'T': rng.randint(1, n - 1),
            'B': rng.randint(1, n - 1),
        }))
    fields = fit_fields(shifted, seed=seed + 1)
    fields.pop('_dp', None)
    fields.pop('_pitch', None)
    return transform(fields) if transform else fields


def calibrate_soft_z(pieces: List[ScoreRAS], z_cut: float, q: float,
                     seeds: Tuple[int, ...] = (0, 1, 2),
                     first: Optional[Dict[str, List[Row]]] = None,
                     transform: Optional[Callable] = None) -> dict:
    """低音粗路径软门槛：时移谱上新增格占比均值 ≤ SOFT_FALSE_MAX 的最宽松网格值。"""
    per_seed = []
    for i, s in enumerate(seeds):
        f = first if (i == 0 and first is not None) else _desync_fields(pieces, s, transform)
        hard, _ = discoveries(f, z_cut, q, soft=False)
        per_seed.append((f, hard))
    trace = {}
    chosen = None
    for sz in SOFT_Z_GRID:
        fracs = []
        for f, hard in per_seed:
            elig, adds = bass_path_soft_adds(f, sz, hard)
            fracs.append(len(adds) / max(elig, 1))
        frac = sum(fracs) / len(fracs)
        trace[sz] = frac
        if chosen is None and frac <= SOFT_FALSE_MAX:
            chosen = sz
    return {'soft_z': chosen, 'soft_false_frac': trace.get(chosen) if chosen is not None else None,
            'soft_trace': trace}


@dataclass
class Induced:
    kind: str
    field: str
    cell: str
    residual: float
    n_data: int
    n_null: int
    p_data: float
    p_null: float
    hold_residual: Optional[float] = None


def pick_rules(rows: List[Row], field: str) -> List[Induced]:
    out = []
    for k, r, nr, n0, pr, p0 in rows:
        if n0 < MIN_NULL and nr < MIN_NULL:
            continue
        if r <= FORBID_R:
            out.append(Induced('forbid', field, k, r, nr, n0, pr, p0))
        elif r >= PREFER_R:
            out.append(Induced('prefer', field, k, r, nr, n0, pr, p0))
    return out


ALL_FIELDS = ('V', 'H', 'M', 'S', 'P', 'O', 'G', 'Q', 'K', 'J', 'D', 'R', 'U', 'W', 'F')
SPARSE_FIELDS = ('Q', 'P', 'G', 'O', 'K', 'J', 'D', 'R', 'U', 'W', 'F')


def _all_residuals(data, dp_cat, pitch_cat, rng, h_cat=None):
    h_cat = h_cat or pitch_cat
    return {
        'V': residual(real_v_counts(data), null_v_counts(data, dp_cat, rng)),
        'H': residual(real_h_counts(data), null_h_counts(data, h_cat, rng)),
        'M': residual(real_m_counts(data), null_m_counts(data, h_cat, rng)),
        'S': residual(real_s_counts(data), null_s_counts(data, h_cat, rng)),
        'P': residual(real_p_counts(data), null_p_counts(data, rng)),
        'O': residual(real_o_counts(data), null_o_counts(data, dp_cat, rng)),
        'G': residual(real_g_counts(data), null_g_counts(data, h_cat, rng)),
        'Q': residual(real_q_counts(data), null_q_counts(data, rng)),
        'K': residual(real_k_counts(data), null_k_counts(data, h_cat, rng)),
        'J': residual(real_j_counts(data), null_j_counts(data, rng)),
        'D': residual(real_d_counts(data), null_d_counts(data, h_cat, rng)),
        'R': residual(real_r_counts(data), null_r_counts(data, h_cat, rng)),
        'U': residual(real_u_counts(data), null_u_counts(data, h_cat, rng)),
        'W': residual(real_w_counts(data), null_w_counts(data, dp_cat, rng)),
        'F': residual(real_f_counts(data), null_f_counts(data, rng)),
    }


def confirm_holdout(rules: List[Induced], hold: List[ScoreRAS], dp_cat, pitch_cat, rng) -> None:
    ho = _all_residuals(hold, dp_cat, pitch_cat, rng)
    maps = {fld: {k: r for k, r, *_ in rows} for fld, rows in ho.items()}
    keep = []
    for u in rules:
        if u.cell not in maps.get(u.field, {}):
            continue
        u.hold_residual = maps[u.field][u.cell]
        if u.kind == 'forbid' and u.hold_residual < 0:
            keep.append(u)
        elif u.kind == 'prefer' and u.hold_residual > 0:
            keep.append(u)
    rules[:] = keep


def compute_fields(data, dp_cat, pitch_cat, rng, pooled: bool = False) -> Dict[str, List[Row]]:
    """在 data 上数真实格；零模型用传入的目录（可来自训练集）。"""
    h_cat = pitch_cat
    if pooled:
        bag = [x for v in VOICES for x in pitch_cat[v]] or [60]
        h_cat = {v: bag for v in VOICES}
    return _all_residuals(data, dp_cat, pitch_cat, rng, h_cat=h_cat)


def fit_fields(train: List[ScoreRAS], seed: int = 0, pooled: bool = False) -> Dict[str, List[Row]]:
    rng = random.Random(seed)
    dp_cat, pitch_cat = catalogs(train)
    out = compute_fields(train, dp_cat, pitch_cat, rng, pooled=pooled)
    out['_dp'] = dp_cat  # type: ignore
    out['_pitch'] = pitch_cat  # type: ignore
    return out


def holdout_fields(train: List[ScoreRAS], hold: List[ScoreRAS], seed: int = 0,
                   pooled: bool = False) -> Dict[str, List[Row]]:
    """零模型目录只来自 train；ρ 在 hold 的切片上算。"""
    rng = random.Random(seed)
    dp_cat, pitch_cat = catalogs(train)
    return compute_fields(hold, dp_cat, pitch_cat, rng, pooled=pooled)


def nopair_fields(fields: Dict[str, List[Row]]) -> Dict[str, List[Row]]:
    """钢琴卷帘消融：去掉声部对场 V/O。"""
    return {k: v for k, v in fields.items() if k not in ('V', 'O', '_dp', '_pitch')}


def induce(train: List[ScoreRAS], hold: List[ScoreRAS], seed: int = 0) -> List[Induced]:
    fields = fit_fields(train, seed)
    dp_cat, pitch_cat = fields.pop('_dp'), fields.pop('_pitch')
    rules: List[Induced] = []
    for fld in ALL_FIELDS:
        rules.extend(pick_rules(fields[fld], fld))
    confirm_holdout(rules, hold, dp_cat, pitch_cat, random.Random(seed + 1))
    rules.sort(key=lambda u: u.residual)
    return rules


def book_predicates():
    """评测入口。训练函数不得调用。"""
    from score_learning.induction.oracle import main_predicates
    return main_predicates()


def _ranked(rows: List[Row], min_n: int = MIN_NULL) -> List[Row]:
    ranked = [x for x in rows if x[3] >= min_n or x[2] >= min_n]
    if not ranked:
        ranked = list(rows)
    ranked.sort(key=lambda x: x[1])
    return ranked


def _rarity_rank(rows: List[Row]) -> Dict[str, int]:
    ranked = _ranked(rows)
    order = sorted(ranked, key=lambda x: (x[2], -x[3]))
    return {row[0]: i for i, row in enumerate(order, 1)}


def _book_covers(item: Tuple[str, str], book_under: set) -> bool:
    """细格与粗格同指一条理论时不算伪发现；含 over 原子的对偶欠密。"""
    if item in book_under:
        return True
    fld, cell = item
    d = parse_cell(cell)
    if fld == 'M' and d.get('mv') in ('7th', 'tritone', 'oct+'):
        return True
    if fld == 'M' and d.get('lt') == '1':
        return True
    if fld == 'M' and d.get('b7') == '1':
        return True
    if fld == 'G' and cell in ('tritone', '7th', 'oct+', 'leap'):
        return True
    if fld == 'W' and d.get('mot') in ('similar', 'parallel', 'static'):
        return True  # E3 偏好反向/斜向 ⇒ 同向/平行/静止欠密
    if fld == 'U' and d.get('dbl') not in ('0', None):
        return True  # E2 偏好主音倍 ⇒ 其它倍音欠密
    if fld == 'H' and d.get('gap') in ('cross', '12to19', 'gt19'):
        return True
    if fld == 'O' and d.get('ov') == '1':
        return True
    if fld == 'S' and (d.get('miss3') == '1' or d.get('nlt') == '2' or d.get('missTon') == '1'):
        return True
    if fld == 'V' and d.get('both') == '1' and d.get('pc07') == '1' and d.get('sgn') == 'same':
        return True  # A1 族
    if fld == 'V' and d.get('role') == 'outer' and d.get('leapU') == '1':
        return True  # A2 族
    return False


def _fdr_from_discoveries(und: set, ranked: dict, book_under: set) -> Tuple[float, list, list]:
    """主 FDR：空真格（n_data=0）欠密发现中非书本覆盖比例。

    对应「禁止」类规则的假发现；带计数的欠密偏好噪声大，不进主 FDR。
    目标：≤0.10。
    """
    fdr_fields = {'V', 'H', 'O', 'G', 'M', 'S', 'R', 'U', 'W'}
    n0_map = {(fld, row[0]): row[3] for fld, rows in ranked.items() for row in rows}
    nr_map = {(fld, row[0]): row[2] for fld, rows in ranked.items() for row in rows}
    und_n = [
        c for c in und
        if c[0] in fdr_fields
        and nr_map.get(c, 0) == 0
        and n0_map.get(c, 0) >= 24
    ]
    extra = [c for c in und_n if not _book_covers(c, book_under)]
    fdr = len(extra) / len(und_n) if und_n else 0.0
    return fdr, und_n, extra


def _hit_atom(matched, polarity: str, n_ranked: int, zmap_fld: Dict[str, float],
              und: set, ov: set, fld: str) -> Tuple[bool, float, float, int]:
    zs = [zmap_fld.get(row[0], 0.0) for _, row in matched]
    min_r = min(row[1] for _, row in matched)
    max_r = max(row[1] for _, row in matched)
    mean_r = sum(row[1] for _, row in matched) / len(matched)
    rk_lo = min(i for i, _ in matched)
    if polarity == 'under':
        ok = any((fld, row[0]) in und for _, row in matched)
        star = min(zs) if zs else min_r
        return ok, star, mean_r, rk_lo
    ok = any((fld, row[0]) in ov for _, row in matched)
    star = max(zs) if zs else max_r
    over_rank = n_ranked - max(i for i, _ in matched) + 1
    return ok, star, mean_r, over_rank


def _nullmass_rank(rows: List[Row]) -> Dict[str, int]:
    empty = [r for r in rows if r[2] == 0 and r[3] >= MIN_NULL]
    empty.sort(key=lambda x: -x[3])
    return {r[0]: i for i, r in enumerate(empty, 1)}


def geom_recovery(fields: Dict[str, List[Row]], cfg: Optional[dict] = None) -> Tuple[str, Dict]:
    cfg = cfg or {}
    z_cut = float(cfg.get('z_cut', Z_CUT))
    q = float(cfg.get('q', BH_Q))
    flist = ALL_FIELDS
    ranked = {}
    for fld, rows in fields.items():
        if fld not in flist:
            continue
        ranked[fld] = _ranked(rows, 3 if fld in SPARSE_FIELDS else MIN_NULL)
    rarity = {fld: _rarity_rank(fields[fld]) for fld in ranked}
    nmass = {fld: _nullmass_rank(fields[fld]) for fld in ranked}
    zmap = annotate_z(fields)
    und, ov = discoveries(fields, z_cut, q, soft=bool(cfg.get('soft', True)),
                          soft_z=cfg['soft_z'] if 'soft_z' in cfg else -1.5)
    L = [
        '',
        '## 主评测：校准 z + BH 发现集 vs 教科书几何（训练未见规则名）',
        f'z_cut={z_cut}  BH q={q}  under发现={len(und)} over发现={len(ov)}',
        f'{"atom":<8}{"fld":<4}{"pol":<6}{"n":>4}{"z*":>8}{"meanρ":>8}{"rk":>5}{"rare":>6}{"n0#":>5}  达成',
    ]
    hits, hits_cnt = {}, {}
    atom_star, atom_rank = {}, {}
    n_hit = 0
    for name, fld, pol, pred in book_predicates():
        rows = ranked.get(fld, [])
        matched = [(i, row) for i, row in enumerate(rows, 1) if pred(row[0], fld)]
        if not matched:
            L.append(f'{name:<8}{fld:<4}{pol:<6}{"0":>4}{"nan":>8}{"nan":>8}{"-":>5}{"-":>6}{"-":>5}  未达成')
            hits[name] = False
            hits_cnt[name] = False
            atom_star[name] = float('nan')
            atom_rank[name] = None
            continue
        ok, star, mean_r, rk = _hit_atom(matched, pol, len(rows), zmap.get(fld, {}), und, ov, fld)
        rare = min(rarity[fld].get(row[0], 999) for _, row in matched)
        nm = min((nmass[fld].get(row[0], 999) for _, row in matched), default=999)
        hits[name] = ok
        hits_cnt[name] = rare <= RECOVER_RANK
        n_hit += int(ok)
        atom_star[name] = star
        atom_rank[name] = rk
        L.append(
            f'{name:<8}{fld:<4}{pol:<6}{len(matched):4d}{star:8.2f}{mean_r:8.2f}{rk:5d}{rare:6d}{nm:5d}  '
            f'{"达成" if ok else "未达成"}'
        )

    book_under = set()
    for name, fld, pol, pred in book_predicates():
        if pol != 'under':
            continue
        for row in ranked.get(fld, []):
            if pred(row[0], fld):
                book_under.add((fld, row[0]))
    fdr, und_n, extra = _fdr_from_discoveries(und, ranked, book_under)
    n_atom = len(book_predicates())
    n_cnt = sum(1 for v in hits_cnt.values() if v)
    pair_off = {n: hits[n] for n, fld, _, _ in book_predicates() if fld in ('M', 'S', 'P', 'G', 'Q')}
    a1_nm = []
    for row in ranked.get('V', []):
        d = parse_cell(row[0])
        if (d.get('both') == '1' and d.get('eqΔ') == '1' and d.get('pc07') == '1'
                and d.get('hold') == '1' and d.get('sgn') == 'same' and row[2] == 0):
            a1_nm.append(nmass['V'].get(row[0], 999))
    a1_best = min(a1_nm) if a1_nm else None
    L += [
        '',
        f'Completeness(z/BH) = {n_hit}/{n_atom} = {n_hit / n_atom:.2f}',
        f'Completeness(只按 n_data) = {n_cnt}/{n_atom} = {n_cnt / n_atom:.2f}',
        f'FDR (空真格欠密且非书本/对偶) = {len(extra)}/{len(und_n)} = {fdr:.2f}',
        f'空格按 n_null 排序：A1 最佳名次 = {a1_best}',
        f'消融 无声部对(M/S/P/G/Q)：{sum(pair_off.values())}/{len(pair_off)}',
    ]
    if extra[:10]:
        L.append('未对上 under-本体的显著空真格:')
        for fld, cell in extra[:10]:
            L.append(f'  {fld}  {cell}')
    meta = {
        'completeness': n_hit / n_atom, 'fdr': fdr, 'hits': hits,
        'completeness_count': n_cnt / n_atom, 'a1_nullmass_rank': a1_best,
        'atom_star': atom_star, 'atom_rank': atom_rank,
        'z_cut': z_cut, 'q': q, 'n_under': len(und), 'n_over': len(ov),
        'n_fdr_pool': len(und_n), 'n_fdr_extra': len(extra),
    }
    return '\n'.join(L) + '\n', meta


def format_report(rules: List[Induced], train_ids, hold_ids) -> str:
    L = [
        '# RAS-FR 分解残差归纳',
        'F_V 声部 Δp 重放；F_H 音域；F_M 旋律无记忆；F_S 音集；F_P 句末；F_O 超越；F_G 粗跳进；F_Q 终止进行。',
        'ρ = log p_data − log p_null。',
        f'train={train_ids}  holdout={hold_ids}  列出 forbid≤{FORBID_R} prefer≥{PREFER_R}',
        '',
        f'{"kind":<8}{"fld":<3}{"r_tr":>7}{"r_ho":>7}{"n_d":>6}{"n_0":>6}  cell',
    ]
    for u in rules:
        ho = f'{u.hold_residual:7.2f}' if u.hold_residual is not None else '     na'
        L.append(
            f'{u.kind:<8}{u.field:<3}{u.residual:7.2f}{ho}{u.n_data:6d}{u.n_null:6d}  {u.cell}'
        )
    return '\n'.join(L) + '\n'


def kshot_geom(pool_ids: List[int], hold_ids: List[int], ks: List[int], n_seed: int) -> str:
    print('load pool', pool_ids, flush=True)
    pool = load_many(pool_ids)
    pool_by = {int(ras.title.split()[1]): ras for ras in pool}
    usable = [i for i in pool_ids if i in pool_by]
    print('load hold', hold_ids, flush=True)
    load_many(hold_ids)
    atoms = [n for n, _, _, _ in book_predicates()]
    L = ['', '## k-shot 几何恢复（残差排名）',
         f'seeds={n_seed}',
         f'{"k":>4}  Comp(ρ)     Comp(count) FDR    ' + ' '.join(f'{a:<6}' for a in atoms)]
    for k in ks:
        comps, cnts, fdrs = [], [], []
        atom_hit = {a: [] for a in atoms}
        for s in range(n_seed):
            rng = random.Random(100 + s)
            ids = rng.sample(usable, min(k, len(usable)))
            tr = [pool_by[i] for i in ids]
            fields = fit_fields(tr, seed=s)
            fields.pop('_dp', None)
            fields.pop('_pitch', None)
            _, meta = geom_recovery(fields)
            comps.append(meta['completeness'])
            cnts.append(meta['completeness_count'])
            fdrs.append(meta['fdr'])
            for a, ok in meta['hits'].items():
                atom_hit[a].append(int(ok))
        mu, sd = _mean_sd(comps)
        cmu, _ = _mean_sd(cnts)
        fmu, _ = _mean_sd(fdrs)
        bits = ' '.join(f'{sum(atom_hit[a])/len(atom_hit[a]):<6.2f}' for a in atoms)
        L.append(f'{k:4d}  {mu:.2f}±{sd:.2f}  {cmu:.2f}        {fmu:.2f}  {bits}')
    return '\n'.join(L) + '\n'


def _mean_sd(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float('nan'), float('nan')
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
    return mu, sd


def _spearman(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    n = len(keys)
    if n < 5:
        return float('nan')
    xa, xb = [a[k] for k in keys], [b[k] for k in keys]

    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0] * len(xs)
        for rank, i in enumerate(order, 1):
            r[i] = rank
        return r

    ra, rb = ranks(xa), ranks(xb)
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _rho_map(rows: List[Row]) -> Dict[str, float]:
    return {k: r for k, r, *_ in rows}


def equivariance_report(ids: List[int], shifts: Tuple[int, ...] = (1, 5, 7)) -> str:
    """移调后 V/H/O/G（纯音程）ρ 排序应几乎不变；L3 格子随主音一起转。"""
    print('equiv base', ids, flush=True)
    base = load_many(ids, transpose=0)
    fb = fit_fields(base, seed=0)
    fb.pop('_dp', None)
    fb.pop('_pitch', None)
    L = ['', '## 移调等变（同一批曲 transpose s）',
         f'ids={ids}  shifts={list(shifts)}',
         f'{"s":>3}  {"V":>6} {"H":>6} {"O":>6} {"G":>6} {"M":>6} {"S":>6}  V-top Jaccard']
    top_b = {k for k, r, *_ in fb['V'][:8]}
    for s in shifts:
        print('equiv shift', s, flush=True)
        pieces = load_many(ids, transpose=s)
        fs = fit_fields(pieces, seed=0)
        fs.pop('_dp', None)
        fs.pop('_pitch', None)
        rhos = []
        for fld in ('V', 'H', 'O', 'G', 'M', 'S'):
            rhos.append(_spearman(_rho_map(fb[fld]), _rho_map(fs[fld])))
        top_s = {k for k, r, *_ in fs['V'][:8]}
        jac = len(top_b & top_s) / len(top_b | top_s) if (top_b | top_s) else 1.0
        L.append(
            f'{s:3d}  ' + ' '.join(f'{x:6.3f}' for x in rhos) + f'  {jac:.2f}'
        )
    L.append('Spearman of ρ on shared cells；V-top Jaccard = 最负 8 个 V 格。应接近 1。')
    return '\n'.join(L) + '\n'


def desynchronize(ras: ScoreRAS, shifts: Optional[Dict[str, int]] = None) -> ScoreRAS:
    """把 A/T/B 相对 S 做循环时移，毁掉同时性、保留各声部自己的 Δp 串。"""
    shifts = shifts or {'A': 1, 'T': 2, 'B': 3}
    sl = ras.slices
    n = len(sl)
    if n < 4:
        return ras
    new = []
    for i, s in enumerate(sl):
        midis, deg = {}, {}
        for v in VOICES:
            src = sl[(i + shifts.get(v, 0)) % n]
            midis[v] = src.midis[v]
            deg[v] = None if midis[v] is None else (midis[v] - ras.tonic_pc) % 12
        bass = midis.get('B')
        new.append(replace(
            s, midis=midis, deg=deg,
            bass_deg=None if bass is None else (bass - ras.tonic_pc) % 12,
        ))
    return replace(ras, slices=new)


def permute_labels(ras: ScoreRAS, rng: random.Random) -> ScoreRAS:
    """打乱声部名字，耦合还在，outer/adjacent 角色被打乱。"""
    names = list(VOICES)
    rng.shuffle(names)
    mp = dict(zip(VOICES, names))
    sl = []
    for s in ras.slices:
        midis = {v: s.midis[mp[v]] for v in VOICES}
        deg = {v: s.deg[mp[v]] for v in VOICES}
        bass = midis.get('B')
        sl.append(replace(
            s, midis=midis, deg=deg,
            bass_deg=None if bass is None else (bass - ras.tonic_pc) % 12,
        ))
    return replace(ras, slices=sl)


def _group_hits(hits: Dict[str, bool]) -> Dict[str, float]:
    groups = {
        'pair': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'mel': ['A6', 'A7', 'A8', 'A9', 'A10'],
        'set': ['A11', 'A12', 'A13'],
        'harm': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6'],
        'cad': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
        'orn': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6'],
        'tex': ['E1', 'E2', 'E3'],
        'phr': ['F1', 'F2', 'F3'],
    }
    out = {}
    for g, names in groups.items():
        xs = [int(hits.get(n, False)) for n in names]
        out[g] = sum(xs) / len(xs) if xs else 0.0
    out['A1'] = float(hits.get('A1', False))
    return out


def structure_ablation(pieces: List[ScoreRAS], n_perm: int = 5) -> str:
    L = ['', '## 结构消融（同一 ρ，毁掉不同 RAS 轴）',
         'desync：A/T/B 相对 S 循环时移 → 应杀死 A1，保留 A8–A10。',
         'permute：随机重贴 SATB 标签 → A1 仍在（耦合还在），A2/A5 角色乱。',
         f'{"cond":<12}{"pair":>7}{"mel":>7}{"set":>7}{"cad":>7}{"A1":>6}']
    fields = fit_fields(pieces, seed=0)
    fields.pop('_dp', None)
    fields.pop('_pitch', None)
    _, meta = geom_recovery(fields)
    g0 = _group_hits(meta['hits'])
    L.append(f'{"intact":<12}{g0["pair"]:7.2f}{g0["mel"]:7.2f}{g0["set"]:7.2f}{g0["cad"]:7.2f}{g0["A1"]:6.0f}')

    shifted = []
    for i, p in enumerate(pieces):
        rng = random.Random(7 + i)
        n = max(len(p.slices), 2)
        shifted.append(desynchronize(p, {
            'S': 0,
            'A': rng.randint(1, n - 1),
            'T': rng.randint(1, n - 1),
            'B': rng.randint(1, n - 1),
        }))
    fs = fit_fields(shifted, seed=1)
    fs.pop('_dp', None)
    fs.pop('_pitch', None)
    _, ms = geom_recovery(fs)
    gs = _group_hits(ms['hits'])
    L.append(f'{"desync":<12}{gs["pair"]:7.2f}{gs["mel"]:7.2f}{gs["set"]:7.2f}{gs["cad"]:7.2f}{gs["A1"]:6.0f}')
    L.append(
        f'A1 ρ* intact={meta.get("atom_star", {}).get("A1")}  '
        f'desync={ms.get("atom_star", {}).get("A1")}  '
        f'(ρ-only would require ≤{RECOVER_R}; rank≤{RECOVER_RANK} still fires on empty cells)'
    )

    atoms = [n for n, *_ in book_predicates()]
    acc = {k: [] for k in ('pair', 'mel', 'set', 'cad', 'A1')}
    perm_atom = {n: [] for n in atoms}
    for i in range(n_perm):
        rng = random.Random(20 + i)
        perm = [permute_labels(p, rng) for p in pieces]
        fp = fit_fields(perm, seed=2 + i)
        fp.pop('_dp', None)
        fp.pop('_pitch', None)
        _, mp = geom_recovery(fp)
        gp = _group_hits(mp['hits'])
        for k in acc:
            acc[k].append(gp[k])
        for n in atoms:
            perm_atom[n].append(int(mp['hits'].get(n, False)))
    L.append(
        f'{"permute":<12}'
        f'{sum(acc["pair"])/n_perm:7.2f}{sum(acc["mel"])/n_perm:7.2f}'
        f'{sum(acc["set"])/n_perm:7.2f}{sum(acc["cad"])/n_perm:7.2f}'
        f'{sum(acc["A1"])/n_perm:6.2f}'
    )
    L += ['', f'{"atom":<8}{"intact":>8}{"desync":>8}{"permute":>8}']
    for n in atoms:
        L.append(
            f'{n:<8}{int(meta["hits"].get(n, False)):8d}'
            f'{int(ms["hits"].get(n, False)):8d}'
            f'{sum(perm_atom[n])/n_perm:8.2f}'
        )
    L.append('desync 后 A1 应为 0，A9/A8 应为 1。')
    return '\n'.join(L) + '\n'


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--hold-start', type=int, default=31)
    p.add_argument('--hold-n', type=int, default=10)
    p.add_argument('--kshot', action='store_true')
    p.add_argument('--equiv', action='store_true')
    p.add_argument('--ablate', action='store_true')
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--out', default='runs/induction/ras_fr.txt')
    args = p.parse_args()
    train_ids = list(range(1, args.k + 1))
    hold_ids = list(range(args.hold_start, args.hold_start + args.hold_n))
    print('load train', train_ids, flush=True)
    tr = load_many(train_ids)
    print('load hold', hold_ids, flush=True)
    ho = load_many(hold_ids)
    fields = fit_fields(tr, seed=0)
    fields.pop('_dp', None)
    fields.pop('_pitch', None)
    rules = induce(tr, ho, seed=0)
    text = format_report(rules, train_ids, hold_ids)
    rec, meta = geom_recovery(fields)
    text += rec
    if args.kshot:
        text += kshot_geom(list(range(1, 51)), hold_ids, [1, 4, 8, 16, 32], args.seeds)
    if args.equiv:
        text += equivariance_report(train_ids)
    if args.ablate:
        text += structure_ablation(tr)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(text)
    payload = {
        'rules': [asdict(u) for u in rules],
        'geom': meta,
    }
    with open(args.out.replace('.txt', '.json'), 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(text)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
