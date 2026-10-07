# -*- coding: utf-8 -*-
"""RAS 全场：L0 音符 / L1 差 / L2 差-差 / Lv 声部对 / L3 主音 / L4 织体音域 / L5 音集。

训练不写罗马数字、终止名、违规号。几何场必须够写回 A1–A13。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

VOICES = ('S', 'A', 'T', 'B')
PAIRS = (('S', 'A'), ('S', 'T'), ('S', 'B'), ('A', 'T'), ('A', 'B'), ('T', 'B'))
ADJ = {('S', 'A'), ('A', 'T'), ('T', 'B')}
OUTER = {('S', 'B')}
DEG_NAMES = {0: '1', 1: '#1', 2: '2', 3: 'b3', 4: '3', 5: '4',
             6: '#4', 7: '5', 8: 'b6', 9: '6', 10: 'b7', 11: '7'}
PC_NAMES = {0: 'C', 1: 'C#', 2: 'D', 3: 'Eb', 4: 'E', 5: 'F',
            6: 'F#', 7: 'G', 8: 'Ab', 9: 'A', 10: 'Bb', 11: 'B'}
DUR_BINS = ((0.26, '16'), (0.51, '8'), (1.01, '4'), (2.01, '2'), (4.01, '1'))


def midi_name(m: Optional[int]) -> str:
    if m is None:
        return 'rest'
    return PC_NAMES[m % 12] + str(m // 12 - 1)


def dur_bin(d: float) -> str:
    for th, name in DUR_BINS:
        if d < th:
            return name
    return 'long'


def perfect_class(iv: int) -> str:
    a = abs(iv)
    if a == 0:
        return 'P1'
    if a % 12 == 7:
        return 'P5' if a == 7 else 'P5c'
    if a % 12 == 0 and a > 0:
        return 'P8' if a == 12 else 'P8c'
    return '-'


def rel_mot(dp_i: int, dp_j: int) -> str:
    if dp_i == 0 and dp_j == 0:
        return 'static'
    if dp_i == 0 or dp_j == 0:
        return 'oblique'
    if (dp_i > 0) != (dp_j > 0):
        return 'contrary'
    if dp_i == dp_j:
        return 'parallel'
    return 'similar'


def metric_label(pos_in_bar: float, bar_len: float) -> str:
    if abs(pos_in_bar) < 1e-6:
        return 'downbeat'
    if bar_len >= 4 and abs(pos_in_bar - 2.0) < 1e-6:
        return 'strong'
    if abs(pos_in_bar - round(pos_in_bar)) < 1e-6:
        return 'weak'
    return 'offbeat'


def estimate_tonic(midis: List[int]) -> Tuple[int, str]:
    maj = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
    mi = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
    hist = [0.0] * 12
    for m in midis:
        hist[m % 12] += 1.0
    s = sum(hist) or 1.0
    hist = [h / s for h in hist]
    best, best_sc, mode = 0, -1e9, 'major'
    for tonic in range(12):
        rot = hist[tonic:] + hist[:tonic]
        sm = sum(a * b for a, b in zip(rot, maj))
        si = sum(a * b for a, b in zip(rot, mi))
        if sm > best_sc:
            best, best_sc, mode = tonic, sm, 'major'
        if si > best_sc:
            best, best_sc, mode = tonic, si, 'minor'
    return best, mode


@dataclass
class Note:
    voice: str
    onset: float
    offset: float
    midi: Optional[int]
    tied: bool = False
    fermata: bool = False

    @property
    def dur(self) -> float:
        return self.offset - self.onset

    @property
    def is_rest(self) -> bool:
        return self.midi is None

    @property
    def pc(self) -> Optional[int]:
        return None if self.midi is None else self.midi % 12


@dataclass
class HorizDelta:
    voice: str
    t0: float
    t1: float
    dp: int
    dpc: int
    ddeg: int
    ddur: float
    dur_ratio: float
    dt: float
    doct: int
    leap: bool
    step: bool
    aug: bool


@dataclass
class FireParams:
    """几何点火参数：与 checker 谓词对齐。"""
    perfect_pc: Tuple[int, ...] = (0, 7)
    a1_both_move: bool = True
    a1_same_pc: bool = True
    a2_leap_gt: int = 2
    a2_same_dir: bool = True
    skip_rests: bool = True
    a1a2_also_beats: bool = True     # 与 check_satb 一样：八分网格 + 拍点网格
    a4a5a8_beats_only: bool = True
    sa_max: int = 12
    at_max: int = 12
    tb_max: int = 19
    allow_unison: bool = True


@dataclass
class PairStep:
    pair: Tuple[str, str]
    t: float
    interval: int
    interval_prev: int
    d_interval: int
    dp_a: int
    dp_b: int
    iv_mod: int
    iv_prev_mod: int
    perfect: str
    order: str
    who_moves: str
    rel_mot: str
    d_parallel: int
    common_tone: bool
    doubling: bool
    crossed: bool
    overlap: bool
    hidden: bool
    attack_rel: str
    role: str
    is_beat: bool


@dataclass
class Slice:
    t: float
    bar: int
    pos_in_bar: float
    metric: str
    midis: Dict[str, Optional[int]]
    attack: Dict[str, bool]
    hold: Dict[str, bool]
    rest: Dict[str, bool]
    deg: Dict[str, Optional[int]]
    pc_multiset: Counter
    n_sounding: int
    bass_deg: Optional[int]
    has_tonic: bool
    has_dom: bool
    has_lt: bool
    has_third_above_bass: bool
    missing_tonic_triad: List[str]
    voice_roles: Dict[str, str]  # tonic/dom/lt/third/other/rest


@dataclass
class ScoreRAS:
    title: str
    tonic_pc: int
    mode_hint: str
    key_sig: int
    meter: str
    bar_len: float
    notes: List[Note]
    slices: List[Slice]
    horiz: Dict[str, List[HorizDelta]]
    d2p: Dict[str, List[int]]
    passing: Dict[str, List[str]]
    pairs: Dict[Tuple[str, str], List[PairStep]]
    range_mm: Dict[str, Tuple[int, int]]
    tessitura: Dict[str, float]
    n_sounding_mean: float
    attack_sync: float
    mot_hist: Dict[str, int]
    melody_voice: str
    pedal_voice: str
    imitation: List[Tuple[str, str, int, float]]
    exchanges: List[str]
    phrase_ends: List[float]


def _diatonic_aug(dp: int) -> bool:
    return abs(dp) in (3, 6, 8, 9, 10)  # coarse: tritone/aug2-ish; 3 may be m3


def _timeline(part) -> List[Note]:
    from music21 import expressions
    voice = '?'
    out: List[Note] = []
    for n in part.flatten().notesAndRests:
        on = float(n.offset)
        off = on + float(n.quarterLength)
        if n.quarterLength <= 0:
            continue
        ferm = False
        if not n.isRest:
            ferm = any(isinstance(e, expressions.Fermata) for e in n.expressions)
        midi = None
        if not n.isRest:
            midi = int(n.pitches[-1].midi) if n.isChord else int(n.pitch.midi)
        tie = getattr(n, 'tie', None)
        if midi is not None and tie is not None and tie.type in ('stop', 'continue') \
                and out and out[-1].midi == midi and abs(out[-1].offset - on) < 1e-6:
            out[-1].offset = off
            out[-1].fermata = out[-1].fermata or ferm
            out[-1].tied = True
            continue
        out.append(Note(voice=voice, onset=on, offset=off, midi=midi,
                        tied=bool(tie and tie.type == 'start'), fermata=ferm))
    return out


def _at(notes: List[Note], t: float) -> Tuple[Optional[Note], bool]:
    for n in notes:
        if n.onset - 1e-6 <= t < n.offset - 1e-6:
            return n, abs(n.onset - t) < 1e-6
    return None, False


def _horiz(notes: List[Note], tonic: int) -> List[HorizDelta]:
    sounding = [n for n in notes if n.midi is not None]
    out = []
    for a, b in zip(sounding, sounding[1:]):
        dp = b.midi - a.midi
        ddur = math.log(max(b.dur, 1e-6)) - math.log(max(a.dur, 1e-6))
        out.append(HorizDelta(
            voice=a.voice, t0=a.onset, t1=b.onset, dp=dp,
            dpc=dp % 12 if dp >= 0 else -((-dp) % 12),
            ddeg=((b.midi - tonic) % 12) - ((a.midi - tonic) % 12),
            ddur=ddur, dur_ratio=b.dur / max(a.dur, 1e-6),
            dt=b.onset - a.onset, doct=(b.midi // 12) - (a.midi // 12),
            leap=abs(dp) >= 5, step=abs(dp) in (1, 2),
            aug=abs(dp) in (6, 8, 9),  # tritone / aug5 / maj6 flagged coarse
        ))
    return out


def _imitation(horiz: Dict[str, List[HorizDelta]]) -> List[Tuple[str, str, int, float]]:
    seq = {v: [h.dp for h in horiz[v]] for v in VOICES}
    hits = []
    for i, a in enumerate(VOICES):
        for b in VOICES[i + 1:]:
            sa, sb = seq[a], seq[b]
            if len(sa) < 3 or len(sb) < 3:
                continue
            best_lag, best_r = 0, -1.0
            for lag in range(1, 5):
                if lag >= len(sa) or lag >= len(sb):
                    break
                xa, xb = sa[:-lag], sb[lag:]
                n = min(len(xa), len(xb))
                if n < 3:
                    continue
                xa, xb = xa[:n], xb[:n]
                mx, my = sum(xa) / n, sum(xb) / n
                num = sum((p - mx) * (q - my) for p, q in zip(xa, xb))
                den = (sum((p - mx) ** 2 for p in xa) * sum((q - my) ** 2 for q in xb)) ** 0.5
                r = num / den if den > 1e-6 else 0.0
                if r > best_r:
                    best_lag, best_r = lag, r
            hits.append((a, b, best_lag, round(best_r, 3)))
    hits.sort(key=lambda x: -x[3])
    return hits


def from_music21(score, title: str = '', t_max: Optional[float] = 8.0, step: float = 0.5) -> ScoreRAS:
    from music21 import meter, key as m21key, stream
    parts = list(score.parts)
    if len(parts) != 4:
        raise ValueError(f'need 4 parts, got {len(parts)}')
    tls = [_timeline(p) for p in parts]
    for v, tl in zip(VOICES, tls):
        for n in tl:
            n.voice = v
    piece_end = min(tl[-1].offset for tl in tls if tl)
    if t_max is None:
        t_max = piece_end
    notes = [n for tl in tls for n in tl if n.onset < t_max - 1e-6]
    for n in notes:
        if n.offset > t_max:
            n.offset = t_max

    ts = score.flatten().getElementsByClass(meter.TimeSignature)
    ts0 = ts[0] if ts else None
    bar_len = float(ts0.numerator) if ts0 and ts0.denominator == 4 else 4.0
    meter_s = f'{ts0.numerator}/{ts0.denominator}' if ts0 else '4/4'
    ks = score.flatten().getElementsByClass(m21key.KeySignature)
    key_sig = int(ks[0].sharps) if ks else 0

    sounding = [n.midi for n in notes if n.midi is not None]
    tonic, mode = estimate_tonic(sounding)
    phrase_ends = [n.onset for n in notes if n.fermata and n.voice == 'S' and n.onset < t_max]

    bars = []
    p0 = parts[0]
    for m in p0.getElementsByClass(stream.Measure):
        bars.append(float(m.offset))
    if not bars:
        bars = [0.0]

    def bar_pos(t: float) -> Tuple[int, float]:
        bi = 0
        for i, st in enumerate(bars):
            if t >= st - 1e-6:
                bi = i
        st = bars[bi]
        return bi + 1, t - st

    slices: List[Slice] = []
    t = 0.0
    end = min(t_max, min(tl[-1].offset for tl in tls if tl))
    while t < end - 1e-6:
        evs = [_at(tl, t) for tl in tls]
        midis, attack, hold, rest, deg = {}, {}, {}, {}, {}
        pcs: List[int] = []
        for v, (ev, att) in zip(VOICES, evs):
            if ev is None or ev.midi is None:
                midis[v] = None
                attack[v] = False
                hold[v] = False
                rest[v] = True
                deg[v] = None
            else:
                midis[v] = ev.midi
                attack[v] = att
                hold[v] = not att
                rest[v] = False
                deg[v] = (ev.midi - tonic) % 12
                pcs.append(ev.midi % 12)
        multi = Counter(pcs)
        sounding_m = {v: m for v, m in midis.items() if m is not None}
        n_snd = len(sounding_m)
        bass_m = sounding_m.get('B')
        bass_deg = (bass_m - tonic) % 12 if bass_m is not None else None
        rels = {(m - tonic) % 12 for m in sounding_m.values()}
        has_tonic = 0 in rels
        has_dom = 7 in rels
        has_lt = 11 in rels
        third_iv = {3, 4}
        has_3b = False
        if bass_m is not None:
            for m in sounding_m.values():
                if abs(m - bass_m) % 12 in third_iv:
                    has_3b = True
        miss = []
        if mode == 'major':
            if 0 not in rels:
                miss.append('1')
            if 4 not in rels:
                miss.append('3')
            if 7 not in rels:
                miss.append('5')
        else:
            if 0 not in rels:
                miss.append('1')
            if 3 not in rels:
                miss.append('b3')
            if 7 not in rels:
                miss.append('5')
        roles = {}
        for v, d in deg.items():
            if d is None:
                roles[v] = 'rest'
            elif d == 0:
                roles[v] = 'tonic'
            elif d == 7:
                roles[v] = 'dom'
            elif d == 11:
                roles[v] = 'lt'
            elif bass_m is not None and midis[v] is not None \
                    and abs(midis[v] - bass_m) % 12 in third_iv:
                roles[v] = 'third'
            else:
                roles[v] = 'other'
        bi, pos = bar_pos(t)
        slices.append(Slice(
            t=t, bar=bi, pos_in_bar=round(pos, 3), metric=metric_label(pos, bar_len),
            midis=midis, attack=attack, hold=hold, rest=rest, deg=deg,
            pc_multiset=multi, n_sounding=n_snd, bass_deg=bass_deg,
            has_tonic=has_tonic, has_dom=has_dom, has_lt=has_lt,
            has_third_above_bass=has_3b, missing_tonic_triad=miss,
            voice_roles=roles,
        ))
        t += step

    horiz = {v: _horiz([n for n in notes if n.voice == v], tonic) for v in VOICES}
    d2p = {v: [horiz[v][i].dp - horiz[v][i - 1].dp for i in range(1, len(horiz[v]))]
           for v in VOICES}
    passing = {v: [] for v in VOICES}
    for v in VOICES:
        hs = horiz[v]
        for i in range(len(hs) - 1):
            a, b = hs[i], hs[i + 1]
            if a.step and b.step and (a.dp > 0) == (b.dp > 0):
                passing[v].append(f'pass@{a.t1:.1f}')
            elif a.step and b.step and (a.dp > 0) != (b.dp > 0):
                passing[v].append(f'neigh@{a.t1:.1f}')

    pairs: Dict[Tuple[str, str], List[PairStep]] = {p: [] for p in PAIRS}
    for i in range(1, len(slices)):
        prev, cur = slices[i - 1], slices[i]
        for a, b in PAIRS:
            ma0, mb0 = prev.midis[a], prev.midis[b]
            ma1, mb1 = cur.midis[a], cur.midis[b]
            if None in (ma0, mb0, ma1, mb1):
                continue
            iv0, iv1 = ma0 - mb0, ma1 - mb1
            dpa, dpb = ma1 - ma0, mb1 - mb0
            who = '+'.join([x for x, d in ((a, dpa), (b, dpb)) if d]) or 'none'
            order0 = f'{a}>{b}' if ma0 > mb0 else f'{b}>{a}'
            order1 = f'{a}>{b}' if ma1 > mb1 else f'{b}>{a}'
            rm = rel_mot(dpa, dpb)
            perf = perfect_class(iv1)
            hidden = (perf in ('P5', 'P8', 'P5c', 'P8c')
                      and rm in ('parallel', 'similar')
                      and (a, b) == ('S', 'B')
                      and abs(dpa) > 2)
            overlap = False
            if (a, b) in ADJ:
                overlap = (mb1 > ma0) or (ma1 < mb0)
            att_a, att_b = cur.attack[a], cur.attack[b]
            if att_a and att_b:
                ar = 'together'
            elif att_a and cur.hold[b]:
                ar = f'{a}_over_{b}'
            elif att_b and cur.hold[a]:
                ar = f'{b}_over_{a}'
            else:
                ar = 'sustain'
            if (a, b) in OUTER:
                role = 'outer'
            elif (a, b) in ADJ:
                role = 'adjacent'
            else:
                role = 'diagonal'
            pairs[(a, b)].append(PairStep(
                pair=(a, b), t=cur.t, interval=iv1, interval_prev=iv0,
                d_interval=iv1 - iv0, dp_a=dpa, dp_b=dpb,
                iv_mod=iv1 % 12, iv_prev_mod=iv0 % 12,
                perfect=perf, order=order1,
                who_moves=who, rel_mot=rm, d_parallel=dpa - dpb,
                common_tone=(ma1 % 12 == ma0 % 12) or (mb1 % 12 == mb0 % 12),
                doubling=(abs(iv1) % 12 == 0), crossed=(order0 != order1),
                overlap=overlap, hidden=hidden, attack_rel=ar, role=role,
                is_beat=(cur.metric != 'offbeat'),
            ))

    range_mm, tess = {}, {}
    for v in VOICES:
        ms = [n.midi for n in notes if n.voice == v and n.midi is not None]
        range_mm[v] = (min(ms), max(ms)) if ms else (0, 0)
        tess[v] = sum(ms) / len(ms) if ms else 0.0
    n_mean = sum(s.n_sounding for s in slices) / max(len(slices), 1)
    sync = sum(1 for s in slices if sum(s.attack.values()) >= 3) / max(len(slices), 1)
    mot = Counter()
    for p in PAIRS:
        for st in pairs[p]:
            mot[st.rel_mot] += 1
    var = {}
    for v in VOICES:
        dps = [h.dp for h in horiz[v]]
        if not dps:
            var[v] = 0.0
        else:
            m = sum(dps) / len(dps)
            var[v] = sum((x - m) ** 2 for x in dps) / len(dps)
    melody = max(var, key=var.get)
    pedal = min(var, key=var.get)

    exchanges = []
    for i in range(len(slices) - 1):
        a, b = slices[i], slices[i + 1]
        for p, q in (('S', 'A'), ('A', 'T'), ('T', 'B'), ('S', 'B')):
            if None in (a.midis[p], a.midis[q], b.midis[p], b.midis[q]):
                continue
            if a.midis[p] % 12 == b.midis[q] % 12 and a.midis[q] % 12 == b.midis[p] % 12 \
                    and a.midis[p] % 12 != a.midis[q] % 12:
                exchanges.append(f'{p}{q}@{b.t:.1f}')

    return ScoreRAS(
        title=title or getattr(score, 'metadata', None) and str(score.metadata) or 'score',
        tonic_pc=tonic, mode_hint=mode, key_sig=key_sig, meter=meter_s, bar_len=bar_len,
        notes=sorted(notes, key=lambda n: (n.onset, VOICES.index(n.voice))),
        slices=slices, horiz=horiz, d2p=d2p, passing=passing, pairs=pairs,
        range_mm=range_mm, tessitura=tess, n_sounding_mean=n_mean,
        attack_sync=sync, mot_hist=dict(mot), melody_voice=melody, pedal_voice=pedal,
        imitation=_imitation(horiz), exchanges=exchanges, phrase_ends=phrase_ends,
    )


def load_chorale(n: int = 1, t_max: Optional[float] = 8.0, transpose: int = 0) -> ScoreRAS:
    from music21 import corpus
    it = corpus.chorales.Iterator(numberingSystem='riemenschneider', returnType='filename')
    paths = list(it)
    path = paths[n - 1]
    sc = corpus.parse(path)
    if transpose:
        sc = sc.transpose(transpose)
    span = 'full' if t_max is None else f'first {t_max:g} QL'
    title = f'Riemenschneider {n}  {path}  {span}  t{transpose}'
    return from_music21(sc, title=title, t_max=t_max)


PAIR_IJ = {('S', 'A'): (0, 1), ('S', 'T'): (0, 2), ('S', 'B'): (0, 3),
           ('A', 'T'): (1, 2), ('A', 'B'): (1, 3), ('T', 'B'): (2, 3)}


def geometry_fire(ras: ScoreRAS, p: Optional[FireParams] = None) -> Dict[str, set]:
    """带参数的几何函数：从 RAS 场点火，不读规则名。"""
    p = p or FireParams()
    a1, a2, a4, a5, a8 = set(), set(), set(), set(), set()

    def mid(s):
        return [s.midis[v] for v in VOICES]

    cols = [s for s in ras.slices if all(s.midis[v] is not None for v in VOICES)]
    if not p.skip_rests:
        cols = list(ras.slices)
    pairs_idx = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for a, b in zip(cols, cols[1:]):
        v1, v2 = mid(a), mid(b)
        if None in v1 or None in v2:
            continue
        t = round(b.t, 3)
        for i, j in pairs_idx:
            d_i, d_j = v2[i] - v1[i], v2[j] - v1[j]
            both = d_i != 0 and d_j != 0
            pc1, pc2 = (v1[i] - v1[j]) % 12, (v2[i] - v2[j]) % 12
            if (not p.a1_both_move or both) and pc2 in p.perfect_pc:
                if (not p.a1_same_pc or pc1 == pc2) and pc1 in p.perfect_pc:
                    a1.add((t, i, j))
        ds, db = v2[0] - v1[0], v2[3] - v1[3]
        same = (ds > 0) == (db > 0)
        if ds != 0 and db != 0 and (not p.a2_same_dir or same) \
                and abs(ds) > p.a2_leap_gt and (v2[0] - v2[3]) % 12 in p.perfect_pc:
            a2.add((t,))
    beats = [s for s in ras.slices
             if all(s.midis[v] is not None for v in VOICES)
             and ((not p.a4a5a8_beats_only) or s.metric != 'offbeat')]
    if p.a1a2_also_beats:
        for a, b in zip(beats, beats[1:]):
            v1, v2 = mid(a), mid(b)
            t = round(b.t, 3)
            for i, j in pairs_idx:
                d_i, d_j = v2[i] - v1[i], v2[j] - v1[j]
                both = d_i != 0 and d_j != 0
                pc1, pc2 = (v1[i] - v1[j]) % 12, (v2[i] - v2[j]) % 12
                if (not p.a1_both_move or both) and pc2 in p.perfect_pc:
                    if (not p.a1_same_pc or pc1 == pc2) and pc1 in p.perfect_pc:
                        a1.add((t, i, j))
            ds, db = v2[0] - v1[0], v2[3] - v1[3]
            same = (ds > 0) == (db > 0)
            if ds != 0 and db != 0 and (not p.a2_same_dir or same) \
                    and abs(ds) > p.a2_leap_gt and (v2[0] - v2[3]) % 12 in p.perfect_pc:
                a2.add((t,))
    for a, b in zip(beats, beats[1:]):
        t = round(b.t, 3)
        v, pv = mid(b), mid(a)
        s, al, tn, bs = v
        cross = not (s >= al >= tn >= bs) if p.allow_unison else not (s > al > tn > bs)
        wide = (s - al > p.sa_max) or (al - tn > p.at_max) or (tn - bs > p.tb_max)
        if cross or wide:
            a5.add((t,))
        if (v[1] > pv[0] or v[0] < pv[1] or v[2] > pv[1] or v[1] < pv[2]
                or v[3] > pv[2] or v[2] < pv[3]):
            a4.add((t,))
        for vi in range(4):
            d = abs(v[vi] - pv[vi])
            if d == 6 or d in (10, 11) or d > 12:
                a8.add((t, vi))
    return {'A1': a1, 'A2': a2, 'A4': a4, 'A5': a5, 'A8': a8}


def format_full(ras: ScoreRAS) -> str:
    tn = PC_NAMES[ras.tonic_pc]
    L: List[str] = [
        f'# {ras.title}',
        f'L3  tonic={tn}  mode={ras.mode_hint}  key_sig_fifths={ras.key_sig}  '
        f'meter={ras.meter}  bar_len={ras.bar_len}  phrase_fermata_t={ras.phrase_ends}',
        f'     melody_voice={ras.melody_voice}  pedal_voice={ras.pedal_voice}',
        '',
        '## L0 音符事件（onset/offset/dur/bin/metric/deg）',
        f'{"v":2} {"on":>5} {"off":>5} {"dur":>4} bin  met        pitch  deg  rest tie ferm',
    ]
    # metric per note from nearest slice
    def met_at(t: float) -> str:
        best = ras.slices[0]
        for s in ras.slices:
            if abs(s.t - t) < abs(best.t - t):
                best = s
        return f'b{best.bar}:{best.metric}'

    for n in ras.notes:
        pitch = midi_name(n.midi)
        d = None if n.midi is None else (n.midi - ras.tonic_pc) % 12
        L.append(
            f'{n.voice:2} {n.onset:5.1f} {n.offset:5.1f} {n.dur:4.1f} {dur_bin(n.dur):<3}  '
            f'{met_at(n.onset):<10} {pitch:<6} {DEG_NAMES.get(d, "-"):<3}  '
            f'{int(n.is_rest)}    {int(n.tied)}   {int(n.fermata)}'
        )

    L += ['', '## L0 同时切片（含 hold=延续，attack=新起音）']
    L.append('t    bar met       S      A      T      B      att    n  pcs')
    for s in ras.slices:
        cells = ' '.join(
            f'{midi_name(s.midis[v]) if s.midis[v] else "rest":<6}'
            + ('*' if s.attack[v] else ('~' if s.hold[v] else ' '))
            for v in VOICES
        )
        att = ''.join(v if s.attack[v] else '.' for v in 'SATB')
        pcs = ''.join(PC_NAMES[p] + str(c) for p, c in sorted(s.pc_multiset.items()))
        L.append(f'{s.t:4.1f} {s.bar:3} {s.metric:<8} {cells} {att}  {s.n_sounding}  {pcs}')

    L += ['', '## L1 声部内差（相继发声音，非网格）']
    L.append('v  t0→t1     Δp  Δdeg  Δdur  ratio  Δt   leap step aug')
    for v in VOICES:
        for h in ras.horiz[v]:
            L.append(
                f'{v}  {h.t0:4.1f}→{h.t1:4.1f}  {h.dp:+3d}  {h.ddeg:+4d}  {h.ddur:+5.2f}  '
                f'{h.dur_ratio:4.2f}  {h.dt:4.1f}  {int(h.leap)}    {int(h.step)}    {int(h.aug)}'
            )
        L.append(f'   Δ²p={ras.d2p[v]}  windows={ras.passing[v] or ["-"]}')

    L += ['', '## L2 + Lv 声部对时间线']
    L.append('pair role     t    iv  d_iv class Rel        Δ∥  who     att        geom')
    for p in PAIRS:
        for st in ras.pairs[p]:
            geom = []
            if st.rel_mot == 'parallel' and st.perfect.startswith('P') and st.perfect != 'P1':
                if st.d_interval == 0:
                    geom.append(f'par{st.perfect}')
            if st.rel_mot == 'contrary' and st.perfect.startswith('P') and st.perfect != 'P1':
                geom.append(f'ctr{st.perfect}')
            if st.hidden:
                geom.append('hidden')
            if st.crossed:
                geom.append('cross')
            if st.overlap:
                geom.append('overlap')
            if st.doubling:
                geom.append('8ve')
            g = '+'.join(geom) if geom else '.'
            L.append(
                f'{p[0]}{p[1]}  {st.role:<8} {st.t:4.1f}  {st.interval:+3d}  {st.d_interval:+4d}  '
                f'{st.perfect:<5} {st.rel_mot:<9} {st.d_parallel:+3d}  {st.who_moves:<7} '
                f'{st.attack_rel:<10} {g}'
            )

    L += ['', '## L4 音域 / 织体']
    for v in VOICES:
        lo, hi = ras.range_mm[v]
        L.append(f'  {v}  {midi_name(lo)}–{midi_name(hi)}  tess={midi_name(int(round(ras.tessitura[v])))}')
    L.append(f'  n_sounding_mean={ras.n_sounding_mean:.2f}  attack_sync(≥3 voices)={ras.attack_sync:.2f}')
    L.append(f'  Rel_mot hist={ras.mot_hist}')

    L += ['', '## L5 同时音集（无名）+ 声部分配']
    L.append('t    deg_set     bass  1/5/7/3rd  miss   S     A     T     B')
    for s in ras.slices:
        degs = ','.join(sorted({DEG_NAMES[d] for d in (
            (m - ras.tonic_pc) % 12 for m in s.midis.values() if m is not None)}))
        flags = ''.join([
            '1' if s.has_tonic else '.',
            '5' if s.has_dom else '.',
            '7' if s.has_lt else '.',
            '3' if s.has_third_above_bass else '.',
        ])
        miss = ','.join(s.missing_tonic_triad) if s.missing_tonic_triad else '-'
        roles = ' '.join(f'{s.voice_roles[v]:<5}' for v in VOICES)
        bd = DEG_NAMES.get(s.bass_deg, '-')
        L.append(f'{s.t:4.1f}  {{{degs:<8}}}  ^{bd:<3}  {flags}      {miss:<5}  {roles}')

    L += ['', '## 延迟关系']
    L.append(f'  imitation (voice pair, lag QL-steps of Δp, corr): {ras.imitation[:4]}')
    L.append(f'  voice-exchange flags: {ras.exchanges or ["-"]}')

    L += ['', '## 覆盖：这些字段能否无学习表达（有场≠这首在响）']
    L += [
        '  A1 平行五八度     Lv parP5/parP8 且 d_iv=0',
        '  A2 隐伏           SB hidden（外声部同类运动 + 女高跳进 + 到达纯音程）',
        '  A3 交叉           Lv cross (order 翻转)',
        '  A4 超越           Lv overlap',
        '  A5 间距           Lv interval 相邻对',
        '  A6 导音解决       L5 role=lt 的声部 + L1 Δp=+1',
        '  A7 七音           L5 has_lt/集合；下行看该声部 L1 Δp<0（七音≠导音，仅几何代理）',
        '  A8–A10 增程/跳    L1 aug/leap/|Δp|>12',
        '  A11 导音重复      L5 两声部同时 role=lt',
        '  A12/A13 缺音      L5 miss 与 has_third_above_bass（相对主音三和弦，不是罗马）',
        '  B/C               L5 集合序列 + 女高 deg + fermata 句末（类型名不进训练）',
        '  D 经过/辅助       L1 windows pass@ / neigh@ + Lv attack_rel 延续',
        '  E 音域织体        L4 range/tess/sync/mot_hist',
        '  F 乐句            L3 phrase_fermata_t（评测可用；训练可改无监督变点）',
    ]
    return '\n'.join(L) + '\n'


def main():
    import argparse
    import os
    p = argparse.ArgumentParser()
    p.add_argument('--chorale', type=int, default=1)
    p.add_argument('--beats', type=float, default=8.0)
    p.add_argument('--out', default='runs/induction/ras_full.txt')
    args = p.parse_args()
    ras = load_chorale(args.chorale, t_max=args.beats)
    text = format_full(ras)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
