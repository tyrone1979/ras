# -*- coding: utf-8 -*-
"""DCRA 规范表示：逐声部事件序列 + D2 上下文 → 四声部 music21 谱 → 冻结解析器。

不依赖旧的同节奏构谱函数。四个声部各有自己的起音/持续/休止序列；
小节、乐句终止（女高音延长记号）与调性由本模块显式写出。

D2 = 调性上下文、拍子与小节起点、乐句边界、四声部联合起音/持续/休止矩阵。
`structure(piece)` 返回 D2 的完整指纹；`assert_same_d2` 是方法的硬不变量。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from score_learning.induction import ras as ras_mod
from score_learning.induction.fri import VOICES

EPS = 1e-6
INSIDE = 0.01  # 查询“结束时刻之前正在发声的音”时向内取的偏移（小于最小时值）
SLICE_STEP = 0.5  # 与冻结解析器 from_music21 的切片步长一致


class StructureViolation(AssertionError):
    pass


@dataclass(frozen=True)
class Event:
    onset: float
    dur: float
    midi: Optional[int]  # None = 休止

    @property
    def end(self) -> float:
        return self.onset + self.dur


@dataclass(frozen=True)
class Piece:
    title: str
    voices: Dict[str, Tuple[Event, ...]]
    bar_starts: Tuple[float, ...]      # 小节起点（照抄原谱；真实曲目中弱起与延长记号处小节可不等长）
    bar_len: float                     # 每小节四分音符数（x/4 拍子）
    phrase_ends: Tuple[float, ...]     # 乐句结束时刻（不含），最后一个 = 全曲长度
    tonic_pc: int
    mode: str                          # 'major' / 'minor'

    @property
    def length(self) -> float:
        return self.phrase_ends[-1]


# ---------------------------------------------------------------- validation

def _on_grid(x: float, g: float) -> bool:
    return abs(x / g - round(x / g)) < EPS


def validate(p: Piece, grid: float = 0.125) -> None:
    if set(p.voices) != set(VOICES):
        raise StructureViolation(f'{p.title}: voices {sorted(p.voices)}')
    L = p.length
    for v in VOICES:
        ev = p.voices[v]
        if not ev:
            raise StructureViolation(f'{p.title}/{v}: empty')
        t = 0.0
        for e in ev:
            if abs(e.onset - t) > EPS or e.dur <= EPS:
                raise StructureViolation(f'{p.title}/{v}: gap/overlap at {e.onset} (expected {t})')
            if not (_on_grid(e.onset, grid) and _on_grid(e.dur, grid)):
                raise StructureViolation(f'{p.title}/{v}: off grid {e}')
            if e.midi is not None and not 0 <= e.midi <= 127:
                raise StructureViolation(f'{p.title}/{v}: midi {e.midi}')
            t = e.end
        if abs(t - L) > EPS:
            raise StructureViolation(f'{p.title}/{v}: ends at {t}, piece length {L}')
    pe = p.phrase_ends
    if list(pe) != sorted(pe) or len(set(pe)) != len(pe) or pe[0] <= 0:
        raise StructureViolation(f'{p.title}: phrase_ends {pe}')
    bs = p.bar_starts
    if not bs or abs(bs[0]) > EPS or any(b - a <= EPS for a, b in zip(bs, bs[1:])) \
            or bs[-1] >= L - EPS:
        raise StructureViolation(f'{p.title}: bar_starts {bs[:4]}...')
    for t in pe:
        s = _sounding(p.voices['S'], t - INSIDE)
        if s is None or s.midi is None:
            raise StructureViolation(f'{p.title}: soprano not sounding at phrase end {t}')


def _sounding(ev: Sequence[Event], t: float) -> Optional[Event]:
    for e in ev:
        if e.onset - EPS <= t < e.end - EPS:
            return e
    return None


# ---------------------------------------------------------------- D2 fingerprint

def onset_matrix(p: Piece, step: float = SLICE_STEP) -> Tuple[Tuple[str, ...], ...]:
    """每个切片、每个声部：'A' 起音 / 'H' 持续 / 'R' 休止（与 from_music21 的判定相同）。"""
    out = []
    t = 0.0
    while t < p.length - EPS:
        row = []
        for v in VOICES:
            e = _sounding(p.voices[v], t)
            if e is None or e.midi is None:
                row.append('R')
            else:
                row.append('A' if abs(e.onset - t) < EPS else 'H')
        out.append(tuple(row))
        t += step
    return tuple(out)


def structure(p: Piece) -> tuple:
    return (round(p.length, 6), tuple(round(b, 6) for b in p.bar_starts), p.bar_len,
            tuple(round(x, 6) for x in p.phrase_ends), p.tonic_pc, p.mode, onset_matrix(p))


def assert_same_d2(orig: Piece, cf: Piece) -> None:
    """硬不变量：反事实与原曲的 D2（含逐切片联合起音/持续/休止矩阵）必须完全一致。"""
    a, b = structure(orig), structure(cf)
    if a == b:
        return
    names = ('length', 'bar_starts', 'bar_len', 'phrase_ends', 'tonic_pc', 'mode', 'onset_matrix')
    bad = [n for n, x, y in zip(names, a, b) if x != y]
    if 'onset_matrix' in bad:
        k = next(i for i, (r, s) in enumerate(zip(a[-1], b[-1])) if r != s) \
            if len(a[-1]) == len(b[-1]) else -1
        raise StructureViolation(f'{orig.title}: D2 differs in {bad}; first matrix slice {k}')
    raise StructureViolation(f'{orig.title}: D2 differs in {bad}')


def ras_onset_matrix(sr, drop_trailing_rest: bool = False) -> Tuple[Tuple[str, ...], ...]:
    m = [tuple('R' if s.rest[v] else ('A' if s.attack[v] else 'H') for v in VOICES)
         for s in sr.slices]
    if drop_trailing_rest:
        while m and all(x == 'R' for x in m[-1]):
            m.pop()
    return tuple(m)


# ---------------------------------------------------------------- score construction

def to_score(p: Piece):
    """规范表示 → music21 Score：手工建小节（支持弱起），跨小节线的音用连线拆分，
    每个乐句最后一个女高音（及同时发声的其他声部）加延长记号。"""
    from music21 import expressions, key, meter, metadata, note, stream, tempo, tie

    validate(p)
    ends = list(p.bar_starts[1:]) + [p.length]
    bars = list(zip(p.bar_starts, ends))
    ferm_at = set(round(t, 6) for t in p.phrase_ends)

    sc = stream.Score()
    sc.metadata = metadata.Metadata()
    sc.metadata.title = p.title
    for v, name in zip(VOICES, ('Soprano', 'Alto', 'Tenor', 'Bass')):
        part = stream.Part(id=v)
        part.partName = name
        ferm_events = {id(e) for e in p.voices[v] if round(e.end, 6) in ferm_at and e.midi is not None}
        for bi, (b0, b1) in enumerate(bars):
            m = stream.Measure(number=bi + 1)
            if bi == 0:
                m.insert(0, tempo.MetronomeMark(number=72))
                m.insert(0, meter.TimeSignature(f'{int(p.bar_len)}/4'))
                m.insert(0, key.Key(_key_name(p.tonic_pc, p.mode)))
            for e in p.voices[v]:
                s, t = max(e.onset, b0), min(e.end, b1)
                if t - s <= EPS:
                    continue
                if e.midi is None:
                    n = note.Rest(quarterLength=t - s)
                else:
                    n = note.Note(e.midi, quarterLength=t - s)
                    first, last = s - e.onset < EPS, e.end - t < EPS
                    if not (first and last):
                        n.tie = tie.Tie('start' if first else ('stop' if last else 'continue'))
                    if first and id(e) in ferm_events:
                        n.expressions.append(expressions.Fermata())
                m.insert(s - b0, n)
            part.insert(b0, m)
        sc.insert(0, part)
    return sc


_PC_NAMES = ('C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'A-', 'A', 'B-', 'B')


def _key_name(pc: int, mode: str) -> str:
    n = _PC_NAMES[pc % 12]
    return n if mode == 'major' else n.lower()


def to_ras(p: Piece):
    """构谱并用冻结解析器解析；调性取规范表示中已有的调性上下文（不重新估计）。"""
    sc = to_score(p)
    est = ras_mod.estimate_tonic
    ras_mod.estimate_tonic = lambda _m: (p.tonic_pc, p.mode)
    try:
        return ras_mod.from_music21(sc, title=p.title, t_max=None)
    finally:
        ras_mod.estimate_tonic = est


def check_roundtrip(p: Piece, sr) -> None:
    """解析结果必须与规范表示逐切片一致：起音矩阵、音高、小节位置、乐句终止。"""
    if ras_onset_matrix(sr) != onset_matrix(p):
        raise StructureViolation(f'{p.title}: parsed onset matrix differs')
    for s in sr.slices:
        for v in VOICES:
            e = _sounding(p.voices[v], s.t)
            if (e.midi if e else None) != s.midis[v]:
                raise StructureViolation(f'{p.title}/{v}: pitch differs at t={s.t}')
        bi = max(i for i, b in enumerate(p.bar_starts) if b <= s.t + EPS)
        if s.bar != bi + 1 or abs(s.pos_in_bar - round(s.t - p.bar_starts[bi], 3)) > EPS:
            raise StructureViolation(f'{p.title}: bar position differs at t={s.t}')
    want = [round(_sounding(p.voices['S'], t - INSIDE).onset, 6) for t in p.phrase_ends]
    if [round(x, 6) for x in sr.phrase_ends] != want:
        raise StructureViolation(f'{p.title}: phrase ends {sr.phrase_ends} vs {want}')
    if (sr.tonic_pc, sr.mode_hint) != (p.tonic_pc, p.mode):
        raise StructureViolation(f'{p.title}: tonal context differs')


# ---------------------------------------------------------------- from parsed real data

def from_ras(sr) -> Piece:
    """已解析的真实曲目（ScoreRAS）→ 规范表示。乐句结束 = 女高音延长记号音的结束时刻。
    曲末四声部同时休止的尾部（不含任何音高）被截去，见 `trailing_rest_rows`。"""
    L = min(max(n.offset for n in sr.notes if n.voice == v) for v in VOICES)
    L = min(L, max(n.offset for n in sr.notes if n.midi is not None))
    voices: Dict[str, Tuple[Event, ...]] = {}
    for v in VOICES:
        ev: List[Event] = []
        for n in sorted((n for n in sr.notes if n.voice == v), key=lambda n: n.onset):
            on, off = n.onset, min(n.offset, L)
            if off - on <= EPS:
                continue
            if ev and ev[-1].end < on - EPS:
                ev.append(Event(ev[-1].end, on - ev[-1].end, None))
            ev.append(Event(on, off - on, n.midi))
        if ev and ev[0].onset > EPS:
            ev.insert(0, Event(0.0, ev[0].onset, None))
        voices[v] = tuple(ev)
    s_notes = [n for n in sr.notes if n.voice == 'S' and n.midi is not None]
    pe = sorted({round(min(n.offset, L), 6) for n in s_notes
                 if any(abs(n.onset - t) < EPS for t in sr.phrase_ends)})
    if not pe or pe[-1] < L - EPS:
        pe.append(round(L, 6))
    bars = sorted({round(s.t - s.pos_in_bar, 6) for s in sr.slices})
    return Piece(title=sr.title, voices=voices, bar_starts=tuple(bars), bar_len=sr.bar_len,
                 phrase_ends=tuple(pe), tonic_pc=sr.tonic_pc, mode=sr.mode_hint)
