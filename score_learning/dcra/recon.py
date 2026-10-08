# -*- coding: utf-8 -*-
"""DCRA 正式干预 (b)：序列级独立声部重建（预注册 `dcra-prereg` 第 3.1 节）。

泄漏边界由结构保证：拟合与生成函数只接收 `VoiceView`——一个声部自己的事件序列加上
共享的 D2 上下文（调性、拍子与小节起点、乐句边界）。`VoiceView` 不含其他声部的任何信息。
本模块不导入任何待检验的模式、格子或原子定义（审计见 `audit_recon.py`）。

模型 P_v（预注册后固定）：
  乐句首音：v 的乐句首音 (主音相对音高 = 音级 + 八度) 分布（按调式；样本 < 20 退回不分调式）。
  其余起音：半音步进 ~ P_v(step | 调式内音级, 拍位强弱, 乐句位置)；
           拍位强弱 = music21 重音权重 ≥ 0.5；乐句位置 ∈ {内部, 倒数第二个起音, 最后一个起音}；
           样本 < 20 时依次去掉乐句位置、拍位强弱，最后用 v 的无条件步进分布。
  音域：v 在拟合数据中的 [最低, 最高] 音；超出音域的候选被拒绝重抽
        （等价于在音域内候选上重新归一化；某一级没有合法候选时退到下一级）。
持续与休止照抄；只在 v 自己起音的位置抽新音高。
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from score_learning.dcra import canon
from score_learning.dcra.canon import Event, Piece

MIN_STATE = 20
R_DEFAULT = 32


@dataclass(frozen=True)
class VoiceView:
    """一个声部 + 共享 D2。字段集合是泄漏边界的一部分，由审计脚本逐项核对。"""
    voice: str
    events: Tuple[Event, ...]
    bar_starts: Tuple[float, ...]
    bar_len: float
    phrase_ends: Tuple[float, ...]
    tonic_pc: int
    mode: str


VOICE_VIEW_FIELDS = ('voice', 'events', 'bar_starts', 'bar_len', 'phrase_ends', 'tonic_pc', 'mode')
assert tuple(f.name for f in fields(VoiceView)) == VOICE_VIEW_FIELDS


def voice_view(p: Piece, v: str) -> VoiceView:
    return VoiceView(voice=v, events=p.voices[v], bar_starts=p.bar_starts, bar_len=p.bar_len,
                     phrase_ends=p.phrase_ends, tonic_pc=p.tonic_pc, mode=p.mode)


# ---------------------------------------------------------------- D2-derived state

@lru_cache(maxsize=None)
def _accent(bar_len: float, pos: float) -> float:
    from music21 import meter
    return meter.TimeSignature(f'{int(bar_len)}/4').getAccentWeight(pos)


def metric_class(view: VoiceView, t: float) -> str:
    bs = view.bar_starts
    i = max(k for k, b in enumerate(bs) if b <= t + canon.EPS)
    pos = t - bs[i]
    if i == 0 and len(bs) > 1 and bs[1] - bs[0] < view.bar_len - canon.EPS:
        pos += view.bar_len - (bs[1] - bs[0])           # 弱起小节：从小节末端对齐
    pos = pos % view.bar_len
    return 'strong' if _accent(view.bar_len, round(pos, 6)) >= 0.5 else 'weak'


def degree(view: VoiceView, midi: int) -> Tuple[str, int]:
    return view.mode, (midi - view.tonic_pc) % 12


def phrase_slots(view: VoiceView) -> List[List[int]]:
    """每个乐句中 v 的起音事件下标（按乐句结束时刻划分）。"""
    out: List[List[int]] = [[] for _ in view.phrase_ends]
    k = 0
    for i, e in enumerate(view.events):
        if e.midi is None:
            continue
        while e.onset >= view.phrase_ends[k] - canon.EPS:
            k += 1
        out[k].append(i)
    return out


def _ppos(j: int, n: int) -> str:
    if j == n - 1:
        return 'final'
    if j == n - 2:
        return 'penult'
    return 'interior'


def attack_states(view: VoiceView):
    """逐起音给出 (事件下标, 是否乐句首音, 状态键)；状态取决于“当前音”，生成时现算。"""
    out = []
    for slots in phrase_slots(view):
        for j, i in enumerate(slots):
            out.append((i, j == 0, view.events[i].onset, _ppos(j, len(slots))))
    return out


# ---------------------------------------------------------------- model

@dataclass
class VoiceModel:
    voice: str
    lo: int
    hi: int
    init: Dict[object, Counter]          # key: mode 或 '*'  → Counter(主音相对音高)
    step: Dict[tuple, Counter]           # key: 状态元组（含各回退级）→ Counter(半音步进)


def _keys(mode_deg, mc, pp):
    return [('L0', mode_deg, mc, pp), ('L1', mode_deg, mc), ('L2', mode_deg), ('L3',)]


def fit_voice(views: Sequence[VoiceView]) -> VoiceModel:
    if not views:
        raise ValueError('no data')
    v = views[0].voice
    if any(w.voice != v for w in views):
        raise ValueError('fit_voice takes views of a single voice')
    init: Dict[object, Counter] = defaultdict(Counter)
    step: Dict[tuple, Counter] = defaultdict(Counter)
    lo, hi = 999, -1
    for w in views:
        prev: Optional[int] = None
        for i, first, t, pp in attack_states(w):
            m = w.events[i].midi
            lo, hi = min(lo, m), max(hi, m)
            if first:
                init[w.mode][m - w.tonic_pc] += 1
                init['*'][m - w.tonic_pc] += 1
            else:
                for k in _keys(degree(w, prev), metric_class(w, t), pp):
                    step[k][m - prev] += 1
            prev = m
    return VoiceModel(voice=v, lo=lo, hi=hi, init=dict(init), step=dict(step))


def _draw(counter: Counter, ok, rng: random.Random) -> Optional[int]:
    cand = [(x, n) for x, n in sorted(counter.items()) if ok(x)]
    if not cand:
        return None
    tot = sum(n for _, n in cand)
    r = rng.random() * tot
    for x, n in cand:
        r -= n
        if r < 0:
            return x
    return cand[-1][0]


def generate_voice(model: VoiceModel, view: VoiceView, rng: random.Random) -> Tuple[Event, ...]:
    if model.voice != view.voice:
        raise ValueError('model/view voice mismatch')
    new = {}
    cur: Optional[int] = None
    for i, first, t, pp in attack_states(view):
        if first:
            c = model.init.get(view.mode)
            ok = lambda r: model.lo <= view.tonic_pc + r <= model.hi
            r = None
            if c is not None and sum(c.values()) >= MIN_STATE:
                r = _draw(c, ok, rng)
            if r is None:
                r = _draw(model.init['*'], ok, rng)
            if r is None:
                raise RuntimeError(f'{view.voice}: no in-range phrase-initial pitch')
            cur = view.tonic_pc + r
        else:
            ok = lambda s: model.lo <= cur + s <= model.hi
            s = None
            for k in _keys(degree(view, cur), metric_class(view, t), pp):
                c = model.step.get(k)
                if c is None or (k[0] != 'L3' and sum(c.values()) < MIN_STATE):
                    continue
                s = _draw(c, ok, rng)
                if s is not None:
                    break
            if s is None:
                raise RuntimeError(f'{view.voice}: no in-range step')
            cur = cur + s
        new[i] = cur
    return tuple(Event(e.onset, e.dur, new.get(i)) if e.midi is not None else e
                 for i, e in enumerate(view.events))


# ---------------------------------------------------------------- collection-level API

def fit(pieces: Sequence[Piece]) -> Dict[str, VoiceModel]:
    from score_learning.induction.fri import VOICES
    return {v: fit_voice([voice_view(p, v) for p in pieces]) for v in VOICES}


def voice_seed(base: int, piece_idx: int, v: str, r: int) -> int:
    """每个 (曲, 声部, 重建) 一个独立随机流：一个声部的结果不依赖其他声部。"""
    return hash((base, piece_idx, 'SATB'.index(v), r)) & 0xFFFFFFFFFFFF


def reconstruct(p: Piece, models: Dict[str, VoiceModel], base_seed: int, piece_idx: int,
                r: int) -> Piece:
    from score_learning.induction.fri import VOICES
    voices = {}
    for v in VOICES:
        rng = random.Random(voice_seed(base_seed, piece_idx, v, r))
        voices[v] = generate_voice(models[v], voice_view(p, v), rng)
    cf = Piece(title=f'{p.title}#cf{r}', voices=voices, bar_starts=p.bar_starts,
               bar_len=p.bar_len, phrase_ends=p.phrase_ends, tonic_pc=p.tonic_pc, mode=p.mode)
    check_reconstruction(p, cf, models)
    return cf


def check_reconstruction(orig: Piece, cf: Piece, models: Dict[str, VoiceModel]) -> None:
    """每次重建后的硬断言：D2 完全一致（含联合起音/持续/休止矩阵）；音高都在声部音域内。"""
    canon.validate(cf)
    canon.assert_same_d2(orig, cf)
    for v, m in models.items():
        for e in cf.voices[v]:
            if e.midi is not None and not m.lo <= e.midi <= m.hi:
                raise canon.StructureViolation(f'{cf.title}/{v}: {e.midi} outside [{m.lo},{m.hi}]')
