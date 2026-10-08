# -*- coding: utf-8 -*-
"""DCRA 合成基准 S1–S6（预注册 `dcra-prereg` 第 5 节），输出规范表示 `canon.Piece`。

S1–S4：沿用冻结的 `synthetic_truth.gen_events`（生成器不改），只把输出转成规范表示，
       显式写出小节（4/4）与乐句边界（每句 6 个四分音符 + 二分音符 = 两小节）。
S5：S2 音高过程；各声部节奏不同但共享拍子；强拍全体起音；D3 = 0（给定 D2）。
S6：D3 = 0；单声部结构含 P_v 以外的成分（乐句拱形轮廓、跳进后反向级进、重复前句步进序列）。

S5、S6 中每个声部用自己的随机流独立生成，生成函数只接收该声部自己的节奏与共享 D2，
不接触其他声部的任何音高。
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from score_learning.dcra.canon import Event, Piece, validate
from score_learning.induction.fri import VOICES
from score_learning.induction.synthetic_truth import (
    DIA, N_PHRASE, PHRASE_Q, RANGE, START, STEP_P, _deg_w, gen_events,
)

BAR_LEN = 4.0
PHRASE_LEN = PHRASE_Q + 2.0          # 6 个四分音符 + 二分音符 = 8 拍
GENS = ('S1', 'S2', 'S3', 'S4', 'S5', 'S6')
PLANTED = {'S1': (), 'S2': (), 'S3': ('A1',), 'S4': ('A1', 'A11', 'A12'), 'S5': (), 'S6': ()}

# S5 节奏参数（预注册后、首次 DCRA 运行前固定）
S5_P_WEAK = {'S': 0.70, 'A': 0.55, 'T': 0.60, 'B': 0.80}    # 弱拍（第 2、4 拍）起音概率
S5_P_EIGHTH = {'S': 0.25, 'A': 0.15, 'T': 0.20, 'B': 0.10}  # 后半拍八分音符起音概率
S5_STRONG_W = {'B': {0: 3.0, 7: 2.5}, 'S': {0: 2.0, 4: 2.0, 7: 2.0},
               'A': {0: 2.0, 4: 2.0, 7: 2.0}, 'T': {0: 2.0, 4: 2.0, 7: 2.0}}

# S6 单声部结构参数
S6_ARCH = 2.0          # 前半句上行权重 ×2、下行 ×0.5；后半句相反
S6_GAP_SEMIS = 4       # 跳进 > 4 半音
S6_GAP_P = 0.7         # 之后以 0.7 概率反向级进
S6_REPEAT_P = 0.3      # 每句（第 2 句起）以 0.3 概率重复一个前句的音阶步进序列


def _bars(length: float) -> Tuple[float, ...]:
    return tuple(float(i * BAR_LEN) for i in range(int(round(length / BAR_LEN))))


def _phrase_ends() -> Tuple[float, ...]:
    return tuple(float((k + 1) * PHRASE_LEN) for k in range(N_PHRASE))


def _voice_rng(rng: random.Random) -> Dict[str, random.Random]:
    return {v: random.Random(rng.getrandbits(64)) for v in VOICES}


def _start(v: str, rng: random.Random) -> int:
    c = min(RANGE[v][1], max(RANGE[v][0], START[v] + 2 * rng.randint(-2, 2)))
    return min(DIA, key=lambda m: abs(m - c))


def _step_weights(v: str, cur: int, end: bool, extra=None) -> Tuple[List[int], List[float]]:
    """S2 的单声部提议分布（与 synthetic_truth._propose 相同），可乘额外权重 extra(k, m)。"""
    i = DIA.index(cur)
    lo, hi = RANGE[v]
    mid = (lo + hi) / 2
    cand, w = [], []
    for k, p in STEP_P.items():
        j = i + k
        if not 0 <= j < len(DIA) or not lo <= DIA[j] <= hi:
            continue
        m = DIA[j]
        if (cur - mid > 4 and k > 0) or (mid - cur > 4 and k < 0):
            p *= 0.5
        x = p * _deg_w('S2', v, m % 12, end)
        if extra is not None:
            x *= extra(k, m)
        cand.append(m)
        w.append(x)
    return cand, w


# ---------------------------------------------------------------- S1–S4

def _from_events(ev, title: str) -> Piece:
    voices = {}
    for vi, v in enumerate(VOICES):
        out, t = [], 0.0
        for e in ev:
            out.append(Event(t, float(e[4]), int(e[vi])))
            t += float(e[4])
        voices[v] = tuple(out)
    pe = _phrase_ends()
    return Piece(title=title, voices=voices, bar_starts=_bars(pe[-1]), bar_len=BAR_LEN,
                 phrase_ends=pe, tonic_pc=0, mode='major')


# ---------------------------------------------------------------- S5

def _s5_rhythm(v: str, rng: random.Random) -> List[Tuple[float, float]]:
    """一个声部的起音时刻与时值（整曲）。强拍（每小节第 1、3 拍）必起音；乐句末二分音符全体起音。"""
    out: List[Tuple[float, float]] = []
    for k in range(N_PHRASE):
        base = k * PHRASE_LEN
        on = []
        for q in range(PHRASE_Q):
            t = base + q
            strong = (t % BAR_LEN) in (0.0, 2.0)
            if strong or rng.random() < S5_P_WEAK[v]:
                on.append(t)
            if rng.random() < S5_P_EIGHTH[v]:
                on.append(t + 0.5)
        on.append(base + PHRASE_Q)            # 句末二分音符
        ends = on[1:] + [base + PHRASE_LEN]
        out.extend((a, b - a) for a, b in zip(on, ends))
    return out


def _s5_voice(v: str, rhythm, rng: random.Random) -> Tuple[Event, ...]:
    cur = _start(v, rng)
    out = []
    first = True
    for on, dur in rhythm:
        end = abs((on % PHRASE_LEN) - PHRASE_Q) < 1e-9
        strong = (on % BAR_LEN) in (0.0, 2.0)
        if not first:
            sw = S5_STRONG_W[v]
            extra = (lambda k, m: sw.get(m % 12, 1.0)) if strong and not end else None
            cand, w = _step_weights(v, cur, end, extra)
            cur = rng.choices(cand, w)[0]
        first = False
        out.append(Event(on, dur, cur))
    return tuple(out)


def gen_s5(rng: random.Random, title: str) -> Piece:
    vr = _voice_rng(rng)
    voices = {v: _s5_voice(v, _s5_rhythm(v, vr[v]), vr[v]) for v in VOICES}
    pe = _phrase_ends()
    return Piece(title=title, voices=voices, bar_starts=_bars(pe[-1]), bar_len=BAR_LEN,
                 phrase_ends=pe, tonic_pc=0, mode='major')


# ---------------------------------------------------------------- S6

def _s6_voice(v: str, rng: random.Random) -> Tuple[Event, ...]:
    """与 S1–S4 相同的同节奏网格（D2 与 S1–S4 一致），单声部内部结构见模块说明。"""
    cur = _start(v, rng)
    lo, hi = RANGE[v]
    out: List[Event] = []
    phrase_steps: List[List[int]] = []      # 每句的音阶步进（DIA 索引差）
    prev_dp: Optional[int] = None
    t = 0.0
    for k in range(N_PHRASE):
        rep = phrase_steps[rng.randrange(len(phrase_steps))] if k and rng.random() < S6_REPEAT_P else None
        steps: List[int] = []
        for q in range(PHRASE_Q + 1):
            end = q == PHRASE_Q
            dur = 2.0 if end else 1.0
            if out:
                i = DIA.index(cur)
                nxt = None
                if rep is not None and q < len(rep):
                    j = i + rep[q]
                    if 0 <= j < len(DIA) and lo <= DIA[j] <= hi:
                        nxt = DIA[j]
                if nxt is None and prev_dp is not None and abs(prev_dp) > S6_GAP_SEMIS \
                        and rng.random() < S6_GAP_P:
                    j = i - (1 if prev_dp > 0 else -1)
                    if 0 <= j < len(DIA) and lo <= DIA[j] <= hi:
                        nxt = DIA[j]
                if nxt is None:
                    up = q < (PHRASE_Q + 1) / 2

                    def arch(kk, _m, up=up):
                        if kk == 0:
                            return 1.0
                        return S6_ARCH if (kk > 0) == up else 1.0 / S6_ARCH

                    cand, w = _step_weights(v, cur, end, arch)
                    nxt = rng.choices(cand, w)[0]
                steps.append(DIA.index(nxt) - i)
                prev_dp = nxt - cur
                cur = nxt
            else:
                steps.append(0)
            out.append(Event(t, dur, cur))
            t += dur
        phrase_steps.append(steps)
    return tuple(out)


def gen_s6(rng: random.Random, title: str) -> Piece:
    vr = _voice_rng(rng)
    voices = {v: _s6_voice(v, vr[v]) for v in VOICES}
    pe = _phrase_ends()
    return Piece(title=title, voices=voices, bar_starts=_bars(pe[-1]), bar_len=BAR_LEN,
                 phrase_ends=pe, tonic_pc=0, mode='major')


# ---------------------------------------------------------------- entry

def make_pieces(gen: str, n: int, seed: int) -> List[Piece]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        title = f'{gen}-{seed}-{i}'
        if gen in ('S1', 'S2', 'S3', 'S4'):
            p = _from_events(gen_events(gen, rng), title)
        elif gen == 'S5':
            p = gen_s5(rng, title)
        elif gen == 'S6':
            p = gen_s6(rng, title)
        else:
            raise ValueError(gen)
        validate(p)
        out.append(p)
    return out


def dataset_seed(gen: str, r: int) -> int:
    return 5000 + 1000 * GENS.index(gen) + r
