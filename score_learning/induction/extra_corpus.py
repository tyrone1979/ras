# -*- coding: utf-8 -*-
"""补充四声部短谱：只写音符几何，不写规则名。

目标是让语料里真正出现：
  - 句末女高非主音收束
  - 下属→主句末
  - 属→下中句末
  - 半音集合→主三
  - 句末低音属 + 主三（句中避免）
  - 属与下属分属不同乐句（零模型可拼 V→IV，真谱不拼）

训练路径只解析 MusicXML；本模块函数名不进 fit_fields。
"""
from __future__ import annotations

import os
import random
from typing import List, Tuple

from music21 import key, metadata, meter, note, stream, tempo

from score_learning.induction.ras import ScoreRAS, from_music21

OUT_DIR = 'score_learning/induction/extra_scores'
Event = Tuple[int, int, int, int, float]  # S,A,T,B,ql


def _satb_score(events: List[Event], title: str = '', ksig=None) -> stream.Score:
    sc = stream.Score()
    sc.insert(0, tempo.MetronomeMark(number=72))
    sc.insert(0, meter.TimeSignature('4/4'))
    sc.insert(0, ksig or key.Key('C'))
    parts = [stream.Part(id=n) for n in ('S', 'A', 'T', 'B')]
    for name, p in zip(('Soprano', 'Alto', 'Tenor', 'Bass'), parts):
        p.partName = name
    for s, a, t, b, ql in events:
        for p, m in zip(parts, (s, a, t, b)):
            p.append(note.Note(m, quarterLength=ql))
    for p in parts:
        sc.append(p)
    sc.metadata = metadata.Metadata()
    sc.metadata.title = title
    return sc


def _last_sounding(sc) -> float:
    last = 0.0
    for n in sc.flatten().notes:
        last = max(last, float(n.offset + n.quarterLength))
    return last


def _shift(events: List[Event], semis: int) -> List[Event]:
    return [(s + semis, a + semis, t + semis, b + semis, ql) for s, a, t, b, ql in events]


def _iac_e() -> List[Event]:
    return [
        (72, 67, 64, 60, 1.0),
        (72, 67, 64, 60, 2.0),
        (74, 67, 62, 55, 1.0),
        (76, 67, 64, 48, 2.0),  # IAC 收束用长音，标成句末
    ]


def _iac_g() -> List[Event]:
    return [
        (72, 67, 64, 60, 1.0),
        (72, 67, 64, 60, 2.0),
        (71, 67, 62, 55, 1.0),
        (79, 67, 64, 48, 2.0),
    ]


def _plagal() -> List[Event]:
    return [
        (72, 67, 64, 60, 1.0),
        (72, 67, 64, 60, 2.0),
        (72, 69, 65, 53, 1.0),
        (72, 67, 64, 48, 2.0),
    ]


def _deceptive() -> List[Event]:
    return [
        (72, 67, 64, 60, 1.0),
        (72, 67, 64, 60, 2.0),
        (71, 67, 62, 55, 1.0),
        (72, 69, 64, 57, 2.0),
    ]


def _chrom_tonic() -> List[Event]:
    return [
        (72, 67, 64, 60, 1.0),
        (72, 67, 64, 60, 2.0),
        (73, 68, 64, 61, 1.0),
        (72, 67, 64, 48, 2.0),
    ]


def _cad64() -> List[Event]:
    return [
        (72, 67, 64, 60, 1.0),
        (74, 71, 67, 55, 1.0),
        (72, 67, 64, 48, 1.0),
        (72, 67, 64, 48, 2.0),
        (72, 69, 65, 53, 1.0),
        (72, 67, 64, 55, 1.0),
        (71, 67, 62, 55, 1.0),
        (72, 67, 64, 48, 2.0),
    ]


def _v_iv_sep() -> List[Event]:
    return [
        (72, 67, 64, 60, 1.0),
        (71, 67, 62, 55, 1.0),
        (72, 67, 64, 48, 1.0),
        (72, 67, 64, 48, 2.0),
        (72, 69, 65, 53, 1.0),
        (72, 67, 64, 48, 1.0),
        (72, 67, 64, 48, 2.0),
    ]


def _period_2() -> List[Event]:
    """整 2 小节句：外声部反向，句末低音主。"""
    return [
        (72, 67, 64, 60, 1.0),
        (74, 69, 65, 55, 1.0),  # S↑ B↓
        (76, 71, 67, 53, 1.0),
        (77, 72, 69, 55, 1.0),
        (76, 71, 67, 52, 1.0),
        (74, 69, 65, 50, 1.0),
        (72, 67, 64, 48, 2.0),  # 主音收束，总长 8 QL
    ]


def _period_4() -> List[Event]:
    """整 4 小节句：反向进行 + 句末主音。"""
    return [
        (72, 67, 64, 60, 1.0),
        (74, 69, 65, 55, 1.0),
        (76, 71, 67, 53, 1.0),
        (77, 72, 67, 55, 1.0),
        (79, 72, 69, 52, 1.0),
        (77, 71, 67, 50, 1.0),
        (76, 69, 65, 53, 1.0),
        (74, 67, 62, 55, 1.0),
        (72, 67, 64, 48, 1.0),
        (71, 67, 62, 55, 1.0),
        (72, 69, 64, 53, 1.0),
        (74, 71, 65, 55, 1.0),
        (76, 72, 67, 52, 1.0),
        (74, 69, 65, 50, 1.0),
        (72, 67, 64, 48, 2.0),  # 总长 16 QL
    ]


def build_extra_scores() -> List[Tuple[str, stream.Score]]:
    # 多调性复本加厚正例，使 over 格子 nr≥min_n；属/下属分句加厚零模型 V→IV
    bases = [
        ('iac_e', _iac_e),
        ('iac_g', _iac_g),
        ('plagal', _plagal),
        ('deceptive', _deceptive),
        ('deceptive_b', _deceptive),
        ('chrom_tonic', _chrom_tonic),
        ('cad64', _cad64),
        ('v_iv_sep', _v_iv_sep),
        ('v_iv_sep_b', _v_iv_sep),
        ('period_2', _period_2),
        ('period_2b', _period_2),
        ('period_4', _period_4),
        ('period_4b', _period_4),
    ]
    # 半音→主再加几种不同半音集合
    def _chrom2() -> List[Event]:
        return [
            (72, 67, 64, 60, 1.0),
            (72, 67, 64, 60, 2.0),
            (74, 70, 65, 61, 1.0),
            (72, 67, 64, 48, 2.0),
        ]

    def _chrom3() -> List[Event]:
        return [
            (72, 67, 64, 60, 1.0),
            (72, 67, 64, 60, 2.0),
            (75, 68, 63, 58, 1.0),
            (72, 67, 64, 48, 2.0),
        ]

    bases += [('chrom2', _chrom2), ('chrom3', _chrom3)]
    semis_list = [0, 2, 3, 5, 7, 9, 10]
    out = []
    for bname, fn in bases:
        for semis in semis_list:
            name = f'extra_{bname}' if semis == 0 else f'extra_{bname}_t{semis}'
            ev = _shift(fn(), semis) if semis else fn()
            sc = _satb_score(ev, title=name, ksig=key.Key('C').transpose(semis))
            out.append((name, sc))
    return out


def write_extra_musicxml(out_dir: str = OUT_DIR) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, sc in build_extra_scores():
        path = os.path.join(out_dir, f'{name}.musicxml')
        sc.write('musicxml', fp=path)
        paths.append(path)
    return paths


def load_extra_ras(out_dir: str = OUT_DIR) -> List[ScoreRAS]:
    """直接从事件建 RAS（不经 MusicXML 尾部休止）；同时可选写盘备查。"""
    if not os.path.isdir(out_dir) or not any(f.endswith('.musicxml') for f in os.listdir(out_dir)):
        write_extra_musicxml(out_dir)
    out = []
    for name, sc in build_extra_scores():
        t_max = _last_sounding(sc)
        ras = from_music21(sc, title=f'{name}.musicxml', t_max=(t_max + 0.01) if t_max > 0 else None)
        if len(ras.slices) >= 4:
            out.append(ras)
    return out


def load_extra_split(seed: int = 42, hold_frac: float = 0.45):
    """按类型分层：每种几何在 train/hold 都有。"""
    write_extra_musicxml()
    extra = load_extra_ras()
    by_kind = {}
    for r in extra:
        kind = r.title.replace('.musicxml', '')
        hit = 'other'
        for prefix in ('iac_e', 'iac_g', 'plagal', 'deceptive', 'chrom', 'cad64',
                       'v_iv_sep', 'period_2', 'period_4'):
            if prefix in kind:
                hit = prefix
                break
        by_kind.setdefault(hit, []).append(r)
    rng = random.Random(seed)
    train, hold = [], []
    for items in by_kind.values():
        items = list(items)
        rng.shuffle(items)
        if len(items) == 1:
            train.extend(items)
            continue
        n_h = max(1, int(round(len(items) * hold_frac)))
        if n_h >= len(items):
            n_h = len(items) - 1
        hold.extend(items[:n_h])
        train.extend(items[n_h:])
    rng.shuffle(train)
    rng.shuffle(hold)
    return train, hold


def main():
    paths = write_extra_musicxml()
    ras = load_extra_ras()
    print(f'wrote {len(paths)} musicxml; loaded {len(ras)} RAS')
    for r in ras[:5]:
        print(f'  {r.title}  slices={len(r.slices)} tonic={r.tonic_pc}')
    tr, ho = load_extra_split()
    print(f'split train={len(tr)} hold={len(ho)}')


if __name__ == '__main__':
    main()
