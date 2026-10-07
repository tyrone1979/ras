# -*- coding: utf-8 -*-
"""冻结教科书几何（只评测）。训练路径不得 import 本模块。

学习算法只见乐谱几何格子；下列名字、罗马数字、终止类型从不进入
fit_fields / compute_fields。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple


def parse_cell(cell: str) -> Dict[str, str]:
    out = {}
    body = cell
    if '|slot=' in body and '/to/' in body:
        body, slot = body.rsplit('|slot=', 1)
        out['slot'] = slot
    for part in body.split('|'):
        if '/to/' in part:
            continue
        if '=' in part:
            k, v = part.split('=', 1)
            out[k] = v
        else:
            out['pair'] = part
    return out


def _split_to(cell: str):
    body = cell
    slot = None
    if '|slot=' in body:
        body, slot = body.rsplit('|slot=', 1)
    if '/to/' not in body:
        return None, None, slot
    a, b = body.split('/to/', 1)
    return a, b, slot


def _on(fld_need, fn):
    def wrap(cell, field):
        if field != fld_need:
            return False
        return fn(parse_cell(cell), cell)
    return wrap


def _on_to(fld_need, fn):
    def wrap(cell, field):
        if field != fld_need:
            return False
        a, b, slot = _split_to(cell)
        if a is None:
            return False
        return fn(parse_cell(a), parse_cell(b), slot, cell)
    return wrap


def _on_q(fn):
    return _on_to('Q', fn)


def _on_k(fn):
    return _on_to('K', fn)


# 只出现在本文件：调内三音集合（相对主音 pc）
_TRIAD = {
    '0.4.7', '0.3.7',
    '2.5.9', '2.5.8',
    '4.7.11', '3.7.10',
    '0.5.9', '0.5.8',
    '2.7.11', '2.7.10',
    '0.4.9', '0.3.8',
    '2.5.11', '2.5.10',
}
_DOM7 = {'2.5.7.11', '2.5.7.10', '2.7.11', '5.7.11', '2.5.11'}


def main_predicates() -> List[Tuple[str, str, str, Callable]]:
    return [
        ('A1', 'V', 'under', _on('V', lambda d, c: d.get('both') == '1' and d.get('eqΔ') == '1'
                                 and d.get('pc07') == '1' and d.get('hold') == '1'
                                 and d.get('sgn') == 'same')),
        ('A2', 'V', 'under', _on('V', lambda d, c: d.get('role') == 'outer' and d.get('both') == '1'
                                 and d.get('sgn') == 'same' and d.get('pc07') == '1'
                                 and d.get('leapU') == '1')),
        ('A3', 'H', 'under', _on('H', lambda d, c: d.get('gap') == 'cross')),
        ('A4', 'O', 'under', _on('O', lambda d, c: d.get('ov') == '1')),
        ('A5', 'H', 'under', _on('H', lambda d, c: (
            (d.get('pair') == 'SA' and d.get('gap') in ('12to19', 'gt19'))
            or (d.get('pair') == 'AT' and d.get('gap') in ('12to19', 'gt19'))
            or (d.get('pair') == 'TB' and d.get('gap') == 'gt19')))),
        ('A6', 'M', 'under', _on('M', lambda d, c: d.get('lt') == '1' and d.get('out') == '1'
                                 and d.get('mv') != '+1')),
        ('A7', 'M', 'under', _on('M', lambda d, c: d.get('b7') == '1' and d.get('mv') not in ('-1', '-2'))),
        ('A8', 'M', 'under', _on('M', lambda d, c: d.get('mv') == 'tritone')),
        ('A9', 'G', 'under', _on('G', lambda d, c: c == '7th')),
        ('A10', 'G', 'under', _on('G', lambda d, c: c == 'oct+')),
        ('A11', 'S', 'under', _on('S', lambda d, c: d.get('nlt') == '2')),
        ('A12', 'S', 'under', _on('S', lambda d, c: d.get('miss3') == '1')),
        ('A13', 'S', 'under', _on('S', lambda d, c: d.get('n7') != '0' and d.get('missTon') == '1')),

        ('B1', 'K', 'over', _on_k(lambda a, b, s, c: a.get('set') in _TRIAD or b.get('set') in _TRIAD)),
        ('B2', 'J', 'under', _on_to('J', lambda a, b, s, c: s == 'int' and (
            (a.get('set') in ('0.4.7', '0.3.7') and a.get('b') == '7')
            or (b.get('set') in ('0.4.7', '0.3.7') and b.get('b') == '7')))),
        ('B3', 'K', 'over', _on_k(lambda a, b, s, c: a.get('set') in _DOM7 or b.get('set') in _DOM7)),
        ('B4', 'K', 'over', _on_k(lambda a, b, s, c: a.get('ch') == '1' and b.get('ch') == '0'
                                  and a.get('b') not in ('0', '7'))),
        ('B5', 'K', 'under', _on_k(lambda a, b, s, c: a.get('b') == '7' and b.get('b') == '5')),
        ('B6', 'K', 'over', _on_k(lambda a, b, s, c: (
            (a.get('b') == '0' and b.get('b') == '5')
            or (a.get('b') == '5' and b.get('b') == '7')
            or (a.get('b') == '7' and b.get('b') == '0')))),

        ('C1', 'Q', 'over', _on_q(lambda a, b, s, c: s == 'fin' and a.get('b') == '7' and b.get('b') == '0'
                                  and b.get('sop') == '0')),
        ('C2', 'Q', 'over', _on_q(lambda a, b, s, c: s == 'fin' and b.get('b') == '0'
                                  and b.get('sop') in ('4', '7'))),
        ('C3', 'Q', 'over', _on_q(lambda a, b, s, c: a.get('b') == '5' and b.get('b') == '0'
                                  and b.get('sop') == '0')),
        ('C4', 'Q', 'over', _on_q(lambda a, b, s, c: b.get('b') == '7')),
        ('C5', 'Q', 'over', _on_q(lambda a, b, s, c: a.get('b') == '7' and b.get('b') == '9')),
        ('C6', 'K', 'over', _on_k(lambda a, b, s, c: a.get('ch') == '1' and b.get('set') in _TRIAD
                                  and a.get('b') not in ('7',) and b.get('ch') == '0')),
        ('C7', 'Q', 'over', _on_q(lambda a, b, s, c: s == 'int' and b.get('b') == '7')),
        ('C8', 'K', 'over', _on_k(lambda a, b, s, c: a.get('ch') == '1' and b.get('b') == '0'
                                  and b.get('set') in ('0.4.7', '0.3.7'))),

        ('D1', 'D', 'over', _on('D', lambda d, c: d.get('st01') == '1' and d.get('st12') == '1'
                                and d.get('same') == '1' and d.get('inn') == '0')),
        ('D2', 'D', 'over', _on('D', lambda d, c: d.get('st01') == '1' and d.get('st12') == '1'
                                and d.get('ret') == '1' and d.get('inn') == '0')),
        ('D3', 'D', 'over', _on('D', lambda d, c: d.get('hov') == '1' and d.get('inn') == '0'
                                and d.get('st12') == '1')),
        ('D4', 'D', 'over', _on('D', lambda d, c: d.get('leap') == '1' and d.get('st12') == '1'
                                and d.get('inn') == '0')),
        ('D5', 'D', 'over', _on('D', lambda d, c: d.get('ant') == '1')),
        # 无 D6（持续音）：ped 是整曲声部属性而非音级局部，零模型保留它，无法检验。

        ('E1', 'R', 'under', _on('R', lambda d, c: (
            (d.get('v') == 'S' and d.get('oct') not in ('5', '6'))
            or (d.get('v') == 'A' and d.get('oct') not in ('4', '5', '6'))
            or (d.get('v') == 'T' and d.get('oct') not in ('4', '5'))
            or (d.get('v') == 'B' and d.get('oct') not in ('3', '4', '5'))))),
        ('E2', 'U', 'over', _on('U', lambda d, c: d.get('dbl') == '0')),
        ('E3', 'W', 'over', _on('W', lambda d, c: d.get('mot') in ('contrary', 'oblique'))),

        ('F1', 'F', 'over', _on('F', lambda d, c: d.get('kind') == 'L' and d.get('bars') in ('2', '4'))),
        ('F2', 'F', 'over', _on('F', lambda d, c: d.get('kind') == 'S' and d.get('slot') == 'fin'
                                and d.get('b') == '0')),
        ('F3', 'F', 'over', _on('F', lambda d, c: d.get('kind') == 'S' and d.get('slot') == 'int'
                                and d.get('b') == '7')),
    ]


def appendix_predicates() -> List[Tuple[str, str, str, Callable]]:
    return []
