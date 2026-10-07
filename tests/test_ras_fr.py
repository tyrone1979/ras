# -*- coding: utf-8 -*-
from score_learning.induction.fri import v_cell, h_cell, g_cell, residual
from collections import Counter


def test_v_cell_interval_shift_invariant():
    c0 = v_cell(2, 2, 7, 7, ('S', 'A'))
    # 全体 +5：差与音程不变
    c1 = v_cell(2, 2, 7, 7, ('S', 'A'))
    assert c0 == c1
    assert 'eqΔ=1' in c0 and 'pc07=1' in c0 and 'hold=1' in c0


def test_h_gap_uses_difference_not_absolute():
    assert h_cell('S', 'A', 72, 60) == h_cell('S', 'A', 77, 65)


def test_g_cell_octave():
    assert g_cell(12) == 'oct+'
    assert g_cell(-12) == 'oct+'
    assert g_cell(6) == 'tritone'


def test_residual_underdense_when_null_has_mass():
    real = Counter({'a': 0, 'b': 10})
    null = Counter({'a': 100, 'b': 10})
    rows = residual(real, null)
    by = {k: r for k, r, *_ in rows}
    assert by['a'] < by['b']
    assert by['a'] < -1.0
