# -*- coding: utf-8 -*-
import inspect

from score_learning.induction import learn, ras
from score_learning.induction.fri import (
    compute_fields, fit_fields, geom_recovery, holdout_fields, nopair_fields,
)
from score_learning.induction.oracle import main_predicates


def test_oracle_not_imported_by_train_path():
    assert 'oracle' not in inspect.getsource(fit_fields)
    assert 'oracle' not in inspect.getsource(compute_fields)
    assert 'oracle' not in inspect.getsource(holdout_fields)
    assert 'oracle' not in inspect.getsource(learn)
    assert 'oracle' not in inspect.getsource(ras)


def test_train_source_has_no_rule_names():
    src = inspect.getsource(compute_fields) + inspect.getsource(fit_fields)
    for banned in ('PAC', 'IAC', '平行五度', 'B1', 'C1', 'D1', 'roman', 'V7', '六四'):
        assert banned not in src


def test_main_ontology_covers_a_to_f():
    names = [n for n, *_ in main_predicates()]
    assert names[:13] == [
        'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
        'A11', 'A12', 'A13',
    ]
    for prefix, n in (('B', 6), ('C', 8), ('D', 5), ('E', 3), ('F', 3)):
        got = [x for x in names if x.startswith(prefix)]
        assert len(got) == n, (prefix, got)


def test_nopair_drops_a1_on_chorales():
    pieces = learn.load_many([1, 2, 3, 4])
    fields = fit_fields(pieces, seed=0)
    fields.pop('_dp', None)
    fields.pop('_pitch', None)
    _, intact = geom_recovery(fields)
    _, dropped = geom_recovery(nopair_fields(fields))
    assert intact['hits']['A1'] is True
    assert dropped['hits']['A1'] is False
    assert dropped['hits']['A8'] is True
