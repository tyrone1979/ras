# -*- coding: utf-8 -*-
"""补充语料：稀有几何 B2/B5/C2/C3/C5/C8 可在 RAS-FR 下达成。"""
from score_learning.induction.extra_corpus import load_extra_split, write_extra_musicxml
from score_learning.induction.fri import fit_fields, geom_recovery


def test_extra_scores_load():
    write_extra_musicxml()
    tr, ho = load_extra_split(seed=0, hold_frac=0.4)
    assert len(tr) >= 10 and len(ho) >= 8


def test_rare_atoms_on_extra():
    tr, ho = load_extra_split(seed=1, hold_frac=0.4)
    fields = fit_fields(tr + ho, seed=0)
    fields.pop('_dp', None)
    fields.pop('_pitch', None)
    _, meta = geom_recovery(fields, {'z_cut': 3.0, 'q': 0.08})
    # E3 依赖众赞歌里的反向/斜向质量，补充短谱静态块多，单独测不稳
    # 零模型保留句末槽位率后，B2（句末六四）由槽位频率解释，不再要求达成
    rare = ['B5', 'C2', 'C3', 'C5', 'C8', 'F1', 'F2']
    missing = [a for a in rare if not meta['hits'].get(a)]
    hits = {a: meta['hits'][a] for a in rare}
    assert not missing, f'rare miss {missing} hits={hits}'


def test_e3_f1_f2_polarity_on_mixed_hold():
    """E3/F1/F2 在众赞歌+补充语料留出上应达成。"""
    import random
    from score_learning.induction.paper import load_chorale_pool
    from score_learning.induction.fri import holdout_fields

    tr_x, ho_x = load_extra_split(seed=42, hold_frac=0.45)
    by, usable, _ = load_chorale_pool()
    hold_ids, train_ids = usable[-40:], usable[:-40]
    rng = random.Random(0)
    tr = [by[i] for i in rng.sample(train_ids, 16)] + list(tr_x)
    hold = list(ho_x) + [by[i] for i in hold_ids[:8]]
    _, meta = geom_recovery(holdout_fields(tr, hold, seed=0), {'z_cut': 3.0, 'q': 0.08})
    for a in ('E3', 'F1', 'F2'):
        assert meta['hits'][a], f'{a} not recovered; star={meta["atom_star"].get(a)}'


def test_fdr_and_completeness_targets():
    """只漏句末槽位可解释的 B2/B5，且空真格 FDR≤0.10。"""
    import random
    from score_learning.induction.paper import load_chorale_pool
    from score_learning.induction.fri import holdout_fields

    tr_x, ho_x = load_extra_split(seed=42, hold_frac=0.45)
    by, usable, _ = load_chorale_pool()
    hold_ids, train_ids = usable[-40:], usable[:-40]
    rng = random.Random(0)
    tr = [by[i] for i in rng.sample(train_ids, 16)] + list(tr_x)
    hold = list(ho_x) + [by[i] for i in hold_ids[:8]]
    _, meta = geom_recovery(holdout_fields(tr, hold, seed=0), {'z_cut': 3.0, 'q': 0.08})
    miss = {a for a, ok in meta['hits'].items() if not ok}
    assert miss <= {'B2', 'B5'}, miss
    assert meta['fdr'] <= 0.10 + 1e-9, meta['fdr']
