# -*- coding: utf-8 -*-
"""预注册之外的探索性基线：二阶对数线性零模型与循环时移零模型。

  venv/bin/python -m score_learning.induction.extra_baselines [n_half] [set ...]   # 每集写一个 JSON

对数线性：同签名格上与真实计数全部两两位边缘一致的 IPF 期望（三阶以上交互才算偏差）。
循环时移：零模型计数 = 同一首留出曲把 A/T/B 各自随机循环时移 R 次后的真实计数，
  即保留每个声部全部旋律、只打乱同时性的数据随机化（swap randomisation 一类）；
  旋律内场（M、G、D）在时移下不变，这些场上它没有信号。
五个评测集、训练集与 final_eval.json 相同；方法共享同一组共同原子。
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

from score_learning.induction import bootstrap as bs
from score_learning.induction import fri
from score_learning.induction.ablation import uniform_fields
from score_learning.induction.baselines import (EVAL_ATOMS, EVAL_FIELDS, atom_auc, indep_fields, loglin_fields,
                                                precision_at_k, productive_fields)
from score_learning.induction.fresh_eval import SEEDS, fresh_ids, load_palestrina_fresh
from score_learning.induction.paper import _desync_piece, _git_head, load_chorale_pool, load_palestrina

OUT = 'runs/induction/paper/extra_baselines.json'
N_SHIFT = 16
TRANSFORMS = {'ras': lambda f: f, 'pmi': indep_fields, 'rarity': uniform_fields,
              'productive': productive_fields, 'loglin': loglin_fields}
METHODS = list(TRANSFORMS) + ['shift']


def shift_piece_counts(hold, seed: int, k: int = N_SHIFT) -> List[Dict[str, Tuple[Counter, Counter]]]:
    rng = random.Random(seed)
    out = []
    for p in hold:
        copies = [_desync_piece(p, rng) for _ in range(k)]
        row = {}
        for f in fri.ALL_FIELDS:
            real_f = getattr(fri, f'real_{f.lower()}_counts')
            null = Counter()
            for q in copies:
                null.update(real_f([q]))
            row[f] = (real_f([p]), null)
        out.append(row)
    return out


def stats(per, per_d, sh, sh_d, idx) -> Dict[str, Dict[str, float]]:
    f, fd = bs.fields_from(per, idx), bs.fields_from(per_d, idx)
    tf = {m: t(f) for m, t in TRANSFORMS.items()}
    tfd = {m: t(fd) for m, t in TRANSFORMS.items()}
    tf['shift'], tfd['shift'] = bs.fields_from(sh, idx), bs.fields_from(sh_d, idx)
    auc = {m: atom_auc(x) for m, x in tf.items()}
    auc_d = {m: atom_auc(x) for m, x in tfd.items()}
    ok = lambda d: {a for a, x in d.items() if x is not None}
    common = EVAL_ATOMS & set.intersection(*[ok(v) for v in auc.values()])
    vert = common & bs._VERT & set.intersection(*[ok(v) for v in auc_d.values()])
    under = {a for a in common if bs._POL[a] == 'under'}
    out = {}
    for m in METHODS:
        pk = precision_at_k(tf[m], flds=EVAL_FIELDS)
        out[m] = {'auc_common': bs._mean(auc[m], common), 'auc_under': bs._mean(auc[m], under),
                  'auc_vert': bs._mean(auc[m], vert),
                  'vert_gap': bs._mean(auc[m], vert) - bs._mean(auc_d[m], vert),
                  'p50_under': pk['under'][50], 'p50_over': pk['over'][50]}
    out['_n_common'], out['_n_vert'] = len(common), len(vert)
    out['_atoms'] = {m: {a: auc[m][a] for a in sorted(common)} for m in METHODS}
    return out


def main():
    n_half = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    only = set(sys.argv[2:])
    with open(bs.FINAL) as fh:
        fin = json.load(fh)
    by, usable, _ = load_chorale_pool()
    tr = [by[i] for i in fin['train_ids']]
    sets = {
        'hold_a': ([by[i] for i in fin['hold_ids']], 0),
        'hold_b': ([by[i] for i in fin['hold_b_ids']], 1),
        'palestrina': (load_palestrina(16), 0),
        'hold_c': ([by[i] for i in fresh_ids(fin, usable)], SEEDS['hold_c']),
        'palestrina2': (load_palestrina_fresh(), SEEDS['palestrina2']),
    }
    out = {'commit': _git_head(), 'final_commit': fin['commit'], 'n_half': n_half, 'n_shift': N_SHIFT,
           'status': 'exploratory (not preregistered)', 'sets': {}}
    for si, (name, (hold, seed)) in enumerate(sets.items()):
        if only and name not in only:
            continue
        t0 = time.time()
        rng_d = random.Random(99)
        hold_d = [_desync_piece(p, rng_d) for p in hold]
        per, per_d = bs.piece_counts(tr, hold, seed), bs.piece_counts(tr, hold_d, seed)
        sh, sh_d = shift_piece_counts(hold, 500 + si), shift_piece_counts(hold_d, 600 + si)
        full = list(range(len(hold)))
        point = stats(per, per_d, sh, sh_d, full)
        rng = random.Random(2026)
        boot = []
        for b in range(n_half):
            boot.append(stats(per, per_d, sh, sh_d, rng.sample(full, len(hold) // 2)))
            if (b + 1) % 50 == 0:
                print(f'  {name} half {b + 1} {time.time() - t0:.0f}s', flush=True)
        summary = {}
        for k in ('auc_common', 'auc_under', 'auc_vert', 'vert_gap', 'p50_under', 'p50_over'):
            s = {m: point[m][k] for m in METHODS}
            for base in METHODS[1:]:
                s[f'diff_{base}'] = point['ras'][k] - point[base][k]
                if n_half and k in ('auc_common', 'auc_under', 'vert_gap'):
                    d = [x['ras'][k] - x[base][k] for x in boot]
                    s[f'diff_{base}_ci'] = bs._ci(d)
            summary[k] = s
            print(f'  {name:<12}{k:<11}' + ' '.join(f'{m} {s[m]:.3f}' for m in METHODS) + (
                '  ' + ' '.join(f'-{b[:4]} {s["diff_" + b]:+.3f} [{s["diff_" + b + "_ci"][0]:+.3f},'
                                f'{s["diff_" + b + "_ci"][1]:+.3f}]' for b in METHODS[1:])
                if f'diff_{METHODS[1]}_ci' in s else ''), flush=True)
        out['sets'][name] = {'n': len(hold), 'n_common': point['_n_common'], 'n_vert': point['_n_vert'],
                             'summary': summary, 'atoms': point['_atoms']}
        print(f'== {name} n={len(hold)} common {point["_n_common"]} vert {point["_n_vert"]} '
              f'{time.time() - t0:.0f}s', flush=True)
        path = OUT.replace('.json', f'_{name}.json')
        with open(path, 'w') as fh:
            json.dump({**out, 'sets': {name: out['sets'][name]}}, fh, indent=1, default=str)
        print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
