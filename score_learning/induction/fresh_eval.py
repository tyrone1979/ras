# -*- coding: utf-8 -*-
"""预注册的一次性评测：冻结方法在从未用作评测对象的新测试集上（papers/rasfr/preregistration.md）。

  venv/bin/python -m score_learning.induction.fresh_eval

新测试集：
  hold_c       可用众赞歌中不属于训练 / 留出 A / 留出 B / 开发集、且编号 > 60 的全部曲目
               （编号 1--60 曾被 RAS-FR 之前的早期脚本加载过，保守排除）；
  palestrina2  music21 Palestrina 四声部作品中跳过已用的前 16 首后的接下来 40 首。
训练集、校准配置与 final_eval.json 完全相同（cfg 直接读取，不重新校准）。
方法代码有未提交改动时拒绝运行。
"""
from __future__ import annotations

import json
import random
import sys
import time

from score_learning.induction import bootstrap as bs
from score_learning.induction.baselines import (
    EVAL_ATOMS, atom_auc, indep_fields, metrics, productive_fields,
)
from score_learning.induction.ablation import uniform_fields
from score_learning.induction.paper import _desync_piece, _git_head, from_music21, load_chorale_pool

OUT = 'runs/induction/paper/fresh_eval.json'
EARLY_MAX_ID = 60
N_PAL2 = 40
METHODS = {'ras': lambda f: f, 'pmi': indep_fields, 'rarity': uniform_fields,
           'productive': productive_fields}
SEEDS = {'hold_c': 2, 'palestrina2': 3}


def fresh_ids(fin: dict, usable) -> list:
    train_pool = usable[:-40]
    rest = [i for i in train_pool if i not in set(fin['train_ids']) and i not in set(fin['hold_b_ids'])]
    dev = set(random.Random(11).sample(rest, 40))
    used = set(fin['train_ids']) | set(fin['hold_ids']) | set(fin['hold_b_ids']) | dev
    return [i for i in usable if i not in used and i > EARLY_MAX_ID]


def load_palestrina_fresh(skip: int = 16, limit: int = N_PAL2):
    """与 load_palestrina 相同的筛选，跳过前 skip 首合格作品（即已用的 16 首）。"""
    from music21 import corpus
    out, ok = [], 0
    for p in corpus.getComposer('palestrina'):
        if len(out) >= limit:
            break
        try:
            sc = corpus.parse(p)
            if len(sc.parts) != 4:
                continue
            ras = from_music21(sc, title=str(p), t_max=None)
            if len(ras.slices) < 8:
                continue
        except Exception:
            continue
        ok += 1
        if ok > skip:
            out.append(ras)
    return out


def main():
    commit = _git_head()
    if commit.endswith('-dirty') and '--allow-dirty' not in sys.argv:
        sys.exit(f'method code is dirty ({commit}); commit before the preregistered evaluation')
    n_half = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000
    with open(bs.FINAL) as fh:
        fin = json.load(fh)
    by, usable, _ = load_chorale_pool()
    tr = [by[i] for i in fin['train_ids']]
    c_ids = fresh_ids(fin, usable)
    out_path = OUT
    if '--smoke' in sys.argv:
        sets = {'hold_c': [by[i] for i in fin['hold_ids'][:12]]}
        n_half, out_path = 3, '/tmp/fresh_eval_smoke.json'
    else:
        sets = {'hold_c': [by[i] for i in c_ids], 'palestrina2': load_palestrina_fresh()}
    out = {'commit': commit, 'final_commit': fin['commit'], 'hold_c_ids': c_ids,
           'palestrina2_titles': [p.title for p in sets.get('palestrina2', [])], 'n_half': n_half,
           'sets': {}}
    for name, hold in sets.items():
        t0 = time.time()
        seed = SEEDS[name]
        print(f'== {name} n={len(hold)}', flush=True)
        rng_d = random.Random(99)
        per = bs.piece_counts(tr, hold, seed)
        per_d = bs.piece_counts(tr, [_desync_piece(p, rng_d) for p in hold], seed)
        full = list(range(len(hold)))
        f, fd = bs.fields_from(per, full), bs.fields_from(per_d, full)
        point = bs.draw_stats(f, fd, METHODS)
        atoms = {m: {'real': atom_auc(t(f)), 'desync': atom_auc(t(fd))} for m, t in METHODS.items()}
        ras_m = metrics(f, fin['cfg'])
        rng = random.Random(2026)
        boot = []
        for b in range(n_half):
            idx = rng.sample(full, len(hold) // 2)
            boot.append(bs.draw_stats(bs.fields_from(per, idx), bs.fields_from(per_d, idx), METHODS))
            if (b + 1) % 100 == 0:
                print(f'  {name} half {b + 1} {time.time() - t0:.0f}s', flush=True)
        summary = {}
        for k in point['ras']:
            s = {m: point[m][k] for m in METHODS}
            for base in [m for m in METHODS if m != 'ras']:
                s[f'diff_{base}'] = point['ras'][k] - point[base][k]
            if k in bs.CI_KEYS:
                s.update({f'{m}_ci': bs._ci([x[m][k] for x in boot]) for m in METHODS})
                for base in [m for m in METHODS if m != 'ras']:
                    d = [x['ras'][k] - x[base][k] for x in boot]
                    s[f'diff_{base}_ci'] = bs._ci(d)
                    s[f'p_le0_{base}'] = sum(v <= 0 for v in d) / len(d)
            summary[k] = s
            line = f'  {name:<12}{k:<11}' + ' '.join(f'{m} {s[m]:.3f}' for m in METHODS)
            if k in bs.CI_KEYS:
                line += '  ' + ' '.join(f'-{b[:3]} {s["diff_" + b]:+.3f} [{s["diff_" + b + "_ci"][0]:+.3f},'
                                        f'{s["diff_" + b + "_ci"][1]:+.3f}]' for b in METHODS if b != 'ras')
            print(line, flush=True)
        ev = sorted(a for a in EVAL_ATOMS if all(atoms[m]['real'].get(a) is not None for m in METHODS))
        out['sets'][name] = {
            'n': len(hold), 'summary': summary, 'atoms': atoms, 'eval_common': ev,
            'ras_discovery': {'completeness': ras_m['completeness'], 'miss': ras_m['miss'],
                              'under': {k: ras_m['enrich']['under'][k] for k in ('n_disc', 'precision', 'base', 'lift')},
                              'over': {k: ras_m['enrich']['over'][k] for k in ('n_disc', 'precision', 'base', 'lift')}},
        }
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=1, default=str)
    print('wrote', out_path, flush=True)


if __name__ == '__main__':
    main()
