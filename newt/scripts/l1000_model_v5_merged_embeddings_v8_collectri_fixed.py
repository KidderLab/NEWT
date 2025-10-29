#!/usr/bin/env python3
"""
L1000 Model V5 with merged embeddings and collectri fix.
Memory-optimized inference with explicit GC and batch control.
"""

import os
import glob
import csv
import argparse
import gc

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras import backend as K

from itertools import chain, combinations
from utils import parallel

# Global holders
CURRENT_EMBEDDINGS = None
pert_sig = None
cpd2target = None
sigvec_all = None
perttype = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train & evaluate L1000 model with multiple embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--cpdlist_file', required=True)
    parser.add_argument('--target_file', required=True)
    parser.add_argument('--sig_file', required=True)
    parser.add_argument('--perttype', required=True, choices=['cDNA', 'shRNA'])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--emb_go', required=True)
    parser.add_argument('--emb_archs4', required=True)
    parser.add_argument('--emb_ppi', default=None)
    parser.add_argument('--emb_msigdb', default=None)
    parser.add_argument('--emb_cellnet', default=None)
    parser.add_argument('--emb_collectri', default=None)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--modeldir', required=True)
    parser.add_argument('--n_cpu', type=int, default=1)
    parser.add_argument('--predict_batch_size', type=int, default=1024,
                        help='Batch size for inference predictions')
    return parser.parse_args()


def get_model(fp_dim, hidden_dim=2048):
    left = keras.Input(shape=(fp_dim,))
    right = keras.Input(shape=(fp_dim,))
    shared = keras.Sequential([
        layers.Dropout(0.25),
        layers.Dense(hidden_dim),
        layers.BatchNormalization(),
        layers.ReLU()
    ])
    out_l = shared(left)
    out_r = shared(right)
    merged = layers.Multiply()([out_l, out_r])
    classifier = keras.Sequential([
        layers.Dense(hidden_dim // 4),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(1)
    ])
    return keras.Model([left, right], classifier(merged))


def compute_weight(matrix):
    sim = cosine_similarity(matrix, matrix)
    sim[np.isnan(sim)] = 0
    mean = sim.mean(axis=1)
    std = sim.std(axis=1)
    weights = (mean - mean.mean()) / np.where(std > 0, std, 1e-9)
    return np.clip(weights, 0, 1).reshape(-1, 1)


def compute_list_emb(gene_list):
    global CURRENT_EMBEDDINGS
    mats, weights = [], []
    for emb in CURRENT_EMBEDDINGS.values():
        dim = next(iter(emb.values())).shape[0]
        mat = np.vstack([emb.get(g, np.zeros(dim)) for g in gene_list])
        mats.append(normalize(mat))
        weights.append(compute_weight(mats[-1]))
    wmax = weights[0]
    for w in weights[1:]:
        wmax = np.maximum(wmax, w)
    concat = np.hstack(mats)
    total = wmax.sum()
    if total < 1e-100:
        return np.zeros(concat.shape[1])
    return (concat * wmax).sum(axis=0) / total


def get_vec(item):
    term_id, genes = item
    return {term_id: compute_list_emb(list(genes))}


def prepare_data(train_cpds):
    global pert_sig, cpd2target, perttype, sigvec_all
    pos_l, pos_r, neg_l, neg_r = [], [], [], []
    freq = {}
    pos_idx = pert_sig[pert_sig.Perturbagen == perttype].index
    tgt_index = pert_sig.loc[pos_idx].groupby('CellLine').groups
    cpd_index = pert_sig[pert_sig.Perturbagen == 'Cpd'].groupby('Name').groups

    # positive
    for cpd in train_cpds:
        if cpd not in cpd_index:
            continue
        for idx in cpd_index[cpd]:
            cl = pert_sig.at[idx, 'CellLine']
            for tidx in tgt_index.get(cl, []):
                if pert_sig.at[tidx, 'Name'] in cpd2target.get(cpd, []):
                    pos_l.append(idx)
                    pos_r.append(tidx)
                    freq[tidx] = freq.get(tidx, 0) + 1

    # negative 1:1
    for cl, tinds in tgt_index.items():
        for tidx in tinds:
            needed = freq.get(tidx, 0)
            if needed <= 0:
                continue
            np.random.shuffle(train_cpds)
            for cpd in train_cpds:
                if pert_sig.at[tidx, 'Name'] in cpd2target.get(cpd, []):
                    continue
                for idx in cpd_index.get(cpd, []):
                    neg_l.append(idx)
                    neg_r.append(tidx)
                    needed -= 1
                    if needed <= 0:
                        break
                if needed <= 0:
                    break

    print(f"#Pos: {len(pos_l)}, #Neg: {len(neg_l)}")
    L = pos_l + neg_l
    R = pos_r + neg_r
    y = np.array([1]*len(pos_l) + [0]*len(neg_l))
    return sigvec_all[L], sigvec_all[R], y


def inference_testset(model, test_cpds, out_dir, batch_sz):
    global pert_sig, cpd2target, perttype, sigvec_all
    line_to_idx = pert_sig[pert_sig.Perturbagen == perttype].groupby('CellLine').groups
    cnt10 = cnt100 = 0
    for i, cpd in enumerate(test_cpds, start=1):
        subset = pert_sig[pert_sig.Name == cpd]
        for cl, grp in subset.groupby('CellLine'):
            tix = line_to_idx.get(cl, [])
            cix = grp.index
            if not len(cix) or not len(tix):
                continue
            L = np.repeat(cix, len(tix))
            R = np.tile(tix, len(cix))
            preds = model.predict([sigvec_all[L], sigvec_all[R]],
                                   batch_size=batch_sz, verbose=0)
            scores = 1/(1+np.exp(-preds.flatten()))
            genes = pert_sig.loc[tix, 'Name'].values
            max_scores = [scores[(R==t).nonzero()[0]].max() for t in tix]
            order = np.argsort(-np.array(max_scores))

            out_file = os.path.join(out_dir, f"{cpd}{cl}_{perttype}.txt")
            with open(out_file, 'w') as fw:
                fw.write("gene\trank\tscore\n")
                for rank, idx in enumerate(order, start=1):
                    fw.write(f"{genes[idx].split('@')[0]}\t{rank}\t{max_scores[idx]:.6f}\n")

            top10 = set(genes[order[:10]])
            top100 = set(genes[order[:100]])
            for tgt in cpd2target.get(cpd, []):
                cnt10 += int(tgt in top10)
                cnt100 += int(tgt in top100)

            del L, R, preds
            gc.collect()

        print(f"{i}/{len(test_cpds)} {cpd}: top10={cnt10}, top100={cnt100}")


if __name__ == '__main__':
    args = parse_args()
    perttype = args.perttype

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    # load embeddings
    def load_emb(path):
        with open(path) as f:
            return {row[0]: np.array(row[1:], dtype=np.float32) for row in csv.reader(f)}

    base = {'go': load_emb(args.emb_go), 'archs4': load_emb(args.emb_archs4)}
    opt = {}
    for label, path in [('ppi', args.emb_ppi), ('msigdb', args.emb_msigdb),
                        ('cellnet', args.emb_cellnet), ('collectri', args.emb_collectri)]:
        if path:
            opt[label] = load_emb(path)

    # build cpd->target map
    df_t = pd.read_csv(args.target_file)
    cpd2target = {}
    for term, genes in zip(df_t.term_name, df_t.Broad_target_gene_id):
        if pd.isna(genes):
            continue
        key = term.split('Cpd:')[1].split(':')[0] + '@'
        for gid in str(genes).split(':'):
            cpd2target.setdefault(key, []).append(gid + '@')
    cpd2target = {k: v for k,v in cpd2target.items() if len(set(v)) <= 5}

    # load signatures
    df_s = pd.read_csv(args.sig_file)
    df_id = pd.read_csv(os.path.join(os.path.dirname(args.target_file), 'term2gene_id.csv'))
    t2g = dict(zip(df_id.term_name, df_id.gene_id.astype(str)))

    # prepare records + tasks
    records, tasks, all_cells = [], [], set()
    for _, row in df_s.iterrows():
        tp = row.term_name.split(':')[0]
        if tp == 'Cpd':
            cpd = row.term_name.split(':')[1].split(':')[0] + '@'
            cl = row.term_name.split('@')[1].split('@')[0]
            if cpd in cpd2target:
                records.append([row.term_id, cpd, cl, row.term_name, tp, set(row.gids.split(','))])
                tasks.append((row.term_id, set(row.gids.split(','))))
                all_cells.add(cl)
        elif tp in ['cDNA', 'shRNA'] and row.term_name in t2g:
            gid = t2g[row.term_name] + '@'
            cl = row.term_name.split('@')[1].split('@')[0]
            if cl in all_cells:
                records.append([row.term_id, gid, cl, row.term_name, tp, set(row.gids.split(','))])
                tasks.append((row.term_id, set(row.gids.split(','))))

    pert_sig = pd.DataFrame(records, columns=['l1k','Name','CellLine','Description','Perturbagen','Signature'])

    # compound list and cross-validation folds
    with open(args.cpdlist_file) as f:
        cpd_list = [l.strip() + '@' for l in f if l.strip()]
    n = len(cpd_list)
    folds = [list(range(int(n*0.2*i), int(n*0.2*(i+1)))) for i in range(5)]

    # iterate embedding combos
    for combo in chain.from_iterable(combinations(opt.keys(), r) for r in range(len(opt)+1)):
        CURRENT_EMBEDDINGS = dict(base)
        for k in combo:
            CURRENT_EMBEDDINGS[k] = opt[k]
        name = 'base_' + '_'.join(sorted(combo)) if combo else 'base'
        odir = os.path.join(args.outdir, name)
        mdir = os.path.join(args.modeldir, name)
        os.makedirs(odir, exist_ok=True)
        os.makedirs(mdir, exist_ok=True)

        # skip if done
        if glob.glob(os.path.join(odir, '*.txt')) and glob.glob(os.path.join(mdir, '*.weights.h5')):
            print(f"Skipping {name}")
            continue

        # compute embeddings
        if args.n_cpu > 1:
            res = parallel.map(get_vec, tasks, n_CPU=args.n_cpu, progress=False)
            sig2vec = {k: v for d in res for k, v in d.items()}
        else:
            sig2vec = {}
            for t in tasks:
                sig2vec.update(get_vec(t))
        sigvec_all = np.vstack([sig2vec[l] for l in pert_sig.l1k])
        del CURRENT_EMBEDDINGS; gc.collect()

        # cross-validation
        for f_idx in range(5):
            train_idx = [i for j in range(5) if j != f_idx for i in folds[j]]
            test_idx = folds[f_idx]
            cpd_train = [cpd_list[i] for i in train_idx]
            cpd_test  = [cpd_list[i] for i in test_idx]

            print(f"Fold {f_idx}: train={len(cpd_train)}, test={len(cpd_test)}")
            xL, xR, y = prepare_data(cpd_train)

            model = get_model(sigvec_all.shape[1])
            model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(from_logits=True))
            model.fit([xL, xR], y, batch_size=1000, epochs=args.epochs, verbose=1)

            wpath = os.path.join(mdir, f"model_{perttype}_fold{f_idx}.weights.h5")
            model.save_weights(wpath)
            inference_testset(model, cpd_test, odir, args.predict_batch_size)

            del xL, xR, y, model
            K.clear_session(); gc.collect()

        del sigvec_all, sig2vec; gc.collect()

    print("Processing complete.")

