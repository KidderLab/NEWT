#!/usr/bin/env python
"""
Runs experiments using the "loose" approach only.

The loose approach always requires the default modality and fills in missing other
modalities (ppi, msigdb, dorothea, and collectri) with zeros if they are missing.
This method retains many more genes compared to the strict approach.

The script:
 - Iterates over combinations of modalities.
 - Trains a RandomForest classifier plus applies t-SNE on the raw embeddings.
 - Optionally trains a Keras fusion model to compute a fused representation.
 - Outputs:
     - t-SNE CSV and PNG files,
     - Fused embeddings text file,
     - An aggregated JSON report,
     - An accuracy ranking file (all based solely on the loose approach).

New modalities "dorothea" and "collectri" are included in various combinations.
"""

import argparse
import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

###############################################################################
# Utility to fix seeds (if desired)
# import random
# random.seed(0)
# np.random.seed(0)
# tf.random.set_seed(0)
###############################################################################

def load_embeddings_csv(filepath, expected_dim=256):
    """
    Reads lines: gene, f1, f2, ...
    Returns a dictionary: { gene -> np.array(embedding) } of length expected_dim.
    Skips rows where conversion fails (e.g. header rows).
    """
    emb = {}
    if not os.path.isfile(filepath):
        print(f"[Warning] {filepath} not found.")
        return emb
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                float(row[1])
            except ValueError:
                continue  # Skip header or malformed row.
            gene = row[0].strip()
            try:
                vals = np.array([float(x) for x in row[1:]], dtype=np.float32)
            except ValueError as e:
                print(f"[Warning] Could not convert row for gene {gene}: {e}")
                continue
            if len(vals) < expected_dim:
                vals = np.pad(vals, (0, expected_dim - len(vals)), 'constant')
            else:
                vals = vals[:expected_dim]
            emb[gene] = vals
    return emb

def load_tissue_file(filepath):
    """
    Tissue file format: Tissue, gene1, gene2, ...
    Returns a dictionary: { tissue -> set(genes) }.
    """
    t2g = {}
    if not os.path.isfile(filepath):
        print(f"[Warning] Tissue file not found: {filepath}")
        return t2g
    with open(filepath, "r") as f:
        for line in f:
            items = line.strip().split(",")
            if len(items) < 2:
                continue
            tissue = items[0].strip()
            genes = {g.strip() for g in items[1:] if g.strip()}
            t2g[tissue] = genes
    return t2g

# Mapping modality names to dimensions.
def modality_dim(mod, cellnet_dim):
    if mod == "default":
        return 512
    elif mod in ["ppi", "msigdb", "dorothea", "collectri"]:
        return 256
    elif mod == "cellnet":
        return cellnet_dim
    else:
        raise ValueError("Unknown modality: " + mod)

###############################################################################
# Build data functions (loose approach only)
###############################################################################

def build_XyGene_for_combo_loose(tissue_gene, default_emb, ppi_emb, msigdb_emb, cellnet_emb, dorothea_emb, collectri_emb, combo, cellnet_dim=256):
    """
    For the "loose" approach:
      - Always require 'default' and fill missing modality with zeros.
    """
    effective_modalities = combo if len(combo) > 0 else ["default"]
    Xlist = []
    ylist = []
    gene_list = []
    for tissue, geneset in tissue_gene.items():
        for g in geneset:
            if "default" in effective_modalities and g not in default_emb:
                continue
            vec_parts = []
            for mod in effective_modalities:
                if mod == "default":
                    vec_parts.append(default_emb[g])
                elif mod == "ppi":
                    vec_parts.append(ppi_emb[g] if g in ppi_emb else np.zeros(modality_dim("ppi", cellnet_dim), dtype=np.float32))
                elif mod == "msigdb":
                    vec_parts.append(msigdb_emb[g] if g in msigdb_emb else np.zeros(modality_dim("msigdb", cellnet_dim), dtype=np.float32))
                elif mod == "cellnet":
                    vec_parts.append(cellnet_emb[g] if g in cellnet_emb else np.zeros(modality_dim("cellnet", cellnet_dim), dtype=np.float32))
                elif mod == "dorothea":
                    vec_parts.append(dorothea_emb[g] if g in dorothea_emb else np.zeros(modality_dim("dorothea", cellnet_dim), dtype=np.float32))
                elif mod == "collectri":
                    vec_parts.append(collectri_emb[g] if g in collectri_emb else np.zeros(modality_dim("collectri", cellnet_dim), dtype=np.float32))
                else:
                    raise ValueError("Unknown modality in loose mode: " + mod)
            xcat = np.concatenate(vec_parts)
            Xlist.append(xcat)
            ylist.append(tissue)
            gene_list.append(g)
    X = np.array(Xlist, dtype=np.float32)
    y = np.array(ylist)
    return X, y, gene_list

###############################################################################
# t-SNE + RandomForest classifier (loose approach)
###############################################################################

def run_rf_tsne(X, y, genes, tissues, combo_name, outdir, approach):
    """
    1) Applies t-SNE on raw X.
    2) Splits data (80/20) into training and testing.
    3) Trains a RandomForest and computes accuracy.
    4) Exports t-SNE CSV and plot (PNG).
    """
    n_points = len(X)
    if n_points < 2:
        print(f"[Warning] {approach}_{combo_name} has only {n_points} points => skipping.")
        return None, None, n_points

    tsne_model = TSNE(n_components=2, random_state=0)
    coords = tsne_model.fit_transform(X)

    tsne_csv = os.path.join(outdir, f"tsne_{approach}_{combo_name}.csv")
    with open(tsne_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gene", "x", "y", "label"])
        for i in range(n_points):
            writer.writerow([genes[i], f"{coords[i,0]:.4f}", f"{coords[i,1]:.4f}", y[i]])

    plt.figure(figsize=(5,5))
    for t in tissues:
        idx = np.where(y == t)[0]
        if len(idx) > 0:
            plt.scatter(coords[idx,0], coords[idx,1], label=t)
    plt.title(f"{approach}_{combo_name}")
    plt.legend()
    figpath = os.path.join(outdir, f"tsne_{approach}_{combo_name}.png")
    plt.savefig(figpath, bbox_inches="tight")
    plt.close()

    idx_all = np.arange(n_points)
    np.random.shuffle(idx_all)
    train_sz = int(0.8 * n_points)
    train_idx = idx_all[:train_sz]
    test_idx = idx_all[train_sz:]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()

    return acc, figpath, n_points

###############################################################################
# Fusion Model building (loose approach only)
###############################################################################

class MultiModalAttentionFusion(tf.keras.layers.Layer):
    def __init__(self, projection_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
    def build(self, input_shape):
        self.proj_layers = [Dense(self.projection_dim, activation='relu') for _ in input_shape]
        self.score_layer = Dense(1)
        super().build(input_shape)
    def call(self, inputs):
        projected = []
        for proj, inp in zip(self.proj_layers, inputs):
            projected.append(proj(inp))
        stacked = tf.stack(projected, axis=1)
        scores = self.score_layer(stacked)
        scores = tf.squeeze(scores, axis=-1)
        w = Activation('softmax')(scores)
        w_exp = tf.expand_dims(w, axis=-1)
        fused = tf.reduce_sum(stacked * w_exp, axis=1)
        return fused

def build_fusion_model_modalities(mods, cellnet_dim, num_classes, fusion_method="attention"):
    """
    Builds a fusion model for the modalities provided in 'mods'.
    Valid modality names: "default", "ppi", "msigdb", "cellnet", "dorothea", "collectri".
    """
    input_layers = {}
    for mod in mods:
        d = modality_dim(mod, cellnet_dim)
        input_layers[mod] = Input(shape=(d,), name=f"{mod}_input")
    inputs_list = [input_layers[mod] for mod in mods]
    
    if fusion_method == "attention":
        fused = MultiModalAttentionFusion(128)(inputs_list)
        penultimate = Dense(128, activation='relu', name="penultimate")(fused)
        out = Dense(num_classes, activation='softmax')(penultimate)
        model = Model(inputs_list, out)
    else:
        branch_outputs = []
        for mod in mods:
            branch = input_layers[mod]
            if mod == "default":
                branch = Dense(256, activation='relu')(branch)
                branch = Dense(128, activation='relu')(branch)
            elif mod in ["ppi", "msigdb", "cellnet", "dorothea", "collectri"]:
                branch = Dense(128, activation='relu')(branch)
                branch = Dense(64, activation='relu')(branch)
            branch_outputs.append(branch)
        merged = Concatenate()(branch_outputs)
        dense1 = Dense(128, activation='relu', name="dense1")(merged)
        penultimate = Dense(64, activation='relu', name="penultimate")(dense1)
        out = Dense(num_classes, activation='softmax')(penultimate)
        model = Model(inputs_list, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def average_attention_weights(model, modality_order, X_modalities):
    """
    Calculates the average attention weights from the MultiModalAttentionFusion layer.
    """
    att_layer = None
    for layer in model.layers:
        if isinstance(layer, MultiModalAttentionFusion):
            att_layer = layer
            break
    if att_layer is None:
        return None
    projected = []
    for proj, X in zip(att_layer.proj_layers, X_modalities):
        projected.append(proj(X))
    stacked = tf.stack(projected, axis=1)
    scores = att_layer.score_layer(stacked)
    scores = tf.squeeze(scores, axis=-1)
    w = tf.nn.softmax(scores, axis=-1)
    mean_w = tf.reduce_mean(w, axis=0).numpy()
    return mean_w

###############################################################################
# Fusion helper for experiments (loose approach only)
###############################################################################

def run_fusion_combo(approach_label, build_func, default_emb, ppi_emb, msigdb_emb, cellnet_emb, dorothea_emb, collectri_emb,
                     tissue_gene, combo, cellnet_dim, outdir, fusion_method, fusion_epochs, fusion_patience, batch_size):
    """
    Runs a fusion experiment using the modalities specified in combo.
    """
    effective_modalities = combo if len(combo) > 0 else ["default"]
    Xall, y_all, gene_all = build_func(tissue_gene, default_emb, ppi_emb, msigdb_emb, cellnet_emb, dorothea_emb, collectri_emb, effective_modalities, cellnet_dim)
    n_points = len(Xall)
    if n_points < 2:
        print(f"[Warning] {approach_label} fusion => only {n_points} points => skipping.")
        return {"n_points": n_points, "fusion_accuracy": None}
    dims = [modality_dim(mod, cellnet_dim) for mod in effective_modalities]
    splits = []
    cum = 0
    for d in dims:
        splits.append(Xall[:, cum:cum+d])
        cum += d
    idx_all = np.arange(n_points)
    np.random.shuffle(idx_all)
    train_sz = int(0.8 * n_points)
    train_idx = idx_all[:train_sz]
    test_idx = idx_all[train_sz:]
    
    X_train_modal = [X_mod[train_idx] for X_mod in splits]
    X_test_modal  = [X_mod[test_idx] for X_mod in splits]
    y_train_lbl = y_all[train_idx]
    y_test_lbl  = y_all[test_idx]
    gene_test   = [gene_all[i] for i in test_idx]

    tissue_set = sorted(list(set(y_all)))
    t2i = {t: i for i, t in enumerate(tissue_set)}
    def to_onehot(lbls):
        arr = np.zeros((len(lbls), len(tissue_set)), dtype=np.float32)
        for i, val in enumerate(lbls):
            arr[i, t2i[val]] = 1.0
        return arr
    y_train_oh = to_onehot(y_train_lbl)
    y_test_oh  = to_onehot(y_test_lbl)
    num_classes = len(tissue_set)

    model_fus = build_fusion_model_modalities(effective_modalities, cellnet_dim, num_classes, fusion_method)
    es = EarlyStopping(monitor="val_accuracy", patience=fusion_patience, restore_best_weights=True)
    model_fus.fit(
        X_train_modal,
        y_train_oh,
        validation_data=(X_test_modal, y_test_oh),
        epochs=fusion_epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=[es]
    )
    loss_fus, acc_fus = model_fus.evaluate(X_test_modal, y_test_oh, verbose=0)
    print(f"{approach_label} fusion with combo {effective_modalities} => test_acc={acc_fus:.4f}")

    # Extract penultimate embeddings.
    penultimate_layer = model_fus.get_layer("penultimate").output
    sub_model = Model(inputs=model_fus.input, outputs=penultimate_layer)
    feats_fus = sub_model.predict(X_test_modal)
    
    test_count = len(gene_test)
    if test_count < 2:
        print(f"[Warning] {approach_label}: Not enough test samples for TSNE, skipping t-SNE plotting.")
        tsne_csv = None
        figpath = None
    else:
        perplexity = min(30, test_count - 1)
        coords = TSNE(n_components=2, random_state=0, perplexity=perplexity).fit_transform(feats_fus)
        tsne_csv = os.path.join(outdir, f"tsne_{approach_label}.csv")
        with open(tsne_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gene", "x", "y", "label"])
            for i in range(test_count):
                writer.writerow([gene_test[i], f"{coords[i,0]:.4f}", f"{coords[i,1]:.4f}", y_test_lbl[i]])
        
        plt.figure(figsize=(5,5))
        for t in tissue_set:
            idx = np.where(y_test_lbl == t)[0]
            if len(idx) > 0:
                plt.scatter(coords[idx,0], coords[idx,1], label=t)
        plt.legend()
        plt.title(f"{approach_label}")
        figpath = os.path.join(outdir, f"tsne_{approach_label}.png")
        plt.savefig(figpath, bbox_inches="tight")
        plt.close()

    fusion_txt = os.path.join(outdir, f"fusion_{approach_label}_embeddings.txt")
    with open(fusion_txt, "w") as f:
        for i in range(test_count):
            rowvals = [f"{feats_fus[i,j]}" for j in range(feats_fus.shape[1])]
            line = gene_test[i] + "\t" + "\t".join(rowvals)
            f.write(line + "\n")

    att_w = None
    if fusion_method == "attention":
        att_w = average_attention_weights(model_fus, effective_modalities, X_test_modal)
        if att_w is not None:
            print(f"Average attention for {approach_label} {effective_modalities}:",
                  att_w.round(3).tolist())

    return {
        "n_points": n_points,
        "fusion_accuracy": float(acc_fus),
        "tsne_plot": figpath,
        "tsne_data": tsne_csv,
        "penultimate_embeddings": fusion_txt,
        "attention_weights": att_w.round(3).tolist() if att_w is not None else None
    }

###############################################################################
# Function to save accuracy ranking.
###############################################################################

def save_accuracy_ranking(final_results, outpath):
    """
    Flattens final_results and writes a tab-separated file ranking experiments by accuracy.
    """
    rows = []
    for approach_name, combos_dict in final_results.items():
        for combo_key, info in combos_dict.items():
            acc = info.get("test_accuracy")
            if acc is not None:
                rows.append((approach_name, combo_key, acc))
            fusion_acc = info.get("fusion_accuracy")
            if fusion_acc is not None:
                rows.append((approach_name, combo_key + " (fusion)", fusion_acc))
    rows.sort(key=lambda x: x[2], reverse=True)
    
    with open(outpath, "w") as f:
        f.write("Approach\tCombo\tAccuracy\n")
        for approach_name, combo_key, acc in rows:
            f.write(f"{approach_name}\t{combo_key}\t{acc:.4f}\n")

###############################################################################
# Main pipeline (loose approach only)
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results", help="Output directory.")
    parser.add_argument("--default_file", default="../data/gene_vec_go_256.csv", help="GO (256).")
    parser.add_argument("--archs4_file", default="../data/gene_vec_archs4_256.csv", help="ARCHS4 (256); together form default (512).")
    parser.add_argument("--ppi_file", default="../data/learned_gene_embeddings_go_graph.csv", help="PPI (256).")
    parser.add_argument("--msigdb_file", default="../data/msigdb_bundle_embeddings_entrez.csv", help="MSigDB (256).")
    parser.add_argument("--cellnet_file", default="../data/cellnet_filtered_entrez_embeddings.csv", help="CellNet embeddings (user dim).")
    parser.add_argument("--dorothea_file", default="../data/dorothea_embeddings_entrez_embeddings.csv", help="Dorothea embeddings (256).")
    parser.add_argument("--collectri_file", default="../data/collectri_embeddings_entrez_embeddings.csv", help="Collectri embeddings (256).")
    parser.add_argument("--cellnet_dim", type=int, default=256, help="Dimension for CellNet.")
    parser.add_argument("--tissue_file", default="data/tissue_specific.txt", help="Tissue CSV.")
    parser.add_argument("--fusion_method", choices=["attention", "multimodal"], default="attention", help="Keras fusion approach.")
    parser.add_argument("--fusion_epochs", type=int, default=30, help="Epochs for Keras fusion.")
    parser.add_argument("--fusion_patience", type=int, default=5, help="EarlyStopping patience.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for fusion.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load tissue file.
    tissue_gene = load_tissue_file(args.tissue_file)
    tissues = sorted(list(tissue_gene.keys()))
    print(f"Found {len(tissues)} tissues, total genes = {sum(len(s) for s in tissue_gene.values())}")

    # 2) Default embeddings (concatenation of GO and ARCHS4).
    go_emb = load_embeddings_csv(args.default_file, 256)
    arch_emb = load_embeddings_csv(args.archs4_file, 256)
    default_emb = {}
    for g in set(list(go_emb.keys()) + list(arch_emb.keys())):
        v_go = go_emb.get(g, np.zeros(256, dtype=np.float32))
        v_ar = arch_emb.get(g, np.zeros(256, dtype=np.float32))
        default_emb[g] = np.concatenate([v_go, v_ar])  # 512

    # 3) Other modalities.
    ppi_emb    = load_embeddings_csv(args.ppi_file, 256)
    msigdb_emb = load_embeddings_csv(args.msigdb_file, 256)
    cellnet_emb= load_embeddings_csv(args.cellnet_file, args.cellnet_dim)
    dorothea_emb = load_embeddings_csv(args.dorothea_file, 256)
    collectri_emb = load_embeddings_csv(args.collectri_file, 256)

    # Classification experiments (loose only).
    combos = [
        [],                               # default only (implicit default)
        ["ppi"],
        ["msigdb"],
        ["cellnet"],
        ["dorothea"],
        ["collectri"],
        ["ppi", "msigdb"],
        ["ppi", "cellnet"],
        ["msigdb", "cellnet"],
        ["ppi", "msigdb", "cellnet"],
        ["dorothea", "cellnet"],
        ["default", "dorothea"],
        ["default", "collectri"],
        ["default", "cellnet", "collectri"],
        ["default", "cellnet", "dorothea", "collectri"],
        ["default", "ppi", "msigdb", "cellnet", "dorothea", "collectri"]  # all modalities
    ]

    def combo_name(c):
        return "default" if len(c) == 0 else "_".join(c)

    results_loose = {}
    def run_combos_for_approach(approach_name, build_func):
        approach_results = {}
        for c in combos:
            cname = combo_name(c)
            X, y, genes = build_func(tissue_gene, default_emb, ppi_emb, msigdb_emb, cellnet_emb, dorothea_emb, collectri_emb, c, args.cellnet_dim)
            n_points = len(X)
            if n_points < 2:
                print(f"[Warning] {approach_name}_{cname} => only {n_points} points => skipping.")
                approach_results[f"{approach_name}_{cname}"] = {"n_points": n_points, "test_accuracy": None, "tsne_plot": None}
                continue
            print(f"\n== {approach_name} combo: {cname}, n_points={n_points}, dims={X.shape[1]} ==")
            acc, figpath, used_n = run_rf_tsne(X, y, genes, tissues, cname, args.outdir, approach_name)
            approach_results[f"{approach_name}_{cname}"] = {"n_points": used_n, "test_accuracy": acc, "tsne_plot": figpath}
        return approach_results

    # Run classification experiments.
    combos_loose_res = run_combos_for_approach("loose", build_XyGene_for_combo_loose)
    for k, v in combos_loose_res.items():
        results_loose[k] = v

    # Fusion experiments.
    orig_combo = ["default", "ppi", "msigdb", "cellnet"]
    fusion_loose_orig = run_fusion_combo("loose_fusion_all", build_XyGene_for_combo_loose,
                                           default_emb, ppi_emb, msigdb_emb, cellnet_emb, dorothea_emb, collectri_emb,
                                           tissue_gene, orig_combo, args.cellnet_dim, args.outdir,
                                           args.fusion_method, args.fusion_epochs, args.fusion_patience, args.batch_size)
    results_loose["loose_fusion_all"] = fusion_loose_orig

    new_fusion_combos = {
        "fusion_dorothea": ["dorothea"],
        "fusion_collectri": ["collectri"],
        "fusion_dorothea_cellnet": ["dorothea", "cellnet"],
        "fusion_default_dorothea": ["default", "dorothea"],
        "fusion_default_collectri": ["default", "collectri"],
        "fusion_default_cellnet_collectri": ["default", "cellnet", "collectri"],
        "fusion_default_cellnet_dorothea_collectri": ["default", "cellnet", "dorothea", "collectri"],
        "fusion_all": ["default", "ppi", "msigdb", "cellnet", "dorothea", "collectri"]  # all modalities
    }
    for key, combo in new_fusion_combos.items():
        fusion_res_loose = run_fusion_combo("loose_" + key, build_XyGene_for_combo_loose,
                                             default_emb, ppi_emb, msigdb_emb, cellnet_emb, dorothea_emb, collectri_emb,
                                             tissue_gene, combo, args.cellnet_dim, args.outdir,
                                             args.fusion_method, args.fusion_epochs, args.fusion_patience, args.batch_size)
        results_loose["loose_" + key] = fusion_res_loose

    final_results = {"loose": results_loose}
    out_json = os.path.join(args.outdir, "report_aggregated.json")
    with open(out_json, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"\nAll results saved to {out_json}")

    ranking_txt = os.path.join(args.outdir, "accuracy_ranking.txt")
    save_accuracy_ranking(final_results, ranking_txt)
    print(f"Accuracy ranking saved to {ranking_txt}")
    print("Done.")

if __name__ == "__main__":
    main()

