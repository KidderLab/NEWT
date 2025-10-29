#!/usr/bin/env python

import os
import re
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.stats import ttest_rel
import logging

# Relative or configurable paths (portable)
FOLDER_DEFAULT = os.path.join("results", "default_shRNA")
FOLDER_IMPROVED_ROOT = os.path.join("results", "results_merged_multimodal_test2_shRNA")
CPD_GENE_PAIRS_CSV = os.path.join("data", "cpd_gene_pairs.csv")

# Optionally, ignore specific cell lines
IGNORE_CELLS = ['VCAP']

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)

def load_compound_targets(csv_path):
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        term = str(row.get("term_name", ""))
        if not term.startswith("Cpd:"):
            continue
        parts = term.split(":")
        if len(parts) < 2:
            continue
        cmpd = parts[1]
        gid = row.get("Broad_target_gene_id")
        if pd.isna(gid):
            continue
        try:
            gid = int(float(gid))
        except ValueError:
            continue
        mapping.setdefault(cmpd, set()).add(gid)
    return mapping

def load_predictions_by_cell_line(folder_path):
    preds = {}
    pattern = re.compile(r'^(BRD-[^@]+)@([^_]+)_shRNA\.txt$')
    for fn in os.listdir(folder_path):
        m = pattern.match(fn)
        if not m:
            continue
        cmpd, line = m.groups()
        df = pd.read_csv(os.path.join(folder_path, fn), sep="\t")
        total = len(df)
        line_dict = preds.setdefault(line, {})
        cmpd_dict = line_dict.setdefault(cmpd, {})
        for _, r in df.iterrows():
            gene, rank = r["gene"], r["rank"]
            frac = rank / total
            if gene not in cmpd_dict or rank < cmpd_dict[gene][0]:
                cmpd_dict[gene] = (rank, frac)
    return preds

def aggregate_predictions(preds_by_line):
    agg = {}
    for line_preds in preds_by_line.values():
        for cmpd, gd in line_preds.items():
            cd = agg.setdefault(cmpd, {})
            for gene, (rk, frac) in gd.items():
                if gene not in cd or rk < cd[gene][0]:
                    cd[gene] = (rk, frac)
    return agg

def compute_recall(mapping, preds):
    fracs = []
    for cmpd, targets in mapping.items():
        if cmpd not in preds:
            continue
        best = None
        for t in targets:
            if t in preds[cmpd]:
                _, f = preds[cmpd][t]
                if best is None or f < best:
                    best = f
        if best is not None:
            fracs.append(best)
    if not fracs:
        return {}
    total = len(fracs)
    return {p: sum(1 for f in fracs if f <= p) / total for p in [0.01, 0.02, 0.03, 0.04, 0.05]}

def get_best_metrics(mapping, preds):
    best_fracs, best_ranks = [], []
    for cmpd, targets in mapping.items():
        if cmpd not in preds:
            continue
        bf, br = None, None
        for t in targets:
            if t in preds[cmpd]:
                rk, frac = preds[cmpd][t]
                if bf is None or frac < bf:
                    bf, br = frac, rk
        if bf is not None:
            best_fracs.append(bf)
            best_ranks.append(br)
    if not best_fracs:
        return {}
    mf = np.mean(best_fracs)
    medf = np.median(best_fracs)
    mrr = np.mean([1.0/r for r in best_ranks])
    ndcg = np.mean([1.0/np.log2(r + 1) for r in best_ranks])
    prec = {k: sum(1 for r in best_ranks if r <= k) / len(best_ranks) for k in range(1, 41)}
    return {"mean_fraction": mf, "median_fraction": medf, "MRR": mrr, "nDCG": ndcg, "precision_at": prec}

# ---- Plotting routines ----

def plot_recall_curve(r_def, r_imp, line, odir):
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = sorted(r_def.keys())
    ax.plot(xs, [r_def[x] for x in xs], 'o-', label='default')
    ax.plot(xs, [r_imp[x] for x in xs], 'o-', label='improved')
    ax.set_xlabel("top % of genes")
    ax.set_ylabel("recall")
    ax.set_title(f"recall vs top % ({line})")
    ax.legend()
    # ensure axes region is square
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(os.path.join(odir, f"recall_{line}.png"), dpi=300)
    plt.close(fig)

def plot_cdf(fr_def, fr_imp, line, odir):
    fig, ax = plt.subplots(figsize=(6, 6))
    data_def = np.sort(fr_def)
    data_imp = np.sort(fr_imp)
    n1, n2 = len(data_def), len(data_imp)
    ax.step(data_def, np.arange(1, n1+1)/n1, where='post', label='default')
    ax.step(data_imp, np.arange(1, n2+1)/n2, where='post', label='improved')
    ax.set_xlabel("best fraction")
    ax.set_ylabel("cumulative proportion")
    ax.set_title(f"cdf ({line})")
    ax.legend()
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(os.path.join(odir, f"cdf_{line}.png"), dpi=300)
    plt.close(fig)

def plot_combined_recall(per_line, odir):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lines = [l for l in per_line if l not in IGNORE_CELLS]
    for idx, line in enumerate(sorted(lines)):
        color = colors[idx % len(colors)]
        r_def = per_line[line]['recall_def']
        r_imp = per_line[line]['recall_imp']
        xs = sorted(r_def.keys())
        ax.plot(xs, [r_def[x] for x in xs], marker='o', linestyle='--', color=color, label=f"default {line}")
        ax.plot(xs, [r_imp[x] for x in xs], marker='o', linestyle='-', color=color, label=f"improved {line}")
    ax.set_xlabel("top % of genes")
    ax.set_ylabel("recall")
    ax.set_title("recall vs top % (all cell lines)")
    ax.legend(fontsize='small', ncol=2)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(os.path.join(odir, "recall_all_cell_lines.png"), dpi=300)
    plt.close(fig)

def plot_combined_cdf(per_line, odir):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lines = [l for l in per_line if l not in IGNORE_CELLS]
    for idx, line in enumerate(sorted(lines)):
        color = colors[idx % len(colors)]
        fr_def = per_line[line]['best_fracs_def']
        fr_imp = per_line[line]['best_fracs_imp']
        if not fr_def or not fr_imp:
            continue
        n1, n2 = len(fr_def), len(fr_imp)
        ax.step(sorted(fr_def), np.arange(1, n1+1)/n1, where='post', linestyle='--', color=color, label=f"default {line}")
        ax.step(sorted(fr_imp), np.arange(1, n2+1)/n2, where='post', linestyle='-', color=color, label=f"improved {line}")
    ax.set_xlabel("best fraction")
    ax.set_ylabel("cumulative proportion")
    ax.set_title("cdf (all cell lines)")
    ax.legend(fontsize='small', ncol=2)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(os.path.join(odir, "cdf_all_cell_lines.png"), dpi=300)
    plt.close(fig)

def plot_mosaic_recall(per_line, odir):
    lines = [l for l in per_line if l not in IGNORE_CELLS]
    if not lines:
        return
    n = len(lines)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4), squeeze=False)
    for idx, line in enumerate(sorted(lines)):
        ax = axes[idx // ncols][idx % ncols]
        r_def = per_line[line]['recall_def']
        r_imp = per_line[line]['recall_imp']
        xs = sorted(r_def.keys())
        ax.plot(xs, [r_def[x] for x in xs], 'o--')
        ax.plot(xs, [r_imp[x] for x in xs], 'o-')
        ax.set_title(line)
        ax.set_box_aspect(1)
    for idx in range(n, nrows * ncols):
        fig.delaxes(axes[idx // ncols][idx % ncols])
    fig.tight_layout()
    fig.savefig(os.path.join(odir, "mosaic_recall.png"), dpi=300)
    plt.close(fig)

def plot_mosaic_cdf(per_line, odir):
    lines = [l for l in per_line if l not in IGNORE_CELLS]
    if not lines:
        return
    n = len(lines)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4), squeeze=False)
    for idx, line in enumerate(sorted(lines)):
        ax = axes[idx // ncols][idx % ncols]
        fr_def = per_line[line]['best_fracs_def']
        fr_imp = per_line[line]['best_fracs_imp']
        if not fr_def or not fr_imp:
            continue
        n1, n2 = len(fr_def), len(fr_imp)
        ax.step(sorted(fr_def), np.arange(1, n1+1)/n1, where='post', linestyle='--')
        ax.step(sorted(fr_imp), np.arange(1, n2+1)/n2, where='post', linestyle='-')
        ax.set_title(line)
        ax.set_box_aspect(1)
    for idx in range(n, nrows * ncols):
        fig.delaxes(axes[idx // ncols][idx % ncols])
    fig.tight_layout()
    fig.savefig(os.path.join(odir, "mosaic_cdf.png"), dpi=300)
    plt.close(fig)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info("Loading compound→target map from %s", CPD_GENE_PAIRS_CSV)
    mapping = load_compound_targets(CPD_GENE_PAIRS_CSV)

    logging.info("Loading default predictions from %s", FOLDER_DEFAULT)
    default_preds = load_predictions_by_cell_line(FOLDER_DEFAULT)

    default_name = os.path.basename(os.path.normpath(FOLDER_DEFAULT))
    improved_dirs = [
        os.path.join(FOLDER_IMPROVED_ROOT, d)
        for d in os.listdir(FOLDER_IMPROVED_ROOT)
        if os.path.isdir(os.path.join(FOLDER_IMPROVED_ROOT, d)) and d != default_name
    ]
    logging.info("Found %d improved subfolders", len(improved_dirs))

    for i, imp_path in enumerate(sorted(improved_dirs), 1):
        name = os.path.basename(imp_path)
        logging.info("(%d/%d) Processing %s", i, len(improved_dirs), name)

        improved_preds = load_predictions_by_cell_line(imp_path)
        out_dir = f"{name}_metrics_shRNA_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)

        cell_lines = sorted(set(default_preds) | set(improved_preds))
        per_line = {}

        for line in cell_lines:
            dp = default_preds.get(line, {})
            ip = improved_preds.get(line, {})
            r_def = compute_recall(mapping, dp)
            r_imp = compute_recall(mapping, ip)
            m_def = get_best_metrics(mapping, dp)
            m_imp = get_best_metrics(mapping, ip)

            bf_def = [
                min((dp.get(c, {}).get(t, (None, np.nan))[1] for t in mapping[c]), default=np.nan)
                for c in mapping if c in dp
            ]
            bf_imp = [
                min((ip.get(c, {}).get(t, (None, np.nan))[1] for t in mapping[c]), default=np.nan)
                for c in mapping if c in ip
            ]
            bf_def = [f for f in bf_def if not np.isnan(f)]
            bf_imp = [f for f in bf_imp if not np.isnan(f)]

            per_line[line] = {
                "recall_def": r_def,
                "recall_imp": r_imp,
                "metrics_def": m_def,
                "metrics_imp": m_imp,
                "best_fracs_def": bf_def,
                "best_fracs_imp": bf_imp
            }

            if r_def and r_imp:
                plot_recall_curve(r_def, r_imp, line, out_dir)
            if bf_def and bf_imp:
                plot_cdf(bf_def, bf_imp, line, out_dir)

        if per_line:
            plot_combined_recall(per_line, out_dir)
            plot_combined_cdf(per_line, out_dir)
            plot_mosaic_recall(per_line, out_dir)
            plot_mosaic_cdf(per_line, out_dir)

        # overall aggregated analysis
        agg_def = aggregate_predictions(default_preds)
        agg_imp = aggregate_predictions(improved_preds)
        or_def = compute_recall(mapping, agg_def)
        or_imp = compute_recall(mapping, agg_imp)

        bf_ag_def = [
            min((agg_def.get(c, {}).get(t, (None, np.nan))[1] for t in mapping[c]), default=np.nan)
            for c in mapping if c in agg_def
        ]
        bf_ag_imp = [
            min((agg_imp.get(c, {}).get(t, (None, np.nan))[1] for t in mapping[c]), default=np.nan)
            for c in mapping if c in agg_imp
        ]
        bf_ag_def = [f for f in bf_ag_def if not np.isnan(f)]
        bf_ag_imp = [f for f in bf_ag_imp if not np.isnan(f)]

        if or_def and or_imp:
            plot_recall_curve(or_def, or_imp, f"overall {name}", out_dir)
        if bf_ag_def and bf_ag_imp:
            plot_cdf(bf_ag_def, bf_ag_imp, f"overall {name}", out_dir)

        logging.info("  ✓ Completed %s (results in %s)", name, out_dir)

    logging.info("All experiments complete.")

if __name__ == "__main__":
    main()

