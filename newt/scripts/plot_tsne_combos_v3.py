#!/usr/bin/env python

import os
import sys
import math
import argparse
import itertools

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def load_embedding(path):
    """Load gene×embedding CSV (no header) into a DataFrame indexed by gene ID."""
    return pd.read_csv(path, index_col=0, header=None, dtype={0: str})

def make_merged_df(embeddings, sources):
    """Concatenate requested embedding DataFrames along columns, filling missing with 0."""
    dfs = [embeddings[src] for src in sources]
    return pd.concat(dfs, axis=1).fillna(0)

def run_tsne_and_save(X, genes, labels, out_png, out_csv, title, fig_sz=(8,6)):
    """
    Run t-SNE, save a global colored scatter + CSV of coords+labels.
    Returns (Y, color_map) for downstream use.
    """
    # Normalize rows then t-SNE
    Xn = normalize(X, axis=1)
    Y  = TSNE(n_components=2, random_state=0).fit_transform(Xn)

    # Save CSV
    df = pd.DataFrame({
        'gene':   genes,
        'tsne1':  Y[:,0],
        'tsne2':  Y[:,1],
        'top_go': labels
    })
    df.to_csv(out_csv, index=False)

    # Build color_map from Matplotlib default cycle
    cats = np.unique(labels)
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {cat: base_colors[i % len(base_colors)]
                 for i, cat in enumerate(cats)}

    # Global scatter
    fig, ax = plt.subplots(figsize=fig_sz)
    for cat in cats:
        mask = labels == cat
        ax.scatter(
            Y[mask,0], Y[mask,1],
            c=[color_map[cat]],
            label=cat, s=10, alpha=0.7
        )
    ax.set_box_aspect(1)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="upper left", bbox_to_anchor=(1,1),
              fontsize="small", frameon=False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return Y, color_map

def run_per_category(Y, labels, color_map, outdir, combo_name, fig_sz=(6,6)):
    """
    For each category, plot only that category’s points in its assigned color.
    """
    for cat, col in color_map.items():
        mask = labels == cat
        if mask.sum() < 2:
            continue
        safe_cat = cat.replace(' ', '_').replace('/', '_')
        out_png = os.path.join(outdir, f"tsne_{combo_name}_{safe_cat}.png")

        fig, ax = plt.subplots(figsize=fig_sz)
        ax.scatter(Y[mask,0], Y[mask,1],
                   c=[col], s=10, alpha=0.7)
        ax.set_box_aspect(1)
        ax.set_title(f"{combo_name}: {cat}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

def run_mosaic(Y, labels, color_map, out_png, combo_name):
    """
    Arrange per-category plots into a mosaic grid for this combination.
    """
    cats = list(color_map.keys())
    n = len(cats)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3, rows * 3),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, cat in zip(axes_flat, cats):
        mask = labels == cat
        ax.scatter(Y[mask,0], Y[mask,1],
                   c=[color_map[cat]], s=10, alpha=0.7)
        ax.set_title(cat, fontsize='small')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)

    # remove any unused axes
    for ax in axes_flat[len(cats):]:
        fig.delaxes(ax)

    fig.suptitle(f"{combo_name} – per‐category mosaic", y=1.02)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="t-SNE for each embedding combo + per‐category & mosaic plots"
    )
    parser.add_argument('--emb_go',       required=True)
    parser.add_argument('--emb_archs4',   required=True)
    parser.add_argument('--emb_ppi',      default=None)
    parser.add_argument('--emb_msigdb',   default=None)
    parser.add_argument('--emb_cellnet',  default=None)
    parser.add_argument('--emb_collectri',default=None)
    parser.add_argument('--go_map',       required=True,
                        help="CSV mapping 'gene','top_go'")
    parser.add_argument('--outdir',       required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load embeddings
    embeddings = {
        'go':     load_embedding(args.emb_go),
        'archs4': load_embedding(args.emb_archs4),
    }
    optional = {
        'ppi':      args.emb_ppi,
        'msigdb':   args.emb_msigdb,
        'cellnet':  args.emb_cellnet,
        'collectri':args.emb_collectri,
    }
    for name, path in optional.items():
        if path:
            embeddings[name] = load_embedding(path)

    # Load GO top-level map
    go_map = pd.read_csv(args.go_map, dtype=str)
    if not {'gene','top_go'}.issubset(go_map.columns):
        print("ERROR: --go_map must have 'gene' and 'top_go' columns", file=sys.stderr)
        sys.exit(1)
    go_map['gene'] = go_map['gene'].str.rstrip('@')
    go_map = go_map.set_index('gene')

    # Build all combinations of optional embeddings
    opt_keys = [k for k,p in optional.items() if p]
    combos = []
    for r in range(len(opt_keys)+1):
        for subset in itertools.combinations(opt_keys, r):
            combo_name = 'base' if not subset else 'base_' + '_'.join(sorted(subset))
            sources    = ['go','archs4'] + list(subset)
            combos.append((combo_name, sources))

    # Process each combination
    for combo_name, sources in combos:
        print(f"Processing combination: {combo_name}")
        merged = make_merged_df(embeddings, sources)

        genes = merged.index.intersection(go_map.index)
        if len(genes) < 2:
            print(f"  → skip '{combo_name}': only {len(genes)} overlapping genes", file=sys.stderr)
            continue

        X      = merged.loc[genes].values
        labels = go_map.loc[genes, 'top_go'].values

        # Global
        png_glob = os.path.join(args.outdir, f"tsne_{combo_name}.png")
        csv_glob = os.path.join(args.outdir, f"tsne_{combo_name}.csv")
        title    = f"t-SNE: {combo_name}"
        Y, cmap  = run_tsne_and_save(
            X, genes.tolist(), labels,
            png_glob, csv_glob, title, fig_sz=(8,6)
        )

        # Per-category
        run_per_category(
            Y, labels, cmap,
            args.outdir, combo_name, fig_sz=(6,6)
        )

        # Mosaic of all categories
        png_mosaic = os.path.join(args.outdir, f"tsne_{combo_name}_mosaic.png")
        run_mosaic(
            Y, labels, cmap,
            png_mosaic, combo_name
        )

if __name__ == "__main__":
    main()

