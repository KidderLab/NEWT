#!/usr/bin/env python3
# =============================================================================
# scanpy_clustering_v16c.py  (diagnostic + robust)
# - Keeps ALL original outputs/plots and file names
# - Headless plotting (Agg) + timestamped outdir
# - Restores 'skip first CSV row' behavior to match original loader
# - Adds verbose logging of how many vectors were loaded per source
# - Fails loudly if improved embeddings are empty (prevents silent degenerate plots)
# =============================================================================

import os
import csv
import argparse
import numpy as np
import pandas as pd
import datetime as _dt

# Headless plotting
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

sc.settings.autoshow = False


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M")


def _final_outdir(base: str | None) -> str:
    stamp = _timestamp()
    if base:
        base = str(base).rstrip("/ ")
        return f"{base}_{stamp}"
    return f"results/scanpy_v16c_{stamp}"


def _log(msg: str):
    print(f"[scanpy_v16c] {msg}")


# ------------------ Embedding IO (restored semantics) ------------------
def load_embeddings_csv(path, dim=None):
    emb = {}
    if not os.path.exists(path):
        _log(f"MISSING: {path}")
        return emb
    with open(path) as f:
        reader = csv.reader(f)
        # ORIGINAL BEHAVIOR: skip first line blindly (header or not)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            gene, *vals = row
            try:
                v = np.array([float(x) for x in vals], dtype=np.float32)
            except ValueError:
                # skip malformed
                continue
            if dim is not None:
                if v.size < dim:
                    v = np.pad(v, (0, dim - v.size), 'constant')
                else:
                    v = v[:dim]
            emb[gene] = v
    _log(f"Loaded {len(emb):,} vectors from {os.path.basename(path)}" )
    return emb


def compute_weight(mat):
    sim = cosine_similarity(mat, mat)
    sim[np.isnan(sim)] = 0
    mu = sim.mean(axis=1)
    sd = mat.std(axis=1)
    w = (mu - mu.mean()) / np.where(sd > 0, sd, 1e-9)
    return np.clip(w, 0, 1).reshape(-1, 1)


def compute_list_emb(genes, sources):
    mats, wts = [], []
    for name, d in sources.items():
        if not d:
            continue
        dim = next(iter(d.values())).shape[0]
        M = np.vstack([d.get(g, np.zeros(dim, dtype=np.float32)) for g in genes])
        M_norm = normalize(M) if M.size else M
        mats.append(M_norm)
        wts.append(compute_weight(M_norm) if M.size else np.zeros((len(genes), 1), dtype=np.float32))
    if not mats:
        return np.zeros(1, dtype=np.float32)
    wmax = wts[0].copy()
    for w in wts[1:]:
        wmax = np.maximum(wmax, w)
    cat = np.hstack(mats)
    tot = wmax.sum()
    return (cat * wmax).sum(axis=0) / (tot if tot > 1e-12 else 1)


def load_all(data_dir):
    go = load_embeddings_csv(os.path.join(data_dir, 'gene_vec_go_256.csv'), 256)
    arch = load_embeddings_csv(os.path.join(data_dir, 'gene_vec_archs4_256.csv'), 256)

    default = {g: np.concatenate([go.get(g, np.zeros(256, dtype=np.float32)),
                                  arch.get(g, np.zeros(256, dtype=np.float32))]).astype(np.float32)
               for g in (set(go) | set(arch))}

    mods = {}
    for name, fname in [
        ('ppi', 'learned_gene_embeddings_go_graph.csv'),
        ('msigdb', 'msigdb_bundle_embeddings_entrez.csv'),
        ('cellnet', 'cellnet_filtered_entrez_embeddings.csv'),
        ('dorothea', 'dorothea_embeddings_entrez_embeddings.csv'),
        ('collectri', 'collectri_embeddings_entrez_embeddings.csv')
    ]:
        path = os.path.join(data_dir, fname)
        emb = load_embeddings_csv(path)
        mods[name] = emb

    _log("Source sizes: " + ", ".join([f"default={len(default):,}"] + [f"{k}={len(v):,}" for k,v in mods.items()]))
    return default, mods


# Marker genes
marker_genes = {
    'B-cell': ['CD79A', 'MS4A1'], 'Plasma': ['IGJ'],
    'T-cell': ['CD3D'], 'NK': ['GNLY', 'NKG7'],
    'Myeloid': ['CST3', 'LYZ'], 'Monocytes': ['FCGR3A'],
    'Dendritic': ['FCER1A']
}


def run_pipeline(adata, rep_key, label, outdir):
    ad = adata.copy()
    sc.pp.neighbors(ad, use_rep=rep_key)
    sc.tl.umap(ad)
    sc.tl.leiden(ad, key_added='clusters', resolution=0.5)
    sc.pl.umap(ad, color='clusters', title=label,
               save=f'_{label}_umap.png', show=False)
    sc.pl.dotplot(ad, marker_genes, groupby='clusters', dendrogram=True,
                  use_raw=True, save=f'_{label}_dotplot.png', show=False)
    sc.pl.stacked_violin(ad, marker_genes, groupby='clusters', dendrogram=True,
                         use_raw=True, save=f'_{label}_stacked_violin.png', show=False)
    df = pd.DataFrame(ad.obsm['X_umap'], index=ad.obs_names,
                      columns=['UMAP1', 'UMAP2'])
    df['cluster'] = ad.obs['clusters'].astype(str)
    df.to_csv(os.path.join(outdir, f'{label}_umap_clusters.csv'))
    return ad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--outdir', default=None,
                        help='Base output directory (timestamp appended automatically); '
                             'if omitted, results/scanpy_v16c_YYYYmmdd_HHMM is used.')
    args = parser.parse_args()

    outdir = _final_outdir(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    sc.settings.figdir = outdir

    # Load PBMC3k (same as original)
    adata = sc.datasets.pbmc3k()

    # QC & preprocess (same as original)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    adata.obs['n_genes'] = adata.obs['n_genes_by_counts']
    adata.obs['percent_mito'] = adata.obs['pct_counts_mt']
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()
    raw_master = adata.raw.to_adata()
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added='clusters', resolution=0.5)

    # Embedding pipelines
    ad_pca = run_pipeline(adata, 'X_pca', 'PCA', outdir)

    default_emb, mods = load_all(args.data_dir)

    # Reconstruct "improved" as original: include default + all mods
    improved_embs = {'default': default_emb}
    improved_embs.update(mods)

    # Safety check to avoid degenerate single-cluster outputs
    nonempty_sources = sum(1 for d in improved_embs.values() if len(d) > 0)
    if nonempty_sources == 0:
        raise RuntimeError("No embeddings loaded for the 'improved' set. Check --data_dir paths and CSV formats.")

    n = adata.n_obs

    # Build Fdef
    if len(default_emb) > 0:
        dim_def = next(iter(default_emb.values())).shape[0]
        Fdef = np.zeros((n, dim_def), dtype=np.float32)
    else:
        Fdef = np.zeros((n, 1), dtype=np.float32)

    # Build Fimp as concatenation across sources (dimension is sum of each first vector length)
    dims_imp = [next(iter(d.values())).shape[0] for d in improved_embs.values() if len(d) > 0]
    dim_imp = int(sum(dims_imp)) if dims_imp else 1
    Fimp = np.zeros((n, dim_imp), dtype=np.float32)

    # Fill cell embeddings
    for i in range(n):
        expr = adata.X[i].A.flatten() if hasattr(adata.X, "A") else np.array(adata.X[i]).flatten()
        topg = np.argsort(expr)[::-1][:100]
        genes = list(adata.var_names[topg])
        if len(default_emb) > 0:
            Fdef[i] = compute_list_emb(genes, {'default': default_emb})
        Fimp[i] = compute_list_emb(genes, improved_embs)

    adata.obsm['X_froGS_def'] = Fdef
    adata.obsm['X_froGS_imp'] = Fimp

    ad_def = run_pipeline(adata, 'X_froGS_def', 'FRoGS_default', outdir)
    ad_imp = run_pipeline(adata, 'X_froGS_imp', 'FRoGS_improved', outdir)

    joint = np.hstack([adata.obsm['X_pca'][:, :20], adata.obsm['X_froGS_imp']])
    adata.obsm['X_joint'] = joint
    ad_joint = run_pipeline(adata, 'X_joint', 'Joint', outdir)

    # Signature scoring
    score_cols = []
    marker_genes = {
        'B-cell': ['CD79A', 'MS4A1'], 'Plasma': ['IGJ'],
        'T-cell': ['CD3D'], 'NK': ['GNLY', 'NKG7'],
        'Myeloid': ['CST3', 'LYZ'], 'Monocytes': ['FCGR3A'],
        'Dendritic': ['FCER1A']
    }
    for lineage, genes in marker_genes.items():
        sn = lineage.replace('-', '_') + '_score'
        sc.tl.score_genes(adata, gene_list=genes, score_name=sn, use_raw=True)
        score_cols.append(sn)
    pd.DataFrame(adata.obs[score_cols]).to_csv(os.path.join(outdir, 'signature_scores.csv'))

    for ad, lab in [(ad_pca, 'PCA'), (ad_def, 'FRoGS_default'), (ad_imp, 'FRoGS_improved'), (ad_joint, 'Joint')]:
        for c in score_cols:
            ad.obs[c] = adata.obs[c].values
        sc.pl.umap(ad, color=score_cols, save=f'_signature_{lab}.png', show=False)
        plt.close('all')

    # Additional exploratory per method
    method_map = {'PCA': ad_pca, 'FRoGS_default': ad_def, 'FRoGS_improved': ad_imp, 'Joint': ad_joint}
    for lab, ad in method_map.items():
        raw_ad = raw_master.copy()
        raw_ad.obs['clusters'] = ad.obs['clusters']
        with plt.rc_context({'figure.figsize': (4.5, 3)}):
            sc.pl.violin(raw_ad, ['CD79A', 'MS4A1'], groupby='clusters',
                         save=f'_CD79A_MS4A1_violin_{lab}.png', show=False)
        with plt.rc_context({'figure.figsize': (4.5, 3)}):
            sc.pl.violin(raw_ad, ['n_genes', 'percent_mito'], groupby='clusters',
                         stripplot=False, inner='box',
                         save=f'_n_genes_percent_mito_violin_{lab}.png', show=False)
        sc.pl.matrixplot(raw_ad, marker_genes, groupby='clusters', dendrogram=True,
                         cmap='Blues', standard_scale='var',
                         save=f'_matrixplot_scaled_{lab}.png', show=False)
        raw_ad.layers['scaled'] = sc.pp.scale(raw_ad.copy(), copy=True).X
        sc.pl.matrixplot(raw_ad, marker_genes, groupby='clusters', dendrogram=True,
                         layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r',
                         save=f'_matrixplot_zscore_{lab}.png', show=False)
        sc.pl.matrixplot(raw_ad, marker_genes, groupby='clusters', cmap='viridis',
                         swap_axes=False, dendrogram=False,
                         save=f'_heatmap_matrix_{lab}.png', show=False)
        sc.pl.matrixplot(raw_ad, marker_genes, groupby='clusters', layer='scaled',
                         vmin=-2, vmax=2, cmap='RdBu_r', swap_axes=True,
                         dendrogram=False, figsize=(11, 4),
                         save=f'_heatmap_scaled_matrix_{lab}.png', show=False)
        sc.pl.tracksplot(raw_ad, marker_genes, groupby='clusters', dendrogram=False,
                         save=f'_tracksplot_{lab}.png', show=False)

    print('Completed all plots and outputs in', outdir)


if __name__ == '__main__':
    main()
