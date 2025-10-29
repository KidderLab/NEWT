#!/usr/bin/env python3
"""
scanpy_clustering_v16c.py

All embeddings (PCA, FRoGS-default, FRoGS-improved, Joint) over PBMC3k,
with comprehensive QC, filtering, HVG, UMAP, clustering, dotplot, stacked violin,
signature scoring, and additional explorations: violin, matrixplot, "heatmap"(via matrixplot),
tracksplot for each method—saved to outdir.

python scanpy_clustering_v16c.py --data_dir ../data

"""
import os
import csv
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import datetime as _dt

# Disable interactive plotting
sc.settings.autoshow = False

# Helper functions

def load_embeddings_csv(path, dim=None):
    emb = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            gene, *vals = row
            try:
                v = np.array([float(x) for x in vals], dtype=np.float32)
            except ValueError:
                continue
            if dim is not None:
                if v.size < dim:
                    v = np.pad(v, (0, dim - v.size), 'constant')
                else:
                    v = v[:dim]
            emb[gene] = v
    return emb


def compute_weight(mat):
    sim = cosine_similarity(mat, mat)
    sim[np.isnan(sim)] = 0
    mu = sim.mean(axis=1)
    sd = mat.std(axis=1)
    w = (mu - mu.mean()) / np.where(sd > 0, sd, 1e-9)
    return np.clip(w, 0, 1).reshape(-1,1)


def compute_list_emb(genes, sources):
    mats, wts = [], []
    for d in sources.values():
        dim = next(iter(d.values())).shape[0]
        M = np.vstack([d.get(g, np.zeros(dim)) for g in genes])
        M_norm = normalize(M)
        mats.append(M_norm)
        wts.append(compute_weight(M_norm))
    wmax = wts[0].copy()
    for w in wts[1:]:
        wmax = np.maximum(wmax, w)
    cat = np.hstack(mats)
    tot = wmax.sum()
    return (cat * wmax).sum(axis=0) / (tot if tot > 1e-12 else 1)


def load_all(data_dir):
    go = load_embeddings_csv(os.path.join(data_dir, 'gene_vec_go_256.csv'), 256)
    arch = load_embeddings_csv(os.path.join(data_dir, 'gene_vec_archs4_256.csv'), 256)
    default = {g: np.concatenate([go.get(g, np.zeros(256)), arch.get(g, np.zeros(256))])
               for g in set(go) | set(arch)}
    mods = {}
    for name, fname in [
        ('ppi','learned_gene_embeddings_go_graph.csv'),
        ('msigdb','msigdb_bundle_embeddings_entrez.csv'),
        ('cellnet','cellnet_filtered_entrez_embeddings.csv'),
        ('dorothea','dorothea_embeddings_entrez_embeddings.csv'),
        ('collectri','collectri_embeddings_entrez_embeddings.csv')]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            mods[name] = load_embeddings_csv(path)
    improved = {'default': default}
    improved.update(mods)
    return default, improved

# Marker genes
marker_genes = {
    'B-cell': ['CD79A','MS4A1'], 'Plasma': ['IGJ'],
    'T-cell': ['CD3D'], 'NK': ['GNLY','NKG7'],
    'Myeloid': ['CST3','LYZ'], 'Monocytes': ['FCGR3A'],
    'Dendritic': ['FCER1A']
}

# Core pipeline

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
                      columns=['UMAP1','UMAP2'])
    df['cluster'] = ad.obs['clusters'].astype(str)
    df.to_csv(os.path.join(outdir, f'{label}_umap_clusters.csv'))
    return ad

if __name__ == "__main__":
    import argparse
    import datetime as _dt
    import os
    import scanpy as sc

    parser = argparse.ArgumentParser(description="Scanpy clustering and figure generation (v16c)")

    # Default output folder with timestamp
    _default_outdir = f"results/scanpy_v16c_{_dt.datetime.now().strftime('%Y%m%d_%H%M')}"
    parser.add_argument(
        "--outdir",
        default=_default_outdir,
        help="Output directory for plots and results (default: timestamped under results/)"
    )

    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing input data (e.g., h5ad files or raw matrix files)"
    )

    args = parser.parse_args()

    # Ensure output directory exists and set Scanpy’s figure path
    os.makedirs(args.outdir, exist_ok=True)
    sc.settings.figdir = args.outdir

    # Example: load dataset (replace with your real loading logic)
    adata = sc.datasets.pbmc3k()

    # QC & preprocess
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
    ad_pca = run_pipeline(adata, 'X_pca', 'PCA', args.outdir)
    default_emb, improved_embs = load_all(args.data_dir)
    n = adata.n_obs
    dim_def = next(iter(default_emb.values())).shape[0]
    dim_imp = sum(next(iter(v.values())).shape[0] for v in improved_embs.values())
    Fdef = np.zeros((n, dim_def), dtype=np.float32)
    Fimp = np.zeros((n, dim_imp), dtype=np.float32)
    for i, cell in enumerate(adata.obs_names):
        expr = adata[i].X.toarray().flatten()
        topg = np.argsort(expr)[::-1][:100]
        genes = adata.var_names[topg]
        Fdef[i] = compute_list_emb(genes, {'default': default_emb})
        Fimp[i] = compute_list_emb(genes, improved_embs)
    adata.obsm['X_froGS_def'] = Fdef
    adata.obsm['X_froGS_imp'] = Fimp

    ad_def = run_pipeline(adata, 'X_froGS_def', 'FRoGS_default', args.outdir)
    ad_imp = run_pipeline(adata, 'X_froGS_imp', 'FRoGS_improved', args.outdir)
    joint = np.hstack([adata.obsm['X_pca'][:, :20], adata.obsm['X_froGS_imp']])
    adata.obsm['X_joint'] = joint
    ad_joint = run_pipeline(adata, 'X_joint', 'Joint', args.outdir)

    # Signature scoring
    score_cols = []
    for lineage, genes in marker_genes.items():
        sn = lineage.replace('-','_') + '_score'
        sc.tl.score_genes(adata, gene_list=genes, score_name=sn, use_raw=True)
        score_cols.append(sn)
    adata.obs[score_cols].to_csv(os.path.join(args.outdir,'signature_scores.csv'))
    for ad, lab in [(ad_pca,'PCA'),(ad_def,'FRoGS_default'),(ad_imp,'FRoGS_improved'),(ad_joint,'Joint')]:
        for c in score_cols:
            ad.obs[c] = adata.obs[c].values
        sc.pl.umap(ad, color=score_cols, save=f'_signature_{lab}.png', show=False)
        plt.close('all')

    # Additional exploratory per method
    method_map = {'PCA': ad_pca, 'FRoGS_default': ad_def,
                  'FRoGS_improved': ad_imp, 'Joint': ad_joint}
    for lab, ad in method_map.items():
        raw_ad = raw_master.copy()
        raw_ad.obs['clusters'] = ad.obs['clusters']
        # Violin: markers
        with plt.rc_context({'figure.figsize': (4.5,3)}):
            sc.pl.violin(raw_ad, ['CD79A','MS4A1'], groupby='clusters', save=f'_CD79A_MS4A1_violin_{lab}.png', show=False)
        # Violin: QC
        with plt.rc_context({'figure.figsize': (4.5,3)}):
            sc.pl.violin(raw_ad, ['n_genes','percent_mito'], groupby='clusters', stripplot=False, inner='box', save=f'_n_genes_percent_mito_violin_{lab}.png', show=False)
        # Matrixplot: scaled by var
        sc.pl.matrixplot(raw_ad, marker_genes, groupby='clusters', dendrogram=True, cmap='Blues', standard_scale='var', save=f'_matrixplot_scaled_{lab}.png', show=False)
        # Matrixplot: z-score layer
        raw_ad.layers['scaled'] = sc.pp.scale(raw_ad.copy(), copy=True).X
        sc.pl.matrixplot(raw_ad, marker_genes, groupby='clusters', dendrogram=True, layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', save=f'_matrixplot_zscore_{lab}.png', show=False)
        # "Heatmap" via matrixplot raw
        sc.pl.matrixplot(raw_ad, marker_genes, groupby='clusters', cmap='viridis', swap_axes=False, dendrogram=False, save=f'_heatmap_matrix_{lab}.png', show=False)
        # "Heatmap" via matrixplot scaled
        sc.pl.matrixplot(raw_ad, marker_genes, groupby='clusters', layer='scaled', vmin=-2, vmax=2, cmap='RdBu_r', swap_axes=True, dendrogram=False, figsize=(11,4), save=f'_heatmap_scaled_matrix_{lab}.png', show=False)
        # Tracksplot
        sc.pl.tracksplot(raw_ad, marker_genes, groupby='clusters', dendrogram=False, save=f'_tracksplot_{lab}.png', show=False)

    print('Completed all plots and outputs in', args.outdir)

