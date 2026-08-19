#!/usr/bin/env python3
"""
aggregate_ct_networks_v2_fixed.py

Aggregates FRoGS L1000 predictions per‐compound across cell lines,
builds a consensus target list by normalized‐rank aggregation
(min, mean, median, or Borda), filters to Probability ≥ 0.8 & top 5%,
annotates with perturbation metadata, gene symbols, and gene
descriptions (via mygene.info), and writes one CSV per input subdir.
"""

import os
import glob
import argparse
import pandas as pd
import mygene

# initialize mygene client
mg = mygene.MyGeneInfo()

def load_cpd_meta(path):
    """Load compound identifiers, names, and Broad annotations from a delimited table."""
    df = pd.read_csv(path, sep=None, engine='python', dtype=str)
    return (
        df[['pert_id','pert_iname','Broad_target']].drop_duplicates()
          .rename(columns={
              'pert_id': 'Query Pert ID',
              'pert_iname': 'Compound Name',
              'Broad_target': 'Compound Broad Annotation'
          })
          .assign(compound=lambda d: d['Query Pert ID'])
    )

def load_symbol_map(path):
    """Load a unique Entrez-to-gene-symbol mapping from a delimited table."""
    df = pd.read_csv(path, sep=None, engine='python', dtype=str)
    df = df.rename(columns={'gene_id':'Entrez Gene ID','Symbol':'Symbol'})
    df['Entrez Gene ID'] = df['Entrez Gene ID'].astype(str)
    return df[['Entrez Gene ID','Symbol']].drop_duplicates()

def compute_norm_ranks(input_dir):
    """
    For each compound–cell‐line file, compute normalized rank per gene
    and record the compound ID.
    """
    parts = []
    for fn in glob.glob(os.path.join(input_dir, '*@*_*.txt')):
        base = os.path.basename(fn).rsplit('.', 1)[0]
        try:
            cpd, _ = base.split('@', 1)
        except ValueError:
            continue
        df = pd.read_csv(fn, sep='\t', usecols=['gene','score'])
        df = df.rename(columns={'gene':'Entrez Gene ID'})
        df['Entrez Gene ID'] = df['Entrez Gene ID'].astype(str)
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        n = len(df)
        df['norm_rank'] = (df.index + 1) / n
        df['compound'] = cpd
        parts.append(df[['compound','Entrez Gene ID','norm_rank']])
    if not parts:
        return pd.DataFrame(columns=['compound','Entrez Gene ID','norm_rank'])
    return pd.concat(parts, ignore_index=True)

def aggregate_consensus(pool, method):
    """
    Aggregate normalized ranks per compound+gene.
    """
    group_fields = ['compound','Entrez Gene ID']
    if method == 'min':
        cons = pool.groupby(group_fields, as_index=False)['norm_rank'].min()
    elif method == 'mean':
        cons = pool.groupby(group_fields, as_index=False)['norm_rank'].mean()
    elif method == 'median':
        cons = pool.groupby(group_fields, as_index=False)['norm_rank'].median()
    elif method == 'borda':
        cons = pool.groupby(group_fields, as_index=False)['norm_rank'].sum()
    else:
        raise ValueError(f"Unknown aggregation method: {method!r}")
    return cons.rename(columns={'norm_rank':'consensus_rank'})

def fetch_gene_descriptions(entrez_ids):
    """Query MyGene.info for descriptions associated with Entrez identifiers."""
    unique = list(dict.fromkeys(entrez_ids))
    out = {}
    for i in range(0, len(unique), 100):
        batch = unique[i:i+100]
        res = mg.getgenes(batch, fields='name', species='human')
        for entry in res:
            eid = str(entry.get('entrezgene') or entry.get('query'))
            out[eid] = entry.get('name','')
    return out

def process_one_dir(input_dir, meta, symb, outdir,
                    min_score=0.8, top_pct=5, agg_method='min'):
    """Aggregate one prediction directory, annotate targets, and write its network CSV."""
    # 1) collapse to max Probability per compound–gene
    parts = []
    for fn in glob.glob(os.path.join(input_dir, '*@*_*.txt')):
        base = os.path.basename(fn).rsplit('.', 1)[0]
        try:
            cpd, _ = base.split('@', 1)
        except ValueError:
            continue
        df = pd.read_csv(fn, sep='\t', usecols=['gene','score'])
        df = df.rename(columns={'gene':'Entrez Gene ID','score':'Probability'})
        df['Entrez Gene ID'] = df['Entrez Gene ID'].astype(str)
        df['compound'] = cpd
        parts.append(df)
    if not parts:
        print(f"[!] skipping {input_dir!r}, no prediction files found")
        return
    allp = pd.concat(parts, ignore_index=True)
    agg_score = allp.groupby(['compound','Entrez Gene ID'], as_index=False)['Probability'].max()

    # 2) build consensus normalized rank per compound
    pool = compute_norm_ranks(input_dir)
    cons = aggregate_consensus(pool, agg_method)

    # 3) merge scores + consensus_rank on both keys
    df = agg_score.merge(cons, on=['compound','Entrez Gene ID'], how='left')

    # 4) compute percentile within each compound
    df['rank2'] = df.groupby('compound')['consensus_rank'] \
                    .rank(method='first', ascending=True)
    df['n_genes'] = df.groupby('compound')['Entrez Gene ID'].transform('count')
    df['pct'] = df['rank2'] / df['n_genes'] * 100

    # 5) filter by Probability & top_pct
    df = df[(df['Probability'] >= min_score) & (df['pct'] <= top_pct)].copy()

    # 6) annotate with metadata + symbol
    out = df.merge(meta, on='compound', how='left') \
            .merge(symb, on='Entrez Gene ID', how='left')
    out['Gene Description'] = ''

    # 7) fetch descriptions and assign
    unique_ids = out['Entrez Gene ID'].dropna().tolist()
    desc_map   = fetch_gene_descriptions(unique_ids)
    out['Gene Description'] = out['Entrez Gene ID'].map(desc_map).fillna('')

    # 8) write final CSV
    final = out[[
        'Query Pert ID',
        'Compound Name',
        'Compound Broad Annotation',
        'Entrez Gene ID',
        'Symbol',
        'Gene Description',
        'Probability'
    ]]
    sub = os.path.basename(os.path.normpath(input_dir))
    fn = f"{sub}_ct_{agg_method}.csv"
    final.to_csv(os.path.join(outdir, fn), index=False)
    print(f"[+] wrote {fn} with {len(final):,} rows")

def main():
    """Parse command-line options and aggregate all selected prediction directories."""
    p = argparse.ArgumentParser()
    p.add_argument('--results-parent', required=True)
    p.add_argument('--export-dir',     required=True)
    p.add_argument('--cpd-gene-pairs', required=True)
    p.add_argument('--term2gene',      required=True)
    p.add_argument('--agg-method',
                   choices=['min','mean','median','borda'],
                   default='min')
    p.add_argument('--min-score', type=float, default=0.8)
    p.add_argument('--top-pct',   type=float, default=5)
    args = p.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)
    meta = load_cpd_meta(args.cpd_gene_pairs)
    symb = load_symbol_map(args.term2gene)

    subs = [os.path.join(args.results_parent, d)
            for d in os.listdir(args.results_parent)
            if os.path.isdir(os.path.join(args.results_parent, d))]
    if not subs:
        subs = [args.results_parent]

    for sd in sorted(subs):
        print(f"[*] Processing {sd}  agg={args.agg_method}")
        process_one_dir(
            sd, meta, symb, args.export_dir,
            min_score=args.min_score,
            top_pct=args.top_pct,
            agg_method=args.agg_method
        )

if __name__=='__main__':
    main()
