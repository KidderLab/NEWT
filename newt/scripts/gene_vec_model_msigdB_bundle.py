#!/usr/bin/env python
#python gene_vec_model_msigdB_bundle.py --input_folder /path/to/gmx_files --outfile msigdb_bundle_embeddings --vector_size 256 --window 5 --epochs 10 --convert

"""
Generate gene embeddings from one or more MSigDB-style gene set files (.gmx).

This script processes all .gmx files from a given input (file or folder). The input files
are expected to be formatted as follows:
  - The first row contains gene set names (one per column).
  - The second row contains "NA" values (which are ignored).
  - The third row onward contains gene symbols associated with each gene set (uneven rows per column).

Each gene set (i.e. each column) in each file is treated as a "sentence" (a list of gene symbols).
All sentences from all files are combined, and a Word2Vec model is trained on the resulting corpus.
Two output files are generated:
  1. One with gene symbols as identifiers.
  2. One with corresponding Entrez IDs (if --convert is provided).

Usage examples:
  Process a folder:
    python gene_vec_model_msigdB_bundle.py --input_folder /path/to/gmx_files --outfile msigdb_bundle_embeddings --vector_size 256 --window 5 --epochs 10 --convert
  
  Process a single file:
    python gene_vec_model_msigdB_bundle.py --input hallmark_gene_sets.gmx --outfile msigdb_bundle_embeddings --vector_size 256 --window 5 --epochs 10 --convert
"""

import argparse
import os
import glob
from gensim.models import Word2Vec
import numpy as np

def load_msigdB_file(input_file, delimiter="\t"):
    """
    Parse a single MSigDB hallmark file (.gmx) and return:
      - sentences: a list of lists, where each inner list contains gene symbols from one gene set (column).
      - gene_set_names: a list of gene set names from the header.
    """
    with open(input_file, 'r') as f:
        lines = [line.rstrip("\n").split(delimiter) for line in f if line.strip()]
    
    if len(lines) < 3:
        raise ValueError(f"The file {input_file} must have at least three rows (header, NA, and at least one row of genes).")
    
    gene_set_names = lines[0]
    # Skip second row (NA) and take rows from the third onward.
    data_rows = lines[2:]
    
    sentences = []
    for col in range(len(gene_set_names)):
        sentence = []
        for row in data_rows:
            if col < len(row):
                gene = row[col].strip()
                if gene and gene.upper() != "NA":
                    sentence.append(gene)
        if sentence:
            sentences.append(sentence)
    print(f"Processed {input_file}: Found {len(sentences)} gene sets.")
    return sentences, gene_set_names

def load_msigdB_inputs(input_path, delimiter="\t"):
    """
    If input_path is a directory, process all *.gmx files in it.
    Otherwise, process the single file.
    Returns a combined list of sentences from all files.
    """
    all_sentences = []
    if os.path.isdir(input_path):
        gmx_files = glob.glob(os.path.join(input_path, "*.gmx"))
        if not gmx_files:
            raise ValueError(f"No .gmx files found in directory: {input_path}")
        for file in gmx_files:
            sentences, _ = load_msigdB_file(file, delimiter)
            all_sentences.extend(sentences)
    elif os.path.isfile(input_path):
        sentences, _ = load_msigdB_file(input_path, delimiter)
        all_sentences.extend(sentences)
    else:
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")
    print(f"Total sentences from all files: {len(all_sentences)}")
    return all_sentences

def train_embeddings(sentences, vector_size=256, window=5, min_count=1, epochs=10):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4, epochs=epochs)
    return model.wv

def save_embeddings(wv, symbol_outfile, entrez_outfile=None, convert=False):
    with open(symbol_outfile, 'w') as f:
        for gene in wv.index_to_key:
            vec = wv[gene]
            line = gene + "," + ",".join(map(str, vec.tolist()))
            f.write(line + "\n")
    print(f"Symbol embeddings saved to {symbol_outfile}")
    
    if convert and entrez_outfile is not None:
        try:
            import mygene
        except ImportError:
            raise ImportError("mygene is required for conversion. Install it via: pip install mygene")
        mg = mygene.MyGeneInfo()
        symbols = wv.index_to_key
        results = mg.querymany(symbols, scopes="symbol", fields="entrezgene", species="human")
        symbol_to_entrez = {}
        for res in results:
            if res.get("notfound"):
                symbol_to_entrez[res["query"]] = res["query"]
            elif "entrezgene" in res:
                symbol_to_entrez[res["query"]] = str(res["entrezgene"])
            else:
                symbol_to_entrez[res["query"]] = res["query"]
        with open(entrez_outfile, 'w') as f:
            for gene in wv.index_to_key:
                entrez = symbol_to_entrez.get(gene, gene)
                vec = wv[gene]
                line = entrez + "," + ",".join(map(str, vec.tolist()))
                f.write(line + "\n")
        print(f"Entrez embeddings saved to {entrez_outfile}")

def main():
    parser = argparse.ArgumentParser(description="Generate gene embeddings from one or more MSigDB (.gmx) files.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input MSigDB hallmark file (.gmx) or folder containing *.gmx files.")
    parser.add_argument("--outfile", type=str, required=True,
                        help="Base output filename for embeddings (without extension).")
    parser.add_argument("--delimiter", type=str, default="\t",
                        help="Delimiter for the input file (default: tab).")
    parser.add_argument("--vector_size", type=int, default=256,
                        help="Embedding vector size (default: 256).")
    parser.add_argument("--window", type=int, default=5,
                        help="Window size for Word2Vec (default: 5).")
    parser.add_argument("--min_count", type=int, default=1,
                        help="Minimum count for Word2Vec (default: 1).")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10).")
    parser.add_argument("--convert", action="store_true",
                        help="If set, convert gene symbols to Entrez IDs and output a separate file.")
    args = parser.parse_args()

    sentences = load_msigdB_inputs(args.input, delimiter=args.delimiter)
    wv = train_embeddings(sentences, vector_size=args.vector_size, window=args.window, min_count=args.min_count, epochs=args.epochs)
    print(f"Trained embeddings for {len(wv.index_to_key)} genes")
    
    symbol_outfile = args.outfile + "_symbol.csv"
    entrez_outfile = args.outfile + "_entrez.csv" if args.convert else None
    
    save_embeddings(wv, symbol_outfile, entrez_outfile, convert=args.convert)

if __name__ == "__main__":
    main()

