#!/usr/bin/env python

"""
Example script to:
1) Load the CellNet/PACNet TF network with Entrez IDs (e.g. human_tf_network_cellnet_converted_entrez.csv).
2) Filter edges by zscore >= <threshold> and absolute correlation >= <threshold>.
3) Train Word2Vec embeddings on the remaining TF->Target edges.
4) Output embeddings to a CSV file.

Usage:
  python train_cellnet_entrez_embeddings.py \
    --input_csv human_tf_network_cellnet_converted_entrez.csv \
    --output_prefix cellnet_filtered \
    --zscore_thr 3 \
    --corr_thr 0.3 \
    --vector_size 128 \
    --window 5 \
    --epochs 10
"""

import argparse
import csv
from gensim.models import Word2Vec

def load_and_filter_tf_network(csv_file, zscore_thr=3.0, corr_thr=0.3):
    """
    Reads a TF network CSV with columns:
      - TF (Entrez ID)
      - Target (Entrez ID)
      - zscore
      - correlation
      - (optional) Regulation
    Filters rows by:
      - zscore >= zscore_thr
      - abs(correlation) >= corr_thr
    Returns a list of edges as pairs [TF_Entrez, Target_Entrez].
    """
    edges = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tf_id = row.get("TF", "").strip()     # Already Entrez ID
            tg_id = row.get("Target", "").strip() # Already Entrez ID
            if not tf_id or not tg_id:
                continue

            # Check zscore
            try:
                zval = float(row["zscore"])
            except:
                zval = 0.0

            # Check correlation
            try:
                cval = float(row["correlation"])
            except:
                cval = 0.0

            if zval >= zscore_thr and abs(cval) >= corr_thr:
                # Keep this edge
                edges.append([tf_id, tg_id])

    return edges

def main():
    parser = argparse.ArgumentParser(description="Filter CellNet TF network by zscore/correlation and train embeddings.")
    parser.add_argument("--input_csv", required=True,
                        help="Path to the Entrez-based TF network CSV (e.g. *_entrez.csv).")
    parser.add_argument("--output_prefix", default="cellnet_embeddings",
                        help="Output prefix for the embeddings CSV.")
    parser.add_argument("--zscore_thr", type=float, default=3.0,
                        help="Minimum zscore threshold (default 3).")
    parser.add_argument("--corr_thr", type=float, default=0.3,
                        help="Minimum absolute correlation threshold (default 0.3).")
    parser.add_argument("--vector_size", type=int, default=128,
                        help="Embedding vector size (default 128).")
    parser.add_argument("--window", type=int, default=5,
                        help="Word2Vec window size (default 5).")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of Word2Vec training epochs (default 10).")
    args = parser.parse_args()

    # 1) Load edges with filtering
    edges = load_and_filter_tf_network(
        csv_file=args.input_csv,
        zscore_thr=args.zscore_thr,
        corr_thr=args.corr_thr
    )
    print(f"Loaded {len(edges)} edges after filtering: zscore >= {args.zscore_thr} and |correlation| >= {args.corr_thr}")

    if not edges:
        print("No edges passed the threshold. Exiting.")
        return

    # 2) Train Word2Vec
    model = Word2Vec(
        edges,
        vector_size=args.vector_size,
        window=args.window,
        min_count=1,
        workers=4,
        epochs=args.epochs
    )
    wv = model.wv
    print(f"Trained embeddings for {len(wv.index_to_key)} unique Entrez gene IDs.")

    # 3) Save embeddings to CSV
    out_file = f"{args.output_prefix}_entrez_embeddings.csv"
    with open(out_file, "w") as f:
        for gene in wv.index_to_key:
            vec = wv[gene]
            line = gene + "," + ",".join(map(str, vec.tolist()))
            f.write(line + "\n")
    print(f"Saved embeddings to {out_file}")

if __name__ == "__main__":
    main()

