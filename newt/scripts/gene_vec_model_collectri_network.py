#!/usr/bin/env python

"""
This script loads a Collectri network file (with Entrez IDs) and trains Word2Vec embeddings.
The input CSV is expected to be comma-delimited with the following header:
    source, target, mor

For each edge:
  - The "mor" value (mode of regulation) is converted to a float.
  - Since no confidence score is provided, a default confidence of 1.0 is assumed.
  - The effective weight is computed as:
        effective_weight = mor
  - The target token is repeated based on:
        replication_count = max(round(abs(effective_weight) * replication_factor), 1)
  - If effective_weight is negative, the target token is prefixed with "inhib_".
Each source gene forms a sentence that starts with its own token, followed by the repeated target tokens.
Word2Vec is then trained on the resulting collection of sentences and the generated embeddings (for all tokens) are saved to a CSV.
"""

import argparse
import csv
from gensim.models import Word2Vec

def load_tf_network_and_build_corpus(csv_file, replication_factor=10, weight_thr=0.0):
    """
    Loads the Collectri network CSV file and builds a training corpus for Word2Vec.
    
    For each edge:
      - Reads the "source", "target", and "mor" values.
      - Computes effective_weight = mor (assuming a default confidence of 1.0).
      - Uses abs(effective_weight) * replication_factor (rounded, minimum one) as the replication count.
      - If effective_weight is negative, prefixes the target token with "inhib_".
    Edges with an absolute effective weight less than weight_thr are ignored.
    
    Returns:
      - A list of sentences (each sentence is a list of tokens) where each source gene's sentence starts with its own token.
    """
    sentences_dict = {}  # key: source token -> list of tokens (sentence)

    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            source_id = row.get("source", "").strip()
            target_id = row.get("target", "").strip()
            mor_str = row.get("mor", "").strip()
            if not source_id or not target_id or not mor_str:
                continue
            try:
                mor = float(mor_str)
            except Exception:
                continue

            # With no confidence column, assume confidence = 1.0 so that effective_weight equals mor.
            effective_weight = mor

            # Filter out edges that do not meet the minimum weight threshold.
            if abs(effective_weight) < weight_thr:
                continue

            # Determine how many times the target token is replicated.
            replication_count = max(round(abs(effective_weight) * replication_factor), 1)
            
            # Mark negative regulation by prefixing the target with "inhib_".
            if effective_weight >= 0:
                token = target_id
            else:
                token = "inhib_" + target_id

            # Start the sentence with the source gene token if not already present.
            if source_id not in sentences_dict:
                sentences_dict[source_id] = [source_id]
            # Append the target token replicated as needed.
            sentences_dict[source_id].extend([token] * replication_count)
    
    # Return all sentences as a list.
    return list(sentences_dict.values())

def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec embeddings from a Collectri network file using the 'mor' value for weighting edges. "
                    "The input CSV should have columns: source, target, mor."
    )
    parser.add_argument("--input_csv", required=True,
                        help="Path to the Collectri network CSV (e.g., collectri_network_human.csv).")
    parser.add_argument("--output_prefix", required=True,
                        help="Output prefix for the embeddings CSV (e.g., collectri_embeddings).")
    parser.add_argument("--replication_factor", type=float, default=10,
                        help="Factor to scale the replication count (default: 10).")
    parser.add_argument("--weight_thr", type=float, default=0.0,
                        help="Minimum absolute effective weight to include an edge (default: 0.0).")
    parser.add_argument("--vector_size", type=int, default=128,
                        help="Word2Vec embedding vector size (default: 128).")
    parser.add_argument("--window", type=int, default=5,
                        help="Word2Vec window size (default: 5).")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10).")
    args = parser.parse_args()

    # Build the training corpus.
    sentences = load_tf_network_and_build_corpus(
        csv_file=args.input_csv,
        replication_factor=args.replication_factor,
        weight_thr=args.weight_thr
    )
    print(f"Built a corpus with {len(sentences)} sentences from the network.")

    if not sentences:
        print("No data in the corpus after filtering. Exiting.")
        return

    # Train the Word2Vec model.
    model = Word2Vec(
        sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=1,
        workers=4,
        epochs=args.epochs
    )
    wv = model.wv
    print(f"Trained embeddings for {len(wv.index_to_key)} unique tokens.")

    # Save the embeddings to a CSV file.
    out_file = f"{args.output_prefix}_entrez_embeddings.csv"
    with open(out_file, "w", newline="") as fout:
        writer = csv.writer(fout)
        header = ["token"] + [f"v{i}" for i in range(args.vector_size)]
        writer.writerow(header)
        for token in wv.index_to_key:
            writer.writerow([token] + wv[token].tolist())
    print(f"Saved embeddings to {out_file}")

if __name__ == "__main__":
    main()

