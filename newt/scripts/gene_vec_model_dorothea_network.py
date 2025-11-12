#!/usr/bin/env python

"""
This script loads a Dorothea TF network file (with Entrez IDs) and trains Word2Vec embeddings.
It uses both the confidence score and mode of regulation (mor) to weight regulatory edges.
For each edge in the input CSV (which is expected to be comma-delimited with header):
    TF, confidence, Target, mor
the script computes an effective weight as:
    effective_weight = (numeric confidence) * (mor)
Then, using a user-supplied replication_factor, the target token is repeated based on:
    replication_count = max(round(abs(effective_weight) * replication_factor), 1)
If effective_weight is negative, the target token is prefixed with "inhib_".
Each transcription factor (TF) forms a sentence that begins with its own token,
followed by the repeated target tokens. Word2Vec is then trained on the collection
of sentences and the resulting embeddings (for all tokens) are saved as a CSV.
"""

import argparse
import csv
from gensim.models import Word2Vec

def load_tf_network_and_build_corpus(csv_file, replication_factor=10, weight_thr=0.0):
    """
    Loads the Dorothea network CSV file and builds a training corpus for Word2Vec.
    
    For each edge:
      - Converts the confidence letter (e.g. A, B, C) to a numeric value.
      - Converts the mor value to a float.
      - Computes effective_weight = (confidence value) * (mor).
      - Uses abs(effective_weight) * replication_factor (rounded, minimum one) as the replication count.
      - If effective_weight is negative, prefixes the target token with "inhib_".
    Edges with an absolute effective weight less than weight_thr are ignored.
    
    Returns:
      - A list of sentences (each sentence is a list of tokens) where each TF's sentence starts with its own token.
    """
    # Define mapping for confidence letters to numeric weights.
    confidence_map = {
        "A": 1.0,
        "B": 0.5,
        "C": 0.333333333,
        "D": 0.25,
        "E": 0.2
    }
    
    sentences_dict = {}  # key: TF token -> list of tokens (sentence)
    
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            tf_id = row.get("TF", "").strip()
            confidence_val = row.get("confidence", "").strip()
            tg_id = row.get("Target", "").strip()
            mor_str = row.get("mor", "").strip()
            if not tf_id or not tg_id or not mor_str:
                continue
            try:
                mor = float(mor_str)
            except Exception as e:
                # Skip row if the mor value is not valid.
                continue

            # Map confidence letter to numeric value (default to 0.2 if not found).
            conf_weight = confidence_map.get(confidence_val, 0.2)
            effective_weight = conf_weight * mor
            
            # Skip weak edges if they do not meet the threshold.
            if abs(effective_weight) < weight_thr:
                continue
            
            # Calculate how many times to repeat this token.
            replication_count = max(round(abs(effective_weight) * replication_factor), 1)
            
            # Determine token variant based on sign.
            if effective_weight >= 0:
                token = tg_id
            else:
                token = "inhib_" + tg_id
            
            # Build or update the sentence for the TF.
            if tf_id not in sentences_dict:
                sentences_dict[tf_id] = [tf_id]  # Start with the TF token.
            sentences_dict[tf_id].extend([token] * replication_count)
    
    # Return the list of sentences.
    return list(sentences_dict.values())

def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec embeddings from a Dorothea network file using confidence and mor values. "
                    "The input CSV should have columns: TF, confidence, Target, mor."
    )
    parser.add_argument("--input_csv", required=True,
                        help="Path to the Dorothea network CSV (e.g., dorothea_network_human_converted_entrez.csv).")
    parser.add_argument("--output_prefix", required=True,
                        help="Output prefix for the embeddings CSV (e.g., ../data/dorothea_embeddings).")
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

