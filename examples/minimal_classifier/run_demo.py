#!/usr/bin/env python3
"""Run a small deterministic NEWT fusion-classifier example on synthetic data.

The example creates two embedding modalities for 60 synthetic genes, writes the
same CSV/tissue formats used by NEWT, reloads them through the production data
loaders, and trains the production multimodal attention model. It verifies
installation and data flow only; its accuracy has no biological interpretation.
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from newt.scripts.classifier_improved_fusion_v13_collectri_dorothea_no_strict import (
    build_XyGene_for_combo_loose,
    build_fusion_model_modalities,
    load_embeddings_csv,
    load_tissue_file,
)


def write_embeddings(path: Path, genes: list[str], matrix: np.ndarray) -> None:
    """Write one NEWT-compatible embedding CSV with a header row."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene", *[f"dim_{i + 1}" for i in range(matrix.shape[1])]])
        for gene, vector in zip(genes, matrix):
            writer.writerow([gene, *[f"{value:.7f}" for value in vector]])


def make_demo_inputs(data_dir: Path, seed: int) -> tuple[Path, Path, Path, Path]:
    """Create deterministic GO, ARCHS4, CellNet, and tissue-label demo inputs."""
    rng = np.random.default_rng(seed)
    tissues = ("Brain", "Muscle", "Spleen")
    genes = [f"DEMO_GENE_{i:03d}" for i in range(60)]
    labels = np.repeat(tissues, 20)

    # Class-specific offsets make the tiny installation check quick and stable.
    go = rng.normal(0.0, 0.15, size=(60, 256)).astype(np.float32)
    archs4 = rng.normal(0.0, 0.15, size=(60, 256)).astype(np.float32)
    cellnet = rng.normal(0.0, 0.15, size=(60, 16)).astype(np.float32)
    for class_index in range(3):
        rows = labels == tissues[class_index]
        go[rows, class_index] += 3.0
        archs4[rows, class_index + 3] += 3.0
        cellnet[rows, class_index] += 3.0

    data_dir.mkdir(parents=True, exist_ok=True)
    go_path = data_dir / "gene_vec_go_demo.csv"
    archs4_path = data_dir / "gene_vec_archs4_demo.csv"
    cellnet_path = data_dir / "cellnet_demo.csv"
    tissue_path = data_dir / "tissue_specific_demo.txt"
    write_embeddings(go_path, genes, go)
    write_embeddings(archs4_path, genes, archs4)
    write_embeddings(cellnet_path, genes, cellnet)
    with tissue_path.open("w", encoding="utf-8") as handle:
        for tissue in tissues:
            members = [gene for gene, label in zip(genes, labels) if label == tissue]
            handle.write(",".join([tissue, *members]) + "\n")
    return go_path, archs4_path, cellnet_path, tissue_path


def main() -> None:
    """Generate demo inputs, train the NEWT fusion model, and save test metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("examples/minimal_classifier/results"),
        help="Directory for generated inputs, metrics, predictions, and model.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Small demo training epoch count.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for data and training.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    go_path, archs4_path, cellnet_path, tissue_path = make_demo_inputs(
        args.outdir / "data", args.seed
    )

    go = load_embeddings_csv(str(go_path), expected_dim=256)
    archs4 = load_embeddings_csv(str(archs4_path), expected_dim=256)
    cellnet = load_embeddings_csv(str(cellnet_path), expected_dim=16)
    default = {
        gene: np.concatenate([go[gene], archs4[gene]])
        for gene in sorted(set(go).intersection(archs4))
    }
    tissue_gene = load_tissue_file(str(tissue_path))
    X, y, genes = build_XyGene_for_combo_loose(
        tissue_gene,
        default,
        {},
        {},
        cellnet,
        {},
        {},
        ["default", "cellnet"],
        cellnet_dim=16,
    )

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, random_state=args.seed, stratify=y
    )
    tissue_order = sorted(set(y))
    tissue_to_index = {label: index for index, label in enumerate(tissue_order)}
    y_onehot = np.eye(len(tissue_order), dtype=np.float32)[
        [tissue_to_index[label] for label in y]
    ]
    X_default, X_cellnet = X[:, :512], X[:, 512:]

    model = build_fusion_model_modalities(
        ["default", "cellnet"],
        cellnet_dim=16,
        num_classes=len(tissue_order),
        fusion_method="attention",
    )
    model.fit(
        [X_default[train_idx], X_cellnet[train_idx]],
        y_onehot[train_idx],
        validation_data=(
            [X_default[test_idx], X_cellnet[test_idx]],
            y_onehot[test_idx],
        ),
        epochs=args.epochs,
        batch_size=12,
        verbose=0,
    )
    loss, accuracy = model.evaluate(
        [X_default[test_idx], X_cellnet[test_idx]], y_onehot[test_idx], verbose=0
    )
    probabilities = model.predict(
        [X_default[test_idx], X_cellnet[test_idx]], verbose=0
    )
    predicted = [tissue_order[index] for index in probabilities.argmax(axis=1)]

    prediction_path = args.outdir / "demo_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene", "observed_tissue", "predicted_tissue"])
        for index, label in zip(test_idx, predicted):
            writer.writerow([genes[index], y[index], label])

    model.save(args.outdir / "demo_fusion_model.keras")
    metrics = {
        "purpose": "installation and data-flow check; not a biological benchmark",
        "seed": args.seed,
        "epochs": args.epochs,
        "n_genes": int(len(X)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
    }
    (args.outdir / "demo_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Demo outputs written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
