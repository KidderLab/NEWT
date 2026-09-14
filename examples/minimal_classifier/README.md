# Minimal NEWT fusion-classifier example

This example is a fast installation and data-flow check using 60 synthetic
genes in three synthetic tissue groups. It calls NEWT's production embedding
loaders, loose-universe data builder, and multimodal attention classifier. The
synthetic accuracy is not a biological benchmark and does not reproduce a
manuscript performance estimate.

## Run from the repository root

After creating and activating the documented `newt_env` environment:

```bash
python examples/minimal_classifier/run_demo.py
```

The default run uses a fixed random seed, three epochs, and CPU execution. It
usually completes in under two minutes after TensorFlow has loaded. A custom
output directory or seed can be supplied:

```bash
python examples/minimal_classifier/run_demo.py \
  --outdir examples/minimal_classifier/results \
  --epochs 3 \
  --seed 7
```

## Expected outputs

```text
examples/minimal_classifier/results/
├── data/
│   ├── gene_vec_go_demo.csv
│   ├── gene_vec_archs4_demo.csv
│   ├── cellnet_demo.csv
│   └── tissue_specific_demo.txt
├── demo_fusion_model.keras
├── demo_metrics.json
└── demo_predictions.csv
```

A successful run prints a JSON summary containing `n_genes: 60` and writes
the paths above. Exact accuracy can vary slightly across TensorFlow platforms
despite fixed seeds.
