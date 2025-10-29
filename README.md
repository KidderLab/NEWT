# NEWT — Neural Embeddings for Wide-spectrum Targeting

<p align="center">
  <img src="images/NEWT.png" alt="NEWT logo" width="300"/>
</p>

**NEWT** is a lightweight, script-first toolkit for training, fusing, and applying multimodal **gene embeddings** to:
- classify tissue or lineage signatures,
- train compound–target models from L1000 data,
- export ATC subnetwork GraphMLs,
- and generate reproducible single-cell clustering figures.

It extends the ideas of **FRoGS** (Chen *et al.*, 2024) but is implemented independently, with a distinct architecture and command-line workflow.

---

## 🔬 Background

Functional representation of gene signatures (FRoGS) introduced a framework for learning **gene embeddings** that capture biological co-functionality.  
**NEWT** generalizes and extends this idea — integrating GO, ARCHS4, MSigDB, CellNet, DoRothEA, CollecTRI, and PPI embeddings — for **multimodal fusion and cross-omics prediction**.

**Reference (FRoGS):**  
Chen, H., King, F.J., Zhou, B. *et al.* *Drug target prediction through deep learning functional representation of gene signatures.*  
**Nature Communications** 15, 1853 (2024). [https://doi.org/10.1038/s41467-024-46089-y](https://doi.org/10.1038/s41467-024-46089-y)  
Source: [https://github.com/chenhcs/FRoGS](https://github.com/chenhcs/FRoGS)

---

## ⚙️ Installation

### 1️⃣ Create the environment
```bash
# from repo root (NEWT/)
mamba env create -f environment.yml
mamba activate newt_env
# or:
conda env create -f environment.yml
conda activate newt_env
```

If you previously created `newt_env` and want a clean slate:
```bash
mamba deactivate
mamba env remove -n newt_env
mamba env create -f environment.yml
mamba activate newt_env
```

### 2️⃣ Install NEWT (editable mode)
```bash
pip install -e .
```

### 3️⃣ For headless plotting
If you run on a cluster or server with no display:
```bash
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
```

---

## 📂 Directory layout

```
NEWT/
├── newt/
│   └── scripts/
│       ├── classifier_improved_fusion_v13_collectri_dorothea_no_strict.py
│       ├── l1000_model_v5_merged_embeddings_v8_collectri_fixed.py
│       ├── recall_improved_shRNA_merged_metrics_cell_line_v8.py
│       └── scanpy_clustering_v16c.py
├── data/
│   ├── gene_vec_go_256.csv
│   ├── gene_vec_archs4_256.csv
│   ├── learned_gene_embeddings_go_graph.csv
│   ├── msigdb_bundle_embeddings_entrez.csv
│   ├── cellnet_filtered_entrez_embeddings.csv
│   ├── dorothea_embeddings_entrez_embeddings.csv
│   ├── collectri_embeddings_entrez_embeddings.csv
│   ├── tissue_specific.txt
│   ├── compound_list_shRNA.txt
│   ├── cpd_gene_pairs.csv
│   └── L1000_PhaseI_and_II.csv
├── results/
├── saved_model/
├── environment.yml
└── README.md
```

---

## 🚀 Usage Examples

> **All paths assume you run from the repo root (`NEWT/`).**  
> Adjust `../` prefixes if running from a subdirectory.

### 🧩 A) Fusion classifier
```bash
python newt/scripts/classifier_improved_fusion_v13_collectri_dorothea_no_strict.py   --outdir results   --fusion_method attention   --fusion_epochs 30   --fusion_patience 5   --batch_size 16   --default_file   ../data/gene_vec_go_256.csv   --archs4_file    ../data/gene_vec_archs4_256.csv   --ppi_file       ../data/learned_gene_embeddings_go_graph.csv   --msigdb_file    ../data/msigdb_bundle_embeddings_entrez.csv   --cellnet_file   ../data/cellnet_filtered_entrez_embeddings.csv   --dorothea_file  ../data/dorothea_embeddings_entrez_embeddings.csv   --collectri_file ../data/collectri_embeddings_entrez_embeddings.csv   --cellnet_dim 256   --tissue_file data/tissue_specific.txt
```

**Outputs:**  
Trains a multimodal attention-based fusion classifier and writes results under `results/fusion_YYYYmmdd_HHMM/`.

**Options:**
| Flag | Description |
|------|--------------|
| `--fusion_method` | Fusion type: `attention`, `concat`, or `mlp` |
| `--fusion_epochs` | Training epochs (default: 30) |
| `--fusion_patience` | Early-stop patience (default: 5) |
| `--batch_size` | Batch size (default: 16) |

---

### 🧪 B) L1000 target model (shRNA)
```bash
python newt/scripts/l1000_model_v5_merged_embeddings_v8_collectri_fixed.py   --cpdlist_file  ../data/compound_list_shRNA.txt   --target_file   ../data/cpd_gene_pairs.csv   --sig_file      ../data/L1000_PhaseI_and_II.csv   --perttype      shRNA   --epochs        60   --emb_go        ../data/gene_vec_go_256.csv   --emb_archs4    ../data/gene_vec_archs4_256.csv   --emb_cellnet   ../data/cellnet_filtered_entrez_embeddings.csv   --emb_collectri ../data/collectri_embeddings_entrez_embeddings.csv   --emb_msigdb    ../data/msigdb_bundle_embeddings_entrez.csv   --emb_ppi       ../data/learned_gene_embeddings_go_graph.csv   --outdir        ../results/results_merged_multimodal_test2_shRNA   --modeldir      ../saved_model/merged_multimodal_test2_shRNA   --predict_batch_size 1024
```

**Outputs:**  
- Writes ranked results `BRD-XXXX@CELL_shRNA.txt` in `--outdir`.  
- Saves trained models in `--modeldir`.

**Options:**  
`--perttype` (`shRNA` or `cDNA`), `--epochs`, `--lr`, `--weight_decay`, `--predict_batch_size`

---

### 📈 C) Recall metrics
```bash
python newt/scripts/recall_improved_shRNA_merged_metrics_cell_line_v8.py
```
Evaluates recall/precision metrics across result folders and writes summary CSVs.

---

### 🧬 D) Scanpy clustering (PBMC3k demo)
```bash
python newt/scripts/scanpy_clustering_v16c.py   --data_dir data/   --outdir results/newt_scanpy_figures
```
Generates timestamped folders with PCA, FRoGS-style embeddings (if available), and Joint plots.  
If embeddings are missing, NEWT falls back to PCA-only analysis and logs missing files.

---

## 📦 Data expectations

Each embedding file is a simple comma-separated CSV:
```
gene,val1,val2,val3,...
TP53,0.12,0.07,0.03,...
```

### Required filenames
- `gene_vec_go_256.csv`
- `gene_vec_archs4_256.csv`
- `learned_gene_embeddings_go_graph.csv`
- `msigdb_bundle_embeddings_entrez.csv`
- `cellnet_filtered_entrez_embeddings.csv`
- `dorothea_embeddings_entrez_embeddings.csv`
- `collectri_embeddings_entrez_embeddings.csv`

### Metadata files
- `tissue_specific.txt`
- `compound_list_shRNA.txt`
- `cpd_gene_pairs.csv`
- `L1000_PhaseI_and_II.csv`

---

## 🧰 Troubleshooting

**Qt/xcb plugin errors:**  
```
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
```

**MISSING messages:**  
The script will show which CSVs are missing. Check that filenames and paths are correct under `--data_dir`.

**Conda/pip solver conflicts:**  
Install everything from `conda-forge` first (`environment.yml`) and use pip only for extras.

---

## 🧾 License

**License:** MIT  
This repository is distinct from FRoGS but follows its general conceptual framework.  
FRoGS (© 2024 Chen *et al.*) is licensed under MIT — attribution included.

**FRoGS source:** [https://github.com/chenhcs/FRoGS](https://github.com/chenhcs/FRoGS)

---

## 🧩 Citation

If you use NEWT in your research, please cite:

> Chen, H., King, F.J., Zhou, B. *et al.* Drug target prediction through deep learning functional representation of gene signatures. *Nature Communications* **15**, 1853 (2024). [https://doi.org/10.1038/s41467-024-46089-y](https://doi.org/10.1038/s41467-024-46089-y)

and acknowledge this repository as:

> **Kidder, B.L.** NEWT: Neural Embeddings for Wide-spectrum Targeting. GitHub (2025). [https://github.com/benjohnsonlab/NEWT](https://github.com/benjohnsonlab/NEWT)

---

## 🤝 Contributing

Contributions and pull requests are welcome!  
Please ensure code passes style and tests before submission:

```bash
ruff check .
pytest -q
```

---

## 💡 Acknowledgments

We thank the authors of **FRoGS** for releasing their tools and pretrained embeddings under the MIT license.  
Their work inspired NEWT’s multimodal embedding design philosophy.

> FRoGS GitHub: [https://github.com/chenhcs/FRoGS](https://github.com/chenhcs/FRoGS)

---

# 📦 E) ATC Subnetwork Export (shRNA)

This step builds ATC-organized GraphML subnetworks from your shRNA compound–target CSVs and also writes per-subnetwork target lists.

### What it does (at a glance)

- Creates a 3-level ATC folder hierarchy and GraphMLs:
  - **Level-1** (A, B, C, …) → filtered “top-10” network + CSVs  
  - **Level-2** (e.g., A01, B02…) → filtered “top-10” network + CSVs  
  - **Level-3** (e.g., L01A, A02B…) → **both** filtered “top-10** and **unfiltered “full” networks + CSVs
- Writes `*_details.csv` with ATC metadata and edges, and `*_targets.csv` (no header) containing the unique targets in each subnetwork.

### Requirements

- Input CSVs in `./ct_network_exports_shRNA` with columns:  
  `Query Pert ID, Compound Name, Symbol, Probability`.
- WHO ATC table at `./data/WHO_ATC_DDD_2024-07-31.csv`. (Path configurable via `ATC_FILE` in the script.)
- Python packages: `pandas`, `networkx`.

### How “top-10 filtered” edges are chosen

For each ATC slice, we form the intersection of:
1. the **top-10 targets per compound** (by `Probability`), and  
2. the **top-10 compounds per target** (by `Probability`).

### Run

From the repo root:

```bash
# 1) Export ATC subnetworks (creates ./ct_network_exports_ATC_subnetworks_shRNA/)
python newt/scripts/export_graphml_ATC_subnetworks_v5_shRNA.py

# 2) (Optional) Move into the export folder and post-process
cd ct_network_exports_ATC_subnetworks_shRNA
python copy_graphml_targets.py
```

> Notes:
> - The script paths can be changed via the constants at the top:  
>   `INPUT_DIR`, `ATC_FILE`, `OUTPUT_DIR`. Defaults are:  
>   `INPUT_DIR="./ct_network_exports_shRNA"`,  
>   `ATC_FILE="../data/WHO_ATC_DDD_2024-07-31.csv"`,  
>   `OUTPUT_DIR="./ct_network_exports_ATC_subnetworks_shRNA"`.
> - Each level writes a GraphML (`*.graphml`) plus:  
>   - `*_details.csv` with columns `[atc_code, atc_name, drug_name, compound, target]`, and  
>   - `*_targets.csv` (no header) with deduplicated targets for that subnetwork.

### Output example (abridged)

```
ct_network_exports_ATC_subnetworks_shRNA/
└── A_Alimentary_tract_and_metabolism/
    ├── A_Alimentary_tract_and_metabolism.graphml
    ├── A_Alimentary_tract_and_metabolism_details.csv
    ├── A_Alimentary_tract_and_metabolism_targets.csv
    └── subgroups/
        └── A01_Stomatological_preparations/
            ├── A01_Stomatological_preparations.graphml
            ├── A01_Stomatological_preparations_details.csv
            ├── A01_Stomatological_preparations_targets.csv
            └── subgroups/
                └── A01A_Stomatological_agents/
                    ├── A01A_Stomatological_agents.graphml         # filtered top-10
                    ├── A01A_Stomatological_agents_details.csv
                    ├── A01A_Stomatological_agents_targets.csv
                    ├── A01A_Stomatological_agents_full.graphml    # full (unfiltered)
                    ├── A01A_Stomatological_agents_full_details.csv
                    └── A01A_Stomatological_agents_full_targets.csv
```
