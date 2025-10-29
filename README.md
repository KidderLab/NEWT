# NEWT — Neural Embeddings for Wide-spectrum Targeting
*A multimodal embedding framework for compound–target prediction and single-cell analysis.*

<p align="center">
  <img src="images/NEWT.png" alt="NEWT logo" width="300"/>
</p>

**NEWT** is a lightweight, script-first toolkit for training, fusing, and applying multimodal **gene embeddings** to:
- Classify tissue or lineage signatures
- Generate reproducible single-cell clustering figures
- Train compound–target models from L1000 data
- Export ATC subnetwork GraphMLs

NEWT reinterprets the principles of FRoGS (Chen et al., 2024) within an independently developed architecture that introduces novel cross-modal fusion mechanisms and an integrated computational pipeline for large-scale target prediction and network analysis.

**NEWT** generalizes and extends this idea — integrating GO, ARCHS4, MSigDB, CellNet, DoRothEA, CollecTRI, and PPI embeddings — for **multimodal fusion and cross-omics prediction**.

---

## ⚙️ Installation

### 1️⃣ Create the environment
```bash
git clone https://github.com/KidderLab/NEWT.git
cd NEWT
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
│       ├── scanpy_clustering_v16c.py
│       └── export_graphml_ATC_subnetworks_v5_shRNA.py
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
└── environment.yml
```

---

## 📦 Data expectations

Each embedding file is a comma-separated CSV:
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

## 🚀 Usage Examples

All examples assume execution from the repo root (`NEWT/`).  
Adjust `../` prefixes if running from within a subdirectory.

---

### 🧩 A) Fusion classifier
Trains a multimodal attention-based fusion model that integrates GO, MSigDB, CellNet, and PPI embeddings to classify tissue or lineage signatures.

```bash
python newt/scripts/classifier_improved_fusion_v13_collectri_dorothea_no_strict.py   --outdir results   --fusion_method attention   --fusion_epochs 30   --fusion_patience 5   --batch_size 16   --default_file ../data/gene_vec_go_256.csv   --archs4_file ../data/gene_vec_archs4_256.csv   --ppi_file ../data/learned_gene_embeddings_go_graph.csv   --msigdb_file ../data/msigdb_bundle_embeddings_entrez.csv   --cellnet_file ../data/cellnet_filtered_entrez_embeddings.csv   --dorothea_file ../data/dorothea_embeddings_entrez_embeddings.csv   --collectri_file ../data/collectri_embeddings_entrez_embeddings.csv   --cellnet_dim 256   --tissue_file data/tissue_specific.txt
```

**Outputs:**  
Stores training results in `results/fusion_YYYYmmdd_HHMM/`, including accuracy, loss plots, and fusion model weights.

---

### 🧬 B) Scanpy clustering (PBMC3k demo)
Runs a lightweight Scanpy pipeline to generate PCA, UMAP, and multimodal clustering visualizations, serving as a test for environment setup and plotting utilities.

```bash
python newt/scripts/scanpy_clustering_v16c.py   --data_dir data/   --outdir results/newt_scanpy_figures
```

**Outputs:**  
Generates UMAP and PCA visualizations saved to `results/newt_scanpy_figures_YYYYMMDD_HHMM/`.  
Useful for validating plotting functions and cluster quality.

---

### 🧪 C) L1000 target model (shRNA)
Builds compound–target prediction models using L1000 gene expression data and multimodal embeddings.

```bash
python newt/scripts/l1000_model_v5_merged_embeddings_v8_collectri_fixed.py   --cpdlist_file  ../data/compound_list_shRNA.txt   --target_file   ../data/cpd_gene_pairs.csv   --sig_file      ../data/L1000_PhaseI_and_II.csv   --perttype      shRNA   --epochs        60   --emb_go        ../data/gene_vec_go_256.csv   --emb_archs4    ../data/gene_vec_archs4_256.csv   --emb_cellnet   ../data/cellnet_filtered_entrez_embeddings.csv   --emb_collectri ../data/collectri_embeddings_entrez_embeddings.csv   --emb_msigdb    ../data/msigdb_bundle_embeddings_entrez.csv   --emb_ppi       ../data/learned_gene_embeddings_go_graph.csv   --outdir        ../results/results_merged_multimodal_test2_shRNA   --modeldir      ../saved_model/merged_multimodal_test2_shRNA   --predict_batch_size 1024
```

**Outputs:**  
Creates ranked compound–target prediction files (`BRD-XXXX@CELL_shRNA.txt`) in `--outdir`, and saves model weights in `--modeldir`.

---

### 📈 D) Recall metrics
Computes recall and precision metrics for all result folders to benchmark performance across multimodal fusion variants.

```bash
python newt/scripts/recall_improved_shRNA_merged_metrics_cell_line_v8.py
```

**Outputs:**  
Generates summary tables of recall, precision, and AUC scores for each experiment under `results/`.

---

### 📦 E) ATC Subnetwork Export (shRNA)
Generates WHO ATC–classified GraphML subnetworks and per-subnetwork target lists from shRNA compound–target predictions.

```bash
# Export ATC subnetworks
python newt/scripts/export_graphml_ATC_subnetworks_v5_shRNA.py

# Optional: move into export folder and copy per-subnetwork target lists
cd ct_network_exports_ATC_subnetworks_shRNA
python copy_graphml_targets.py
```

**Outputs:**  
Creates hierarchical ATC folders (levels 1–3) with top‑10 filtered and full GraphMLs plus corresponding CSVs.

---

## 🧰 Troubleshooting

**Qt/xcb plugin errors:**  
```
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
```

**Missing files:**  
The script logs missing embeddings or metadata if paths are incorrect.

---

## 🧾 License

MIT License © 2025 Kidder Lab.  
FRoGS (© 2024 Chen *et al.*) is licensed under MIT — attribution included.  
Source: [https://github.com/chenhcs/FRoGS](https://github.com/chenhcs/FRoGS)

---

## 🧩 Citation

If you use NEWT in your research, please cite:

> Chen, H., King, F.J., Zhou, B. *et al.* Drug target prediction through deep learning functional representation of gene signatures. *Nature Communications* **15**, 1853 (2024). [https://doi.org/10.1038/s41467-024-46089-y](https://doi.org/10.1038/s41467-024-46089-y)

and acknowledge this repository as:

> **Kidder, B.L.** NEWT: Neural Embeddings for Wide-spectrum Targeting. GitHub (2025). [https://github.com/benjohnsonlab/NEWT](https://github.com/benjohnsonlab/NEWT)

---

## 💡 Acknowledgments

We thank the authors of FRoGS for releasing their tools and pretrained embeddings under the MIT license.  
Their work inspired NEWT’s multimodal embedding design philosophy.

> FRoGS GitHub: [https://github.com/chenhcs/FRoGS](https://github.com/chenhcs/FRoGS)

## 🧩 Contact

For academic collaboration or inquiries:  
**Benjamin Kidder, PhD**  
Associate Professor, Department of Oncology  
Karmanos Cancer Institute / Wayne State University
