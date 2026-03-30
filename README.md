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

**NEWT** reimagines the FRoGS (Chen et al., 2024) framework within a new architecture that integrates multiple embedding sources (GO, ARCHS4, MSigDB, CellNet, DoRothEA, CollecTRI, and PPI), enabling multimodal fusion and scalable cross-omics prediction.

---

## ⚙️ Installation

### 1️⃣ Install Git LFS before cloning

This repository uses **Git Large File Storage (Git LFS)** for large embedding and data files.  
You must install Git LFS **before cloning** the repository to ensure all files download correctly.

**Option 1, install via mamba or conda**
```bash
mamba install -c conda-forge git-lfs
# or
conda install -c conda-forge git-lfs
git lfs install
```

**Option 2, install system-wide (recommended for WSL or cluster environments)**
```bash
sudo apt update
sudo apt install git-lfs -y
git lfs install
```

**Verify installation**
```bash
git lfs --version
```

---

### 2️⃣ Clone repository and create the environment
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

---

### 3️⃣ Install NEWT 
```bash
python -m pip install -e .

mamba install -c conda-forge scanpy seaborn matplotlib scikit-learn pandas numpy gensim
mamba install -c conda-forge leidenalg
```

---

### ⚠️ If you cloned before installing Git LFS
```bash
git lfs pull
git restore --source=HEAD :/
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
│       ├── export_graphml_ATC_subnetworks_v5_shRNA.py
│       ├── gene_vec_model_cellnet_tf_network.py
│       ├── gene_vec_model_dorothea_network.py
│       ├── gene_vec_model_collectri_network.py
│       └── gene_vec_model_msigdB_bundle.py
│
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
│
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
- `gene_to_go_top.csv`
- `term2gene_id.csv`

### Metadata files
- `tissue_specific.txt`
- `compound_list_shRNA.txt`
- `cpd_gene_pairs.csv`
- `L1000_PhaseI_and_II.csv`

---

# Training Embeddings

This section describes how to generate NEWT’s embedding CSVs for CellNet, DoRothEA, CollecTRI, and MSigDB. Each script trains a Word2Vec model on network or gene set “sentences” and writes a comma-separated file with one row per gene.

## cellnet human TF network

The CellNet workflow filters TF→target edges by `zscore` and absolute `correlation`, then learns embeddings from the remaining edges. Input CSV must contain `TF`, `Target`, `zscore`, and `correlation` columns, all using Entrez IDs. Output is `<output_prefix>_entrez_embeddings.csv`.

```bash
# minimal wrapper
python newt/scripts/gene_vec_model_cellnet_tf_network.py --input_csv data/human_tf_network_cellnet_converted_entrez.csv \
  --output_prefix ../data/cellnet_filtered

# with thresholds and training params
python newt/scripts/gene_vec_model_cellnet_tf_network.py \
  --input_csv data/human_tf_network_cellnet_converted_entrez.csv \
  --output_prefix data/cellnet_filtered \
  --zscore_thr 3 \
  --corr_thr 0.3 \
  --vector_size 128 \
  --window 5 \
  --epochs 10
```

## dorothea human TF network

DoRothEA uses both the confidence grade and mode of regulation. Confidence letters are mapped to numeric weights, which are multiplied by `mor`. Negative regulation is encoded by prefixing targets with `inhib_`. Sentences begin with the TF token followed by repeated target tokens based on a replication rule. Output is `<output_prefix>_entrez_embeddings.csv`.

```bash
python newt/scripts/gene_vec_model_dorothea_network.py \
  --input_csv data/dorothea_network_human_converted_entrez.csv \
  --output_prefix data/dorothea_embeddings \
  --replication_factor 10 \
  --weight_thr 0.0 \
  --vector_size 128 \
  --window 5 \
  --epochs 10
```

## collectri human TF network

CollecTRI uses the mode of regulation as the effective edge weight. Negative edges are marked with `inhib_`. Sentences start with the source TF, then repeated target tokens based on `abs(mor) * replication_factor`. Output is `<output_prefix>_entrez_embeddings.csv`.

```bash
python newt/scripts/gene_vec_model_collectri_network.py \
  --input_csv data/collectri_network_human_converted_entrez.csv \
  --output_prefix data/collectri_embeddings \
  --replication_factor 10 \
  --weight_thr 0.0 \
  --vector_size 128 \
  --window 5 \
  --epochs 10
```

## msigdb bundle embeddings

Provide one or more `.gmx` files. Each column in a `.gmx` file is treated as a gene set sentence, and all sets across files are combined to train embeddings. The script writes two files, symbols and Entrez, when `--convert` is set. Entrez conversion requires `mygene`.

```bash
# install once if you want Entrez conversion
pip install mygene

# folder of .gmx files
# install once if you want Entrez conversion
pip install mygene

# folder of .gmx files
python gene_vec_model_msigdB_bundle.py \
  --input ../data/ \
  --outfile ../data/msigdb_bundle_embeddings \
  --vector_size 256 \
  --window 5 \
  --epochs 10 \
  --convert

# outputs:
#   ../data/msigdb_bundle_embeddings_symbol.csv
#   ../data/msigdb_bundle_embeddings_entrez.csv
t

# outputs:
#   ../data/msigdb_bundle_embeddings_symbol.csv
#   ../data/msigdb_bundle_embeddings_entrez.csv
```

## notes and tips

- All TF network inputs must already use Entrez IDs for TF and target gene identifiers. The DoRothEA script translates confidence letters to numeric weights internally and encodes inhibitory edges, so you do not need to pre-process those aspects.
- The CellNet script requires both `zscore` and `correlation` fields, and applies thresholds before training. This step reduces noise and defines the training corpus.
- All four scripts write a single CSV per run containing one row per gene token and the embedding vector, which is exactly what the classifier and L1000 model loaders expect.


---

## 🚀 Usage Examples

All examples assume execution from the repo root (`NEWT/`).  
Adjust `../` prefixes if running from within a subdirectory.

---

### 🧩 A) Fusion classifier
Trains a multimodal attention-based fusion model that integrates GO, MSigDB, CellNet, and PPI embeddings to classify tissue or lineage signatures.

```bash
cd newt

newt classifier \
  --outdir results/classifier \
  --fusion_method attention \
  --fusion_epochs 30 \
  --fusion_patience 5 \
  --batch_size 16 \
  --default_file data/gene_vec_go_256.csv \
  --archs4_file data/gene_vec_archs4_256.csv \
  --ppi_file data/learned_gene_embeddings_go_graph.csv \
  --msigdb_file data/msigdb_bundle_embeddings_entrez.csv \
  --cellnet_file data/cellnet_filtered_entrez_embeddings.csv \
  --dorothea_file data/dorothea_embeddings_entrez_embeddings.csv \
  --collectri_file data/collectri_embeddings_entrez_embeddings.csv \
  --cellnet_dim 256 \
  --tissue_file data/tissue_specific.txt
```

**Or** 

```bash
python newt/scripts/classifier_improved_fusion_v13_collectri_dorothea_no_strict.py \
  --outdir results/classifier \
  --fusion_method attention \
  --fusion_epochs 30 \
  --fusion_patience 5 \
  --batch_size 16 \
  --default_file data/gene_vec_go_256.csv \
  --archs4_file data/gene_vec_archs4_256.csv \
  --ppi_file data/learned_gene_embeddings_go_graph.csv \
  --msigdb_file data/msigdb_bundle_embeddings_entrez.csv \
  --cellnet_file data/cellnet_filtered_entrez_embeddings.csv \
  --dorothea_file data/dorothea_embeddings_entrez_embeddings.csv \
  --collectri_file data/collectri_embeddings_entrez_embeddings.csv \
  --cellnet_dim 256 \
  --tissue_file data/tissue_specific.txt
```

**Outputs:**  
Stores training results in `results/fusion_YYYYmmdd_HHMM/`, including accuracy, loss plots, and fusion model weights.

---

### 🧬 B) Scanpy clustering (PBMC3k demo)
Runs a lightweight Scanpy pipeline to generate PCA, UMAP, and multimodal clustering visualizations, serving as a test for environment setup and plotting utilities.

```bash
python newt/scripts/scanpy_clustering_v16c.py \
  --data_dir data/ \
  --outdir results/newt_scanpy_figures
```

**Outputs:**  
Generates UMAP and PCA visualizations saved to `results/newt_scanpy_figures_YYYYMMDD_HHMM/`.  
Useful for validating plotting functions and cluster quality.

---

### 🧪 C) L1000 target model (shRNA)
Builds compound–target prediction models using L1000 gene expression data and multimodal embeddings.

```bash
newt l1000 \
  --cpdlist_file  data/compound_list_shRNA.txt \
  --target_file   data/cpd_gene_pairs.csv \
  --sig_file      data/L1000_PhaseI_and_II.csv \
  --perttype      shRNA \
  --epochs        60 \
  --emb_go        data/gene_vec_go_256.csv \
  --emb_archs4    data/gene_vec_archs4_256.csv \
  --emb_ppi       data/learned_gene_embeddings_go_graph.csv \
  --emb_msigdb    data/msigdb_bundle_embeddings_entrez.csv \
  --emb_cellnet   data/cellnet_filtered_entrez_embeddings.csv \
  --emb_collectri data/collectri_embeddings_entrez_embeddings.csv \
  --outdir        results/results_merged_multimodal_test2_shRNA \
  --modeldir      saved_model/merged_multimodal_test2_shRNA \
  --predict_batch_size 1024
```
**Or** 

```bash
python newt/scripts/l1000_model_v5_merged_embeddings_v8_collectri_fixed.py \
  --cpdlist_file  data/compound_list_shRNA.txt \
  --target_file   data/cpd_gene_pairs.csv \
  --sig_file      data/L1000_PhaseI_and_II.csv \
  --perttype      shRNA \
  --epochs        60 \
  --emb_go        data/gene_vec_go_256.csv \
  --emb_archs4    data/gene_vec_archs4_256.csv \
  --emb_cellnet   data/cellnet_filtered_entrez_embeddings.csv \
  --emb_collectri data/collectri_embeddings_entrez_embeddings.csv \
  --emb_msigdb    data/msigdb_bundle_embeddings_entrez.csv \
  --emb_ppi       data/learned_gene_embeddings_go_graph.csv \
  --outdir        results/results_merged_multimodal_test2_shRNA \
  --modeldir      saved_model/merged_multimodal_test2_shRNA \
  --predict_batch_size 1024
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

> ⚠️ This step requires prior aggregation of NEWT prediction outputs into compound–target CSV networks.

#### 1️⃣ Aggregate NEWT predictions → network CSVs
```bash
python aggregate_ct_networks_v2_fixed.py \
  --results-parent results/results_merged_multimodal_test2_shRNA/ \
  --export-dir ct_network_exports_shRNA \
  --cpd-gene-pairs data/cpd_gene_pairs.csv \
  --term2gene data/term2gene_id.csv \
  --agg-method min
```

#### 2️⃣ Export ATC subnetworks
```bash
python newt/scripts/export_graphml_ATC_subnetworks_v5_shRNA.py
```

---

### ⚠️ WHO ATC data requirement

This step requires the WHO ATC/DDD classification file (e.g. `WHO_ATC_DDD_2024-07-31.csv`).

- This file is **not distributed with the repository** due to licensing restrictions.
- Users must download it directly from:
  https://www.whocc.no/atc_ddd_index/
- Place the file in the `data/` directory before running the export script.

---

**Outputs:**  
Creates hierarchical ATC folders (levels 1–3) with top-10 filtered and full GraphMLs, along with corresponding compound–target CSV files and optional per-subnetwork target lists.

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

> **Kidder, B.L.** NEWT: Neural Embeddings for Wide-spectrum Targeting. GitHub (2025). [https://github.com/KidderLab/NEWT](https://github.com/KidderLab/NEWT)

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
