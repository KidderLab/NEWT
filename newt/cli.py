import click
from ._utils import run_vendored, run_external

CTX = dict(help_option_names=["-h", "--help"], ignore_unknown_options=True, allow_extra_args=True)

@click.group(context_settings=CTX)
@click.version_option(version="0.1.1", prog_name="NEWT")
def main():
    "NEWT — Network-Embedded Weighting Toolkit: unified CLI for multimodal perturbation analytics."

def passthrough(name, args):
    code = run_vendored(name, list(args))
    raise SystemExit(code)

@main.command(context_settings=CTX, name="classifier")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def classifier_cmd(args):
    "Run the classifier + fusion pipeline (attention supported)."
    passthrough("classifier_improved_fusion_v13_collectri_dorothea_no_strict.py", args)

@main.command(context_settings=CTX, name="l1000")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def l1000_cmd(args):
    "Run the L1000 merged-embeddings model with collectri fix."
    passthrough("l1000_model_v5_merged_embeddings_v8_collectri_fixed.py", args)

@main.command(context_settings=CTX, name="atc-shrna")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def atc_shrna_cmd(args):
    "Export ATC subnetworks (shRNA) to GraphML/CSV."
    passthrough("export_graphml_ATC_subnetworks_v5_shRNA.py", args)

@main.command(context_settings=CTX, name="tsne-combos")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def tsne_combos_cmd(args):
    "Generate t-SNEs across embedding combos and per-GO mosaics."
    passthrough("plot_tsne_combos_v3.py", args)

@main.command(context_settings=CTX, name="recall")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def recall_cmd(args):
    "Run recall_improved_shRNA_merged_metrics_cell_line_v8.py if present in CWD or PATH."
    code = run_external("recall_improved_shRNA_merged_metrics_cell_line_v8.py", list(args))
    raise SystemExit(code)

@main.command(context_settings=CTX, name="cluster")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def cluster_cmd(args):
    "Run Scanpy-based clustering and plotting."
    code = run_external("scanpy_clustering_v16c.py", list(args))
    from ._utils import run_vendored
    code = run_vendored("scanpy_clustering_v16c.py", list(args))
    raise SystemExit(code)

if __name__ == "__main__":
    main()
