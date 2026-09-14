"""Command-line entry points for the packaged NEWT workflows.

Commands forward their arguments to the corresponding scripts bundled with
the package. Run ``newt COMMAND --help`` to inspect command-specific options.
"""

import click
from ._utils import run_vendored

CTX = dict(help_option_names=["-h", "--help"], ignore_unknown_options=True, allow_extra_args=True)

@click.group(context_settings=CTX)
@click.version_option(version="0.1.1", prog_name="NEWT")
def main():
    """Run the unified NEWT command-line interface."""

def passthrough(name, args):
    """Execute a bundled NEWT script and exit with its return code."""
    code = run_vendored(name, list(args))
    raise SystemExit(code)

@main.command(context_settings=CTX, name="classifier")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def classifier_cmd(args):
    """Run the classifier and multimodal-fusion pipeline."""
    passthrough("classifier_improved_fusion_v13_collectri_dorothea_no_strict.py", args)

@main.command(context_settings=CTX, name="l1000")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def l1000_cmd(args):
    """Run the L1000 target model using merged gene embeddings."""
    passthrough("l1000_model_v5_merged_embeddings_v8_collectri_fixed.py", args)

@main.command(context_settings=CTX, name="atc-shrna")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def atc_shrna_cmd(args):
    """Export shRNA-derived ATC subnetworks to GraphML and CSV."""
    passthrough("export_graphml_ATC_subnetworks_v5_shRNA.py", args)

@main.command(context_settings=CTX, name="tsne-combos")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def tsne_combos_cmd(args):
    """Generate t-SNE diagnostics across embedding combinations."""
    passthrough("plot_tsne_combos_v3.py", args)

@main.command(context_settings=CTX, name="recall")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def recall_cmd(args):
    """Run target-recall evaluation."""
    passthrough("recall_improved_shRNA_merged_metrics_cell_line_v8.py", args)

@main.command(context_settings=CTX, name="cluster")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def cluster_cmd(args):
    """Run Scanpy-based downstream clustering diagnostics and plots."""
    passthrough("scanpy_clustering_v16c.py", args)

if __name__ == "__main__":
    main()
