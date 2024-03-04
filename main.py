import click

from experiments.run import run_gnn_training
from gnn4ua.datasets.loader import GeneralisationModes, Targets
from gnn4ua.local_feature_extractor import generate_local_motif
from gnn4ua.run_glgexplainer import run_glgexplainer


@click.group(name="GLGExplainer for Universal Algebras")
def cli():
    ...


@cli.command(
    help="Train BlackBoxGNN and save the results"
)
def train_gnns():
    run_gnn_training()

@cli.command(
    help="Extracts local motifs from the classifier to be used as a basis for "
         "explanations"
)
@click.option(
    '--task',
    type=click.Choice(
        ['Distributive', 'Modular', 'Meet_SemiDistributive', 'Join_SemiDistributive',
         'multilabel']
    ),
    default='Distributive',
    help='Task to extract motifs for'
              )
@click.option(
    '--generalisation_mode',
    type=click.Choice(['weak', 'strong']),
    default='strong',
    help='Generalisation mode used to train the GNN'
)
@click.option("--n_epochs", type=int, default=100,
              help="Number of training epochs for PGExplainer")
def extract_motifs(task: str, generalisation_mode: str, n_epochs: int):
    task = Targets[task]
    generalisation_mode = GeneralisationModes[generalisation_mode]

    generate_local_motif(task, generalisation_mode, n_epochs)


@cli.command(
    help="Train GLGExplainer"
)
@click.option(
    '--task',
    type=click.Choice(
        ['Distributive', 'Modular', 'Meet_SemiDistributive', 'Join_SemiDistributive',
         'multilabel']
    ),
    default='Distributive',
    help='Task to extract motifs for'
              )
@click.option(
    '--generalisation_mode',
    type=click.Choice(['weak', 'strong']),
    default='strong',
    help='Generalisation mode used to train the GNN'
)
def train_explainer(task: str, generalisation_mode: str):
    task = Targets[task]
    generalisation_mode = GeneralisationModes[generalisation_mode]

    run_glgexplainer(task, generalisation_mode)


if __name__ == '__main__':
    cli()
