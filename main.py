from typing import Literal

import click

from experiments.run import run_gnn_training
from gnn4ua.datasets.loader import GeneralisationModes, Targets
from gnn4ua.local_feature_extractor import generate_local_motif
from gnn4ua.run_glgexplainer import run_glgexplainer


@click.group(name="GLGExplainer for Universal Algebras")
def cli():
    ...


@cli.command(
    help="Train BlackBoxGNN and save the results_binary"
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
        ['Distributive', 'Modular', 'Meet_SemiDistributive', 'Join_SemiDistributive', 'SemiDistributive',
         'multilabel', 'QuasiCancellitive']
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
@click.option("--n_epochs", type=int, default=1,
              help="Number of training epochs for PGExplainer")
@click.option("--seed", type=click.Choice(['102', '106', '270']), default='102')
@click.option("--explainer", type=click.Choice(['GNNExplainer', 'PGExplainer']),
              default='GNNExplainer')
def extract_motifs(task: str, generalisation_mode: str, n_epochs: int, seed: str,
                   explainer: str):
    task = Targets[task]
    generalisation_mode = GeneralisationModes[generalisation_mode]

    generate_local_motif(task, generalisation_mode, n_epochs, seed=seed,
                         explainer=explainer)


@cli.command(
    help="Train GLGExplainer"
)
@click.option(
    '--task',
    type=click.Choice(
        ['Distributive', 'Modular', 'Meet_SemiDistributive', 'Join_SemiDistributive', 'SemiDistributive',
         'multilabel', 'QuasiCancellitive']
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
@click.option("--seed", type=click.Choice(['102', '106', '270']), default='102')
@click.option("--explainer", type=click.Choice(['GNNExplainer', 'PGExplainer']), default='GNNExplainer')
@click.option("--n-prototypes", type=int, default=8)
@click.option("--n-motifs", type=click.IntRange(min=1, max=5), default=5)
def train_explainer(task: str, generalisation_mode: str,
                    seed: Literal['102', '106', '270'], explainer: str,
                    n_prototypes: int, n_motifs: int) -> None:
    task = Targets[task]
    generalisation_mode = GeneralisationModes[generalisation_mode]

    run_glgexplainer(task, generalisation_mode, seed, explainer, n_prototypes, n_motifs)


if __name__ == '__main__':
    cli()
