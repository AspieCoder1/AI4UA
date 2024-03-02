import click

from gnn4ua.datasets.loader import GeneralisationModes, Targets
from gnn4ua.local_feature_extractor import generate_local_motif


@click.group()
def cli():
    ...


@cli.command()
@click.option('--task',
              type=click.Choice(['Distributive', 'Modular', 'Meet_SemiDistributive',
                                 'Join_SemiDistributive', 'multilabel']),
              default='Distributive'
              )
@click.option('--generalisation_mode', type=click.Choice(['weak', 'strong']),
              default='strong')
def extract_motifs(task: str, generalisation_mode: str):
    task = Targets[task]
    generalisation_mode = GeneralisationModes[generalisation_mode]

    generate_local_motif(task, generalisation_mode)


if __name__ == '__main__':
    cli()
