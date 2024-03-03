import os

import click
import numpy as np
import torch
from torch import nn
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import PGExplainer
from torch_geometric.explain.config import (
    ExplanationType,
    ModelConfig,
    ModelTaskLevel,
    ModelMode,
    ModelReturnType,
    MaskType,
)
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from gnn4ua.datasets.loader import (LatticeDataset, Targets,
                                    GeneralisationModes, )
from gnn4ua.models import BlackBoxGNN


def generate_motifs(model: nn.Module, train_data, test_data,
                    root: str = 'local_features/PGExplainer',
                    task: Targets = Targets.Distributive,
                    generalisation_mode: GeneralisationModes = GeneralisationModes.strong):
    """
    Uses PGExplainer to generate motif features from the datasets.

    :param model:
    :param data:
    :return:
    """
    path = f'{root}/{task}_{generalisation_mode}'

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    config = ModelConfig(
        task_level=ModelTaskLevel.graph,
        mode=ModelMode.multiclass_classification,
        return_type=ModelReturnType.raw,
    )

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=10),
        explanation_type=ExplanationType.phenomenon,
        model_config=config,
        edge_mask_type=MaskType.object,
    )

    assert isinstance(explainer.algorithm, PGExplainer)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1)

    for epoch in range(10):
        for train_sample in tqdm(train_loader):
            explainer.algorithm.train(
                epoch,
                model,
                train_sample.x,
                train_sample.edge_index,
                target=train_sample.y.float(),
                index=0,
                batch=train_sample.batch
            )

    explain_list_train = []
    explain_list_train_classes = []

    for train_sample in tqdm(train_loader):
        out = explainer(train_sample.x, train_sample.edge_index,
                        target=train_sample.y.float(),
                        batch=train_sample.batch,
                        index=0)

        explain_list_train.append(to_dense_adj(out.edge_index, edge_attr=out.edge_attr))
        explain_list_train_classes.append(torch.argmax(train_sample.y).item())

    np.savez_compressed(f'{path}/x_train', *explain_list_train)
    np.save(f'{path}/y_train', np.array(explain_list_train_classes))

    explain_list_test = []
    explain_list_test_classes = []
    for test_sample in tqdm(test_loader):
        out = explainer(test_sample.x, test_sample.edge_index,
                        target=test_sample.y.float(),
                        batch=test_sample.batch,
                        index=0)
        print(out.edge_mask)
        explain_list_test.append(to_dense_adj(out.edge_index, edge_attr=out.edge_mask))
        explain_list_test_classes.append(torch.argmax(test_sample.y).item())

    np.savez_compressed(f'{path}/x_test', *explain_list_test)
    np.save(f'{path}/y_test', np.array(explain_list_test_classes))


def generate_local_motif(task: Targets, generalisation_mode: GeneralisationModes):
    n_layers = 8
    emb_size = 16

    train_data = LatticeDataset(root="data", target=task,
                                generalisation_mode=generalisation_mode,
                                split='train')
    test_data = LatticeDataset(root="data", target=task,
                               generalisation_mode=generalisation_mode,
                               split='test')

    gnn = BlackBoxGNN(train_data.num_features, emb_size, train_data.num_classes,
                      n_layers)

    gnn.load_state_dict(torch.load(
        f'experiments/results/task_{task}/models/BlackBoxGNN_generalization_{generalisation_mode}_seed_102_temperature_1_embsize_16.pt'))

    generate_motifs(gnn, train_data, test_data,
                    task=task,
                    generalisation_mode=generalisation_mode)


@click.command('generate-local-motifs')
@click.option('--task',
              type=click.Choice(['Distributive', 'Modular', 'Meet_SemiDistributive',
                                 'Join_SemiDistributive', 'multilabel']),
              default='Distributive'
              )
@click.option('--generalisation_mode', type=click.Choice(['weak', 'strong']),
              default='strong')
def main(task: str, generalisation_mode: str) -> None:
    task = Targets[task]
    generalisation_mode = GeneralisationModes[generalisation_mode]

    generate_local_motif(task, generalisation_mode)



if __name__ == "__main__":
    main()
