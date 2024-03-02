import os

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
    ThresholdType, ThresholdConfig,
)
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from gnn4ua.datasets.loader import (LatticeDataset, Targets,
                                    GeneralisationModes, )
from gnn4ua.models import BlackBoxGNN


def generate_motifs(model: nn.Module, train_data, test_data, task, root):
    """
    Uses PGExplainer to generate motif features from the datasets.

    :param model:
    :param data:
    :return:
    """
    path = f'{root}/{task}'

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    config = ModelConfig(
        task_level=ModelTaskLevel.graph,
        mode=ModelMode.binary_classification,
        return_type=ModelReturnType.raw,
    )

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=1),
        explanation_type=ExplanationType.phenomenon,
        model_config=config,
        edge_mask_type=MaskType.object,
        threshold_config=ThresholdConfig(
            threshold_type=ThresholdType.topk_hard,
            value=16
        )
    )

    assert isinstance(explainer.algorithm, PGExplainer)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1)

    for epoch in range(1):
        for train_sample in tqdm(train_loader):
            explainer.algorithm.train(
                epoch,
                model,
                train_sample.x,
                train_sample.edge_index,
                target=train_sample.y,
                index=0,
                batch=train_sample.batch
            )

    explain_list_train = []
    explain_list_train_classes = []
    for train_sample in tqdm(train_loader):
        out = explainer(train_sample.x, train_sample.edge_index, target=train_sample.y,
                        batch=train_sample.batch,
                        index=0)

        motif = out.get_explanation_subgraph()

        explain_list_train.append(to_dense_adj(motif.edge_index))
        explain_list_train_classes.append(train_sample.y.item())

    np.savez_compressed(f'{path}/x_train', *explain_list_train)
    np.save(f'{path}/y_train', np.array(explain_list_train_classes))

    explain_list_test = []
    explain_list_test_classes = []
    for test_sample in tqdm(test_loader):
        out = explainer(test_sample.x, test_sample.edge_index, target=test_sample.y,
                        batch=test_sample.batch,
                        index=0)

        motif = out.get_explanation_subgraph()

        explain_list_test.append(to_dense_adj(motif.edge_index))
        explain_list_test_classes.append(test_sample.y.item())

    np.savez_compressed(f'{path}/x_test', *explain_list_test)
    np.save(f'{path}/y_test', np.array(explain_list_test_classes))


def main():
    n_layers = 8
    emb_size = 16

    train_data = LatticeDataset(root="../experiments/data", target=Targets.Distributive,
                                generalisation_mode=GeneralisationModes.weak,
                                split='train')
    test_data = LatticeDataset(root="../experiments/data", target=Targets.Distributive,
                               generalisation_mode=GeneralisationModes.weak,
                               split='test')

    gnn = BlackBoxGNN(train_data.num_features, emb_size, train_data.num_classes,
                      n_layers)

    gnn.load_state_dict(torch.load(
        '../experiments/results/task_Distributive/models/BlackBoxGNN_generalization_strong_seed_102_temperature_1_embsize_16.pt'))

    generate_motifs(gnn, train_data, test_data, root='local_features/PGExplainer',
                    task=Targets.Distributive)


if __name__ == "__main__":
    main()
