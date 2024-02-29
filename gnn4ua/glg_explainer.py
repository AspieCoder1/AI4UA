import networkx as nx
import torch
from matplotlib import pyplot as plt
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
from torch_geometric.utils import to_networkx

from gnn4ua.datasets.loader import (LatticeDataset, Targets,
                                    GeneralisationModes, )
from gnn4ua.models import BlackBoxGNN


def generate_motifs(model: nn.Module, train_data, test_data):
    """
    Uses PGExplainer to generate motif features from the datasets.

    :param model:
    :param data:
    :return:
    """
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
    )

    assert isinstance(explainer.algorithm, PGExplainer)

    for epoch in range(1):
        for train_sample in DataLoader(train_data, batch_size=1):
            explainer.algorithm.train(
                epoch,
                model,
                train_sample.x,
                train_sample.edge_index,
                target=train_sample.y,
                index=0,
                batch=train_sample.batch
            )

    test_sample = next(iter(DataLoader(test_data, batch_size=1, shuffle=False)))
    return explainer(test_sample.x, test_sample.edge_index, target=test_sample.y,
                     batch=test_sample.batch,
                     index=0)


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

    motifs = generate_motifs(gnn, train_data, test_data)

    print(motifs.get_explanation_subgraph().edge_index)

    G = to_networkx(motifs.get_explanation_subgraph())

    print(G)

    ax = plt.subplot(111)
    nx.draw(G, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
