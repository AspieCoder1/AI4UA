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
from torch_geometric.utils import to_undirected

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
        algorithm=PGExplainer(epochs=10),
        explanation_type=ExplanationType.phenomenon,
        model_config=config,
        edge_mask_type=MaskType.object,
        threshold_config=ThresholdConfig(
            threshold_type=ThresholdType.topk_hard,
            value=16
        )
    )

    assert isinstance(explainer.algorithm, PGExplainer)

    for train_sample in DataLoader(train_data, batch_size=1, shuffle=True):
        for epoch in range(10):
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
    print(motifs.edge_mask)

    new_edge_index, new_edge_mask = to_undirected(edge_index=motifs.edge_index,
                                                  edge_attr=motifs.edge_mask,
                                                  reduce='mean')
    print(new_edge_mask)
    motifs.update({'edge_index': new_edge_index, 'edge_mask': new_edge_mask})
    motifs = motifs.threshold(threshold_type=ThresholdType.hard, value=0.1)
    print(motifs.edge_mask)
    motifs.visualize_graph()


if __name__ == "__main__":
    main()
