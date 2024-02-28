import torch
from torch import nn
from torch_geometric.data import Data
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

from gnn4ua.datasets.loader import load_data
from gnn4ua.models import BlackBoxGNN


def generate_motifs(model: nn.Module, data: Data):
    """
    Uses PGExplainer to generate motif features from the datasets.

    :param model:
    :param data:
    :return:
    """
    config = ModelConfig(
        task_level=ModelTaskLevel.graph,
        mode=ModelMode.multiclass_classification,
        return_type=ModelReturnType.raw,
    )

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=30),
        explanation_type=ExplanationType.phenomenon,
        model_config=config,
        edge_mask_type=MaskType.object,
    )

    assert isinstance(explainer.algorithm, PGExplainer)

    for epoch in range(30):
        explainer.algorithm.train(
            epoch,
            model,
            data.x,
            data.edge_index,
            target=data.y,
            index=0,
            batch=data.batch
        )

    return explainer(data.x, data.edge_index, target=data.y, index=0)


def main():
    dataset = 'samples_50_saved'
    label_name = 'Distributive'
    generalization = 'strong',
    random_state = 102
    max_size_train = 8
    max_prob_train = 0.8
    n_layers = 8
    emb_size = 16

    data = load_data(dataset, label_name=label_name,
                     root_dir='../gnn4ua/datasets/',
                     generalization=generalization,
                     random_state=random_state,
                     max_size_train=max_size_train,
                     max_prob_train=max_prob_train)

    gnn = BlackBoxGNN(data.x.shape[1], emb_size, data.y.shape[1], n_layers)

    gnn.load_state_dict(torch.load(
        '../experiments/results/task_Distributive/models/BlackBoxGNN_generalization_strong_seed_102_temperature_1_embsize_16.pt'))

    motifs = generate_motifs(gnn, data)

    print(motifs.get_explanation_subgraph())


if __name__ == "__main__":
    main()
