import json
from typing import Literal

import click
import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx

import glgexplainer.utils as utils
from glgexplainer.local_explainations import read_lattice, lattice_classnames
from glgexplainer.models import LEN, GLGExplainer, LEEmbedder
from gnn4ua.datasets.loader import Targets, GeneralisationModes


def read_lattice_dataset(task: Targets, mode: GeneralisationModes,
                         split: Literal["train", "test"] = 'train'):
    motifs = []
    with np.load(f'local_features/PGExplainer/{task}_{mode}/x_{split}.npz') as data:
        for value in data.values():
            motifs.append(np.squeeze(value, 0))

    labels = np.load(f'local_features/PGExplainer/{task}_{mode}/y_{split}.npy')

    print(labels)

    return motifs, labels, list(range(len(motifs)))


def run_glgexplainer(task: Targets, generalisation_mode: GeneralisationModes):
    DATASET_NAME = task

    click.secho("Loading hyperparameters...", fg="blue", bold=True)
    with open(f"config/{DATASET_NAME}_params.json") as json_file:
        hyper_params = json.load(json_file)

    click.secho("Processing datasets...", fg="blue", bold=True)
    adjs_train, edge_weights_train, ori_classes_train, belonging_train, summary_predictions_train, le_classes_train = read_lattice(
        target=task,
        mode=generalisation_mode,
        split='train'
    )
    adjs_test, edge_weights_test, ori_classes_test, belonging_test, summary_predictions_test, le_classes_test = read_lattice(
        target=task,
        mode=generalisation_mode,
        split='test'
    )

    device = "cpu"  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([
        T.NormalizeFeatures(),
    ])

    click.secho("Setup datasets...", fg="blue", bold=True)
    dataset_train = utils.LocalExplanationsDataset("data_glg", adjs_train, "same",
                                                   transform=transform,
                                                   y=le_classes_train,
                                                   belonging=belonging_train,
                                                   task_y=ori_classes_train)
    dataset_test = utils.LocalExplanationsDataset("data_glg", adjs_test, "same",
                                                  transform=transform,
                                                  y=le_classes_test,
                                                  belonging=belonging_test,
                                                  task_y=ori_classes_test)

    train_group_loader = utils.build_dataloader(dataset_train, belonging_train,
                                                num_input_graphs=128)
    test_group_loader = utils.build_dataloader(dataset_test, belonging_test,
                                               num_input_graphs=256)

    torch.manual_seed(42)

    torch.manual_seed(42)
    len_model = LEN(hyper_params["num_prototypes"],
                    hyper_params["LEN_temperature"],
                    remove_attention=hyper_params["remove_attention"]).to(device)
    le_model = LEEmbedder(num_features=hyper_params["num_le_features"],
                          activation=hyper_params["activation"],
                          num_hidden=hyper_params["dim_prototypes"]).to(device)
    expl = GLGExplainer(len_model,
                        le_model,
                        device=device,
                        hyper_params=hyper_params,
                        classes_names=lattice_classnames,
                        dataset_name=DATASET_NAME,
                        num_classes=len(
                            train_group_loader.dataset.data.task_y.unique())
                        ).to(device)

    click.secho("Train GLGExplainer...", fg="blue", bold=True)
    expl.iterate(train_group_loader, test_group_loader, plot=False)
    expl.inspect(test_group_loader)

    # change assign function to a non-discrete one just to compute distance between local expls. and prototypes
    # useful to show the materialization of prototypes based on distance
    click.secho("Plotting prototypes...", fg='blue', bold=True)
    expl.hyper["assign_func"] = "sim"

    x_train, emb, concepts_assignement, y_train_1h, le_classes, le_idxs, belonging = expl.get_concept_vector(
        test_group_loader,
        return_raw=True)
    expl.hyper["assign_func"] = "discrete"

    proto_names = {
        0: "BA",
        1: "Wheel",
        2: "Mix",
        3: "Grid",
        4: "House",
        5: "Grid",
    }
    torch.manual_seed(42)
    fig = plt.figure(figsize=(15, 5 * 1.8))
    n = 0
    for p in range(expl.hyper["num_prototypes"]):
        idxs = le_idxs[concepts_assignement.argmax(-1) == p]
        # idxs = idxs[torch.randperm(len(idxs))]    # random
        sa = concepts_assignement[concepts_assignement.argmax(-1) == p]
        idxs = idxs[torch.argsort(sa[:, p], descending=True)]
        for ex in range(min(5, len(idxs))):
            n += 1
            ax = plt.subplot(expl.hyper["num_prototypes"], 5, n)
            G = to_networkx(dataset_test[int(idxs[ex])], to_undirected=True,
                            remove_self_loops=True)
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, node_size=20, ax=ax, node_color="orange")
            ax.axis("on")
            plt.box(False)

    for p in range(expl.hyper["num_prototypes"]):
        plt.subplot(expl.hyper["num_prototypes"], 5, 5 * p + 1)
        plt.ylabel(f"$P_{p}$\n", size=25, rotation="horizontal",
                   labelpad=50)

    plt.show()
