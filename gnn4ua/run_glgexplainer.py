import csv
import json
import os
from typing import Literal

import click
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx

import glgexplainer.utils as utils
from glgexplainer.local_explainations import (read_lattice, pattern_names,
                                              generate_lattice_classnames, )
from glgexplainer.models import LEN, GLGExplainer, LEEmbedder
from gnn4ua.datasets.loader import Targets, GeneralisationModes


def run_glgexplainer(task: Targets, generalisation_mode: GeneralisationModes,
                     seed: Literal['102', '106', '270'], explainer: str,
                     n_prototypes: int, n_motifs: int):
    DATASET_NAME = task

    click.secho(
        f"RUNNING {explainer.upper()} ON {DATASET_NAME.upper()}-{generalisation_mode.upper()} (SEED {seed} NUM_MOTIFS {n_motifs} NUM_PROTOTYPES {n_prototypes})",
        fg='blue', bold=True, underline=True)

    click.secho("Loading hyperparameters...", bold=True)
    with open(f"config/{DATASET_NAME}_params.json") as json_file:
        hyper_params = json.load(json_file)

    hyper_params['num_prototypes'] = n_prototypes

    click.secho("Processing datasets...", bold=True)
    adjs_train, edge_weights_train, ori_classes_train, belonging_train, summary_predictions_train, le_classes_train = read_lattice(
        seed=seed,
        explainer=explainer,
        target=task,
        mode=generalisation_mode,
        split='train',
        n_motifs=n_motifs
    )

    adjs_test, edge_weights_test, ori_classes_test, belonging_test, summary_predictions_test, le_classes_test = read_lattice(
        seed=seed,
        explainer=explainer,
        target=task,
        mode=generalisation_mode,
        split='test',
        n_motifs=n_motifs
    )

    device = "cpu"  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([
        T.NormalizeFeatures(),
    ])

    click.secho("Setup datasets...", bold=True)
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

    click.secho("Creating plot directory...", bold=True)
    plot_dir = f'{explainer}_plots/{seed}/{task}-{generalisation_mode}'
    os.makedirs(plot_dir, exist_ok=True)

    click.secho("Creating GLGExplainer instance...", bold=True)
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
                        classes_names=generate_lattice_classnames(n_motifs=n_motifs),
                        dataset_name=DATASET_NAME,
                        num_classes=len(
                            train_group_loader.dataset.task_y.unique()),
                        plot_dir=plot_dir
                        ).to(device)

    click.secho("Train GLGExplainer...", bold=True)
    expl.iterate(train_group_loader, test_group_loader, plot=True)

    click.secho("Test GLGExplainer...", bold=True)
    results = expl.inspect(test_group_loader, plot=True)

    click.secho("Writing results...", bold=True)
    csv_exists = os.path.exists(f'{explainer}_results.csv')

    with open(f'{explainer}_results.csv', 'a+') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['task', 'mode', 'seed',
                                            'num_prototypes',
                                            'motifs',
                                            'logic_acc',
                                            'logic_acc_clf', 'concept_purity',
                                            'concept_purity_std', 'LEN_fidelity',
                                            'formula_0', 'formula_1'])
        row = results | {'task': task, 'mode': generalisation_mode, 'seed': seed,
                         'num_prototypes': hyper_params["num_prototypes"],
                         'motifs': '+'.join(pattern_names[:n_motifs])}
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)

        click.secho("Creating graph prototypes plot...", bold=True)
        plot_prototypes(expl, test_group_loader, dataset_test, plot_dir)

        click.secho("Creating explanations plot...", bold=True)
        plot_example_explanations(task, generalisation_mode, plot_dir, seed)


def plot_prototypes(expl: GLGExplainer, test_group_loader, dataset_test, plot_dir):
    expl.hyper["assign_func"] = "sim"

    x_train, emb, concepts_assignement, y_train_1h, le_classes, le_idxs, belonging = expl.get_concept_vector(
        test_group_loader,
        return_raw=True)
    expl.hyper["assign_func"] = "discrete"

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

    plt.savefig(f"{plot_dir}/protype_examples.pdf", dpi=300, bbox_inches="tight")


def plot_example_explanations(target, mode, plot_dir, seed):
    data = np.load(f"local_features/GNNExplainer/{seed}/{target}_{mode}/x_train.npz")
    y = np.load(f"local_features/GNNExplainer/{seed}/{target}_{mode}/y_train.npy")
    adjs = list(data.values())

    fig, axs = plt.subplots(9, 9, figsize=(15, 15))
    axs = axs.flatten()

    for i in range(81):
        adj = adjs[i].squeeze()
        adj[adj > 0] = 1
        G = nx.Graph(adj, undirected=True)
        G.remove_edges_from(nx.selfloop_edges(G))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=axs[i], node_color="orange", node_size=10)
        axs[i].set_title(f'Class={y[i]}')
    plt.savefig(f"{plot_dir}/example_explanations.pdf", dpi=300, bbox_inches="tight")
