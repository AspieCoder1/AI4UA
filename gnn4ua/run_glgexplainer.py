import csv
import json
import os
from typing import Literal

import click
import torch
import torch_geometric.transforms as T

import glgexplainer.utils as utils
from glgexplainer.local_explainations import read_lattice, lattice_classnames
from glgexplainer.models import LEN, GLGExplainer, LEEmbedder
from gnn4ua.datasets.loader import Targets, GeneralisationModes


def run_glgexplainer(task: Targets, generalisation_mode: GeneralisationModes,
                     seed: Literal['102', '106', '270']):
    DATASET_NAME = task

    click.secho(
        f"RUNNING GLGEXPLAINER ON {DATASET_NAME.capitalize()}-{generalisation_mode.capitalize()} (SEED {seed})",
        fg='blue', bold=True, underline=True)

    click.secho("Loading hyperparameters...", bold=True)
    with open(f"config/{DATASET_NAME}_params.json") as json_file:
        hyper_params = json.load(json_file)

    click.secho("Processing datasets...", bold=True)
    adjs_train, edge_weights_train, ori_classes_train, belonging_train, summary_predictions_train, le_classes_train = read_lattice(
        seed=seed,
        explainer='GNNExplainer',
        target=task,
        mode=generalisation_mode,
        split='train'
    )

    adjs_test, edge_weights_test, ori_classes_test, belonging_test, summary_predictions_test, le_classes_test = read_lattice(
        seed=seed,
        explainer='GNNExplainer',
        target=task,
        mode=generalisation_mode,
        split='test'
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

    click.secho("Creating plot directory", bold=True)
    plot_dir = f'GLGExplainer_plots/{seed}/{task}-{generalisation_mode}'
    os.makedirs(plot_dir, exist_ok=True)

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
                            train_group_loader.task_y.unique()),
                        plot_dir=plot_dir
                        ).to(device)

    click.secho("Train GLGExplainer...", bold=True)
    expl.iterate(train_group_loader, test_group_loader, plot=True)

    click.secho("Test GLGExplainer...", bold=True)
    results = expl.inspect(test_group_loader, plot=True)

    click.secho("Writing results...", bold=True)
    csv_exists = os.path.exists('GLGExplainer_results.csv')

    with open('GLGExplainer_results.csv', 'a+') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['task', 'mode', 'seed', 'logic_acc',
                                            'logic_acc_clf', 'concept_purity',
                                            'concept_purity_std', 'LEN_fidelity',
                                            'formula_len_0', 'formula_len_1'])
        row = results | {'task': task, 'mode': generalisation_mode, 'seed': seed}
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)
