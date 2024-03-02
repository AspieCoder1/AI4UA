import json
from typing import Literal

import numpy as np
import torch
import torch_geometric.transforms as T

import glgexplainer.utils as utils
from glgexplainer.models import LEN, LEEmbedder, GLGExplainer
from glgexplainer.utils import LocalExplanationsDataset
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

    with open(f"config/{DATASET_NAME}_params.json") as json_file:
        hyper_params = json.load(json_file)

    train_adjs, train_labels, belonging_train = read_lattice_dataset(task,
                                                                     generalisation_mode,
                                                                     split='train')
    test_adjs, test_labels, belonging_test = read_lattice_dataset(task,
                                                                  generalisation_mode,
                                                                  split='test')

    device = "cpu"  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([
        T.NormalizeFeatures(),
    ])

    dataset_train = LocalExplanationsDataset(
        "glg_data",
        adjs=train_adjs,
        feature_type='same',
        belonging=belonging_train,
        y=train_labels,
        task_y=train_labels,
        transform=transform
    )
    dataset_test = LocalExplanationsDataset(
        "glg_data",
        adjs=test_adjs,
        feature_type='same',
        belonging=belonging_test,
        y=train_labels,
        task_y=test_labels,
        transform=transform
    )

    train_group_loader = utils.build_dataloader(dataset_train, belonging_train,
                                                num_input_graphs=128)
    test_group_loader = utils.build_dataloader(dataset_test, belonging_test,
                                               num_input_graphs=256)
    print(dataset_train.data)

    torch.manual_seed(42)

    len_model = LEN(hyper_params["num_prototypes"],
                    hyper_params["LEN_temperature"],
                    remove_attention=hyper_params["remove_attention"]).to(device)
    le_model = LEEmbedder(num_features=hyper_params["num_le_features"],
                          activation=hyper_params["activation"],
                          num_hidden=hyper_params["dim_prototypes"]).to(device)
    expl = GLGExplainer(len_model,
                        le_model,
                        device,
                        hyper_params=hyper_params,
                        classes_names=['None distributive', 'Distributive'],
                        dataset_name=DATASET_NAME,
                        num_classes=2
                        ).to(device)

    expl.iterate(train_group_loader, test_group_loader, plot=False)
    expl.inspect(test_group_loader)
