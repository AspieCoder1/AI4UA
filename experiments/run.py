import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import (MultilabelAUROC,
                                         MultilabelAccuracy, BinaryAccuracy,
                                         BinaryAUROC, )
from tqdm import trange

from gnn4ua.datasets.loader import LatticeDataset, Targets, GeneralisationModes

sys.path.append('..')
from gnn4ua.models import BlackBoxGNN, GraphNet

# torch.autograd.set_detect_anomaly(True)
seed_everything(42)


def run_gnn_training():
    # hyperparameters
    random_states = np.random.RandomState(42).randint(0, 1000, size=5)
    dataset = 'samples_50_saved'
    temperature = 1
    targets = [Targets.multilabel, Targets.Distributive, Targets.Modular,
               Targets.Meet_SemiDistributive, Targets.Join_SemiDistributive]
    generalisation_modes = [GeneralisationModes.strong, GeneralisationModes.weak]
    train_epochs = 200
    emb_size = 16
    learning_rate = 0.001
    max_size_train = 8
    max_prob_train = 0.8
    n_layers = 8
    internal_loss_weight = 0.1

    # we will save all results_binary in this directory
    results_dir = f"results/"
    os.makedirs(results_dir, exist_ok=True)

    results = []
    cols = ['task', 'generalisation', 'model', 'test_auc', 'train_auc', 'n_layers',
            'temperature', 'emb_size', 'learning_rate', 'train_epochs',
            'max_size_train', 'max_prob_train']

    for target in targets:
        model_dir = os.path.join(results_dir, f'task_{target}/models/')
        os.makedirs(model_dir, exist_ok=True)
        metrics_dir = os.path.join(results_dir, f'metrics/')
        os.makedirs(metrics_dir, exist_ok=True)

        for generalisation in generalisation_modes:
            for state_id, random_state in enumerate(random_states):

                train_data = LatticeDataset(root="data", target=target,
                                            generalisation_mode=generalisation)
                test_data = LatticeDataset(root="data/", target=target,
                                           generalisation_mode=generalisation,
                                           split='test')

                train_metrics = MetricCollection({
                    "accuracy": MultilabelAccuracy(
                        num_labels=train_data.num_classes) if target is Targets.multilabel else BinaryAccuracy(),
                    "auroc": MultilabelAUROC(
                        num_labels=train_data.num_classes) if target is Targets.multilabel else BinaryAUROC()
                }, prefix="train/")
                test_metrics = train_metrics.clone(prefix="test/")
                loss_form = torch.nn.BCEWithLogitsLoss()

                # load data and set up cross-validation
                # data = load_data(dataset, label_name=label_name,
                #                  root_dir='../gnn4ua/datasets/',
                #                  generalisation=generalisation,
                #                  random_state=random_state,
                #                  max_size_train=max_size_train,
                #                  max_prob_train=max_prob_train)
                # train_index, test_index = data.train_mask, data.test_mask
                # print(sum(train_index), sum(test_index))

                # reset model weights for each fold
                models: List[GraphNet] = [
                    # HierarchicalGCN(train_data.num_features, emb_size,
                    #                 train_data.num_classes,
                    #                 n_layers),
                    # GCoRe(train_data.num_features, emb_size, train_data.num_classes,
                    #       n_layers),
                    BlackBoxGNN(train_data.num_features, emb_size,
                                train_data.num_classes, n_layers),
                ]

                for gnn in models:
                    train_metrics.reset()
                    test_metrics.reset()
                    model_path = os.path.join(model_dir,
                                              f'{gnn.__class__.__name__}_generalization_{generalisation}_seed_{random_state}_temperature_{temperature}_embsize_{emb_size}.pt')
                    print(
                        f'Running {gnn.__class__.__name__} on {target} ({generalisation} generalisation) [seed {state_id + 1}/{len(random_states)}]')

                    if not os.path.exists(model_path):
                        # train model
                        optimizer = torch.optim.AdamW(gnn.parameters(),
                                                      lr=learning_rate)


                        gnn.train()
                        for _epoch in trange(train_epochs):
                            for data in DataLoader(train_data, batch_size=2048,
                                                        shuffle=True):
                                optimizer.zero_grad()
                                x, edge_index, batch = data.x, data.edge_index, data.batch

                                out = gnn.forward(x,
                                                  edge_index,
                                                  batch)

                                if isinstance(gnn, BlackBoxGNN):
                                    y_pred = out
                                else:
                                    y_pred, node_concepts, graph_concepts = out

                                train_metrics(y_pred, data.y)

                                # compute loss
                                main_loss = loss_form(y_pred,
                                                      data.y.float())
                                internal_loss = 0
                                if gnn.__class__.__name__ == 'HierarchicalGCN':
                                    internal_loss = gnn.internal_loss(graph_concepts,
                                                                      data.y.float(),
                                                                      loss_form)
                                internal_loss = internal_loss * internal_loss_weight
                                loss = main_loss + internal_loss

                                loss.backward()
                                optimizer.step()

                            # Test set performance
                            for data in DataLoader(test_data, batch_size=2048,
                                                   shuffle=False):
                                y_pred, _, _ = gnn.forward(data.x, data.edge_index,
                                                           data.batch)
                                # y_pred.sigmoid()
                                test_metrics(y_pred, data.y)
                        torch.save(gnn.state_dict(), model_path)

                    # get model predictions
                    gnn.load_state_dict(torch.load(model_path))
                    gnn.eval()
                    train_metrics.reset()
                    test_metrics.reset()

                    for data in DataLoader(train_data, batch_size=2048,
                                           shuffle=True):
                        x, edge_index, batch = data.x, data.edge_index, data.batch
                        y_pred, _, _ = gnn.forward(x,
                                                   edge_index,
                                                   batch)

                        train_metrics(y_pred, data.y)

                    for data in DataLoader(test_data, batch_size=1024,
                                           shuffle=False):
                        y_pred, _, _ = gnn.forward(data.x, data.edge_index,
                                                   data.batch)
                        y_pred.sigmoid()
                        test_metrics(y_pred, data.y)
                    metrics_test = test_metrics.compute()
                    metrics_train = train_metrics.compute()

                    print(metrics_test)

                    results.append(
                        [target, generalisation, gnn.__class__.__name__,
                         metrics_test["test/auroc"].item(),
                         metrics_train["train/auroc"].item(), n_layers, temperature, emb_size,
                         learning_rate,
                         train_epochs, max_size_train, max_prob_train])
                    pd.DataFrame(results, columns=cols).to_csv(
                        os.path.join(metrics_dir, 'auc.csv'))


if __name__ == '__main__':
    run_gnn_training()
