import os
import os.path as osp
from enum import StrEnum, auto
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset, Data, download_url


class Targets(StrEnum):
    Distributive = 'Distributive'
    Modular = 'Modular'
    Meet_SemiDistributive = 'Meet_SemiDistributive'
    Join_SemiDistributive = 'Join_SemiDistributive'
    SemiDistributive = 'SemiDistributive'
    multilabel = 'multilabel'


class GeneralisationModes(StrEnum):
    weak = auto()
    strong = auto()


class LatticeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
                 target: Targets = Targets.Distributive,
                 generalisation_mode: GeneralisationModes = GeneralisationModes.strong,
                 split: Literal['train', 'test'] = 'train'):
        self.target = target
        self.generalisation_mode = generalisation_mode
        self.split = split
        self.split_frac = 0.8 if self.split == 'train' else 0.2
        self.max_train_size = 8
        super().__init__(root, transform, pre_transform, pre_filter)

        assert target in Targets, f'{target} is not a valid target class'
        assert generalisation_mode in GeneralisationModes, f'{generalisation_mode} is not a valid generalisation mode'
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, f'{self.target}_{self.generalisation_mode}', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'{self.target}_{self.generalisation_mode}',
                        'processed')

    @property
    def raw_file_names(self) -> str:
        return 'samples_50_saved.json'

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.split}.pt'

    def download(self):
        download_url(
            url='https://drive.google.com/uc?export=download&id=1KP67FfoS_0IjuwSmV_8QODvDmL5SLhwp',
            folder=self.raw_dir, filename='samples_50_saved.json')

    def create_graph(self, row):
        adj_matrix = np.array(row['Adj_matrix'])
        symmetric_matrix_with_self_loops = adj_matrix + np.eye(
            len(adj_matrix)) + adj_matrix.T
        edge_indices = torch.nonzero(
            torch.tensor(symmetric_matrix_with_self_loops, dtype=torch.float)
        ).tolist()

        edge_index = torch.tensor(edge_indices).t()
        x = torch.ones((edge_index.max().item() + 1, 1))

        if self.target is Targets.multilabel:
            label_names = list(set(row.index).difference(
                ['ID', 'Cardinality', 'LoE_matrix', 'Adj_matrix']))
            y = torch.LongTensor([row[label_names].to_list()])
        else:
            label = row[self.target]
            y = torch.LongTensor([label])
        return Data(x=x, edge_index=edge_index, y=y)

    def process(self):
        df = pd.read_json(osp.join(self.raw_dir, self.raw_file_names))

        if self.generalisation_mode is GeneralisationModes.strong:
            df_small_lattices = df[df['Cardinality'] < self.max_train_size]
            df_large_lattices = df[df['Cardinality'] > self.max_train_size]
            df_medium_lattices_train, df_medium_lattices_test = train_test_split(
                df[df['Cardinality'] == self.max_train_size], train_size=0.8,
                random_state=42)

            # Construct the train and test datasets
            df_train = pd.concat([df_small_lattices, df_medium_lattices_train])
            df_test = pd.concat([df_large_lattices, df_medium_lattices_test])
        else:
            df_train, df_test = train_test_split(df, train_size=0.8, random_state=42)

        df_split = df_train if self.split == 'train' else df_test

        data_list = df_split.apply(self.create_graph, axis=1).to_list()
        self.save(data_list, self.processed_paths[0])

    @property
    def num_features(self) -> int:
        return 1

    @property
    def num_classes(self) -> int:
        if self.target is Targets.multilabel:
            return 5
        return 2


# Convert to InMemoryDataset to do loading sensibly and also handles the minibatching for us

def load_data(dataset_name, label_name, root_dir='../gnn4ua/datasets/', generalization='strong',
              random_state=42, max_size_train=40, max_prob_train=0.8):
    file_name = os.path.join(root_dir, f'{dataset_name}_{label_name}.joblib')

    if not os.path.exists(file_name):
        file_name = os.path.join(root_dir, dataset_name+'.json')
        if not os.path.exists(file_name):
            raise FileNotFoundError(f'File {file_name} does not exist')

        df = pd.read_json(file_name)
        adjacency_matrices = df['Adj_matrix'].values.tolist()

        # Compute the indices of the nonzero elements in the adjacency matrices
        edge_indices = []
        labels = []
        batch = []
        train_mask = []
        test_mask = []
        n_nodes_seen = 0
        for i, matrix in enumerate(adjacency_matrices):
            # generate the indices of the nonzero elements in the adjacency matrix
            matrix = np.array(matrix)
            symmetric_matrix_with_self_loops = matrix + np.eye(len(matrix)) + matrix.T
            # safety check: the matrix should be symmetric and with a maximum value of 1
            assert np.allclose(symmetric_matrix_with_self_loops, symmetric_matrix_with_self_loops.T, rtol=1e-05, atol=1e-08)
            assert np.max(symmetric_matrix_with_self_loops) == 1

            # get edge indexes
            matrix_indices = torch.nonzero(torch.tensor(symmetric_matrix_with_self_loops, dtype=torch.float)).tolist()
            matrix_indices_shifted = [(index[0] + n_nodes_seen, index[1] + n_nodes_seen) for index in matrix_indices]
            edge_indices.extend(matrix_indices_shifted)

            # generate the labels
            if label_name == 'multilabel':
                label_names = list(set(df.columns).difference(['ID', 'Cardinality', 'LoE_matrix', 'Adj_matrix']))
                label_values = df[label_names].values[i].astype(int)
                labels.append(torch.LongTensor(label_values))
            else:
                label = df[label_name].values[i].astype(int)
                labels.append(torch.LongTensor([1-label, label]))

            # generate the batch index
            batch.extend(torch.LongTensor([i] * len(matrix)))

            # generate the train/test masks
            if generalization == 'strong':
                if len(matrix) > max_size_train:
                    is_train_graph = False
                elif len(matrix) < max_size_train:
                    is_train_graph = True
                elif len(matrix) == max_size_train:
                    if np.random.RandomState(random_state*(i+1)).rand() < max_prob_train:
                        is_train_graph = True
                    else:
                        is_train_graph = False
            else:
                is_train_graph = True if np.random.RandomState(random_state*(i+1)).rand() < max_prob_train else False
            train_mask.extend(torch.BoolTensor([is_train_graph]))
            test_mask.extend(torch.BoolTensor([not is_train_graph]))

            n_nodes_seen += len(matrix)

        # Create the edge index tensor
        edge_index = torch.tensor(edge_indices).t()
        x = torch.ones((edge_index.max().item() + 1, 1))
        y = torch.vstack(labels)
        batch = torch.hstack(batch)
        train_mask = torch.hstack(train_mask)
        test_mask = torch.hstack(test_mask)

        data = pyg.data.Data(x=x, edge_index=edge_index, y=y, batch=batch, train_mask=train_mask, test_mask=test_mask)
        data.validate(raise_on_error=True)

        joblib.dump(data, os.path.join(root_dir, f'{dataset_name}_{label_name}_{generalization}.joblib'))
    else:
        data = joblib.load(os.path.join(root_dir, f'{dataset_name}_{label_name}_{generalization}.joblib'))
        data.validate(raise_on_error=True)
    return data


if __name__ == '__main__':
    test = LatticeDataset(root="data_test", target=Targets.multilabel, split='test')
    print(next(iter(test)))
