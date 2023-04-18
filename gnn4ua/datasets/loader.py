import json

import joblib
import numpy as np
import pandas as pd
import torch
import os
import numpy as np
import torch_geometric as pyg


def load_data(dataset_name, label_name, root_dir='../gnn4ua/datasets/'):
    file_name = os.path.join(root_dir, f'{dataset_name}_{label_name}.joblib')

    if not os.path.exists(file_name):
        file_name = os.path.join(root_dir, dataset_name+'.json')
        if not os.path.exists(file_name):
            raise FileNotFoundError(f'File {file_name} does not exist')

        df = pd.read_json(file_name, lines=True)
        adjacency_matrices = df['Adj_matrix'].values.tolist()

        # Compute the cumulative number of nodes from previous matrices
        cumulative_nodes = [0] + list(
            torch.cumsum(torch.tensor([len(matrix) for matrix in adjacency_matrices]), dim=0).numpy())

        # Compute the indices of the nonzero elements in the adjacency matrices
        edge_indices = []
        labels = []
        batch = []
        for i, matrix in enumerate(adjacency_matrices):
            matrix_indices = torch.nonzero(torch.tensor(matrix, dtype=torch.float)).tolist()
            matrix_indices_shifted = [(index[0] + cumulative_nodes[i], index[1] + cumulative_nodes[i]) for index in
                                      matrix_indices]
            edge_indices.extend(matrix_indices_shifted)
            label = df[label_name].values[i].astype(int)
            labels.append(torch.LongTensor([1-label, label]))
            batch.extend(torch.LongTensor([i] * len(matrix)))

        # Create the edge index tensor
        edge_index = torch.tensor(edge_indices).t()
        edge_index = pyg.utils.add_self_loops(edge_index)[0]
        x = torch.ones((edge_index.max().item() + 1, 1))
        y = torch.vstack(labels)
        batch = torch.hstack(batch)

        data = pyg.data.Data(x=x, edge_index=edge_index, y=y, batch=batch)
        data.validate(raise_on_error=True)

        joblib.dump(data, os.path.join(root_dir, f'{dataset_name}_{label_name}.joblib'))
    else:
        data = joblib.load(os.path.join(root_dir, f'{dataset_name}_{label_name}.joblib'))
        data.validate(raise_on_error=True)
    return data