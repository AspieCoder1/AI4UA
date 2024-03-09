import numpy as np
import pandas as pd
from utils import prepare_dataset_json

from gnn4ua.datasets.lattice import Lattice

if __name__ == '__main__':
    df = pd.read_json("samples_50_saved_old.json")

    LoE_matrices = df['LoE_matrix'].map(lambda x: np.array(x))

    lattices = list(map(lambda x: Lattice(x), LoE_matrices))
    prepare_dataset_json(lattices)
