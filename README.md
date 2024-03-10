# L65 Project Repository

_Forked from [AI4UA repository](https://github.com/fragiannini/AI4UA) and uses the code
from [Global Explainability of GNNs via Logic Combination of Learned Concepts repo](https://github.com/steveazzolin/gnn_logic_global_expl) to implement GLGExplainer._

Repository for the L65 Geometric Deep Learning mini-project completed by Luke
Braithwaite and Matthew Hattrup during Lent term 2024.

## Requirements

- pytorch
- pytorch-geometric
- networkx
- numpy

# Torch geometric installation
Follow instructions in https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html.

> pip install torch_geometric
> 
> pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

## Running the code

The pipeline consists of the following steps all of which can be performed using the cli
included in `main.py`.
They are:

1. Generating the lattice dataset.
2. Training the GNN to classify the lattice properties you would like.
3. Training a local explanation extractor to generate the explanation subgraphs from the
   input lattices.
4. Train GLGExplainer to use the extracted local explanations.

## Generating the lattice dataset

This can be performed by running the file `gnn4Uua/datasets/runner`.
Currently the CLI does not support this operation.

## Training the GNNs

Run the following command to train the GNNs on each task

```
python main.py train-gnns
```

## Extracting local explanations

Run the following to extract the local explanations using GNNExplainer

```
python main.py extract-motifs --task=Distributive --generalisation_mode=strong --seed=102 --n_epochs=100
```

## Training GLGExplainer

Run the following command to train GLGExplainer on a specific task

```
python main.py train-explainer --task=Distributive --generalisation_mode=strong --seed=102
```

