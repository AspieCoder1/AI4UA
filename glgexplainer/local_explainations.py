import copy
from typing import Literal

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism

from gnn4ua.datasets.loader import GeneralisationModes, Targets
from .utils import normalize_belonging

base = "../local_explanations/"

lattice_classnames = ['N5', 'M3', 'DIAMOND', 'N5+M3', 'N5+DIAMOND', 'M3+DIAMOND', 'ALL',
                      'OTHER']

# Patterns for Lattice extraction
pattern_M3 = nx.Graph()
pattern_M3.add_edges_from([
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 4),
    (3, 4)
])
pattern_N5 = nx.Graph()
pattern_N5.add_edges_from([
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 4)
])
pattern_diamond = nx.Graph()
pattern_diamond.add_edges_from([
    (0, 1),
    (0, 2),
    (1, 3),
    (1, 4)
])


def elbow_method(weights, index_stopped=None, min_num_include=7, backup=None):
    sorted_weights = sorted(weights, reverse=True)
    sorted_weights = np.convolve(sorted_weights, np.ones(min_num_include),
                                 'valid') / min_num_include

    stop = np.mean(sorted_weights) if backup is None else backup  # backup threshold
    for i in range(len(sorted_weights) - 2):
        if i < min_num_include:
            continue
        if sorted_weights[i - 1] - sorted_weights[i] > 0.0:
            if sorted_weights[i - 1] - sorted_weights[i] >= 40 * (
                    sorted_weights[0] - sorted_weights[i - 2]) / 100 + (
                    sorted_weights[0] - sorted_weights[i - 2]):
                stop = sorted_weights[i]
                if index_stopped is not None:
                    index_stopped.append(stop)
                break
    return stop


def assign_class_lattice(pattern_matched):
    if len(pattern_matched) == 0:
        return 7

    if len(pattern_matched) == 1:
        return pattern_matched[0]

    if 0 in pattern_matched and 1 in pattern_matched:
        return 3
    if 0 in pattern_matched and 2 in pattern_matched:
        return 4
    if 1 in pattern_matched and 2 in pattern_matched:
        return 5
    return 6


def label_explanation_lattice(G_orig, return_raw=False):
    pattern_matched = []
    for i, pattern in enumerate([pattern_N5, pattern_M3, pattern_diamond]):
        GM = isomorphism.GraphMatcher(G_orig, pattern)
        if GM.subgraph_is_isomorphic():
            pattern_matched.append(i)
    if return_raw:
        return pattern_matched
    else:
        return assign_class_lattice(pattern_matched)


def read_lattice(explainer="GNNExplainer", target: Targets = Targets.Distributive,
                 mode: GeneralisationModes = GeneralisationModes.strong,
                 split: Literal['train', 'test'] = 'train', min_num_include: int = 7,
                 evaluate_method=False):
    base_path = f"local_features/{explainer}/{target}_{mode}/"
    adjs, edge_weights, index_stopped = [], [], []
    ori_adjs, ori_edge_weights, ori_classes, belonging, ori_predictions = [], [], [], [], []
    precomputed_embeddings, gnn_embeddings = [], []
    total_graph_labels, total_cc_labels, le_classes = [], [], []

    global summary_predictions
    summary_predictions = {"correct": [], "wrong": []}

    labels = np.load(f'{base_path}/y_{split}.npy', allow_pickle=True)

    with np.load(f'{base_path}/x_{split}.npz', allow_pickle=True) as data:
        num_multi_shapes_removed, num_class_relationship_broken, cont_num_iter, num_iter = 0, 0, 0, 0
        for idx, adj in enumerate(data.values()):
            adj = adj.squeeze()
            G_orig = nx.Graph(adj)
            cut = elbow_method(np.triu(adj).flatten(), index_stopped,
                               min_num_include)

            masked = copy.deepcopy(adj)
            masked[masked <= cut] = 0
            masked[masked > cut] = 1
            G = nx.Graph(masked)
            added = 0
            graph_labels = label_explanation_lattice(G_orig, return_raw=True)
            summary_predictions["correct"].append(assign_class_lattice(graph_labels))
            total_cc_labels.append([])
            cc_labels = []
            for cc in nx.connected_components(G):
                G1 = G.subgraph(cc)
                if not nx.diameter(G1) == len(G1.edges()):  # if is not a line
                    cc_lbl = label_explanation_lattice(G1, return_raw=True)
                    added += 1
                    cc_labels.extend(cc_lbl)
                    total_cc_labels[-1].extend(cc_lbl)
                    adjs.append(nx.to_numpy_array(G1))
                    edge_weights.append(nx.get_edge_attributes(G1, "weight"))
                    belonging.append(idx)
                    le_classes.append(assign_class_lattice(cc_lbl))

            if not total_cc_labels[-1]:
                del total_cc_labels[-1]
            if added:
                if graph_labels != []: total_graph_labels.append(graph_labels)
                num_iter += 1
                ori_adjs.append(adj)
                ori_edge_weights.append(nx.get_edge_attributes(G, "weight"))
                ori_classes.append(labels[idx])  # c | gnn_pred
                for lbl in graph_labels:
                    if lbl not in cc_labels:
                        num_class_relationship_broken += 1
                        break
    belonging = normalize_belonging(belonging)
    if evaluate_method:
        evaluate_cutting(ori_adjs, adjs)
        print("num_class_relationship_broken: ", num_class_relationship_broken,
              " num_multi_shapes_removed:", num_multi_shapes_removed)
    return adjs, edge_weights, ori_classes, belonging, summary_predictions, le_classes  # (total_graph_labels, total_cc_labels)


def evaluate_cutting(ori_adjs, adjs):
    num_shapes = 0
    for i, adj in enumerate(ori_adjs):
        G = nx.Graph(adj)

        # count original patterns
        for pattern in [pattern_N5, pattern_M3, pattern_diamond]:
            GM = isomorphism.GraphMatcher(G, pattern)
            match = list(GM.subgraph_isomorphisms_iter())
            if len(match) > 0:
                num_shapes += 1

    num_preserved = 0
    num_multipleshapes = 0
    for i, adj in enumerate(adjs):
        G = nx.Graph(adj)
        for cc in nx.connected_components(G):
            if len(cc) > 2:
                G1 = G.subgraph(cc)
                pattern_found = False
                for pattern in [pattern_N5, pattern_M3, pattern_diamond]:
                    GM = isomorphism.GraphMatcher(G1, pattern)
                    match = list(GM.subgraph_isomorphisms_iter())
                    if len(match) > 0:
                        if pattern_found:
                            num_multipleshapes += 1
                        num_preserved += 1
                        pattern_found = True
    print(
        f"Num shapes: {num_shapes}, Num Preserved: {num_preserved}, Ratio: {round(num_preserved / num_shapes, 3)}, Num Multipleshapes: {num_multipleshapes}")
    return round(num_preserved / num_shapes, 3)