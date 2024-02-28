import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import itertools
from lattice import Lattice
from torch import Tensor


def transitive_closure(mat):
    n = np.size(mat[0])

    for i in range(n):
        mat_trans = np.where(np.matmul(mat, mat) > 0., 1., 0.)
        if np.array_equal(mat_trans, mat):
            return mat_trans
        mat = mat_trans
    return mat


# TODO: remove isomorphic lattices in future versions
def has_isomorphic(latt, latt_list):
    G = nx.from_numpy_matrix(latt.adj, create_using=nx.DiGraph)
    for mat in latt_list:
        G_mat = nx.from_numpy_matrix(mat.adj, create_using=nx.DiGraph)
        if nx.is_isomorphic(G, G_mat):
            return True
    return False


def plot_graph_from_lattice(lattice):
    G = nx.DiGraph(Tensor.numpy(lattice.adj))
    nx.draw(G, labels={i: str(i) for i in range(lattice.size)}, pos=nx.planar_layout(G))
    plt.show()


def lattices_generator_per_cardinality(n, sampling, num_lattices_to_sample):
    lattices_list = []
    domain_pairs = [[i, j] for i in range(1, n - 1) for j in range(i + 1, n - 1)]
    num_of_pairs = len(domain_pairs)  # # number of pairs to define possible functions

    if sampling:
        tuple_taken = len(lattices_list)
        while tuple_taken < num_lattices_to_sample:
            candidate = np.random.choice([0, 1], size=num_of_pairs)
            new_matrix = np.triu(np.ones([n, n]))
            for j, p in enumerate(domain_pairs):
                new_matrix[p[0], p[1]] = candidate[j]
            new_matrix = transitive_closure(new_matrix)
            new_lattice = Lattice(new_matrix)
            # if not has_isomorphic(new_matrix, matrices_list):
            if new_lattice.is_a_lattice:
                lattices_list.append(new_lattice)
                tuple_taken += 1
    else:
        assignments = itertools.product([0, 1], repeat=num_of_pairs)
        for assignment in assignments:
            new_matrix = np.triu(np.ones([n, n]))
            for j, p in enumerate(domain_pairs):
                new_matrix[p[0], p[1]] = assignment[j]
            new_matrix = transitive_closure(new_matrix)
            new_lattice = Lattice(new_matrix)
            # if not has_isomorphic(new_matrix, matrices_list):
            if new_lattice.is_a_lattice:
                lattices_list.append(new_lattice)
    return lattices_list


def generate_all_lattices(n, max_cardinality_to_generate_all, num_lattices_to_sample):
    print("SETTING: ", "Generate lattices up to", n, "elements;", " Max cardinality to generate all:",
          max_cardinality_to_generate_all, "; Number of samples:", num_lattices_to_sample)
    lattices_list = []
    sampling = False
    for m in range(2, n + 1):
        print("generating lattices with ", m, " elements")
        if m > max_cardinality_to_generate_all and not sampling:
            sampling = True
        lattices_list += lattices_generator_per_cardinality(m, sampling, num_lattices_to_sample)
    return lattices_list


def prepare_dataset_json(lattices):
    # FIELD: ID, graph_cardinality; LoE_mat; Adj_mat; distribut; modularity; meet_semiDistributive; join_semiDistributive; semiDistributive;
    with open("samples_50_saved.json", "w") as outfile:
        outfile.write("[")
        for i, lattice in enumerate(lattices):
            dictionary = {
                "ID": "G" + str(i),
                "Cardinality": lattice.size,
                "LoE_matrix": (lattice.loe).tolist(),
                "Adj_matrix": (lattice.adj).tolist(),
                "Distributive": lattice.dist,
                "Modular": lattice.mod,
                "Meet_SemiDistributive": lattice.meet_semi_dist,
                "Join_SemiDistributive": lattice.join_semi_dist,
                "SemiDistributive": lattice.semi_dist,
                # "Agruesian_n2": lattice.agru_n2
            }
            # create and write json lattice
            json_object = json.dumps(dictionary)
            if i + 1 < len(lattices):
                outfile.write(json_object + ",\n")
            else:
                outfile.write(json_object + "\n")
        outfile.write("]")
    outfile.close()
