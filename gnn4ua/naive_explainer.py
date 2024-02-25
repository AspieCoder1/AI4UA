from typing import List

from .datasets.lattice import Lattice


def brute_force_explainer(pos_examples: List[Lattice],
                          neg_examples: List[Lattice]) -> Lattice:
    """
    Brute force explanation function to find minimal omitted lattice
    :return:
    """

    min_lattice = neg_examples[0]

    for neg_ex in neg_examples:
        for pos_ex in pos_examples:
            if neg_ex in pos_ex:
                break
        if neg_ex.size < min_lattice.size:
            min_lattice = neg_ex

    return min_lattice
