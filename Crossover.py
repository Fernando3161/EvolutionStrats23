import numpy as np
import random

import Organism

def crossover(crossover_function):
    if crossover_function == 1:
        crossover =

def n_point(parents_two):
    parent1 = parents_two[0]
    parent2 = parents_two[1]
    point = random.randint(1, len(parent1.val) - 1)  # 3
    genes1 = parent1.val
    genes2 = parent2.val
    genes_child1_a = genes1[0:point]
    genes_child1_b = genes2[point:]
    genes_child_1 = np.concatenate((genes_child1_a, genes_child1_b), axis=None)
    genes_child2_a = genes2[0:point]
    genes_child2_b = genes1[point:]
    genes_child_2 = np.concatenate((genes_child2_a, genes_child2_b), axis=None)
    #genes_child_2 = genes_child2_a + genes_child2_b

    return genes_child_1, genes_child_2

def intermediate_recombination(p: [Organism], small_rho: int) -> np.array:
    parents: [Organism] = []
    sum = 0
    for i in range(small_rho):
        parent: Organism = random.choice(p)
        parents.append(parent)
        sum += parents[i].x
    x = sum / small_rho
    return x