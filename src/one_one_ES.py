import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


from common.functions import create_genes
from common.Organism import Organism

# Exercise 1.1
# 1+1 ES
# Exercises
# 1. Implement a (1+1)-ES with Gaussian mutation, step size σ = 1.0,
# and optimize the Sphere function with N = 2, 10, 100.
# 2. Test different numbers of generations, e.g., 10, 100, and 1000.

# Mutation
def create_mutation(parent, sigma=None):
    """Mutation adds the parent values to a random value of the sigma
    Args:
        parent_genes (Organism): An Organism acing as parent
        sigma (double): Scale factor for mutation

    Returns:
        array: array with genes of the mutation
    """
    if sigma is None:
        sigma = parent.sigma

    random_list = np.random.randn(len(parent.genes))
    mutation = sigma * random_list
    child = Organism(genes = mutation, sigma = sigma, func=parent.func)
    #child.calc_fitness()
    return child

"""
def create_child(parent, sigma):
    # Mutiert den parent und kreiert ein child
    child = mutation(parent, sigma)
    return child
"""
# Selection
def selection(parent, child):
    """Selects between a parent and one children

    Args:
        parent (Organism): A parent organism
        child (Organism): A child organism

    Returns:
        Organism: organism with the best selection
    """
    # wählt zwischen parent und child das element mit der kleineren Fitness
    if parent.fitness<child.fitness:
        
        return parent
    else:
        
        return child

def one_one_ES(N=5, SIGMA = 0.1, FUNC = 0, MAX_GENS = 1000, APPLY_LIMIT = False):
    # Initialisation
    parent_genes = create_genes(N)
    parent = Organism(genes =parent_genes, sigma = SIGMA, func = FUNC, generation = 0 )
    
    solution_list = []
    solution_list.append(parent.fitness)

    generation = 0
    while generation < MAX_GENS:
        generation += 1
        child = create_mutation(parent)
        parent = selection(parent, child)
        solution_list.append(parent.fitness)

        # fancy exit algorithm (because I am bored)
        limit = int(MAX_GENS/100)+1
        if len(solution_list) > limit and APPLY_LIMIT:
            mean_before = np.mean(solution_list[-limit:-2])
            mean_after = np.mean(solution_list[-limit+1:-1])
            if mean_after == mean_before:
                generation = MAX_GENS

    # Save the results
    return solution_list

def main():
    # Study of the functions using several configuration parameters
    N = [2,5,10]
    FUNC = [0,1,2]
    results = {}

    for n, f in [(x,y) for x in N for y in FUNC]:
        fitness = one_one_ES(N=n, SIGMA = 0.1, FUNC = f, MAX_GENS = 1000, APPLY_LIMIT = False)
        results[str((n,f))]=fitness

    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(os.getcwd(),"src","results", f"one_one.csv"))



if __name__ == '__main__':
    main()