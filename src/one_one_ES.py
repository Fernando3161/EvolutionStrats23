import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from common.functions import create_genes
from common.organism import Organism


def one_one_ES(n=5, sigma=0.1, func=0, generations=1000, 
               APPLY_LIMIT=False, rechenberg = False):
    # Initialisation
    parent_genes = create_genes(n)
    parent = Organism(genes=parent_genes, sigma=sigma, func=func, generation=0)
    parent.calc_fitness()
    genes_ = [parent.genes]
    fitness_ = [parent.fitness]
    sigma_ = [parent.sigma]

    generation = 0
    while generation < generations:
        generation += 1
        child_genes = parent.genes + sigma*np.random.randn(n)
        child = Organism(genes=child_genes, sigma=sigma,
                         func=func, generation=generation)
        child.calc_fitness()
        if rechenberg:
            d = np.sqrt(n+1)
            faktor= 1 if child.fitness < parent.fitness else 0
            sigma=sigma*np.exp(1/d*(faktor-1/5))
        if child.fitness < parent.fitness:
            parent = child
        # fancy exit algorithm (because I am bored)
        limit = int(generations/100)+1
        if len(fitness_) > limit and APPLY_LIMIT:
            mean_before = np.mean(fitness_[-limit:-2])
            mean_after = np.mean(fitness_[-limit+1:-1])
            if mean_after == mean_before:
                generation = generations

        genes_.append(parent.genes)
        fitness_.append(parent.fitness)
        sigma_.append(parent.sigma)

    results = {"genes": genes_, "fitness": fitness_, "sigma": sigma_}
    return results


def main_one_one_ES():
    # Study of the functions using several configuration parameters
    N = [2, 5, 10]
    FUNC = [0, 1, 2]
    RECH = [False, True]

    for n, f,r in [(x, y, z) for x in N for y in FUNC for z in RECH]:
        results = one_one_ES(n=n, sigma=0.05, func=f,
                             generations=10000, APPLY_LIMIT=False,
                             rechenberg=r)
        df = pd.DataFrame.from_dict(results)
        path = os.path.join(os.getcwd(), "results", "one_one_es")
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        # Create a new directory because it does not exist
        if r:
            filename = f"one_one_es_F_{f}_n_{n}_rech.csv"
        else:
            filename = f"one_one_es_F_{f}_n_{n}.csv"
        df.to_csv(os.path.join(os.getcwd(), "results",
                  "one_one_es", filename),
                  header=True,)
        print(f"Done for Function {f} for N = {n}, Rechenberg = {r}")


if __name__ == '__main__':
    main_one_one_ES()
