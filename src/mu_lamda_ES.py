import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from common.functions import create_genes
from common.organism import Organism
"""
This function should contain the followwing posibilites:
mu = 1 
Recombination based on 2 or more partners
Comma or Plus selection
Kill parents if they are OLD generations old
"""


def mu_lamda_ES(n=5, mu=10, lam=20, partners=2, method=",", comb = "cros",
                sigma=0.1, func=0, max_age=None, generations=1000):
    # Initialisation
    parent_genes_ = [create_genes(n) for _ in range(mu)]

    parent_ = [Organism(genes=p, sigma=sigma,
                        func=func, generation=0) for p in parent_genes_]
    [parent.calc_fitness() for parent in parent_]
    genes_ = [[list(parent.genes) for parent in parent_]]
    fitness_ = [[parent.fitness for parent in parent_]]
    fitness_best_ = [min(fitness_[0])]
    sigma_ = [[p.sigma for p in parent_][0]]

    generation = 0
    while generation < generations:
        generation += 1
        child_ = []
        for _ in range(lam):
            # Crossover
            if comb=="cros":
                child_parents = random.sample(parent_, partners)
                child_parents_genes = [c_p.genes for c_p in child_parents]
                child_genes = sum(child_parents_genes)/partners
                # -----------
            # Dominant

            if comb == "dom":
                child_parents = random.sample(parent_, partners)
                child_genes = []
                for py in range(n):
                    gene_pool = [c_p.genes[py] for c_p in child_parents]
                    gene = random.sample(gene_pool,1)[0]
                    child_genes.append(gene)
                child_genes=np.array(child_genes)

            child_genes = child_genes + sigma*np.random.randn(n)
            child = Organism(genes=child_genes, sigma=sigma,
                            func=func, generation=generation)
            child.calc_fitness()
            child_.append(child)

        if method == "c":
            child_ = sorted(child_, key=lambda x: x.fitness, reverse=False)
            parent_ = child_[0:mu]
        elif method == "p":
            child_ = child_+parent_
            if max_age:
                child_ = [
                    child for child in child_ if child.generation-generation <= max_age]
            child_ = sorted(child_, key=lambda x: x.fitness, reverse=False)
            parent_ = child_[0:mu]

        genes_.append([list(p.genes) for p in parent_])
        fitness_.append([p.fitness for p in parent_])
        fitness_best_.append(min([p.fitness for p in parent_]))
        sigma_.append(parent_[0].sigma)

    results = {"genes": genes_, "fitness_best": fitness_best_, "sigma": sigma_}
    return results


def main_mu_lamda_ES():
    # Study of the functions using several configuration parameters
    N = [2, 5, 10]
    FUNC = [0, 1, 2]
    METHOD = ["c", "p"]
    COMB = ["cros", "dom"]
    for n, f, m, co in [(x, y, z, w) for x in N for y in FUNC for z in METHOD for w in COMB]:
        results = mu_lamda_ES(n=n, mu=5, lam=30, 
                              partners=2, method=m,
                              comb=co,
                              sigma=0.1, func=f,
                              generations=2000)
        df = pd.DataFrame.from_dict(results)
        path = os.path.join(os.getcwd(), "results", "mu_lam_es")
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        # Create a new directory because it does not exist
        df.to_csv(os.path.join(os.getcwd(), "results",
                  "mu_lam_es", f"mu_lam_es_F_{f}_n_{n}_m_{m}_c_{co}.csv"),
                  header=True,)
        print(f"Done for Function {f} for N = {n} for Method = {m} for Comb = {co}")


if __name__ == '__main__':
    main_mu_lamda_ES()
