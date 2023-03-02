import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from common.functions import create_genes
from common.organism import Organism


def one_lam_SA_ES(n=5, lam= 20, sigma=0.1, func=0, generations=1000):
    # Initialisation
    tau = 1/np.sqrt(n)
    parents_=[]
    parent_genes = create_genes(n)
    parent = Organism(genes=parent_genes, sigma=sigma, func=func, generation=0)
    parent.calc_fitness()
    parents_.append(parent)
    genes_ = [parent.genes ]
    fitness_ = [parent.fitness]
    sigma_ = [parent.sigma]

    generation = 0
    while generation < generations:
        generation += 1
        child_ = []
        for _ in range(lam):

            psi_k = tau*np.random.randn()
            z_k = np.random.randn(n)
            sigma = parent.sigma
            sigma_k = sigma*np.exp(psi_k)
            child_genes = parent.genes+ sigma_k*z_k
            child = Organism(genes=child_genes, sigma=sigma_k,
                         func=func, generation=generation)
            child.calc_fitness()
            child_.append(child)
        
        child_=sorted(child_, key=lambda x: x.fitness, reverse=False)
        parent = child_[0]

        #sigma = [parent.sigma for parent in parents_]   

        genes_.append(parent.genes)
        fitness_.append(parent.fitness)
        sigma_.append(parent.sigma)

    results = {"genes": genes_, "fitness": fitness_, "sigma": sigma_}
    return results


def main_one_lam_SA_ES():
    # Study of the functions using several configuration parameters
    N = [2, 5, 10]
    FUNC = [0, 1, 2]

    for n, f in [(x, y) for x in N for y in FUNC]:
        results = one_lam_SA_ES(n=n, lam = 20, sigma=0.5, func=f,
                             generations=10000)
        df = pd.DataFrame.from_dict(results)
        path = os.path.join(os.getcwd(), "results", "one_lam_sa_es")
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        # Create a new directory because it does not exist
        filename = f"F_{f}_n_{n}.csv"
        df.to_csv(os.path.join(path, filename),
                  header=True,)
        print(f"Done for Function {f} for N = {n}")


if __name__ == '__main__':
    main_one_lam_SA_ES()
