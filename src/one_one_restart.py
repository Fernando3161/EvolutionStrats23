import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from common.functions import create_genes
from common.organism import Organism


def one_one_ES(n=5, parent_genes = None, sigma=0.1, func=0, generations=1000,
               APPLY_LIMIT=False, rechenberg=False):
    # Initialisation
    if parent_genes is None:
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
            faktor = 1 if child.fitness < parent.fitness else 0
            sigma = sigma*np.exp(1/d*(faktor-1/5))
        if child.fitness < parent.fitness:
            parent = child

        genes_.append(parent.genes)
        fitness_.append(parent.fitness)
        sigma_.append(parent.sigma)

        # fancy exit algorithm (because I am bored)
        limit = int(generations/10)+1
        if len(fitness_) > limit and APPLY_LIMIT:
            before_ = np.mean(fitness_[-limit:-2])
            after_ = np.mean(fitness_[-limit+1:-1])
            if np.array_equal(before_,after_):
                generation = generations

    results = {"genes": genes_, "fitness": fitness_, "sigma": sigma_}
    return results


def main_restart():
    # Study of the functions using several configuration parameters
    N = [2, 5, 10]
    FUNC = [0, 1, 2]
    RECH = [False, True]

    for n, f, rech in [(x, y, z) for x in N for y in FUNC for z in RECH]:
        restart = []
        start_point = []
        final_point = []
        final_conv = []
        parent_genes = None
        best_fit_curve = None
        for r in range(100):
            res_restart = one_one_ES(n=n, parent_genes = parent_genes,
                                     sigma=0.05, func=f,
                                     generations=1000, APPLY_LIMIT=False,
                                     rechenberg=True)
            restart.append(r)
            start_point.append(res_restart["genes"][0])
            final_point.append(res_restart["genes"][-1])
            final_conv.append(res_restart["fitness"][-1])
            parent_genes =res_restart["genes"][-1]

            #save the best convergence flow
  
            if len(final_conv)>1:
                if final_conv[-1]<= min(final_conv):
                    best_fit_curve = res_restart["fitness"]


            # if it is convergin to the same point, next one is far away
            if np.array_equal(final_point[-5:-1], final_point[-6:-2]):
                distance_genes= create_genes(dimensions=n, space = 10)
                parent_genes = parent_genes + distance_genes
            else:
                parent_genes = None

        results = {"restart": restart,
                   "start_point": start_point,
                   "final_point": final_point,
                   "final_conv": final_conv
                   }
        df = pd.DataFrame.from_dict(results)
        path = os.path.join(os.getcwd(), "results", "one_one_es_restart")
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        # Create a new directory because it does not exist
        if rech:
            filename = f"one_one_es_restart_F_{f}_n_{n}_rech.csv"
        else:
            filename = f"one_one_es_restart_F_{f}_n_{n}.csv"
        df.to_csv(os.path.join(path, filename),
                  header=True,)
        
        pd.DataFrame(best_fit_curve).to_csv(os.path.join(path, "results_" + filename))
        print(f"Done for Function {f} for N = {n}, Rechenberg = {rech}, Restart")


if __name__ == '__main__':
    main_restart()
