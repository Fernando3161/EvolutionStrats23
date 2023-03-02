import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.special import gamma as gamma
from common.functions import create_genes
from common.organism import Organism


def one_lam_evol(n=5, lam= 20, sigma=0.1, func=0, generations=1000):
    # Initialisation
    c_sig= np.sqrt(1/(n))
    d = np.sqrt(n)
    s_sig= np.zeros(n)
    
    parent_genes = create_genes(n)
    parent = Organism(genes=parent_genes, sigma=sigma, func=func, generation=0)
    parent.calc_fitness()
    #parents_=[parent]
    genes_ = [parent.genes ]
    fitness_ = [parent.fitness]
    sigma_ = [parent.sigma]

    generation = 0
    while generation < generations:
        generation += 1
        child_ = []
        for _ in range(lam):
            z_k = np.random.randn(n)
            sigma = parent.sigma
            child_genes = parent.genes+ sigma*z_k
            child = Organism(genes=child_genes, sigma=sigma,z=z_k,
                         func=func, generation=generation)
            child.calc_fitness()
            child_.append(child)
            
        #child_.append(parent)
        child_=sorted(child_, key=lambda x: x.fitness, reverse=False)
        #if child_[0].fitness<parent.fitness:

        parent = child_[0]

        #different formulation:
        # https://eldorado.tu-dortmund.de/bitstream/2003/5427/1/137.pdf
        s_sig = (1-c_sig)*s_sig + np.sqrt((2-c_sig)*c_sig)*parent.z

        e_l = np.sqrt(2)*gamma((n+1)/2)/gamma(n/2)
        parent.sigma  = parent.sigma*np.exp(1/2/d/n * ((np.linalg.norm(s_sig))**2-n))

        #parent.sigma = sigma
        #sigma = [parent.sigma for parent in parents_]   

        genes_.append(parent.genes)
        fitness_.append(parent.fitness)
        sigma_.append(parent.sigma)

    results = {"genes": genes_, "fitness": fitness_, "sigma": sigma_}
    return results


def main_one_lam_evol():
    # Study of the functions using several configuration parameters
    N = [2, 5, 10]
    FUNC = [0, 1, 2]

    for n, f in [(x, y) for x in N for y in FUNC]:
        results = one_lam_evol(n=n, lam = 20, sigma=0.5, func=f,
                             generations=int(2*10**(np.sqrt(n)+1)))
        df = pd.DataFrame.from_dict(results)
        path = os.path.join(os.getcwd(), "results", "one_lam_evol")
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        # Create a new directory because it does not exist
        filename = f"F_{f}_n_{n}.csv"
        df.to_csv(os.path.join(path, filename),
                  header=True,)
        print(f"Done for Function {f} for N = {n}")


if __name__ == '__main__':
    main_one_lam_evol()
