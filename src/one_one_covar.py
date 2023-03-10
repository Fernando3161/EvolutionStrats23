import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.special import gamma as gamma
from common.functions import create_genes
from common.organism import Organism


def one_one_covar(n=5, lam= 20, sigma=0.1, func=0, generations=1000, alpha = 20, kappa = 30):
    alpha = 10
    d = np.sqrt(n+1)

    covar = np.identity(n)

    # 1 Parent
    parent_genes = create_genes(dimensions=n,space=5)
    parent = Organism(genes = parent_genes, sigma =sigma, func = func)
    parent.calc_fitness()

    #Data saving
    parent_list = [parent.genes]
    fitness_list = [parent.fitness]
    sigma_list = [parent.sigma]
    covar_max_list = [covar]
    sigma_vector_list=[]
    iter = 0
    A = []

    while iter <= generations: 
        iter +=1
        if iter%kappa == 0 and len(A)>=alpha:
            covar = np.cov(np.transpose(A))
        sigma_vector = sigma* np.random.multivariate_normal(mean=[0 for _ in range(n)], cov = covar)
        child_genes = parent.genes + sigma_vector
        
        # New child
        child = Organism(genes = child_genes, func=func)
        child.calc_fitness()

        # Rechenberg selector
        if child.fitness<=parent.fitness:
            rech= 4/5
        else:
            rech = -1/5
        sigma = sigma*np.exp(1/d*rech)

        if child.fitness<parent.fitness:
            parent = child
            if len(A)>=alpha:
                A = A[1:alpha]
                A.append(parent.genes)
            else:
                A.append(parent.genes)

                
        parent_list.append(parent.genes)
        fitness_list.append(parent.fitness)
        sigma_list.append(sigma)
        covar_max_list.append(covar)
        sigma_vector_list.append(sigma_vector)
    #print(sigma_list)
    sigma_vector_list.append(sigma_vector_list[-1])

    results = {"genes": parent_list, "fitness": fitness_list, "sigma": sigma_list, "covar_mx":covar_max_list }
    return results


def main_one_one_covar():
    # Study of the functions using several configuration parameters
    N = [2, 5, 10]
    FUNC = [0, 1, 2]

    for n, f in [(x, y) for x in N for y in FUNC]:
        results = one_one_covar(n=n, lam = 20, sigma=0.5, func=f,
                             generations=int(5000),
                             alpha= 20,
                             kappa= 20)

        df = pd.DataFrame.from_dict(results)
        path = os.path.join(os.getcwd(), "results", "one_one_covar")
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        # Create a new directory because it does not exist
        filename = f"F_{f}_n_{n}.csv"
        df.to_csv(os.path.join(path, filename),
                  header=True,)
        print(f"Done for Function {f} for N = {n}")


if __name__ == '__main__':
    main_one_one_covar()
