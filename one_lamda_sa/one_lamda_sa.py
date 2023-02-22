import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

current_folder = "C:/Users/ferna/PycharmProjects/evolAlgo01/one_lamda_sa"
import sys
sys.path.append("C:/Users/ferna/PycharmProjects/evolAlgo01")
from one_one_ES import create_parent, create_child, selection, calc_fitness    

ITERATIONS = 1000 #100, 1000
GENOMES = 2  # N
METHOD = 0  # ["sphere","rosenbrock", "rastring"]
LAMDA = 20
SIGMA_START = 1

#1 given tau = 1/sqrt(N)
tau = 1/np.sqrt(GENOMES)

# 2 initialize x E R^N, sigma  E R^+
# in this case only 1 parent
class Organism(object):
    """General data class to store information
    """

    def __init__(self, genes=None, sigma=None,func=None) -> None:
        self.genes = genes
        self.sigma = sigma
        self.func = func
        self.fitness = None

    
    def calc_fitness(self):
        self.fitness=calc_fitness(parent=self.genes, func = self.func)

# Initialize the data saving
for METHOD, LAMDA in [(x,y) for x in [0,1,2] for y in [5,10,20,40]]:
    parent_genes = create_parent(dimensions=GENOMES,size=10)
    sigma = SIGMA_START

    # Create a parent Organism 
    parent = Organism(genes = parent_genes, sigma = sigma, func = METHOD)
    parent.calc_fitness()
    parent_list = [parent.genes]
    fitness_list = [parent.fitness]
    sigma_list = [parent.sigma]

    for iter in range(ITERATIONS):
        child_list = []
        for k in range(LAMDA):
            #psi_k = tau*RAND(0,1)
            psi_k = tau*np.random.randn()
            
            #z_k = VECTOR -> RAND(0,1)
            z_k = np.array([np.random.randn() for _ in range(GENOMES)])
            # sigma based on the random psi
            sigma_k = sigma*np.exp(psi_k)

            # x_k = Children genes
            x_k = parent.genes + sigma_k*z_k

            # Create a children with all the properties
            child_k = Organism(genes = x_k, sigma = sigma_k, func= METHOD)
            child_k.calc_fitness()
            child_list.append(child_k)

        # select best children
        children_sorted=sorted(child_list, key=lambda x: x.fitness, reverse=False)
        if children_sorted[0].fitness<parent.fitness:
            parent= children_sorted[0]
            sigma = parent.sigma

        parent_list.append(parent.genes)
        fitness_list.append(parent.fitness)
        sigma_list.append(parent.sigma)


    results = {"sigmas": sigma_list,
                "parents": parent_list,
                "best_fit": fitness_list,
                }

    df = pd.DataFrame.from_dict(results)
   
    df.to_csv(os.path.join(current_folder,"results", f"one_lamda_sa_M_{METHOD}_L_{LAMDA}.csv"))
    print(f"Done for Method {METHOD} for {LAMDA} children")