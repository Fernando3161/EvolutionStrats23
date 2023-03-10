import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import statistics

from common.functions import create_genes
from common.organism import Organism


def meta_one_one_es(
        meta_iters=100,
        internal_iter=2500,
        repetitions=50,
        first_sigma=0.5,
        tau=0.2,
        func = 0,  # ["sphere","rosenbrock", "rastring"]
        genes=2):
    # METHOD = 0  # ["sphere","rosenbrock", "rastring"]

    BASE_SIGMA = first_sigma
    #median_fitness_preview = None
    meta_iter_count = 0
    while meta_iter_count <= meta_iters:
        meta_iter_count += 1
        best_fitness = []
        if meta_iter_count == 1:
            sigma = BASE_SIGMA
        else:
            sigma = BASE_SIGMA * np.exp(tau * np.random.randn())
        for _ in range(repetitions):
            generation = 0
            parent_genes = create_genes(dimensions=genes, space=10)
            parent = Organism(genes=parent_genes, sigma=sigma,
                                    func=func, generation=generation)
            parent.calc_fitness()
            max_generation = internal_iter  # 10, 100, 1000
            while generation < max_generation:
                generation += 1
                child_genes = parent.genes + sigma*np.random.randn(genes)
                child = Organism(genes=child_genes, sigma=sigma,
                                    func=func, generation=generation)
                child.calc_fitness()
                if child.fitness < parent.fitness:
                    parent = child

            best_fitness.append(parent.fitness)

        median_fitness = statistics.median(best_fitness)
        lowest_fitness = min(best_fitness)

        if meta_iter_count == 1:
            # Save first values for comparison
            median_fitnes_preview = median_fitness
            sigmas_list = [sigma]
            sigma_iter_list = [1]
            median_fitness_list = [median_fitness]
            lowest_fitness_list = [lowest_fitness]

        else:
            if median_fitness < median_fitnes_preview:
                # replace the base sigma for next iteration
                BASE_SIGMA = sigma
                # replace the median fitness for comparison
                median_fitnes_preview = median_fitness
                # save data
            sigmas_list.append(sigma)
            sigma_iter_list.append(meta_iter_count)
            median_fitness_list.append(median_fitnes_preview)
            lowest_fitness_list.append(lowest_fitness)

        results = {"sigmas": sigmas_list,
                   "sigma_iter": sigma_iter_list,
                   "medians": median_fitness_list,
                   "best_fit": lowest_fitness_list,
                   }

        df = pd.DataFrame.from_dict(results)

    return df


def main_meta():
    for f,n in [(f,g) for f in [0,1,2] for g in [2, 5, 10]]:
        df = meta_one_one_es(
            meta_iters=200,
            internal_iter=2000,
            repetitions=20,
            first_sigma=0.5,
            tau=0.2,
            genes=n,
            func=f)
        path = os.path.join(os.getcwd(), "results", "one_one_es_meta")
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        # Create a new directory because it does not exist
        filename = f"one_one_es_meta_F_{f}_n_{n}.csv"
        df.to_csv(os.path.join(path, filename),
                  header=True,)
        print(f"Done for Function {f} for N = {n}, META - ES ")


if __name__ == '__main__':
    main_meta()
