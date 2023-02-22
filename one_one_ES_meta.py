# import random
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from one_one_ES import create_parent, create_child, selection, calc_fitness

if __name__ == '__main__':
    SIGMA_ITERATIONS = 500
    INTERNAL_ITERATIONS = 2500
    LIST_LENGTH = 100
    FIRST_SIGMA = 0.5
    TAU = 0.2
    GENES = 2
    # METHOD = 0  # ["sphere","rosenbrock", "rastring"]

    for METHOD in [0, 1, 2]:
        BASE_SIGMA = FIRST_SIGMA
        median_fitness_preview = None
        sigma_iter_count = 0
        while sigma_iter_count <= SIGMA_ITERATIONS:
            sigma_iter_count += 1
            best_fitness = []
            if sigma_iter_count == 1:
                sigma = BASE_SIGMA
            else:
                sigma =  BASE_SIGMA * np.exp(TAU * np.random.randn())
            for i in range(LIST_LENGTH):
                parent = create_parent(dimensions=GENES, size=10)
                generation = 0
                max_generation = INTERNAL_ITERATIONS  # 10, 100, 1000
                solution_list = []
                while generation < max_generation:
                    generation += 1
                    child = create_child(parent, sigma)
                    parent = selection(parent, child, func=METHOD)
                    solution_list.append(calc_fitness(parent, func=METHOD))

                    # exit algorithm for faster calculation
                    if len(solution_list) > 11:
                        mean_before = np.mean(solution_list[-11:-2])
                        mean_after = np.mean(solution_list[-10:-1])
                        if mean_after == mean_before:
                            generation = max_generation
                            
                best_fitness.append(calc_fitness(parent, func=METHOD))

            median_fitness = statistics.median(best_fitness)
            lowest_fitness = min(best_fitness)

            if sigma_iter_count == 1:
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

                    #replace the median fitness for comparison
                    median_fitnes_preview = median_fitness
                    
                    #save data
                    sigmas_list.append(sigma)
                    sigma_iter_list.append(sigma_iter_count)
                    median_fitness_list.append(median_fitness)
                    lowest_fitness_list.append(lowest_fitness)
                    

        results = {"sigmas": sigmas_list,
                   "sigma_iter": sigma_iter_list,
                   "medians": median_fitness_list,
                   "best_fit": lowest_fitness_list,
                   }

        df = pd.DataFrame.from_dict(results)

        df.to_csv(f"rechenberg_{METHOD}.csv")
        print(f"Done for Method {METHOD}")
