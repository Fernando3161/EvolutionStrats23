import random
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from datetime import timedelta

from Organism import Organism
import fitness as f
import selection as s
import mutation as m


def create_parents(mu, N, sigma, scaling_factor):
    parents: [Organism] = []
    for k in range(mu):
        z_k = np.random.rand(N)
        parentX =  z_k * scaling_factor
        parent = Organism(fit=fitn(parentX), x = parentX, sigma=sigma, z_k=z_k)
        parents.append(parent)
    return parents


def create_children(best_parent, sigma, fitn, generation, n):
    children: [Organism] = []

    for i in range(lambd):
        child = muta(best_parent, sigma, fitn, generation, n)
        children.append(child)
    return children

def create_children_cross(lambd, parents, rho, sigma, fitn, generation, n):
    children: [Organism] = []

    for i in range(lambd):
        child = muta(crossover(parents, rho), sigma, fitn, generation, n)
        children.append(child)
    return children


# ToDo: Implement all crossover functions
# ToDO: Dominant Recombination
# ToDO: Intermediate Recombination
def crossover(Parents, rho):
    "Intermediate crossover function"
    newparents: [Organism] = []
    child: [Organism] = []
    sum_x = 0
    sum_sigma = 0
    for i in range(rho):
      newparent:  Organism = random.choice(Parents)
      newparents.append(newparent)
      sum_x += newparents[i].x
      sum_sigma += newparents[i].sigma
    child_x = sum_x / rho
    child_sigma = sum_sigma / rho
    child = Organism(fit=fitn(child_x), x=child_x, born=generation, sigma=child_sigma)
    return child


if __name__ == '__main__':
    n = 2  # Genomes / dimensions
    mu = 1  # Parents
    lambd = 100  # Offsprings
    s_sigma = np.zeros(n) # Initial evolution path
    rho = 2
    scaling_factor = 10 # scaling factor for the initial parents
    crossover_function = 0  # ["none", "intermediate_recombination","multi_recombination"]
    print(f'With: μ={mu}, λ={lambd} and {n} dimension(s).')

    c_sigma = np.sqrt(1 / (n + 1))
    d = 1 + np.sqrt(1 / n)

    "Iteration over fitness functions"
    for fitn in [f.sphere, f.rastrigen, f.rosenbruck, f.doublesum]: #[f.sphere, f.rastrigen, f.rosenbruck, f.doublesum]:
        print(f'\n')
        print(f'-------------------- Results for {fitn.__name__} fitness function. --------------------')

        "Iteration over mutation functions"
        for muta in [m.dr_self_adap]:  # [m.gaus_muta, m.self_adap, m.dr_self_adap, m.evol_path]

            "Iteration over selection functions"
            for selec in [s.comma]: #[s.plus, s.comma, s.comma_1]:

                generation = 0 # Set generation counter back to zero
                max_generation = 1000 # Set maximum generation
                sigma = 1 / n # Mutation rates (also called stepsize)
                solution_list = [] # Set list of best solutions back to empty
                sigma_list = [] # Set list of best solutions sigmas back to empty

                parents = create_parents(mu, n, sigma, scaling_factor)
                best_parent = sorted(parents, key=lambda x: x.fit)[0]

                dt1 = datetime.datetime.now() # Get first timestemp for duration calculation
                while generation < max_generation:
                    generation += 1

                    "If no crossover is needed proceed with mutation of selected best parents."
                    if crossover_function == 0:
                        old_parent = best_parent
                        children = create_children(old_parent, sigma, fitn, generation, n)

                        parents = selec(mu, parents, children)
                        best_parent = sorted(parents, key=lambda x: x.fit, reverse=False)[0]
                        solution_list.append(best_parent.fit)


                        "Evolution path initialisation"
                        if muta == m.evol_path:
                            if best_parent.fit < old_parent.fit:
                                old_parent = best_parent
                                s_sigma = (1 - c_sigma) * s_sigma + c_sigma * best_parent.z_k  # eq. 9
                                sigma = best_parent.sigma * np.exp((c_sigma / d) * (np.linalg.norm(s_sigma) / n - 1))  # eq. 10

                            s_sigma = (1 - c_sigma) * s_sigma + c_sigma * old_parent.z_k  # eq. 9
                            sigma = old_parent.sigma

                    else:
                        "If crossover is wanted, use create_children_cross function"
                        children = create_children_cross(lambd, parents, rho, sigma, fitn, generation, n)
                        parents = selec(mu, parents, children)

                        "Documentation of the best fitness of each generation for the plot"
                        best_parent = sorted(parents, key=lambda x: x.fit)[0]
                        solution_list.append(best_parent.fit)

                    "Sigma documentation for the plot"
                    try:
                        sigma_list.append(sum(best_parent.sigma))
                    except TypeError:
                        sigma_list.append(best_parent.sigma)

                    # ToDO: make this listing more variable
                    "Exit if no improvement of results"
                    # if len(solution_list) > 21:
                    #     mean_before = np.mean(solution_list[-21:-2])
                    #     mean_after = np.mean(solution_list[-20:-1])
                    #     if mean_after == mean_before:
                    #         max_generation = generation

                dt2 = datetime.datetime.now() # Get last timestemp for duration calculation
                dt = dt2 - dt1 # Get timestep delta
                "Exit print after finishing"
                if dt.min < timedelta(seconds=60):
                    if selec == s.comma or s.comma_1:
                        if muta == m.gaus_muta:
                            print(f'(1, λ) Gaussian mutation: {best_parent.fit} from generation: {best_parent.born} Duration: {dt.seconds} seconds')
                        elif muta == m.self_adap:
                            print(f'(1, λ) Self-Adaptation: {best_parent.fit} from generation: {best_parent.born} Duration: {dt.seconds} seconds')
                        elif muta == m.dr_self_adap:
                            print(f'(1, λ) De-randomized SA: {best_parent.fit} from generation: {best_parent.born} Duration: {dt.seconds} seconds')
                        elif muta == m.evol_path:
                            print(f'(1, λ) Evolution path: {best_parent.fit} from generation: {best_parent.born} Duration: {dt.seconds} seconds')
                    else:
                        print(f'{selec.__name__} & {muta.__name__}: {best_parent.fit} from generation: {best_parent.born} Duration: {dt.min} seconds')
                else:
                    if selec == s.comma or s.comma_1:
                        if muta == m.gaus_muta:
                            print(
                                f'(1, λ) Gaussian mutation: {best_parent.fit} from generation: {best_parent.born} Duration: {dt.min} seconds')
                        elif muta == m.self_adap:
                            print(
                                f'(1, λ) Self-Adaptation (SA): {best_parent.fit} from generation: {best_parent.born} Duration: {dt.min} seconds')
                        elif muta == m.dr_self_adap:
                            print(
                                f'(1, λ) De-randomized SA: {best_parent.fit} from generation: {best_parent.born} Duration: {dt.min} seconds')
                        elif muta == m.evol_path:
                            print(
                                f'(1, λ) Evolution path: {best_parent.fit} from generation: {best_parent.born} Duration: {dt.min} seconds')
                    else:
                        print(f'{selec.__name__} & {muta.__name__}: {best_parent.fit} from generation: {best_parent.born} Duration: {dt.min} seconds')

                "Plot generation"
                x = []
                for i in range(generation):
                    x.append(i)

                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()
                ax1.plot(x, solution_list, 'blue')
                ax2.plot(x, sigma_list, 'red')
                ax1.set_xlabel('Iterations')
                ax1.set_ylabel('Fitness', color='blue')
                ax1.set_yscale("log")
                ax2.set_ylabel('Average sigma', color='red')

                 # plt.show()

                "Plot saving"
                plt.savefig(f'Plots/{fitn.__name__}_{muta.__name__}_{selec.__name__}.png')
                #plt.savefig(f'Plots/{fitn.__name__}_{selec.__name__}_{muta.__name__}.png')