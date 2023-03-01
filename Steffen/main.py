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
        z_k = np.random.rand(n)
        x_k =  z_k * scaling_factor
        parent = Organism(fit=fitn(x_k), x = x_k, sigma=sigma, z_k=z_k) # Hier
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
    n = 10  # Genomes / dimensions
    mu = 20  # Parents
    lambd = 100  # Offsprings
    s_sigma = np.zeros(n) # Initial evolution path
    rho = 2
    scaling_factor = 10 # scaling factor for the initial parents
    print(f'With: μ={mu}, λ={lambd} and {n} dimension(s).')

    c_sigma = np.sqrt(1 / (n + 1))
    d = 1 + np.sqrt(1 / n)

    plot_generation = 0 # Do we want plots?

    "Iteration over fitness functions"
    for fitn in [f.sphere, f.rastrigen, f.rosenbruck, f.doublesum]: #[f.sphere, f.rastrigen, f.rosenbruck, f.doublesum]:
        print(f'\n')
        print(f'-------------------- Results for {fitn.__name__} fitness function. --------------------')

        "Iteration over mutation functions"
        for muta in [m.gaus_muta, m.self_adap, m.dr_self_adap, m.evol_path]:  # [m.gaus_muta, m.self_adap, m.dr_self_adap, m.evol_path]

            "Iteration over selection functions"
            for selec in [s.comma_1]: #[s.plus, s.comma, s.comma_1]:
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

                    old_parent = best_parent

                    children = create_children(old_parent, sigma, fitn, generation, n)

                    new_parents = selec(mu, parents, children)
                    best_parent = sorted(new_parents, key=lambda x: x.fit, reverse=False)[0]
                    if best_parent.fit < old_parent.fit:
                        old_parent = best_parent

                    solution_list.append(best_parent.fit)

                    "Selection process defines new sigma - is not part of selection function."
                    # ToDo: Can this be part of the selection function
                    if muta == m.self_adap or muta == m.dr_self_adap:
                        sigma = old_parent.sigma
                    elif muta == m.evol_path:
                        z = old_parent.z_k
                        sigma = old_parent.sigma
                        s_sigma = (1 - c_sigma) * s_sigma + c_sigma * z  # eq. 9
                        sigma = best_parent.sigma * np.exp((c_sigma / d) * (np.linalg.norm(s_sigma) / np.sqrt(n-1) - 1))  # eq. 10

                    "Sigma documentation for the plot"
                    try:
                        sigma_list.append(sum(old_parent.sigma))
                    except TypeError:
                        sigma_list.append(old_parent.sigma)

                    "Exit if no improvement of results"
                    if len(solution_list) > 100: # 21
                        mean_before = np.mean(solution_list[-100:-2]) # [-21:-2])
                        mean_after = np.mean(solution_list[-99:-1]) # [-20:-1])
                        if mean_after == mean_before:
                            max_generation = generation

                dt2 = datetime.datetime.now() # Get last timestemp for duration calculation
                dt = dt2 - dt1 # Get timestep delta
                "Exit print after finishing"
                if dt.min < timedelta(seconds=60):
                    if selec == s.plus:
                        if muta == m.gaus_muta:
                            print(f'(1+λ) Gaussian mutation: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.seconds} seconds')
                        elif muta == m.self_adap:
                            print(f'(1+λ) Self-Adaptation: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.seconds} seconds')
                        elif muta == m.dr_self_adap:
                            print(f'(1+λ) De-randomized SA: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.seconds} seconds')
                        elif muta == m.evol_path:
                            print(f'(1+λ) Evolution path: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.seconds} seconds')
                    elif selec == s.comma:
                        if muta == m.gaus_muta:
                            print(f'(1,λ) Gaussian mutation: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.seconds} seconds')
                        elif muta == m.self_adap:
                            print(f'(1,λ) Self-Adaptation: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.seconds} seconds')
                        elif muta == m.dr_self_adap:
                            print(f'(1,λ) De-randomized SA: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.seconds} seconds')
                        elif muta == m.evol_path:
                            print(f'(1,λ) Evolution path: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.seconds} seconds')
                    elif selec == s.comma_1:
                        if muta == m.gaus_muta:
                            print(
                                f'(1,1) Gaussian mutation: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.seconds} seconds')
                        elif muta == m.self_adap:
                            print(
                                f'(1,1) Self-Adaptation: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.seconds} seconds')
                        elif muta == m.dr_self_adap:
                            print(
                                f'(1,1) De-randomized SA: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.seconds} seconds')
                        elif muta == m.evol_path:
                            print(
                                f'(1,1) Evolution path: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.seconds} seconds')
                else:
                    print(f'{selec.__name__} & {muta.__name__}: {old_parent.fit} from generation: {old_parent.born} Duration: {dt.min} seconds')

                "Plot generation"
                if plot_generation == 1:
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
                    if best_parent.fit < 0.000001:
                        plt.savefig(f'Plots/{fitn.__name__}_{selec.__name__}_{muta.__name__}_convert.png')
                    else:
                        plt.savefig(f'Plots/{fitn.__name__}_{muta.__name__}_{selec.__name__}.png')