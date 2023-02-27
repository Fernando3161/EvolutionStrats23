import random
import numpy as np
import matplotlib.pyplot as plt

from Organism import Organism
import fitness as f
import selection as s
import mutation as m

def create_parents(mu, N, sigma, scaling_factor):
    parents: [Organism] = []

    for k in range(mu):
      parentX = np.random.rand(N) * scaling_factor
      parent = Organism(fit=func(parentX), x = parentX, sigma=sigma)
      parents.append(parent)
    return parents


def create_children(lambd, parents, crossover_function, rho, sigma, func, generation, N):
    children: [Organism] = []
    for i in range(lambd):
        child = muta(crossover(parents, crossover_function, rho), sigma, func, generation, N)
        children.append(child)
    return children

# ToDo: Implement all crossover functions
def crossover(Parents, crossover_function, rho):
    # Intermediate crossover function
    if crossover_function == 0:
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
        child = Organism(fit=func(child_x), x=child_x, born=generation, sigma=child_sigma)
    else:
      print(f'Missing crossover function')
    return child


if __name__ == '__main__':
    N = 100  # Genomes / dimensions
    mu = 20  # Parents
    lambd = 100  # Offsprings
    sigma = 1 / N  # mutation rates (also called stepsize)
    rho = 2
    scaling_factor = 10 # scaling factor for the initial parents
    crossover_function = 0  # ["intermediate_recombination","multi_recombination"]
    max_generation = 1000

    # Iteration over fitness functions
    for func in [f.sphere, f.rastrigen, f.rosenbruck, f.doublesum]:
        print(f'----- Solving for {func.__name__} fitness function. -----')

        # Iteration over selection functions
        for selec in [s.comma]: #[s.plus, s.comma]:
            print(f'Using a {selec.__name__}-selection.')

            # Iteration over mutation functions
            for muta in [m.self_adap, m.dr_self_adap]: # ["n_point", "self_adap", "dr_self_ad", "dr_self_ad2"]
                print(f'Using a {muta.__name__}-selection.')

                parents = create_parents(mu, N, sigma, scaling_factor)
                best_parent = sorted(parents, key=lambda x: x.fit)[0]
                print(f'Best parent fitness: {best_parent.fit}')

                generation = 0
                solution_list = []
                sigma_list = []
                # while best_parent.fit < 0.5:
                while generation < max_generation:
                    generation += 1
                    children = create_children(lambd, parents, crossover_function, rho, sigma, func, generation, N)

                    parents = selec(mu, parents, children)

                    # Documentation of the best fitness of each generation for the plot
                    best_parent = sorted(parents, key=lambda x: x.fit)[0]
                    #print(f'Generation: {generation} Best fitness: {best_parent.fit}')

                    solution_list.append(best_parent.fit)

                    # Sigma documentation for the plot
                    try:
                        sigma_list.append(sum(best_parent.sigma))
                    except TypeError:
                        sigma_list.append(best_parent.sigma)

                    # ToDO: make this listing more variable
                    if len(solution_list) > 21:
                        mean_before = np.mean(solution_list[-21:-2])
                        mean_after = np.mean(solution_list[-20:-1])
                        if mean_after == mean_before:
                            print(f'No improvement exit.')
                            generation = max_generation

                print(f'Generation: {generation} Best fitness: {best_parent.fit} from Generation: {best_parent.born}')

                #plot generation
                x = []
                for i in range(generation):
                    x.append(i)

                fig, ax1 = plt.subplots()


                ax2 = ax1.twinx()
                ax1.plot(x, solution_list, 'blue')
                ax2.plot(x, sigma_list, 'red')

                ax1.set_xlabel('Iterations')
                ax1.set_ylabel('Fitness', color='blue')
                ax2.set_ylabel('Average sigma', color='red')

                plt.show()


