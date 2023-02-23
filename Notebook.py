import random
import numpy as np
import matplotlib.pyplot as plt

def init(N, scaling_factor):
  x =  np.random.rand(N) * scaling_factor
  return x

class Organism:
    fit: float
    born: int
    x: np.array
    tau: float
    sigma: float

    def __init__(self, fit: float, x: np.array, sigma: float, born: int = 0):
        self.fit = fit
        self.born = born
        self.x = x
        self.tau = tau
        self.sigma = sigma

def fitness(x, fitness_function):
  #["sphere", "rastrigen", "rosenbruck"]
  if fitness_function == 0:
    fitness = np.dot(x, x)
  elif fitness_function == 1:
    a = 10
    fitness = a * len(x) + sum([(x**2 - a * np.cos(2 * np.pi * x)) for x in x])
  elif fitness_function == 2:
    a = 1
    b = 100
    fitness = 0
    for index, y in enumerate(x):
      if index == N-1:
        break
      fitness += b * (x[index + 1] - y ** 2) ** 2 + (a - y) ** 2
  else:
    print('No fitness function given')
  return fitness

def crossover(Parents, crossover_function, rho):
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
        child = Organism(fit=fitness(child_x, fitness_function), x=child_x, born=generation, sigma=child_sigma)
    else:
      print(f'Missing crossover function')
    return child

def mutation(child, mutation_function, tau, sigma):
  # mutation_function = 0 # ["1-dimensional", "z dimensional"]
  if mutation_function == 0:
    child_x = child.x + np.random.normal(0, sigma, len(child.x))
    child =  Organism(fit=fitness(child_x, fitness_function), x=child_x, born=generation, sigma=sigma)
  elif mutation_function == 1:
    e_k = tau * np.random.randn(N) # eq. 5
    z_k = np.random.randn(N) # eq. 6
    sigma_k = child.sigma * np.exp(e_k) # eq. 7 # parent.sigma
    child_x = child.x + sigma_k * z_k # eq. 8
    child = Organism(fit=fitness(child_x, fitness_function), x=child_x, born=generation, sigma=sigma_k)
  return child

def create_parents(mu, N, fitness_function, tau, sigma, scaling_factor):
    print(f'Creating initial parent generation.')
    parents: [Organism] = []

    for k in range(mu):
      parentX = init(N, scaling_factor)
      parent = Organism(fit=fitness(parentX, fitness_function), x = parentX, sigma=sigma)
      parents.append(parent)

    print(f'Finished creating initial parent generation.')
    return parents


def create_children(lambd, parents, mutation_function, crossover_function, rho, tau, sigma):
    children: [Organism] = []
    if mutation_function == 0:
        for i in range(lambd):
            child = mutation(crossover(parents, crossover_function, rho), mutation_function, tau, sigma)
            children.append(child)
    elif mutation_function == 1:
        for i in range(lambd):
            child = mutation(crossover(parents, crossover_function, rho), mutation_function, tau, sigma)
            children.append(child)
    return children

def selection(selection_function, mu, parents, children):
  if selection_function == 0:
    population: [Organism] = parents + children
    population = sorted(population, key=lambda x: x.fit)

    new_parents = list(population)[:mu]
  return new_parents


if __name__ == '__main__':
    N = 10  # Genomes / dimensions
    mu = 20  # Parents
    lambd = 100  # Offsprings
    sigma = 1 / N  # mutation rates (also called stepsize)
    tau = 1 / np.sqrt(N)
    rho = 2
    scaling_factor = 10 # scaling factor for the initial parents
    fitness_function = 2  # ["sphere", "rastrigen", "rosenbruck"]
    crossover_function = 0  # ["intermediate_recombination","multi_recombination"]
    selection_function = 0  # ["plus", "comma"]
    mutation_function = 1 # ["1-dimensional", "z dimensional"]

    parents = create_parents(mu, N, fitness_function, tau, sigma, scaling_factor)
    best_parent = sorted(parents, key=lambda x: x.fit)[0]
    print(f'Best parent fitness: {best_parent.fit}')

    generation = 0
    solution_list = []
    sigma_list = []
    #while best_parent.fit < 0.5:
    while generation < 100:
        generation += 1
        children = create_children(lambd, parents, mutation_function, crossover_function, rho, tau, sigma)
        parents = selection(selection_function, mu, parents, children)
        best_parent = sorted(parents, key=lambda x: x.fit)[0]
        #print(f'Generation: {generation} Best fitness: {best_parent.fit}')

        solution_list.append(best_parent.fit)
        if mutation_function == 0:
            sigma_list.append(best_parent.sigma)
        elif mutation_function == 1:
            #ToDo: This does not work if the sigma is 1-dimensional (parents of 0 generation are bestparents)
            try:
                sigma_list.append(sum(best_parent.sigma))
            except TypeError:
                sigma_list.append(sigma)
        else:
            print(f'Sigma List Error')

    print(f'Generation: {generation} Best fitness: {best_parent.fit} from Generation: {best_parent.born}')

    # plot generation
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


