import random
import numpy as np
import matplotlib.pyplot as plt


def create_parent(genomes=2, max_value=10):
    # Baut eine Liste mit N zufälligen Elementen aus der Normalverteilung / Gaussian
    # FP: The first element is not gaussian distributed, is evenly distributed
    parent = np.random.randn(N)
    #parent = [max_value * random.random() for x in range(genomes)]
    return parent


def fitness(parent, fitness_function=0):
    # defines the fitness by the spheric function
    if fitness_function == 0:
        fitness = np.dot(parent, parent)
    # defines the fitness by the Rosenbrock function
    elif fitness_function == 1:
        a = 1
        b = 100
        x = parent
        fitness = 0
        for index, y in enumerate(x):
            if index != len(x)-1:
                fitness += b * (y**2 - x[index +1])**2 + (y - 1)**2
        #print(f'Fitness function Rosenbrock not given.')
    # defines the fitness by the Rastring function
    elif fitness_function == 2:
        a = 10
        x = parent # np.array something
        fitness = a * len(x) + sum([(x**2 - a * np.cos(2 * np.pi * x)) for x in x])
    elif fitness_function == 3:
        print(f'Fitness function Doublesum not given.')
    else:
        print(f'Fitness function not given.')
    return fitness


class organism:
    fit: float
    born: int
    val: None

    def __init__(self, genomes=1, max_value=10):
        self.max_value = max_value
        self.genomes = genomes
        self.born = 0

    def create_genes(self):
        self.val = create_parent(self.genomes, self.max_value)

    def calc_fit(self):
        self.fit = fitness(self.val, fitness_function)


def create_parents(mu, max_value, genomes, fitness_function=0):
    parents = []
    for i in range(mu):
        org = organism(genomes, max_value)  # create an organism instance with fit=0, born=0, and the random val array
        org.create_genes()
        org.calc_fit()
        parents.append(org)  # add the organism instance to the list
    return parents


def create_2_children(parents):
    # 1-Point Crossover
    # choose a number between 1 and leng(genomes)
    parents_two = random.sample(parents, 2)
    genes_child_1, genes_child_2 = crossover(parents_two)
    genes_child_1_mutated = mutate(genes_child_1)
    genes_child_2_mutated = mutate(genes_child_2)

    child1 = organism(genomes=len(genes_child_1_mutated))
    child1.val = genes_child_1_mutated
    child1.calc_fit()

    child2 = organism(genomes=len(genes_child_2_mutated))
    child2.val = genes_child_2_mutated
    child2.calc_fit()

    return child1, child2


def create_all_children(parents, lam=100):
    children_list = []
    while len(children_list) < lam:
        child1, child2 = create_2_children(parents)
        children_list.append(child1)
        children_list.append(child2)

    children_list = children_list[0:100]
    return children_list


def crossover(parents_two):
    parent1 = parents_two[0]
    parent2 = parents_two[1]
    point = random.randint(1, len(parent1.val) - 1)  # 3
    genes1 = parent1.val
    genes2 = parent2.val
    genes_child1_a = genes1[0:point]
    genes_child1_b = genes2[point:]
    genes_child_1 = np.concatenate((genes_child1_a, genes_child1_b), axis=None)
    genes_child2_a = genes2[0:point]
    genes_child2_b = genes1[point:]
    genes_child_2 = np.concatenate((genes_child2_a, genes_child2_b), axis=None)
    #genes_child_2 = genes_child2_a + genes_child2_b

    return genes_child_1, genes_child_2


def mutate(genes_child):
    random_list = [np.random.normal(loc=0.0, scale=1.0) for x in range(len(genes_child))]
    # print(random_list)
    sigma = 1 / len(genes_child)
    sigma_list = [sigma * r for r in random_list]
    return [a + b for a, b in zip(genes_child, sigma_list)]

    child1 = organism(genomes=len(genes_child_1))
    child1.val = genes_child_1
    child1.calc_fit()

    child2 = organism(genomes=len(genes_child_2))
    child2.val = genes_child_2
    child2.calc_fit()

    return [child1, child2]


if __name__ == '__main__':
    N = 10  # Genomes
    mu = 20  # Parents
    lam = 100  # Offsprings
    sigma = 1 / N  # mutation rates (also called stepsize)
    fitness_function = 1 #["sphere", "rosenbrock", "rastrigen", "doublesum"]
    selection = ["plus", "comma"]
    max_generation = 100

    parents = create_parents(mu, genomes=N, max_value=10, fitness_function=fitness_function)

    solution_list = []
    generation = 0
    while generation < max_generation:
        generation += 1
        children = create_all_children(parents, lam)
        all_pop = parents + children

        sorted_pop = sorted(all_pop, key=lambda x: x.fit, reverse=False)
        new_parents = sorted_pop[0:mu]
        parents = new_parents
        best_solution = parents[0].fit
        print("Generation", generation, "fit", best_solution)

        # add stagnation evolution termination factor
        solution_list.append(best_solution)
        #ToDO: make this listing more variable
        if len(solution_list) > 21:
            mean_before = np.mean(solution_list[-21:-2])
            mean_after = np.mean(solution_list[-20:-1])
            if mean_after == mean_before:
                generation = max_generation

        # plot generation
    plt.plot(solution_list, color="blue", )
    plt.show()

    # TODO implement the others
    # n-point Crossover
    # Arithmetic Crossover/ Intermediate Recombination
    # Dominant Crossover
    # Fitness functions: Sphere, Doublesum, Rosenbrock, Rastigen
    # Selection: Plus, Comma
    # Rechenberg rule - adaptive stepsize
    # restarts
    # wilcoxon test, student t test

    # TODO implement the others
    # n-point Crossover
    # Arithmetic Crossover/ Intermediate Recombination
    # Dominant Crossover
