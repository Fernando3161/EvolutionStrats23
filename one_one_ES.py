import random
import numpy as np
import matplotlib.pyplot as plt


# Exercise 1.1
# 1+1 ES
# Exercises
# 1. Implement a (1+1)-ES with Gaussian mutation, step size σ = 1.0,
# and optimize the Sphere function with N = 2, 10, 100.
# 2. Test different numbers of generations, e.g., 10, 100, and 1000.

def create_parent(dimensions=2, size=10):
    # Baut eine Liste mit N zufälligen Elementen aus der Normalverteilung / Gaussian
    # FP: The first element is not gaussian distributed, is evenly distributed
    # parent = np.random.randn(N)
    parent = size*np.random.rand(dimensions)
    # parent = [size * random.random() for x in range(dimensions)]
    return parent


def calc_fitness(parent, func):
    func_list =  ["sphere","rosenbrock", "rastring"]
    func = func_list[func]
    if func not in ["sphere", "rosenbrock", "rastring"]:
        raise ValueError("Fitness function not recognized")
    # Fitnessfunktion ist die Multiplikation aller N Elemente
    if func == "sphere":
        fitness = np.dot(parent, parent)
        return fitness

    if func == "rosenbrock":
        fitness = 0
        for i in range(len(parent) - 1):
            x = parent[i]
            y = parent[i + 1]
            fitness += 100 * (x * x - y) ** 2 + (x - 1) ** 2
        return fitness

    if func == "rastring":
        a = 10
        n = len(parent)
        fitness = a * n
        for i in range(len(parent)):
            x = parent[i]
            fitness += x * x - a * np.cos(2 * np.pi * x) # wikipedia says minus!!
        return fitness

# Mutation
def mutation(parent, sigma):
    # Die Mutation addiert auf die Werte der Eltern zufällige Elemente aus der
    # Normalverteilung mit einer Schrittgröße von sigma
    # Check that the N generated is based on a gaussian distribution of 1
    # FP: This list of adding list concatenates, does not add (python is weird)
    # FP: The random normal distribution must have a standard deviation of 1
    random_list = np.random.randn(len(parent))
    # print(random_list)
    sigma_list = sigma * random_list
    mutation = parent+sigma_list
    assert(len(parent)==len(mutation))
    return mutation
    #[a + b for a, b in zip(parent, sigma_list)]
    # return parent + sigma * np.random.randn(N)


def create_child(parent, sigma):
    # Mutiert den parent und kreiert ein child
    child = mutation(parent, sigma)
    return child

# Selection
def selection(parent, child, func):
    # wählt zwischen parent und child das element mit der kleineren Fitness
    if calc_fitness(parent,func=func) < calc_fitness(child, func=func):
        # print(f'parent is better')
        return parent
    else:
        # print(f'child is better')
        return child



if __name__ == '__main__':
    N = 5  # 10. 100
    sigma = 0.1
    method = 2 # ["sphere","rosenbrock", "rastring"]
    #method = "rosenbrock" # rosenbrock, rastring
    #method = "rastring" # rosenbrock, rastring

    # Initialisation
    parent = create_parent(N)
    print(f'parent: {parent}')
    solution_list = []
    solution_list.append(calc_fitness(parent, func=method))

    generation = 0
    max_generation = 1000  # 10, 100, 1000
    while generation < max_generation:
        generation += 1
        child = create_child(parent, sigma)
        parent = selection(parent, child, func=method)
        solution_list.append(calc_fitness(parent, func=method))

        # fancy exit algorithm (because I am bored)
        if len(solution_list) > 11:
            mean_before = np.mean(solution_list[-11:-2])
            mean_after = np.mean(solution_list[-10:-1])
            if mean_after == mean_before:
                generation = max_generation


    print("Last children: ")
    print(parent)
    print("Last fitness: ")
    print(calc_fitness(parent, func=method))
    print("Iterations: ")
    print(len(solution_list))

    # We can do this prettier
    plt.plot(solution_list, color="blue")
    method_t  = ["sphere","rosenbrock", "rastring"][method]
    plt.title(method_t.capitalize())
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Value")
    if method in ["rosenbrock", "rastring"]:
        plt.yscale("log")
    plt.show()


