import numpy as np

def calc_fitness(child, func=None):
    func_list =  ["sphere","rosenbrock", "rastrigin"]
    func = func_list[func]
    if func not in ["sphere", "rosenbrock", "rastrigin"]:
        raise ValueError("Fitness function not recognized")
    # Fitnessfunktion ist die Multiplikation aller N Elemente
    if func == "sphere":
        fitness = np.dot(child.x, child.x)
        return fitness

    if func == "rosenbrock":
        fitness = 0
        for i in range(len(child.x) - 1):
            x = child.x[i]
            y = child.x[i + 1]
            fitness += 100 * (x * x - y) ** 2 + (x - 1) ** 2
        return fitness

    if func == "rastrigin":
        a = 10
        n = len(child.x)
        fitness = a * n
        for i in range(n):
            x = child.x[i]
            fitness += x * x - a * np.cos(2 * np.pi * x) # wikipedia says minus!!
        return fitness
    return

def sphere(x):
    """
    Sphere fitness function.

    Parameters
    ----------
    x (array): Values for all genomes of an individual organism.

    Returns
    -------
    fitness (float): Fitness value of an individual.
    """
    fitness = np.dot(x, x)
    return fitness


def rastrigen(x):
    """
    Rastrigen fitness function.

    Parameters
    ----------
    x (array): Values for all genomes of an individual organism.

    Returns
    -------
    fitness (float): Fitness value of an individual.
    """
    fitness = 10 * len(x) + sum([x ** 2 + 10 * np.cos(2 * np.pi * x) for x in x])
    return fitness


def rosenbruck(x):
    """
    Rosenbruck fitness function.

    Parameters
    ----------
    x (array): Values for all genomes of an individual organism.

    Returns
    -------
    fitness (float): Fitness value of an individual.
    """
    fitness = sum([100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)])
    return fitness


def doublesum(x):
    """
    Doublesum fitness function.

    Parameters
    ----------
    x (array): Values for all genomes of an individual organism.

    Returns
    -------
    fitness (float): Fitness value of an individual.
    """
    fitness = sum([sum(x[:i]) ** 2 for i, _ in enumerate(x)])
    return fitness