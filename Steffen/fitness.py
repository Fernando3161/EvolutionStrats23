import numpy as np

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