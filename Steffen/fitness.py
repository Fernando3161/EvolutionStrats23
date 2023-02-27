import numpy as np

def sphere(x):
    fitness = np.dot(x, x)
    return fitness

def rastrigen(x):
    fitness = 10 * len(x) + sum([(x ** 2 + 10 * np.cos(2 * np.pi * x)) for x in x])
    return fitness

def rosenbruck(x):
    fitness = sum([100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)])
    return fitness

def doublesum(x):
    fitness = sum([sum(x[:i]) ** 2 for i, _ in enumerate(x)])
    return fitness