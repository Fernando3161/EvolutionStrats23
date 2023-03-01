import numpy as np

def create_genes(dimensions=2, space=10, positives = False):
    """Creates a list with N random elements normally distributed

    Args:
        dimensions (int, optional): Number of genes. Defaults to 2.
        space (int, optional): Average of the random generated values. Defaults to 10.

    Returns:
        array: Array n*1 with the genes
    """

    genes = space*np.random.rand(dimensions)
    if positives:
        genes = np.absolute(genes)
    return genes


def calc_fitness(genes, func=None):
    """Calculates the fitness of the genes

    Args:
        genes (array): Array with the genes
        func (int): Integer to look the function

    Raises:
        ValueError: If the fitness function is not found

    Returns:
        double: fitness value of the function
    """

    func_list =  ["sphere","rosenbrock", "rastrigin"]
    func = func_list[func]
    if func not in ["sphere", "rosenbrock", "rastrigin"]:
        raise ValueError("Fitness function not recognized")
    # Fitnessfunktion ist die Multiplikation aller N Elemente
    if func == "sphere":
        fitness = np.dot(genes, genes)
        return fitness

    if func == "rosenbrock":
        fitness = 0
        for i in range(len(genes) - 1):
            x = genes[i]
            y = genes[i + 1]
            fitness += 100 * (x * x - y) ** 2 + (x - 1) ** 2
        return fitness

    if func == "rastrigin":
        a = 10
        n = len(genes)
        fitness = a * n
        for i in range(n):
            x = genes[i]
            fitness += x * x - a * np.cos(2 * np.pi * x) # wikipedia says minus!!
        return fitness