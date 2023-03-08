from Organism import Organism


def plus(mu, parents, children):
    """
    Uses plus selection for the population of new parents.

    Parameters
    ----------
    mu (int): Parent population size
    parents (list): List of individual organisms (class) representing the current parent population.
    children (list): List of individual organisms (class) representing the current children population.

    Returns
    -------
    new_parents (list): List of individual organisms (class) representing the new parent population.
    """
    population: [Organism] = parents + children
    population = sorted(population, key=lambda x: x.fit)
    new_parents = list(population)[:mu]
    return new_parents


# ToDo: parents optional parameter
def comma(mu, parents, children):
    """
    Uses comma selection for the population of new parents.

    Parameters
    ----------
    mu (int): Parent population size
    parents (list): List of individual organisms (class) representing the current parent population.
    children (list): List of individual organisms (class) representing the current children population.

    Returns
    -------
    new_parents (list): List of individual organisms (class) representing the new parent population.
    """
    population: [Organism] = children
    population = sorted(population, key=lambda x: x.fit, reverse=False)

    new_parents = list(population)[:mu]
    return new_parents

# ToDo: parents optional parameter
def comma_1(mu, parents, children):
    """
    Uses comma selection for the population of new parents.

    Parameters
    ----------
    mu (int): Parent population size
    parents (list): List of individual organisms (class) representing the current parent population.
    children (list): List of individual organisms (class) representing the current children population.

    Returns
    -------
    new_parents (list): List of individual organisms (class) representing the new parent population.
    """
    mu = 1
    population: [Organism] = children
    population = sorted(population, key=lambda x: x.fit)

    new_parents = list(population)[:mu]
    return new_parents