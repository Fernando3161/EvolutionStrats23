from Organism import Organism


def plus(mu, parents, children):
    population: [Organism] = parents + children
    population = sorted(population, key=lambda x: x.fit)
    new_parents = list(population)[:mu]
    return new_parents

def comma(mu, parents, children):
    population: [Organism] = children
    population = sorted(population, key=lambda x: x.fit)

    new_parents = list(population)[:mu]
    return new_parents