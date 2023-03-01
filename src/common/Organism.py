import numpy as np
from .functions import calc_fitness

class Organism(object):
    """General data class to store information
    """

    def __init__(self, genes=None, sigma=None,func=None, generation = None) -> None:
        self.genes = genes
        self.sigma = sigma
        self.func = func
        self.fitness = None
        self.generation = generation
        if genes is not None and func is not None:
            self.calc_fitness()

    
    def calc_fitness(self):
        """Calculates the fitness of the genes
        """
        self.fitness=calc_fitness(genes=self.genes, func = self.func)