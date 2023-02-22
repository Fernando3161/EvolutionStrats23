import numpy as np

class Organism:
    val: np.array
    fit: float
    born: int
    #fitness_method: int

    def __init__(self, val: np.array, fit: float, born: int = 0): #, fitness_method: int = 0):
        self.val = val
        self.fit = fit
        self.born = born
        #self.fitness_method = fitness_method

    #def create_genes(self):
    #    self.val = [max_value * random.random() for x in range(genomes)]

    #def calc_fit(self):
    #    self.fit = fitness(self.val, fitness_function)