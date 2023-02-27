import numpy as np

class Organism:
    fit: float
    born: int
    x: np.array
    tau: float
    sigma: float

    def __init__(self, fit: float, x: np.array, sigma: float, born: int = 0):
        self.fit = fit
        self.born = born
        self.x = x
        self.sigma = sigma