import numpy as np

class Organism:
    """
    A class to represent an individual organism.

    ...
    Attributes
    ----------
    fit : Fitness of organism.
    born : Generation of creation
    x (array): Values for all genomes of an individual organism.
    tau :
    sigma (float) or (array): Scalar of mutation aka step length.
    s_sigma : Evolution path
    z_k : Random vector
    """
    fit: float
    born: int
    x: np.array
    tau: float
    sigma: float
    z_k: np.array

    def __init__(self, fit: float, x: np.array, sigma: float, born: int = 0, z_k : np.array = 0):
        """
        Constructs all the necessary attributes for the organism.

        Parameters
        ----------
        fit (float): Fitness of organism.
        x (array): Values for all genomes of an individual organism.
        sigma (float) or (array): Scalar of mutation aka step length.
        born (int): Generation of organisms creation.
        """

        self.fit = fit
        self.born = born
        self.x = x
        self.sigma = sigma
        self.z_k = z_k