import numpy as np
from Organism import Organism


def gaus_muta(child, sigma, fitn, generation, n):
    """
    Uses the self-adaptation for the mutation of genomes.

    Parameters
    ----------
    child (class): Individual Organsim.
    sigma (float) or (array): Scalar of mutation aka step length.
    fitn (func): Fitness function.
    generation (int): Current generation
    n (int): Dimensions/genomes each individual has

    Returns
    -------
    child (class): Individual Organsim.
    """
    x_k = child.x + sigma * np.random.randn(n)  # eq. 6.1
    child = Organism(fit=fitn(x_k), x=x_k, born=generation, sigma=sigma)
    return child

# ToDo: N optional parameter
def self_adap(child, sigma, fitn, generation, n):
    """
    Uses the self-adaptation for the mutation of genomes.

    Parameters
    ----------
    child (class): Individual Organsim.
    sigma (float) or (array): Scalar of mutation aka step length.
    fitn (func): Fitness function.
    generation (int): Current generation
    n (int): Dimensions/genomes each individual has

    Returns
    -------
    child (class): Individual Organsim.
    """
    tau = 1 / np.sqrt(n)
    e_k = tau * np.random.randn(n)  # eq. 5
    z_k = np.random.randn(n)  # eq. 6
    sigma_k = child.sigma * np.exp(e_k)  # eq. 7 # sometimes better, don't know why
    # sigma_k = sigma * np.exp(e_k)  # eq. 7 # parent.sigma
    x_k = child.x + sigma_k * z_k  # eq. 8
    child = Organism(fit=fitn(x_k), x=x_k, born=generation, sigma=sigma_k, z_k=z_k)
    return child


# ToDo: Check mutlivariate Self-Adaptation
def multi_reco(child, sigma, fitn, generation, n):
    """
    Uses the multi-recombination for the mutation of genomes.

    Parameters
    ----------
    child (class): Individual Organsim.
    sigma (float) or (array): Scalar of mutation aka step length.
    fitn (func): Fitness function.
    generation (int): Current generation
    n (int): Dimensions/genomes each individual has

    Returns
    -------
    child (class): Individual Organsim.
    """
    tau_i = 1 / n ** (1/4)
    e_k = tau_i * np.random.randn(n)  # eq. 5
    z_k = np.random.randn(n)  # eq. 6
    #sigma_k = child.sigma * np.exp(e_k)  # eq. 7 # parent.sigma <- wouldn't this be better?
    sigma_k = sigma * np.exp(e_k) * np.exp(e_k) # eq. 7 # parent.sigma
    x_k = child.x + sigma_k * z_k  # eq. 8
    child = Organism(fit=fitn(x_k), x=x_k, born=generation, sigma=sigma_k, z_k=z_k)
    return child


def dr_self_adap(child, sigma, fitn, generation, n):
    """
    Uses the derandomized self-adaptation for the mutation of genomes.

    Parameters
    ----------
    child (class): Individual Organsim.
    sigma (float) or (array): Scalar of mutation aka step length.
    fitn (func): Fitness function.
    generation (int): Current generation
    n (int): Dimensions/genomes each individual has

    Returns
    -------
    child (class): Individual Organsim.
    """
    tau = 1 / 3
    d = np.sqrt(n)
    d_i = n
    e_k = tau * np.random.randn()  # eq. 5
    z_k = np.random.randn(n)  # eq. 6
    x_k = child.x + np.exp(e_k) * sigma * z_k  # eq. 7
    sigma_k = sigma * np.exp(1/d_i) * (np.linalg.norm(z_k) / np.sqrt(n-1) - 1) * np.exp(e_k / d) # eq. 8 Lehrbuch # Problem 1
    child = Organism(fit=fitn(x_k), x=x_k, born=generation, sigma=sigma_k)
    return child


def evol_path(child, sigma, fitn, generation, n):
    z_k = np.random.randn(n)
    x_k = child.x + sigma * z_k  # eq. 6.1
    child = Organism(fit=fitn(x_k), x=x_k, born=generation, sigma=sigma, z_k=z_k)
    return child