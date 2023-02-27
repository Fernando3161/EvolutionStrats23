import numpy as np
from Organism import Organism



def n_point(child, sigma, func, generation, N):
    child_k = child.x + np.random.normal(0, sigma, len(child.x))
    child = Organism(fit=func(child_k), x=child_k, born=generation, sigma=sigma)
    return child


def self_adap(child, sigma, func, generation, N):
    tau = 1 / np.sqrt(N)
    e_k = tau * np.random.randn(N)  # eq. 5
    z_k = np.random.randn(N)  # eq. 6
    sigma_k = child.sigma * np.exp(e_k)  # eq. 7 # parent.sigma
    x_k = child.x + sigma_k * z_k  # eq. 8
    child = Organism(fit=func(x_k), x=x_k, born=generation, sigma=sigma_k)
    return child


#ToDo: scheint nicht zu funktionieren
def dr_self_adap(child, sigma, func, generation, N):
    tau = 1 / 3
    d = np.sqrt(N)
    d_i = N
    e_k = tau * np.random.randn(N)  # eq. 5
    z_k = np.random.randn(N)  # eq. 6
    x_k = child.x + np.exp(e_k) * sigma * z_k  # eq. 7
    sigma_k = sigma * np.exp(1/d_i) * (np.linalg.norm(z_k)/ N - 1) * np.exp(e_k/d) # eq. 8 Lehrbuch
    child = Organism(fit=func(x_k), x=x_k, born=generation, sigma=sigma_k)
    return child


# Funktion von Lea
#ToDo: Testen
def dr_self_adap2(child, sigma, func, generation, N):
    gen = 100
    lambd = 100
    parent = Organism(fit=func(child.x), x=child.x, born=generation, sigma=sigma)
    best = parent

    tau = 1/3
    n = N
    d = 1.3 #np.sqrt(N)
    di = 10 #N

    for g in range(gen):
        pop = []
        for _ in range(lambd):
            xi = tau * np.random.randn(1)  # positive random number scalded by tau
            z = np.random.randn(n)  # vector containing random integers which will be scaled by sigma
            sig_i = parent[1] * np.exp(xi)  # new individual sigma that is derived by old sigma scaled by e^xi
            offspring = parent[0] + sig_i * z
            sig_i *= np.exp((np.linalg.norm(z) ** 2 / n - 1) / di) * np.exp(xi / d)  # de-randomization
            pop.append((offspring, sig_i, func(offspring)))  # safe the offspring, its sigma, z and fitness
            # evaluation & selection
        parent = sorted(pop, reverse=False, key=lambda x: x[2])[0]
        if parent[2] < best[2]:
            best = parent
        # Print results of current gen for user
        #if print_results:
         #   print(f"The top result of Gen {g} is", parent[2])
    return best