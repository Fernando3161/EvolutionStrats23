import numpy as np
import matplotlib.pyplot as plt
from Organism import Organism
import random

# define the initialization function
def init_population(mu, n):
    population = []
    for i in range(mu):
        x = np.random.rand(n)
        population.append(x)
    return population


# define the variation function
def variation(population,t):
    n = len(population[0].x)
    offspring = np.zeros(n)
    parent1, parent2 = np.random.choice(len(population), 2, replace=False)
    for i in range(n):
        if np.random.rand() < 0.5:
            offspring[i] = population[parent1].x[i]
        else:
            offspring[i] = population[parent2].x[i]
    x = offspring
    offspring = Organism(x=x,fit=zdt1(x),born=t,sigma=0.2)
    return offspring

def variation2(population,t):
    sigma = 0.2
    n = len(population[0].x)
    f = np.random.randint(0,len(population))
    x = population[f].x + sigma * np.random.randn(n)
    offspring = Organism(x=x, fit=x, born=t, sigma=sigma)
    return offspring
def is_dominated(x1, x2):
    """Returns True if x1 dominates x2, False otherwise."""
    for i in range(len(x1)):
        if x1[i] < x2[i]:
            return False
    return True

def dominated_individuals2(population):
    D = []
    for i in range(len(population)):
        num = 0
        for j in range(len(population)):
            if i != j and is_dominated(population[i].x,population[j].x):
                num += 1
        D.append(num)
    return D


# define the dominated individuals function
def dominated_individuals(population):
    D = []
    for i, x in enumerate(population):
        if all(x.x <= x.x) and any(x.x < x.x):
            D.append(i)
    return D


# define the dz function
def dz(x, population):
    n = len(population[0].x)
    dom_count = 0
    for i in range(len(population)):
        if all(x.x <= population[i].x) and any(x.x < population[i].x):
            dom_count += 1
    dz_value = dom_count / len(population)
    return dz_value


# define the Delta S function
def delta_S(x, population):
    n = len(population[0].x)
    distances = []
    for i in range(len(population)):
        dist = np.linalg.norm(x.x - population[i].x)
        distances.append(dist)
    delta_S_value = np.min(distances)
    return delta_S_value

# define the multi-objective function ZDT 1
def zdt1(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9 / (n - 1) * np.sum(x[1:])
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.array([f1, f2])

def create_population(mu,n):
    population = []
    i = 1
    while i <= mu:
        i += 1
        x = np.random.rand(n)
        parent = Organism(x=x,fit=zdt1(x),sigma=0.2)
        population.append(parent)
    return population

def sms_ES(mu=20, lambd=1,n=5, max_generations=1000, sigma=0.2):
    t = 0 # Generation counter
    population = create_population(mu,n)

    while t < max_generations:
        # variation
        for i in range(lambd):
            o = variation2(population,t)
            population.append(o)


        # find best individual
        while len(population) > mu:
            # dominated individuals
            D = dominated_individuals2(population)

            if len(D) != 0:
                maxi=0
                a_star=0
                for (i,b) in enumerate(D):
                    if b>=maxi:
                        maxi=b
                        a_star=i
            else:
                a_star = np.argmin([delta_S(x, population) for x in population])

            # update population
            population.pop(a_star)

        # update generation counter
        t += 1
        sigma *= 0.1
    return population



if __name__ == '__main__':
    generations = 1000
    mu = 20
    n = 10
    lambd = 20
    sigma = 0.2

    last_population = sms_ES(mu,lambd,n,generations,sigma)
    y_1 = []
    y_2 = []
    for i in range(len(last_population)):
        y_1.append(last_population[i].fit[0])
        y_2.append(last_population[i].fit[1])

    # Find the Pareto optimal solutions
    pareto_front = []
    for i in range(len(y_1)):
        is_pareto = True
        for j in range(len(y_1)):
            if y_1[j] <= y_1[i] and y_2[j] <= y_2[i] and (y_1[j] < y_1[i] or y_2[j] < y_2[i]):
                is_pareto = False
                break
        if is_pareto:
            pareto_front.append((y_1[i], y_2[i]))

    # Sort the Pareto optimal solutions by f1
    pareto_front.sort()

    # Plot the Pareto front
    plt.plot([point[0] for point in pareto_front], [point[1] for point in pareto_front], 'b--')
    plt.scatter(y_1, y_2)
    for point in pareto_front:
        plt.annotate('({:.1f}, {:.1f})'.format(point[0], point[1]), point)
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('Pareto Front')
    plt.show()