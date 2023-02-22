import numpy as np

def fitness(parent, fitness_function=0):
    # defines the fitness by the spheric function
    if fitness_function == 0:
        fitness = np.dot(parent, parent)
    # defines the fitness by the Rosenbrock function
    elif fitness_function == 1:
        a = 1
        b = 100
        x = parent
        fitness = 0
        for index, y in enumerate(x):
            if index != len(x)-1:
                fitness += b * (y**2 - x[index +1])**2 + (y - 1)**2
        #print(f'Fitness function Rosenbrock not given.')
    # defines the fitness by the Rastring function
    elif fitness_function == 2:
        a = 10
        x = parent # np.array something
        fitness = a * len(x) + sum([(x**2 - a * np.cos(2 * np.pi * x)) for x in x])
    elif fitness_function == 3:
        print(f'Fitness function Doublesum not given.')
    else:
        print(f'Fitness function not given.')
    return fitness