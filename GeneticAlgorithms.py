import numpy as np
from deap import base, creator, tools, algorithms
import random
import FuzzyAlgorithms
import Tools

def initIndividual(icls, content):
    return icls(content)

def initPopulation(pcls, ind_init, min, max, size_of_initial_population):
    contents = []
    for i in range(size_of_initial_population):
        param = []
        for j in range(len(min)):
            r = random.uniform(min[j], max[j])
            p = random.uniform(min[j], r)
            q = random.uniform(r, max[j])
            param.append([r, p, q])
        contents.append(Tools.transform_parameters_to_indyvidual(param))
    return pcls(ind_init(c) for c in contents)

def checkBounds(min, max, delta):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
               for i in range(len(min)):
                   for j in range(3):
                       if child[3*i + j] > max[i]:
                           child[3*i + j] = max[i] - delta
                       elif child[3*i + j] < min[i]:
                           child[3*i + j] = min[i] + delta
                   if child[1] > child[0]:
                       child[1] = child[0] - delta / 2
                   if child[2] < child[0]:
                       child[2] = child[0] + delta / 2
            return offspring
        return wrapper
    return decorator

def update_loss_of_indyvidual(indyvidual, X, y_bin, min, max, population, hof):
    parameters = Tools.transform_indyvidual_to_parameters(indyvidual)
    output = FuzzyAlgorithms.aggregated_output(X, parameters, min, max)
    rmse = Tools.RMSE(output, y_bin)
    indyvidual.fitness.values = rmse,
    hof.update(population)
    return rmse

def run_genetic_algorithm(X, y_bin, delta, min, max, cxpb, mutpb, start_population_size, size_of_offspring, number_of_epochs):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("individual_guess", initIndividual, creator.Individual)
    toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess)
    toolbox.register("select", tools.selTournament, size_of_offspring, size_of_offspring)
    toolbox.register("mate", tools.cxSimulatedBinary, eta=1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.decorate("mate", checkBounds(min, max, delta))
    toolbox.decorate("mutate", checkBounds(min, max, delta))
    population = toolbox.population_guess(min, max, start_population_size)
    hof = tools.HallOfFame(1)
    avg_error_on_population = []
    for i in range(number_of_epochs):
        indyvidual_errors = []
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        for indyvidual in offspring:
            indyvidual_errors.append(update_loss_of_indyvidual(indyvidual, X, y_bin, min, max, population, hof))
        population[:] = offspring
        avg_error_on_population.append(np.mean(indyvidual_errors))
        print("Epoch : {} avg RMSE for population : {} hof : {}".format(i + 1, np.mean(avg_error_on_population[len(avg_error_on_population) - 1]), hof[0]))
    return hof[0]
