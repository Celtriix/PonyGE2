import pandas as pd
import numpy as np
from multiprocessing import Pool

from algorithm.parameters import params
from fitness.evaluation import evaluate_fitness
from operators.initialisation import initialisation
from stats.stats import get_stats, stats
from utilities.algorithm.initialise_run import pool_init
from utilities.stats import trackers


def search_loop():
    """
    This is a standard search process for an evolutionary algorithm. Loop over
    a given number of generations.
    
    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """

    if params['MULTICORE']:
        # initialize pool once, if multi-core is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Initialise population
    individuals = initialisation(params['POPULATION_SIZE'])

    if params["USE_DIVERSITY"]:
        yhat = []
        x = params["TRAIN_DATA_X"]
        for i in range(len(individuals)):
            yhat_ind = eval(individuals[i].phenotype)
            if type(yhat_ind) == int:
                yhat_ind = yhat_ind*np.ones(params["TRAIN_DATA_Y"].shape[0])
            yhat.append(yhat_ind)
        params["YHAT_CORRECT"] = (yhat == params["TRAIN_DATA_Y"])


    # Evaluate initial population
    individuals = evaluate_fitness(individuals)

    # Generate statistics for run so far
    get_stats(individuals)

    # Traditional GE
    if params["SAVE_POP"] and params["USE_DIVERSITY"]:
        phenotypes = [None]*((params["GENERATIONS"]+1)*params["POPULATION_SIZE"])
        phenotypes[0:params["POPULATION_SIZE"]] = [x.phenotype for x in individuals]
        fitness_acc = np.zeros((params["GENERATIONS"]+1)*params["POPULATION_SIZE"])
        fitness_acc[0:params["POPULATION_SIZE"]] = [x.fitness[0] for x in individuals]
        fitness_Q = np.zeros((params["GENERATIONS"]+1)*params["POPULATION_SIZE"])
        fitness_Q[0:params["POPULATION_SIZE"]] = [x.fitness[1] for x in individuals]
        gen = np.zeros((params["GENERATIONS"]+1)*params["POPULATION_SIZE"])
    elif params["SAVE_POP"]:
        phenotypes = [None]*((params["GENERATIONS"]+1)*params["POPULATION_SIZE"])
        phenotypes[0:params["POPULATION_SIZE"]] = [x.phenotype for x in individuals]
        fitness = np.zeros((params["GENERATIONS"]+1)*params["POPULATION_SIZE"])
        fitness[0:params["POPULATION_SIZE"]] = [x.fitness for x in individuals]
        gen = np.zeros((params["GENERATIONS"]+1)*params["POPULATION_SIZE"])
    
    for generation in range(1, (params['GENERATIONS']+1)):
        stats['gen'] = generation

        # New generation
        individuals = params['STEP'](individuals)
        
        if params["SAVE_POP"] and params["USE_DIVERSITY"]:
            first_ind = (generation)*params["POPULATION_SIZE"]
            last_ind = (generation+1)*params["POPULATION_SIZE"]
            phenotypes[first_ind:last_ind] = [x.phenotype for x in individuals]
            fitness_acc[first_ind:last_ind] = [x.fitness[0] for x in individuals]
            fitness_Q[first_ind:last_ind] = [x.fitness[1] for x in individuals]
            gen[first_ind:last_ind] = (generation)*np.ones(params["POPULATION_SIZE"])
        elif params["SAVE_POP"]:
            first_ind = (generation)*params["POPULATION_SIZE"]
            last_ind = (generation+1)*params["POPULATION_SIZE"]
            phenotypes[first_ind:last_ind] = [x.phenotype for x in individuals]
            fitness[first_ind:last_ind] = [x.fitness for x in individuals]
            gen[first_ind:last_ind] = (generation)*np.ones(params["POPULATION_SIZE"])            
            
    if params["USE_DIVERSITY"]:
        yhat = []
        x = params["TEST_DATA_X"]
        for i in range(len(individuals)):
            yhat_ind = eval(individuals[i].phenotype)
            if type(yhat_ind) == int:
                yhat_ind = yhat_ind*np.ones(params["TEST_DATA_Y"].shape[0])
            yhat.append(yhat_ind)
        params["YHAT_CORRECT"] = (yhat == params["TEST_DATA_Y"])
    
    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()
    
    if params["SAVE_POP"] and params["USE_DIVERSITY"]:
        dir = params["SAVE_LOC"] + params["FILE_NAME"]
        pd_pop = pd.DataFrame(list(zip(gen.astype(int), 
                                       np.round(fitness_acc,3),
                                       np.round(fitness_Q,3),
                                       phenotypes)),
                                  columns = ["Generation", "Fitness", "Q-Statistic", "Phenotype"])
        # Remove Invalids
        pd_pop = pd_pop.loc[(pd_pop["Fitness"] != np.nan)]
        # Remove Duplicates
        pd_pop = pd_pop.drop_duplicates("Phenotype")
        pd_pop.to_csv(dir, sep = ";", index = False)
    elif params["SAVE_POP"]:
        dir = params["SAVE_LOC"] + params["FILE_NAME"]
        pd_pop = pd.DataFrame(list(zip(gen.astype(int), np.round(fitness,3), phenotypes)),
                                  columns = ["Generation", "Fitness", "Phenotype"])
        # Remove Invalids
        pd_pop = pd_pop.loc[(pd_pop["Fitness"] != np.nan)]
        # Remove Duplicates
        pd_pop = pd_pop.drop_duplicates("Phenotype")
        pd_pop.to_csv(dir, sep = ";", index = False)

    return individuals


def search_loop_from_state():
    """
    Run the evolutionary search process from a loaded state. Pick up where
    it left off previously.

    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """

    individuals = trackers.state_individuals

    if params['MULTICORE']:
        # initialize pool once, if multi-core is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Traditional GE
    for generation in range(stats['gen'] + 1, (params['GENERATIONS'] + 1)):
        stats['gen'] = generation

        # New generation
        individuals = params['STEP'](individuals)

    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()

    return individuals
