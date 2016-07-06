# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, time, numpy, heapq, pandas, copy, math

class GeneticAlgorithm(object):
    """Represents a genetic algorigthm structure."""
    def __init__(self, individual_factory, population_size=100, reproduction=0.15, crossover=0.8, mutation=0.05, elitism=False, termination_criteria=None):
        self.individual_factory = individual_factory
        self.population_size = population_size
        self.reproduction = reproduction
        self.crossover = crossover
        self.mutation = mutation
        self.elitism = elitism
        if termination_criteria is None:
            self.termination_criteria = ga.NumberOfGenerationsTerminationCriteria()
        else:
            self.termination_criteria = termination_criteria

        self.population = []
        self.population_fitness = []
        self.generation_info = pandas.DataFrame([], columns=["generation", "mean", "min", "std"])
        self.generation = 0

    def set_params(self, **params):
        if params.has_key("population_size"):
            self.population_size = params["population_size"]

        if params.has_key("operators_rate"):
            self.reproduction = params["operators_rate"][0]
            self.crossover = params["operators_rate"][1]
            self.mutation = params["operators_rate"][2]

        if params.has_key("elitism"):
            self.elitism = params["elitism"]

        if params.has_key("termination_criteria"):
            self.termination_criteria = params["termination_criteria"]

        # Reset data
        self.population = []
        self.population_fitness = []
        self.generation_info = pandas.DataFrame([], columns=["generation", "mean", "min", "std"])
        self.generation = 0

    def init_population(self):
        """Initializes a population of individuals using a provided function."""
        for idx in xrange(0, self.population_size):
            individual = self.individual_factory.create()
            self.population.append(individual)

        self.population_fitness = numpy.asarray(map(lambda individual: individual.get_fitness(), self.population))

        # In order to roulette wheel selection work with negative values, 
        # we sum all fitness values to the absolute value of the most negative plus one
        most_negative = self.population_fitness.min()
        self.normalized_fitness = numpy.asarray(map(lambda fitness: 1/math.pow(fitness+numpy.absolute(most_negative)+1, 1), self.population_fitness))
        s = float(self.normalized_fitness.sum())
        self.normalized_fitness = numpy.asarray(map(lambda fitness: fitness/s, self.normalized_fitness))
        #print self.population_fitness.min()
        #print self.population_fitness
        #print self.normalized_fitness

    def evolve(self):
        """Evolves individuals (solutions) until some termination criteria is satisfied;"""
        self.generation = 0
        start_time = time.time()

        # while the termination criteria is not satisfied, makes another generation
        while not self.termination_criteria.satisfied(self.generation, time.time()-start_time, self.population):
            self.generation += 1
            #print str(self.generation)
            next_generation = []

            if self.elitism:
                # Keeps the 10% best individuals
                best_individuals = heapq.nsmallest(int(0.1*self.population_size), self.population, lambda individual: individual.get_fitness())
                next_generation += copy.deepcopy(best_individuals)

            # select genetic operation probabilistically
            # this is a roulette wheel selection
            operations = numpy.random.choice(['reproduction', 'crossover', 'mutation'], size=self.population_size, p=[self.reproduction, self.crossover, self.mutation]).tolist()
            individuals = numpy.random.choice(self.population, p=self.normalized_fitness, size=2*self.population_size, replace=True).tolist()

            while len(next_generation) < self.population_size:
                operation = operations.pop()
                individual = individuals.pop()
                individual.get_fitness() # enforce fitness calculation

                if operation == 'reproduction':
                    next_generation.append(individual)
                elif operation == 'crossover':
                    individual2 = individuals.pop()
                    individual2.get_fitness() # enforce fitness calculation
                    individual1, individual2 = individual.crossover(individual2)
                    next_generation.append(individual1)
                    next_generation.append(individual2)
                elif operation == 'mutation':
                    individual1 = individual.mutate()
                    next_generation.append(individual1)

            self.population = next_generation
            self.population_fitness = numpy.asarray(map(lambda individual: individual.get_fitness(), self.population))
            most_negative = self.population_fitness.min()
            self.normalized_fitness = numpy.asarray(map(lambda fitness: 1/math.pow(fitness+numpy.absolute(most_negative)+1, 1), self.population_fitness))
            s = float(self.normalized_fitness.sum())
            self.normalized_fitness = numpy.asarray(map(lambda fitness: fitness/s, self.normalized_fitness))

            mean = numpy.mean(self.population_fitness)
            std = numpy.std(self.population_fitness)
            min = self.population_fitness.min()

            info_mean = pandas.DataFrame([[self.generation, mean, min, std]], columns=["generation", "mean", "min", "std"])
            self.generation_info = self.generation_info.append(info_mean, ignore_index=True)

    def result(self):
        """Returns one best solution."""
        return min(self.population, key=lambda individual: individual.get_fitness())

    def get_generation_info(self):
        """Returns minimun fitness, average fitness, and std deviation of each generation."""
        return self.generation_info