# -*- coding: utf-8 -*-

import abc

class Individual(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, genotype, fitness_evaluator):
        self.genotype = genotype
        self.fitness_evaluator = fitness_evaluator
        self.fitness = None

    def get_genotype(self):
        return self.genotype

    def get_fitness_evaluator(self):
        return self.fitness_evaluator

    def get_fitness(self):
        if self.fitness is None:
            self.fitness = self.fitness_evaluator.evaluate(self)

        return self.fitness

    @abc.abstractmethod
    def mutate(self):
        pass

    @abc.abstractmethod
    def crossover(self, another_individual):
        pass