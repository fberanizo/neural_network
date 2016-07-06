# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, cvrp, numpy, struct, math

class ClassicalIndividual(ga.Individual):
    """Represents a solution of a vehicle routing problem."""
    def __init__(self, genotype, fitness_evaluator):
        super(cvrp.ClassicalIndividual, self).__init__(genotype, fitness_evaluator)

    def mutate(self):
        """Performs a mutation where two values in the chromosome are swaped."""
        genotype = numpy.array(self.genotype, copy=True)
        [idx1, idx2] = numpy.random.randint(0, len(genotype), 2)
        aux = self.genotype[idx1]
        numpy.put(genotype, [idx1], [genotype[idx2]])
        numpy.put(genotype, [idx2], [aux])
        return cvrp.ClassicalIndividual(genotype, self.fitness_evaluator)

    def crossover(self, another_individual):
        size = len(filter(lambda x: x != 'X', another_individual.get_genotype()))
        # repeats crossover until gets a valid solution
        while True:
            individual1, individual2 = self.one_point_crossover(another_individual)
            size1 = len(filter(lambda x: x != 'X', numpy.unique(individual1.get_genotype())))
            size2 = len(filter(lambda x: x != 'X', numpy.unique(individual2.get_genotype())))
            #print "size1: "+str(size1)+", size2: "+str(size2)+", size: "+str(size)
            if size1 == size2 == size:
                return individual1, individual2

    def one_point_crossover(self, another_individual):
        """All data beyond a select index in either individual genotype is swapped between the two parent genotypes.""" 
        size = len(another_individual.get_genotype())
        genotype1 = numpy.zeros(size, dtype=another_individual.get_genotype().dtype)
        genotype2 = numpy.zeros(size, dtype=another_individual.get_genotype().dtype)
        idx = numpy.random.randint(1, size)
        numpy.put(genotype1, range(0, idx), another_individual.get_genotype()[0:idx])
        numpy.put(genotype1, range(idx, size), self.get_genotype()[idx:size])
        numpy.put(genotype2, range(0, idx), self.get_genotype()[0:idx])
        numpy.put(genotype2, range(idx, size), another_individual.get_genotype()[idx:size])

        return cvrp.ClassicalIndividual(genotype1, self.fitness_evaluator), cvrp.ClassicalIndividual(genotype2, self.fitness_evaluator)

ga.Individual.register(ClassicalIndividual)
