# -*- coding: utf-8 -*-

import sys, os, time
sys.path.insert(0, os.path.abspath('..'))

import ga, numpy, math

class EightQueensFitnessEvaluator(ga.FitnessEvaluator):
    # evaluation criteria is counting how many queens threaten each other
    def evaluate(self, individual):
        genotype = individual.get_genotype()
        threaten_count = 0
        for col1 in xrange(0, 8):
            row1 = genotype[col1]
            for col2 in xrange(col1+1, 8):
                row2 = genotype[col2]
                if row1 == row2:
                    threaten_count += 1
                if math.fabs(col2 - col1) == math.fabs(row2 - row1):
                    threaten_count += 1

        return 144 - threaten_count

ga.FitnessEvaluator.register(EightQueensFitnessEvaluator)

class EightQueensIndividual(ga.Individual):
    def mutate(self):
        idx = numpy.random.randint(0, 8)
        value = numpy.random.randint(0, 8)
        numpy.put(self.genotype, [idx], [value])
        self.fitness = self.fitness_evaluator.evaluate(self)

    def crossover(self, another_individual):
        genotype1 = numpy.zeros(8)
        genotype2 = numpy.zeros(8)

        idx = numpy.random.randint(1, 7)
        
        numpy.put(genotype1, range(0, idx), another_individual.get_genotype()[0:idx])
        numpy.put(genotype1, range(idx, 8), self.get_genotype()[idx:8])

        numpy.put(genotype2, range(0, idx), self.get_genotype()[0:idx])
        numpy.put(genotype2, range(idx, 8), another_individual.get_genotype()[idx:8])

        return EightQueensIndividual(genotype1, self.fitness_evaluator), EightQueensIndividual(genotype2, self.fitness_evaluator)


ga.Individual.register(EightQueensIndividual)

class EightQueensIndividualFactory(ga.IndividualFactory):
    # An individual is represented by an array where each element is a row of a board
    # and the value of the element is the column where the element is at that row
    def create(self):
        genotype = numpy.random.random_integers(low=0, high=7, size=8)
        fitness_evaluator = EightQueensFitnessEvaluator()
        return EightQueensIndividual(genotype, fitness_evaluator)

ga.IndividualFactory.register(EightQueensIndividualFactory)

class EightQueensTerminationCriteria(ga.TerminationCriteria):
    # Termination criteria is satisfied when we place eight chess queens on 
    # an 8×8 chessboard so that no two queens threaten each other
    def satisfied(self, generation, execution_time, population):
        for individual in population:
            genotype = individual.get_genotype()
            satisfied = True
            for col1 in xrange(0, 8):
                row1 = genotype[col1]
                for col2 in xrange(col1+1, 8):
                    row2 = genotype[col2]
                    if row1 == row2:
                        satisfied = False
                        break

                    if math.fabs(col2 - col1) == math.fabs(row2 - row1):
                        satisfied = False
                        break
                if not satisfied:
                    break
            
            if satisfied:
                return True
        return False

ga.TerminationCriteria.register(EightQueensTerminationCriteria)

start_time = time.time()
solver = ga.GeneticAlgorithm(population_size=300)
solver.init_population(EightQueensIndividualFactory())
solver.evolve(EightQueensTerminationCriteria(), reproduction=0.3, crossover=0.6, mutation=0.1)
individual = solver.result()
genotype = individual.get_genotype()
#genotype = numpy.asarray([ 3,  5,  7,  1,  6,  0,  2,  4])
for i in xrange(0, 8):
    for j in xrange(0, 8):
        if genotype[j] == i:
            sys.stdout.write('Q ')
        else:
            sys.stdout.write('# ')
    sys.stdout.write('\n')

print time.time() - start_time

