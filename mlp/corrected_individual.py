# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, cvrp, numpy, struct, math, itertools, random

class CorrectedIndividual(ga.Individual):
    """Represents a solution of a vehicle routing problem."""
    def __init__(self, genotype, fitness_evaluator, capacity, distances, demand):
        super(cvrp.CorrectedIndividual, self).__init__(genotype, fitness_evaluator)
        self.capacity = capacity
        self.distances = distances
        self.demand = demand

    def mutate(self):
        """Simple Random Mutation:
           Moves a customer within the same solution.
        """
        genotype = numpy.array(self.genotype, copy=True)
        # Selects and deletes a customer
        p = map(lambda x: 0.0 if x == 'X' else 1.0, genotype)
        s = sum(p)
        k = numpy.random.choice(genotype, p=map(lambda x: x/s, p))
        genotype = filter(lambda x: True if x not in [k] else False, genotype)
        
        # performs best insertion
        c = self.best_insertion(genotype, int(k), int(k))
        genotype = numpy.insert(genotype, c, k)

        # applies correction operator
        genotype = self.repair(genotype)

        return cvrp.CorrectedIndividual(genotype, self.fitness_evaluator, self.capacity, self.distances, self.demand)

    def crossover(self, another_individual):
        """Simple Random Crossover:
           Randomly selects a subroute from P2 and inserts it into P1."""
        p1 = numpy.array(self.get_genotype(), copy=True)
        p2 = numpy.array(another_individual.get_genotype(), copy=True)
        individual1 = self.simple_random_crossover(p1, p2)

        p1 = numpy.array(self.get_genotype(), copy=True)
        p2 = numpy.array(another_individual.get_genotype(), copy=True)
        individual2 = self.simple_random_crossover(p2, p1)

        return individual1, individual2

    def simple_random_crossover(self, p1, p2):
        # randomly select a subroute from P2
        routes = [list(y) for x, y in itertools.groupby(p2, lambda z: z == 'X') if not x]
        route = random.choice(routes)
        indexes = numpy.random.randint(0, len(route), 2)
        subroute = route[min(indexes):max(indexes)+1]

        # delete the members of the subroute from the P1
        p1 = filter(lambda x: True if x not in subroute else False, p1)

        # performs best insertion
        c = self.best_insertion(p1, int(subroute[0]), int(subroute[-1]))
        p1 = numpy.insert(p1, c, subroute)

        # applies correction operator
        p1 = self.repair(p1)

        return cvrp.CorrectedIndividual(p1, self.fitness_evaluator, self.capacity, self.distances, self.demand)

    def best_insertion(self, genotype, k1, kn):
        """Finds out a inserction position for customer k that minimizes replacement cost."""
        costs = []
        for idx in xrange(0, len(genotype)-1):
            m = 1 if genotype[idx] == 'X' else int(genotype[idx])
            m_plus_one = 1 if genotype[idx + 1] == 'X' else int(genotype[idx + 1])
            c = self.distances.item((m-1, m_plus_one-1)) - self.distances.item((m-1, k1-1)) - self.distances.item((kn-1, m_plus_one-1))
            costs.append(c)
        max_val = max(costs)
        max_idx = costs.index(max_val)
        return max_idx

    def repair(self, genotype):
        # calculates the demand of each route
        routes = [list(y) for x, y in itertools.groupby(genotype, lambda z: z == 'X') if not x]
        demands = map(lambda r: sum(map(lambda c: self.demand[int(c)], r)), routes)

        # obtains routes with max and min demand
        j_max = demands.index(max(demands))
        j_min = demands.index(min(demands))

        # if capacity is exceeded
        if max(demands) > self.capacity:
            # picks a customer from route jMax and inserts it at then end of route jMin
            k = numpy.random.choice(routes[j_max])
            routes[j_max].remove(k)
            routes[j_min].append(k)
        return list(itertools.chain(*list(sum([[i, 'X'] for i in routes], [])[:-1])))

        # Repair(offspring)
        # jMax <- the index of the route in the offspring with largest total demand
        # maxDem <- the total demand of route jMax 
        # jMin <- the index of the route in the offspring with smallest total demand
        # if maxDem > K
        #   randomly choose a customer from route jMax
        #   delete the customer from route jMin
        #   insert the customer at the end of route jMin
        # return offspring

ga.Individual.register(CorrectedIndividual)
