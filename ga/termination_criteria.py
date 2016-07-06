# -*- coding: utf-8 -*-

import abc

class TerminationCriteria(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def satisfied(self, number_of_generations, execution_time, population):
        return True

class ExecutionTimeTerminationCriteria(TerminationCriteria):
    def __init__(self, time_in_seconds=5.0):
        super(ExecutionTimeTerminationCriteria, self).__init__()
        self.time_in_seconds = time_in_seconds

    def satisfied(self, number_of_generations, execution_time, population):
        """Termination criteria is satisfied when execution time is greater than the specified value."""
        return True if execution_time >= self.time_in_seconds else False

TerminationCriteria.register(ExecutionTimeTerminationCriteria)

class NumberOfGenerationsTerminationCriteria(TerminationCriteria):
    def __init__(self, number_of_generations=100):
        super(NumberOfGenerationsTerminationCriteria, self).__init__()
        self.number_of_generations = number_of_generations

    def satisfied(self, number_of_generations, execution_time, population):
        """Termination criteria is satisfied when number of generations is reached."""
        return True if number_of_generations >= self.number_of_generations else False

TerminationCriteria.register(NumberOfGenerationsTerminationCriteria)