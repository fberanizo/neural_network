# -*- coding: utf-8 -*-

import abc

class FitnessEvaluator(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def evaluate(self, individual):
        return 0.0