# -*- coding: utf-8 -*-

import abc, ga

class IndividualFactory(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def create(self):
        return ga.Individual()