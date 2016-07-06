# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import ga, cvrp, numpy, struct, math

# TODO: Fazer criterio de parada baseado em numero de geracoes sem melhora
# TODO: Fazer criterio de parada baseado em conhecimento sobre o otimo da funcao fitness (quando chegar a uma distancia X do otimo)