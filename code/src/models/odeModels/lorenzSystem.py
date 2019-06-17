#DKE Green
#2018

from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import math

import sys
libraryPath = '../../../'
sys.path.append(libraryPath)

from src.models.modelMixins.scipyODEIntegrator import ScipyODEIntegrator


class LorenzSystem(object):
    def __init__(self):

        #https://en.wikipedia.org/wiki/Lorenz_system
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0/3.0

        self.odeCalculator = ScipyODEIntegrator(odeDerivative=self.odeDerivative)

    def calculateDuDt(self,u):

        u_1 = u[0]
        u_2 = u[1]
        u_3 = u[2]

        drvU1 = self.sigma*(u_2-u_1)
        drvU2 = u_1*(self.rho-u_3) - u_2
        drvU3 = u_1*u_2 - self.beta*u_3

        return np.array([drvU1,drvU2,drvU3])


    def odeDerivative(self,t,y):

        return self.calculateDuDt(y)
