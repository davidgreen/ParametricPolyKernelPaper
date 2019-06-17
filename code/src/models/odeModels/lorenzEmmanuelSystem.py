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

class LorenzEmmanuelSystem(object):
    def __init__(self,dofs):

        self.dofs = dofs


        self.assembleRHSMatrix()

        self.odeCalculator = ScipyODEIntegrator(odeDerivative=self.odeDerivative)


    def setForcingFunction(self,func):
        self.fFunc = func


    def setFFactor(self,value):
        self.fFactor = value

    def assembleRHSMatrix(self):

        #Need to make a second order poly features model

        # self.secondOrderPolyPairs = int(self.dofs*(self.dofs-1)/2)
        # self.totalNumDoFs = self.secondOrderPolyPairs

        #You need a 'poly features matrix'
        #And you need indices into that

        #You want to take an outer product then just take say the lower triangle

        #[x1][x1 x2 x3] = [ x1x1 x1x2 x1x3 ]
        #[x2]           = [ x2x1 x2x2 x2x3 ]
        #[x3]           = [ x3x1 x3x2 x3x3 ]

        #For any i,j pair gives the index to the poly features vector
        #The problem with just using the lower triangular is that
        #it is annoying to look up who belongs to what index


        self.polyIndexGridOP = np.zeros((self.dofs,self.dofs))
        self.polyIndexListOP = []
        for i in range(0,self.dofs):
            for j in range(0,self.dofs):
                self.polyIndexGridOP[i,j] = len(self.polyIndexListOP)
                self.polyIndexListOP.append([i,j])

        self.rhsMatrix = np.zeros((self.dofs,self.dofs*self.dofs))

        self.rhsDofsMatrix = np.zeros((self.dofs,self.dofs))

        for i in range(0,self.dofs):

            leftPos = int(self.polyIndexGridOP[i,(i-1)%self.dofs])
            rightPos = int(self.polyIndexGridOP[i,(i+1)%self.dofs])
            twoLeftPos = int(self.polyIndexGridOP[i,(i-2)%self.dofs])
            iiPos = int(self.polyIndexGridOP[i,i])

            self.rhsMatrix[i,i] = -1

            self.rhsDofsMatrix[i,(i+1)%self.dofs] = 1#3.5
            self.rhsDofsMatrix[i,(i-1)%self.dofs] = 1#3.5
            self.rhsDofsMatrix[i,(i+2)%self.dofs] = -1
            self.rhsDofsMatrix[i,(i-2)%self.dofs] = -1

        print(self.rhsMatrix)


    def calculateRHS(self,yIn,t):


        outDrvs = np.zeros(self.dofs)

        for i in range(0,self.dofs):
            iPos = i
            leftPos = (i - 1) % self.dofs
            leftleftPos = (i - 2) % self.dofs
            rightPos = (i + 1) % self.dofs
            rightRightPos = (i + 2) % self.dofs


            outDrvs[i] = (yIn[rightPos]-yIn[leftleftPos])*yIn[leftPos] - yIn[iPos] + self.fFunc(t,yIn)

        return outDrvs


    def odeDerivative(self,t,y):
        return self.calculateRHS(yIn=y,t=t)
