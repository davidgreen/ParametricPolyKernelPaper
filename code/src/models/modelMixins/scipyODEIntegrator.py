#DKE Green
#2018

from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from io import BytesIO
from scipy.integrate import ode, solve_ivp

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

class ScipyODEIntegrator(object):
    def __init__(self,odeDerivative):
        self.odeDerivative = odeDerivative

        self.startTime = 0.0
        self.endTime = 1.0

        self.numberEvaluationIntervals = 10

    def setTimeSpan(self,startTime,endTime):
        self.startTime = startTime
        self.endTime = endTime

    def setNumberEvaluationIntervals(self,numberEvaluationIntervals):
        self.numberEvaluationIntervals = numberEvaluationIntervals


    def solveIVP(self,y0,method='RK45'):
        times = np.linspace(self.startTime,self.endTime,self.numberEvaluationIntervals)

        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
        solution = solve_ivp(fun=self.odeDerivative,
            t_span=[self.startTime,self.endTime],
            y0=y0, method=method, t_eval=times, dense_output=False, events=None, vectorized=False)

        self.previousSolution = solution

        retValues = solution.y.transpose()
        retTimes = solution.t

        print("Solver message:")
        print(solution.message)


        return retValues,retTimes


    def generateDataStream(self,initVals,numTimes,timeInterval,method='RK45'):
        print("Generating data stream initVals: %s numTimes: %s timeInterval: %s" % (initVals,numTimes,timeInterval))

        self.setTimeSpan(startTime=timeInterval[0],endTime=timeInterval[1])
        self.setNumberEvaluationIntervals(numTimes)
        data,times = self.solveIVP(y0=initVals,method=method)

        dtVals = np.zeros((len(times)-1,1))
        for i in range(0,len(times)-1):
            dtVals[i,0] = times[i+1]-times[i]

        retDict = {
            'data': data,
            'times': times,
            'dtVals': dtVals
        }


        return retDict
