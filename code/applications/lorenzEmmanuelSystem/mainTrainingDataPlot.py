#DKE Green
#2018

from __future__ import print_function
from __future__ import division


import sys
libraryPath = '../../'
sys.path.append(libraryPath)

from src.models.odeModels.lorenzEmmanuelSystem import LorenzEmmanuelSystem
from views.plotHandler import PlotHandler

import argparse

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

import os
import sys
import math
# import tensorflow as tf
import time


RANDOM_SEED = 482 #48 best so far
# tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#https://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow
def main():

    numDofs = 8
    odeModel = LorenzEmmanuelSystem(dofs=numDofs)
    odeModel.setForcingFunction(func=lambda t,y: 5.0)

    plotHandler = PlotHandler()



    initVals = np.zeros(numDofs)
    initVals = np.random.normal(0,3,numDofs)
    initVals2 = np.random.normal(0,3,numDofs)


    print("starting solver")

    endTime = 20.0
    dt = 0.001#0.01 #0.25

    itegratorType = 'RK45'#'BDF'#'BDF'

    numTimes = endTime/dt
    retDict = odeModel.odeCalculator.generateDataStream(initVals=initVals,numTimes=numTimes,timeInterval=[0,endTime],method=itegratorType)


    trainingTime = 20.0
    numTimesStepsFroKernel = int(trainingTime/dt)
    selectionSize = 20000 #20000#numTimesStepsFroKernel#1000

    dataTimes = np.array(retDict['times'][0:selectionSize])
    dataValues = np.array(retDict['data'])[0:selectionSize,:]

    plotHandler.plotTrainingData(endTime=20.0,data=dataValues)

    return time.time()



if __name__ == '__main__':


    startTime = time.time()
    endTime = main()
    print("Runtime: %s seconds" % (endTime - startTime))
