#DKE Green
#2018

from __future__ import print_function
from __future__ import division


import sys
libraryPath = '../../'
sys.path.append(libraryPath)

from src.models.odeModels.lorenzEmmanuelSystem import LorenzEmmanuelSystem
from src.models.odeModels.lorenzEmmanuelFromANNDropFromDrv import LorenzEmmanuelFromANNDropFromDrv
from src.models.odeModels.lorenzEmmanuelANNPolyFeat import LorenzEmmanuelANNPolyFeat

from views.plotHandler import PlotHandler


import argparse

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

import os
import sys
import math
import tensorflow as tf
import time


RANDOM_SEED = 482 #48 best so far
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def trainModel(nnModel,dataIn,xsp1,xsp2,xsp3,xsp4,xsp5,dt):

    nnModel.trainValues(numRepeats=1000,lr=0.1,xs=dataIn,xsp1=xsp1,xsp2=xsp2,xsp3=xsp3,xsp4=xsp4,xsp5=xsp5,dh=dt)
    nnModel.trainValues(numRepeats=2000,lr=0.01,xs=dataIn,xsp1=xsp1,xsp2=xsp2,xsp3=xsp3,xsp4=xsp4,xsp5=xsp5,dh=dt)
    nnModel.trainValues(numRepeats=200,lr=0.001,xs=dataIn,xsp1=xsp1,xsp2=xsp2,xsp3=xsp3,xsp4=xsp4,xsp5=xsp5,dh=dt)


#https://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow
def main():

    print("Init System")
    numDofs = 8
    odeModel = LorenzEmmanuelSystem(dofs=numDofs)

    odeModel.setForcingFunction(func=lambda t,y: 5.0)

    initVals = np.zeros(numDofs)
    initVals = np.random.normal(0,3,numDofs)
    initVals2 = np.random.normal(0,3,numDofs)

    print("Start solver")

    endTime = 20.0
    dt = 0.001

    itegratorType = 'RK45'

    numTimes = endTime/dt
    retDict = odeModel.odeCalculator.generateDataStream(initVals=initVals,numTimes=numTimes,timeInterval=[0,endTime],method=itegratorType)

    trainingTime = 20.0
    numTimesStepsFroKernel = int(trainingTime/dt)
    dataTimes = np.array(retDict['times'][0:numTimesStepsFroKernel])
    dataValues = np.array(retDict['data'])[0:numTimesStepsFroKernel,:]


    selectionSize = 20000

    print("Init ANNs")

    nnModelKernel60 = LorenzEmmanuelFromANNDropFromDrv(numDofs,name="60",wwSize=60)
    nnModelKernel80 = LorenzEmmanuelFromANNDropFromDrv(numDofs,name="80",wwSize=80)
    nnModelKernel100 = LorenzEmmanuelFromANNDropFromDrv(numDofs,name="100",wwSize=100)
    nnModelPoly = LorenzEmmanuelANNPolyFeat(numDofs)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    nnModelKernel60.setSession(sess)
    nnModelKernel80.setSession(sess)
    nnModelKernel100.setSession(sess)
    nnModelPoly.setSession(sess)

    print("Prepare training data")
    #have to remember that the ORDER of the data DOES matter
    order = 5
    dataIn = np.zeros((selectionSize-order,numDofs))
    xsp1 = np.zeros((selectionSize-order,numDofs))
    xsp2 = np.zeros((selectionSize-order,numDofs))
    xsp3 = np.zeros((selectionSize-order,numDofs))
    xsp4 = np.zeros((selectionSize-order,numDofs))
    xsp5 = np.zeros((selectionSize-order,numDofs))
    for i in range(0,selectionSize-order):
        dataIn[i,:] = dataValues[i,:]
        xsp1[i,:] = dataValues[i+1,:]
        xsp2[i,:] = dataValues[i+2,:]
        xsp3[i,:] = dataValues[i+3,:]
        xsp4[i,:] = dataValues[i+4,:]
        xsp5[i,:] = dataValues[i+5,:]



    print("Train models")
    trainModel(nnModelKernel60,dataIn,xsp1,xsp2,xsp3,xsp4,xsp5,dt)
    trainModel(nnModelKernel80,dataIn,xsp1,xsp2,xsp3,xsp4,xsp5,dt)
    trainModel(nnModelKernel100,dataIn,xsp1,xsp2,xsp3,xsp4,xsp5,dt)
    trainModel(nnModelPoly,dataIn,xsp1,xsp2,xsp3,xsp4,xsp5,dt)


    print("Prepare output traces")

    newInit = initVals2


    polyDictOut = nnModelPoly.odeCalculator.generateDataStream(initVals=newInit,numTimes=numTimes,timeInterval=[0,endTime],method=itegratorType)
    kernel60DictOut = nnModelKernel60.odeCalculator.generateDataStream(initVals=newInit,numTimes=numTimes,timeInterval=[0,endTime],method=itegratorType)
    kernel80DictOut = nnModelKernel80.odeCalculator.generateDataStream(initVals=newInit,numTimes=numTimes,timeInterval=[0,endTime],method=itegratorType)
    kernel100DictOut = nnModelKernel100.odeCalculator.generateDataStream(initVals=newInit,numTimes=numTimes,timeInterval=[0,endTime],method=itegratorType)

    trueModelOut = odeModel.odeCalculator.generateDataStream(initVals=newInit,numTimes=numTimes,timeInterval=[0,endTime],method=itegratorType)

    print("Save output traces")

    np.save("dataOutPoly", np.array(polyDictOut['data']))
    np.save("dataOutKernel60", np.array(kernel60DictOut['data']))
    np.save("dataOutKernel80", np.array(kernel80DictOut['data']))
    np.save("dataOutKernel100", np.array(kernel100DictOut['data']))
    np.save("dataTrueModelOut", np.array(trueModelOut['data']))


    print("Plot outputs - just for debug")

    plt.figure("train")
    plt.imshow(np.rot90(dataValues), aspect='auto')


    plt.figure("polyDictOut")
    plt.imshow(np.rot90(polyDictOut['data']), aspect='auto')

    plt.figure("kernel60DictOut")
    plt.imshow(np.rot90(kernel60DictOut['data']), aspect='auto')


    plt.figure("kernel80DictOut")
    plt.imshow(np.rot90(kernel80DictOut['data']), aspect='auto')

    plt.figure("trueModelOut")
    plt.imshow(np.rot90(trueModelOut['data']), aspect='auto')

    finalTime = time.time()

    plt.show()


    return finalTime








if __name__ == '__main__':
    startTime = time.time()
    finalTime = main()
    print("Runtime: %s seconds" % (finalTime - startTime))
