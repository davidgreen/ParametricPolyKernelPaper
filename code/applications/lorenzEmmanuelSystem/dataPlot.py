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
import time


def calcErr(true,data):
    trueRow = np.average(true,1)
    dataRow = np.average(data,1)

    return np.cumsum(np.sqrt((trueRow - dataRow)**2.0))


RANDOM_SEED = 482 #48 best so far
# tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#https://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow
def main():

    plotHandler = PlotHandler()

    #load the data

    times = np.linspace(0,20,20000)

    print("Load data")
    dataTrue = np.load("dataTrueModelOut.npy")
    kernel60Data = np.load("dataOutKernel60.npy")
    kernel80Data = np.load("dataOutKernel80.npy")
    kernel100Data = np.load("dataOutKernel100.npy")

    polyModelData = np.load("dataOutPoly.npy")

    print("Calc errors")
    errKernel60 = calcErr(dataTrue,kernel60Data)
    errKernel80 = calcErr(dataTrue,kernel80Data)
    errKernel100 = calcErr(dataTrue,kernel100Data)
    errPoly = calcErr(dataTrue,polyModelData)


    plotTimes = times[1:]
    plt.plot(plotTimes,errKernel60[1:],label="60")
    plt.plot(plotTimes,errKernel80[1:],label="80")
    plt.plot(plotTimes,errKernel100[1:],label="100")
    plt.plot(plotTimes,errPoly[1:],label="poly")

    plt.yscale('log')
    plt.xscale('log')


    plt.legend()
    plt.show()

    with open("errorDataOutput.dat", "w") as f:
      f.write("Time K60 K80 K100 Poly \r\n")
      for i in range(0,19999):
          strs = []
          strs.append(str(plotTimes[i]))
          strs.append(str(errKernel60[i+1]))
          strs.append(str(errKernel80[i+1]))
          strs.append(str(errKernel100[i+1]))
          strs.append(str(errPoly[i+1]))
          strs.append("\r\n")
          outStr = " ".join(strs)
          f.write(outStr)



    plotHandler.plotMatrix(endTime=20,data=kernel100Data,outputName="kernel100Test")
    plotHandler.plotMatrix(endTime=20,data=dataTrue,outputName="dataTrue")


    endTime = time.time()
    return endTime









if __name__ == '__main__':

    startTime = time.time()

    endTime = main()

    print("Runtime: %s seconds" % (endTime - startTime))
