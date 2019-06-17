#DKE Green
#2019

"""
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from nnManager import NNManager
from nnManagerStandard import NNManagerStandard
import tensorflow as tf

from sklearn.kernel_ridge import KernelRidge



def train(net,xs,ys):
    net.trainValues(numRepeats=1000,lr=0.01,xs=xs,ys=ys)
    finalLoss = net.trainValues(numRepeats=1000,lr=0.001,xs=xs,ys=ys)

    return finalLoss



def trueHiddenFunction(x):
    return (x-1.)*(x+1.)*(x+0.5)

def relErr(approx,v):
    return abs((approx-v)/v)

def main():

    np.random.seed(100)
    tf.random.set_random_seed(100)


    print("Generate test data")

    trainXRange = [-2.,2.]
    testXRange = [-5.,5.]

    print("=" * 80)
    print("Generate training data")
    print("=" * 80)

    trueXs = np.linspace(trainXRange[0],trainXRange[1],10000)
    trueYs = trueHiddenFunction(trueXs)

    numTrainingPoints = 25
    subsetChoice = np.random.choice(len(trueXs), numTrainingPoints)

    trainingXs = np.zeros(numTrainingPoints)
    trainingYs = np.zeros(numTrainingPoints)

    for i in range(0,numTrainingPoints):
        trainingXs[i] = trueXs[subsetChoice[i]]
        trainingYs[i] = trueYs[subsetChoice[i]]


    trainingXs = np.reshape(trainingXs,(-1,1))
    trainingYs = np.reshape(trainingYs,(-1,1))


    print("=" * 80)
    print("Generate test data")
    print("=" * 80)

    testXs = np.linspace(testXRange[0],testXRange[1],100)
    testYs = trueHiddenFunction(testXs)
    testXs = np.reshape(testXs,(-1,1))

    print("=" * 80)
    print("Init ANNs")
    print("=" * 80)

    nnPoly2 = NNManager("polyModel2",2)
    nnPoly3 = NNManager("polyModel3",3)
    nnPoly4 = NNManager("polyModel4",4)

    nnStandard = NNManagerStandard("eluModel")

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    nnStandard.setSession(sess)
    nnPoly2.setSession(sess)
    nnPoly3.setSession(sess)
    nnPoly4.setSession(sess)

    standardFinalLoss = train(net=nnStandard,xs=trainingXs,ys=trainingYs)
    poly2FinalLoss = train(net=nnPoly2,xs=trainingXs,ys=trainingYs)
    poly3FinalLoss = train(net=nnPoly3,xs=trainingXs,ys=trainingYs)
    poly4FinalLoss = train(net=nnPoly4,xs=trainingXs,ys=trainingYs)

    print("Standard final loss: %e" % (standardFinalLoss))
    print("Poly2 final loss: %e" % (poly2FinalLoss))
    print("Poly3 final loss: %e" % (poly3FinalLoss))
    print("Poly4 final loss: %e" % (poly4FinalLoss))


    standardNetPredictionYs = nnStandard.predict(testXs)
    poly2NetPredictionYs = nnPoly2.predict(testXs)
    poly3NetPredictionYs = nnPoly3.predict(testXs)
    poly4NetPredictionYs = nnPoly4.predict(testXs)



    print("=" * 80)
    print("KRR")
    print("=" * 80)

    clf = KernelRidge(alpha=0.1, coef0=10.0, degree=3, gamma=None, kernel='poly',kernel_params=None)
    clf.fit(trainingXs, trainingYs)

    clfPredY = clf.predict(testXs)[:,0]


    clfDiff = clfPredY-testYs
    clfLoss = np.average(clfDiff**2.0)

    print("clfLoss loss: %e" % (clfLoss))


    print("=" * 80)
    print("Plots")
    print("=" * 80)

    plt.scatter(trainingXs,trainingYs)
    plt.plot(testXs,standardNetPredictionYs,label="standard")

    plt.plot(testXs,poly2NetPredictionYs,label="poly2")
    plt.plot(testXs,poly3NetPredictionYs,label="poly3")
    plt.plot(testXs,poly4NetPredictionYs,label="poly4")

    plt.plot(testXs,testYs,label="true")

    plt.plot(testXs,clfPredY,label="clfPredY")

    plt.legend()
    plt.show()

    print("=" * 80)
    print("Output training data")
    print("=" * 80)

    print("X trainY")
    for i in range(0,len(trainingXs)):
        print("%e %e" % (trainingXs[i],trainingYs[i]))


    print("=" * 80)
    print("Output test data")
    print("=" * 80)

    print("X trueY poly2Y poly3Y poly4Y standardY clfY")
    for i in range(0,len(testXs)):
        print("%e %e %e %e %e %e %e" % (testXs[i],testYs[i],poly2NetPredictionYs[i],poly3NetPredictionYs[i],poly4NetPredictionYs[i],standardNetPredictionYs[i],clfPredY[i]))


    print("=" * 80)
    print("Output error data")
    print("=" * 80)

    print("X poly2Y poly3Y poly4Y standardY clfY")
    for i in range(0,len(testXs)):
        ty = testYs[i]

        p2 = relErr(poly2NetPredictionYs[i],ty)
        p3 = relErr(poly3NetPredictionYs[i],ty)
        p4 = relErr(poly4NetPredictionYs[i],ty)
        s = relErr(standardNetPredictionYs[i],ty)
        clfV = relErr(clfPredY[i],ty)

        print("%e %e %e %e %e %e" % (testXs[i],p2,p3,p4,s,clfV))








if __name__ == "__main__":

  main()
