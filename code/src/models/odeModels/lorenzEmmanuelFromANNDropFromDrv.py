#DKE Green
#2018

from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from io import BytesIO

import math

import tensorflow as tf

import sys
libraryPath = '../../../'
sys.path.append(libraryPath)

from src.models.modelMixins.scipyODEIntegrator import ScipyODEIntegrator


#https://stackoverflow.com/questions/42737619/output-of-numpy-diff-is-wrong
#https://datascience.stackexchange.com/questions/15032/how-to-do-batch-inner-product-in-tensorflow


class LorenzEmmanuelFromANNDropFromDrv(object):
    def __init__(self,numDofs,name,wwSize):
        #https://github.com/jswhit/pyks/blob/master/KS.py
        #'https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf'

        self.numDofs = numDofs
        self.name = name
        self.wwSize = wwSize


        self.highPrec = False
        if self.highPrec:
            self.dtypeVal = tf.float64
            self.npTypeVal = np.float64
        else:
            self.dtypeVal = tf.float32
            self.npTypeVal = np.float32

        self.__initNetwork()

        self.odeCalculator = ScipyODEIntegrator(odeDerivative=self.odeDerivative)


        # self.dvs = dvs



    def __initNetwork(self):
        print("init network")

        self.__initComputationNetwork()
        self.__initTrainingVars()






    def __evalF(self,uVals):

            nn = tf.multiply( self.nn1(uVals) + self.aVar,self.nn2(uVals) + self.bVar ) + self.cVar #BETTER


            f = self.denseOut(nn)

            return f




    def __initComputationNetwork(self):
        #input placeholders

        with tf.variable_scope("varscope_"+self.name, reuse=tf.AUTO_REUSE):

            self.inputDim = self.numDofs
            self.inputX = tf.placeholder(self.dtypeVal, shape=[None, self.inputDim])


            ### =========================================
            ### poly features
            ### =========================================
            ww = self.wwSize

            inputX = self.inputX

            reg = 0.0#1e-09 #no
            reg2 =0.#1e-09

            self.nn1 = tf.layers.Dense(ww,kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=reg),dtype=self.dtypeVal,use_bias=False)
            self.nn2 = tf.layers.Dense(ww,kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=reg),dtype=self.dtypeVal)

            self.nn3 = tf.layers.Dense(ww,kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=reg),dtype=self.dtypeVal)

            self.aVar = tf.get_variable("aVar_"+self.name,initializer=tf.constant(1.,self.dtypeVal),trainable=True,dtype=self.dtypeVal)
            self.bVar = tf.get_variable("bVar_"+self.name,initializer=tf.constant(1.,self.dtypeVal),trainable=True,dtype=self.dtypeVal)
            self.cVar = tf.get_variable("cVar_"+self.name,initializer=tf.constant(1.,self.dtypeVal),trainable=True,dtype=self.dtypeVal)


            #
            self.denseOut = tf.layers.Dense(self.inputDim,dtype=self.dtypeVal,kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=reg2))
            ### =========================================



    def adamsBashforth(self):
        #trap rule
        self.evalUP1 = self.inputX+(self.inputDH)*(self.outputF)

        self.evalUP2 = self.evalUP1+(3.0*self.inputDH/2.0)*(self.outputFP1) - (1.0*self.inputDH/2.0)*(self.outputF)
        self.evalUP3 = self.evalUP2+(3.0*self.inputDH/2.0)*(self.outputFP2) - (1.0*self.inputDH/2.0)*(self.outputFP1)
        self.evalUP4 = self.evalUP3+(3.0*self.inputDH/2.0)*(self.outputFP3) - (1.0*self.inputDH/2.0)*(self.outputFP2)


        self.lossTerm = tf.reduce_mean(tf.square(self.evalUP1-self.outputTrueUP1))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP2-self.outputTrueUP2))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP3-self.outputTrueUP3))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP4-self.outputTrueUP4))



    def adamsBashforth2(self):
        #trap rule
        self.evalUP1 = self.inputX+(self.inputDH)*(self.outputF)

        self.evalUP2 = self.outputTrueUP1+(3.0*self.inputDH/2.0)*(self.outputFP1) - (1.0*self.inputDH/2.0)*(self.outputF)

        self.evalUP3 = self.outputTrueUP2+(3.0*self.inputDH/2.0)*(self.outputFP2) - (1.0*self.inputDH/2.0)*(self.outputFP1)

        self.evalUP4 = self.outputTrueUP3+(3.0*self.inputDH/2.0)*(self.outputFP3) - (1.0*self.inputDH/2.0)*(self.outputFP2)


        self.lossTerm = tf.reduce_mean(tf.square(self.evalUP1-self.outputTrueUP1))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP2-self.outputTrueUP2))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP3-self.outputTrueUP3))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP4-self.outputTrueUP4))


    def adamsMoulton(self):
        self.evalUP1 = self.inputX+(self.inputDH)*(self.outputFP1)

        self.evalUP2 = self.evalUP1+(1.0*self.inputDH/2.0)*(self.outputFP2 + self.outputFP1)

        self.evalUP3 = self.evalUP2+(self.inputDH)*((5./12.)*self.outputFP3 + (2./3.)*self.outputFP2 - (1./12.)*self.outputFP1)

        self.evalUP4 = self.evalUP3+(self.inputDH)*((9./24.)*self.outputFP4 + (19./24.)*self.outputFP3 - (5./24.)*self.outputFP2 + (1./24.)*self.outputFP1)

        self.evalUP5 = self.evalUP4+(self.inputDH)*((251./720.)*self.outputFP5 + (646./720.)*self.outputFP4 - (264./720.)*self.outputFP3 + (106./720.)*self.outputFP2 - (19./720.)*self.outputFP1)


        self.lossTerm = tf.reduce_mean(tf.square(self.evalUP1-self.outputTrueUP1))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP2-self.outputTrueUP2))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP3-self.outputTrueUP3))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP4-self.outputTrueUP4))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP5-self.outputTrueUP5))



    def adamsMoulton2(self):
        self.evalUP1 = self.inputX+(self.inputDH)*(self.outputFP1)

        self.evalUP2 = self.outputTrueUP1+(1.0*self.inputDH/2.0)*(self.outputFP2 + self.outputFP1)

        self.evalUP3 = self.outputTrueUP2+(self.inputDH)*((5./12.)*self.outputFP3 + (2./3.)*self.outputFP2 - (1./12.)*self.outputFP1)

        self.evalUP4 = self.outputTrueUP3+(self.inputDH)*((9./24.)*self.outputFP4 + (19./24.)*self.outputFP3 - (5./24.)*self.outputFP2 + (1./24.)*self.outputFP1)

        self.evalUP5 = self.outputTrueUP4+(self.inputDH)*((251./720.)*self.outputFP5 + (646./720.)*self.outputFP4 - (264./720.)*self.outputFP3 + (106./720.)*self.outputFP2 - (19./720.)*self.outputFP1)


        # self.lossTerm = tf.reduce_mean(tf.square(self.evalUP1-self.outputTrueUP1))
        # self.lossTerm += tf.reduce_mean(tf.square(self.evalUP2-self.outputTrueUP2))
        self.lossTerm = tf.reduce_mean(tf.square(self.evalUP3-self.outputTrueUP3))
        # self.lossTerm += tf.reduce_mean(tf.square(self.evalUP4-self.outputTrueUP4))
        # self.lossTerm += tf.reduce_mean(tf.square(self.evalUP5-self.outputTrueUP5))


    def bdf(self):


        self.evalUP1 = self.inputX+(self.inputDH)*(self.outputFP1)

        self.evalUP2 = (4.0/3.0)*self.evalUP1 - (1.0/3.0)*self.inputX + (2.0/3.0)*self.inputDH*self.outputFP2

        self.evalUP3 = (18.0/11.0)*self.evalUP2 - (9.0/11.0)*self.evalUP1 + (2.0/11.0)*self.inputX + (6.0/11.0)*self.inputDH*self.outputFP3

        self.evalUP4 = (48.0/25.0)*self.evalUP3 - (36.0/25.0)*self.evalUP2 + (16.0/25.0)*self.evalUP1 - (3.0/25.0)*self.inputX + (12.0/25.0)*self.inputDH*self.outputFP3

        self.evalUP5 = (300.0/25.0)*self.evalUP4 + (48.0/25.0)*self.evalUP3 - (36.0/25.0)*self.evalUP2 + (16.0/25.0)*self.evalUP1 - (3.0/25.0)*self.inputX + (12.0/25.0)*self.inputDH*self.outputFP3



        self.lossTerm = tf.reduce_mean(tf.square(self.evalUP1-self.outputTrueUP1))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP2-self.outputTrueUP2))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP3-self.outputTrueUP3))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP4-self.outputTrueUP4))
        self.lossTerm += tf.reduce_mean(tf.square(self.evalUP5-self.outputTrueUP5))

    def __initTrainingVars(self):

        print("init training loss etc...")

        self.outputTrueUP1 = tf.placeholder(self.dtypeVal, shape=[None, self.inputDim])
        self.outputTrueUP2 = tf.placeholder(self.dtypeVal, shape=[None, self.inputDim])
        self.outputTrueUP3 = tf.placeholder(self.dtypeVal, shape=[None, self.inputDim])
        self.outputTrueUP4 = tf.placeholder(self.dtypeVal, shape=[None, self.inputDim])
        self.outputTrueUP5 = tf.placeholder(self.dtypeVal, shape=[None, self.inputDim])


        self.outputF = self.__evalF(self.inputX)
        self.outputFP1 = self.__evalF(self.outputTrueUP1)
        self.outputFP2 = self.__evalF(self.outputTrueUP2)
        self.outputFP3 = self.__evalF(self.outputTrueUP3)
        self.outputFP4 = self.__evalF(self.outputTrueUP4)
        self.outputFP5 = self.__evalF(self.outputTrueUP5)

        self.inputDH = tf.placeholder(self.dtypeVal, shape=[None, self.inputDim])

        self.adamsMoulton2()


        self.lr = tf.get_variable("lr_"+self.name,initializer=tf.constant(0.01,self.dtypeVal),trainable=False,dtype=self.dtypeVal)

        self.loss = self.lossTerm

        if not self.highPrec:
            regLoss = tf.losses.get_regularization_loss()
            self.loss += regLoss


        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train = self.optimizer.minimize(self.loss)




    def setSession(self,sess):
        self.sess = sess


    def initSession(self):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        self.sess = sess


    def predict(self,xs):

        dtype = self.npTypeVal

        feed_input = {
            self.inputX: xs.astype(dtype),
        }

        outputVars = [self.outputF]
        outputValues = self.sess.run(outputVars,feed_dict=feed_input)
        return outputValues[0]



    def trainValues(self,numRepeats,lr,xs,xsp1,xsp2,xsp3,xsp4,xsp5,dh):

    	previousLoss = np.inf

        numTrainingVals = xs.shape[0]-1

        dhInput = np.zeros((xs.shape[0],self.inputDim))
        dhInput += dh

        dtype = self.npTypeVal
        feed_input = {
            self.inputDH: dhInput.astype(dtype),
            self.outputTrueUP1: xsp1.astype(dtype),
            self.outputTrueUP2: xsp2.astype(dtype),
            self.outputTrueUP3: xsp3.astype(dtype),
            self.outputTrueUP4: xsp4.astype(dtype),
            self.outputTrueUP5: xsp5.astype(dtype),
            self.inputX: xs.astype(dtype),
            self.lr: lr
        }

        for b in range(0,numRepeats):
            _,val_l = self.sess.run([self.train,self.loss],feed_dict=feed_input)
            print("b: %i loss: %e" % (b,val_l))
            previousLoss = val_l




    def odeDerivative(self,t,y):

        evalKernel = self.predict(xs=y.reshape(1,-1))
        evalKernel = evalKernel.reshape(-1)

        return evalKernel
