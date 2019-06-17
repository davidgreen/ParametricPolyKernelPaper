#DKE Green
#2019

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class NNManager(object):
  def __init__(self,name,polyOrder):

    self.name = name
    self.polyOrder = polyOrder
    self.__initNetwork()


  def __initNetwork(self):
    print("init network")

    self.__initComputationNetwork()
    self.__initTrainingVars()


  def __initComputationNetwork(self):
    self.inputDim = 1

    self.inputX = tf.placeholder(tf.float32, shape=[None, self.inputDim])

    dense1 = tf.layers.Dense(20)

    out2 = dense1(self.inputX)
    out3 = out2
    for i in range(1,self.polyOrder):
        out3 *= out2
    nn = out3



    outputWidth = 1
    nn = tf.layers.dense(nn,outputWidth)
    self.output = nn




  def __initTrainingVars(self):

    print("init training loss etc...")

    self.outputTrue = tf.placeholder(tf.float32, shape=[None, self.inputDim])

    self.lossTerm = tf.reduce_mean(tf.square(self.output-self.outputTrue))

    self.lr = tf.get_variable("lr_"+self.name,initializer=tf.constant(0.01,tf.float32),trainable=False)

    self.loss = self.lossTerm

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

    feed_input = {
      self.inputX: xs
    }

    outputVars = [self.output]
    outputValues = self.sess.run(outputVars,feed_dict=feed_input)
    return outputValues[0]


  def trainValues(self,numRepeats,lr,xs,ys):

    previousLoss = np.inf

    feed_input = {
      self.inputX: xs,
      self.outputTrue: ys,
      self.lr: lr
    }

    for b in range(0,numRepeats):

      _,val_l = self.sess.run([self.train,self.loss],feed_dict=feed_input)
      previousLoss = val_l

    return previousLoss #last loss
