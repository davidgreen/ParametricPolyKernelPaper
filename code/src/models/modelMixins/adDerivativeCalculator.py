#DKE Green
#2018

from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

import ad
from ad import adnumber
from ad.admath import *


class ADDerivativeCalculator(object):
    def __init__(self):
        pass


    def calculateDerivates(self,func,xs):
        retYs = np.zeros(len(xs))
        retDYs = np.zeros(len(xs))
        retDDYs = np.zeros(len(xs))

        for i in range(0,len(xs)):
            currX = xs[i]

            adCurrX = adnumber(currX)
            ys = func(adCurrX)

            retYs[i] = ys.x
            retDYs[i] = ys.d(adCurrX)
            retDDYs[i] = ys.d2(adCurrX)

        return retYs,retDYs,retDDYs
