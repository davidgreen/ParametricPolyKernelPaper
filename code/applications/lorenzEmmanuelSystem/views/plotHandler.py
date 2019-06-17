#DKE Green
#2018

from __future__ import print_function
from __future__ import division


import matplotlib.pyplot as plt
import numpy as np




class PlotHandler(object):
    def __init__(self):
        self.plotSetup()




    def plotSetup(labelSize=28):
    #https://github.com/spyder-ide/spyder/issues/3606
    #have to sudo apt-get install dvipng

        #Direct input
        plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
        #Options
        params = {'text.usetex' : True,
    #              'font.size' : 11,
                  'font.family' : 'lmodern',
                  'font.size'   : 12,#16,
                  'text.latex.unicode': True,
                  'axes.labelsize': 12,#16, #labelSize,
                  'axes.titlesize':12,#16,
                  #'text.fontsize': 20,
                  'legend.fontsize': 12,#16,#16,
                  'xtick.labelsize': 14,#20,#20,
                  'ytick.labelsize': 14,#20,#20
                  }
        plt.rcParams.update(params)







    def plotTrainingData(self,endTime,data):

        # https://stackoverflow.com/questions/13384653/imshow-extent-and-aspect
        plotLim = 8.0

        cmap = "viridis"

        numPlots = 1

        figSizeVal = (6,4)

        extent=[0,endTime,9-0.5,0+0.5]

        f = plt.figure(1,figsize=figSizeVal)

        #lbrt = [0.15,0.2,0.96,0.9]
        lbrt = [0.15,0.14,0.96,0.9]

        plt.subplots_adjust(left=lbrt[0], bottom=lbrt[1], right=lbrt[2], top=lbrt[3],wspace=None, hspace=None)


        im1= plt.imshow(np.rot90(data),cmap=cmap, clim=(-plotLim, plotLim),aspect='auto',interpolation="None",extent=extent)

        ax = plt.gca().set_yticks(np.arange(1, 9, 1));


        cbar1 = plt.colorbar(im1, ax=plt.gca(),ticks=[-8, -4, 0, 4,8])
        cbar1.ax.set_yticklabels(['-8.0', '-4.0', '0.0', '4.0','8.0'])
        #plt.show()

        plt.gca().set_xlabel('Time, $t$')
        # plt.gca().set_ylabel('Degree of freedom, $i$')
        # plt.gca().set_ylabel('Variable index, $i$')
        plt.gca().set_ylabel('Variable index, $i$, for $u_i$')

        format = 'png'
        dpiVal = 300

        f.savefig('trainingDataFig.' + format, format=format, dpi=dpiVal)
        plt.show()





    def plotMatrix(self,endTime,data,outputName):

        # https://stackoverflow.com/questions/13384653/imshow-extent-and-aspect
        plotLim = 8.0
        cmap = "viridis"
        numPlots = 1
        figSizeVal = (6,4)
        extent=[0,endTime,9-0.5,0+0.5]

        f = plt.figure(1,figsize=figSizeVal)

        #lbrt = [0.15,0.2,0.96,0.9]
        lbrt = [0.15,0.14,0.96,0.9]

        plt.subplots_adjust(left=lbrt[0], bottom=lbrt[1], right=lbrt[2], top=lbrt[3],wspace=None, hspace=None)


        im1= plt.imshow(np.rot90(data),cmap=cmap, clim=(-plotLim, plotLim),aspect='auto',interpolation="None",extent=extent)

        ax = plt.gca().set_yticks(np.arange(1, 9, 1));


        cbar1 = plt.colorbar(im1, ax=plt.gca(),ticks=[-8, -4, 0, 4,8])
        cbar1.ax.set_yticklabels(['-8.0', '-4.0', '0.0', '4.0','8.0'])
        #plt.show()

        plt.gca().set_xlabel('Time, $t$')
        # plt.gca().set_ylabel('Degree of freedom, $i$')
        plt.gca().set_ylabel('Variable index, $i$, for $u_i$')

        format = 'png'
        dpiVal = 300

        f.savefig(outputName + '.' + format, format=format, dpi=dpiVal)
        plt.show()


    def plotErrMatrix(self,endTime,data,outputName):

        # https://stackoverflow.com/questions/13384653/imshow-extent-and-aspect
        plotLim = 10.0
        cmap = "viridis"
        numPlots = 1
        figSizeVal = (6,4)
        extent=[0,endTime,9-0.5,0+0.5]

        f = plt.figure(1,figsize=figSizeVal)

        #lbrt = [0.15,0.2,0.96,0.9]
        lbrt = [0.15,0.14,0.96,0.9]

        plt.subplots_adjust(left=lbrt[0], bottom=lbrt[1], right=lbrt[2], top=lbrt[3],wspace=None, hspace=None)


        im1= plt.imshow(np.rot90(data),cmap=cmap, clim=(0, plotLim),aspect='auto',interpolation="None",extent=extent)

        ax = plt.gca().set_yticks(np.arange(1, 9, 1));


        cbar1 = plt.colorbar(im1, ax=plt.gca(),ticks=[0,2,4,6,8])
        cbar1.ax.set_yticklabels(['0.0', '2.0', '4.0', '6.0','8.0'])
        #plt.show()

        plt.gca().set_xlabel('Time, $t$')
        # plt.gca().set_ylabel('Degree of freedom, $i$')
        plt.gca().set_ylabel('Variable index, $i$, for $u_i$')

        format = 'png'
        dpiVal = 300

        f.savefig(outputName + '.' + format, format=format, dpi=dpiVal)
        plt.show()
