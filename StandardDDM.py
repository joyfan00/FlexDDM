# packages 
# import pandas as pd
import numpy as np
import random
import math
import numba as nb
from Model import Model
from file_input import *
from variables import Variables
import pandas as pd
import sys

"""
This class represents the standard drift diffusion model 
"""

class StandardDDM(Model):

    param_number = 4
    global bounds
    global data
    parameter_names = ['alpha', 'beta', 'delta', 'tau']
    variables = Variables()


    def __init__(self):
        """
        Initializes a standard diffusion model object. 
        """
        self.data = getRTData()
        self.bounds = [(0,10),(0,1),(1,20),(0,min(self.data['rt']))]
        super().__init__(self.param_number, self.bounds, self.parameter_names)

    # @staticmethod
    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation (alpha, beta, delta, tau, dt=Variables.DT, var=Variables.VAR, nTrials=Variables.NTRIALS, noiseseed=Variables.NOISESEED):
        choicelist = [np.nan]*nTrials
        rtlist = [np.nan]*nTrials
        congruencylist = ['congruent']*int(nTrials//2) + ['incongruent']*int(nTrials//2) 
        np.random.seed(noiseseed)
        updates = np.random.normal(loc=0, scale=.01, size=10000)
        for n in np.arange(0, nTrials):
            t = tau # start the accumulation process at non-decision time tau
            evidence = beta # start our evidence at initial-bias beta
            while evidence < alpha and evidence > -alpha: # keep accumulating evidence until you reach a threshold
                evidence += np.random.choice(updates) # add one of the many possible updates to evidence
                t += dt # increment time by the unit dt
            if evidence > alpha:
                choicelist[n] = 1 # choose the upper threshold action
            else:
                choicelist[n] = 0  # choose the lower threshold action
            rtlist[n] = t
        return (np.arange(1, nTrials+1), choicelist, rtlist, congruencylist)
