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
This class is a specific DSTP model class. 
"""

class DMC (Model):

    param_number = 7
    global bounds
    global data
    parameter_names = ['alpha', 'beta', 'tau', 'shape', 'characteristic_time', 'peak_amplitude', 'mu_c']
    variables = Variables()


    def __init__(self):
        """
        Initializes a DMC model object. 
        """
        self.data = getRTData()
        self.bounds = [(0,10),(0,1),(1,20),(1,10),(0.001,10),(0,1),(0,min(self.data['rt']))]
        super().__init__(self.param_number, self.bounds, self.parameter_names)

    # @staticmethod
    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(alpha, beta, tau, shape, characteristic_time, peak_amplitude, mu_c, dt=Variables.DT, var=Variables.VAR, nTrials=Variables.NTRIALS, noiseseed=Variables.NOISESEED):
        """
        Performs simulations for DMC model.
        @parameters (dict): contains all variables and associated values for DMC models- 
        keys include: alpha, beta, tau, shape, characteristic_time, peak_amplitude, mu_c
        @dt (float): change in time 
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """

        choicelist = [np.nan]*nTrials
        rtlist = [np.nan]*nTrials
        # choicelist = []
        # rtlist = []
        np.random.seed(noiseseed)
        update_jitter = np.random.normal(loc=0, scale=var, size=10000)

        ### Add congruent list (make first half congruent)
        # it assumes an even number of trials within each job -> make better 
        # randomly decide the last element to be congruent or incongruent 
        congruence_list = ['congruent'] * (nTrials // 2) + ['incongruent'] * (nTrials // 2)

        for n in np.arange(0, nTrials):
        # for n in nb.prange(0, nTrials):
            # congruence
            isCongruent = False
            if n < nTrials / 2:
                isCongruent = True
            t = tau # start the accumulation process at non-decision time tau
            evidence = beta*alpha/2 - (1-beta)*alpha/2
            np.random.seed(n)
            while (evidence < alpha/2 and evidence > -alpha/2): # keep accumulating evidence until you reach a threshold
                if not isCongruent:
                    delta = (-peak_amplitude * np.exp(-(t / characteristic_time)) *
                            np.power(((t * np.exp(1)) / ((shape - 1) * characteristic_time)), (shape - 1)) * (((shape - 1) / t) - (1 / characteristic_time))) + mu_c
                else:
                    delta = (peak_amplitude * np.exp(-(t / characteristic_time)) *
                            np.power(((t * np.exp(1)) / ((shape - 1) * characteristic_time)), (shape - 1)) * (((shape - 1) / t) - (1 / characteristic_time))) + mu_c
                evidence += delta*dt + np.random.choice(update_jitter)
                t += dt # increment time by the unit dt
                if evidence > alpha/2:
                    # choicelist.append(1) # choose the upper threshold action
                    # rtlist.append(t)
                    choicelist[n] = 1
                    rtlist[n] = t
                elif evidence < -alpha/2:
                    # choicelist.append(0) # choose the lower threshold action
                    # rtlist.append(t)
                    choicelist[n] = 0
                    rtlist[n] = t

        return (np.arange(1, nTrials+1), choicelist, rtlist, congruence_list)
    