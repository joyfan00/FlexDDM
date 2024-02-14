# packages 
# import pandas as pd
import numpy as np
import random
import math
import numba as nb
from .Model import Model
import pandas as pd
import sys

"""
This class is a specific DSTP model class. 
"""

class DMC (Model):

    param_number = 7
    global bounds
    global data
    parameter_names = ['alpha', 'beta', 'mu_c', 'shape', 'characteristic_time', 'peak_amplitude', 'tau']

    DT = 0.01
    VAR = 0.1
    NTRIALS = 100
    NOISESEED = 50

    def __init__(self, path="S1FlankerData.csv"):
        """
        Initializes a DMC model object. 
        """
        self.data = self.getRTData(path)
        self.bounds = [(0,20),(0,1),(-10,10),(1,10),(0.001,10),(0,1),(np.exp(-10),min(self.data['rt']))]
        super().__init__(self.param_number, self.bounds, self.parameter_names)

    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(alpha, beta, mu_c, shape, characteristic_time, peak_amplitude, tau, dt=DT, var=VAR, nTrials=NTRIALS, noiseseed=NOISESEED):
        """
        Performs simulations for DMC model.
        @alpha (float): threshold
        @beta (float): initial bias
        @mu_c (float):
        @shape (float):
        @characteristic_time (float):
        @peak_amplitude (float):
        @tau (float): non-decision time
        @dt (float): change in time 
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """

        choicelist = [np.nan]*nTrials
        rtlist = [np.nan]*nTrials
        np.random.seed(noiseseed)
        update_jitter = np.random.normal(loc=0, scale=var, size=10000)

        # Creates congruency list with first half of trials being congruent and the following being incongruent
        congruence_list = ['congruent'] * (nTrials // 2) + ['incongruent'] * (nTrials // 2)

        for n in np.arange(0, nTrials):
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
                    choicelist[n] = 1
                    rtlist[n] = t
                elif evidence < -alpha/2:
                    choicelist[n] = 0
                    rtlist[n] = t

        return (np.arange(1, nTrials+1), choicelist, rtlist, congruence_list)
    