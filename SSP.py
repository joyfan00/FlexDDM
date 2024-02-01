# packages 
from Model import Model
from variables import Variables
from file_input import *
# import pandas as pd
import numba as nb
import numpy as np
import math
import random
from scipy.stats import norm
# from file_input import *

"""
This class is a specific SSP model class. 
"""

class SSP(Model):

    param_number = 6
    global bounds
    global data
    parameter_names = ['alpha', 'beta', 'p', 'sd_0', 'sd_r', 'tau']
    
    DT = 0.01
    VAR = 0.1
    NTRIALS = 100
    NOISESEED = 50

    def __init__(self):
        """
        Initializes a SSP model object. 
        """
        self.data = getRTData()
        self.bounds = [(0,20),(0,1),(0,2),(0,10),(0,100),(0,min(self.data['rt']))]
        super().__init__(self.param_number, self.bounds, self.parameter_names)

    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(alpha, beta, p, sd_0, sd_r, tau, dt=DT, var=VAR, nTrials=NTRIALS, noiseseed=NOISESEED):
        """
        Performs simulations for SSP model.
        @alpha (float): threshold
        @beta (float): initial bias
        @p (float):
        @sd_0 (float):
        @sd_r (float):
        @tau (float): non-decision time
        @dt (float): change in time 
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """
        choicelist = [np.nan]*nTrials
        rtlist = [np.nan]*nTrials

        # Creates congruency list with first half of trials being congruent and the following being incongruent
        congruencylist = ['congruent']*int(nTrials//2) + ['incongruent']*int(nTrials//2) 
        np.random.seed(noiseseed)

        noise = np.random.normal(loc=0, scale=.01, size=10000)
        for n in np.arange(0, nTrials):
            t = tau # start the accumulation process at non-decision time tau
            evidence = beta*alpha/2 - (1-beta)*alpha/2 # start our evidence at initial-bias beta
            np.random.seed(n)
            while evidence < alpha/2 and evidence > -alpha/2: # keep accumulating evidence until you reach a threshold
                sd = sd_0 - (sd_r * (t-tau))
                if sd <= 0.001:
                    sd = 0.001
                s_ta = ((1 + math.erf((.5 - 0) / sd / np.sqrt(2))) / 2) - ((1 + math.erf((-.5 - 0) / sd / np.sqrt(2))) / 2)
                s_fl = 1 - s_ta
                if congruencylist[n] == 'incongruent':
                    delta = s_ta*p - s_fl*p
                else:
                    delta = s_ta*p + s_fl*p
                evidence += (delta*dt + np.random.choice(noise)) # add one of the many possible updates to evidence
                t += dt # increment time by the unit dt
            if evidence > alpha/2:
                choicelist[n] = 1 # choose the upper threshold action
            else:
                choicelist[n] = 0 # choose the lower threshold action
            rtlist[n] = t
        return (np.arange(1, nTrials+1), choicelist, rtlist, congruencylist)