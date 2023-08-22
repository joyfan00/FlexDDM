# packages 
# import pandas as pd
import numpy as np
import random
import math
from Model import Model
from file_input import *
import pandas as pd

"""
This class is a specific DSTP model class. 
"""

class DMC (Model):

    param_number = 7
    global bounds

    def __init__(self):
        """
        Initializes a DMC model object. 
        """
        data = getRTData()
        self.bounds = [(0,1),(0,1),(1,20),(0,10),(0,10),(0,1),(0,min(data['rt']))]
        super().__init__(self.param_number, self.bounds)

    def model_simulation(self, parameters, dt, var, nTrials, noiseseed):
        """
        Performs simulations for DMC model.
        @parameters (dict): contains all variables and associated values for DMC models- 
        keys include: alpha, beta, tau, shape, characteristic_time, peak_amplitude, mu_c
        @dt (float): change in time 
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """
        # quick check to make sure that dictionary is inputted correctly 
        if (len(parameters) != self.param_number):
            print('Dictionary input is not correct.')
        
        # define variables 
        alpha = parameters['alpha']
        beta = parameters['beta']
        tau = parameters['tau']
        shape = parameters['shape']
        characteristic_time = parameters['characteristic_time']
        peak_amplitude = parameters['peak_amplitude']
        mu_c = parameters['mu_c']

        choicelist = []
        rtlist = []
        np.random.seed(noiseseed)
        update_jitter = np.random.normal(loc=0, scale=var, size=10000)

        ### Add congruent list (make first half congruent)
        congruence_list = ['congruent'] * (nTrials // 2) + ['incongruent'] * (nTrials // 2)
        iter = 0
        for n in range(0, nTrials):
            # congruence
            isCongruent = False
            if n < nTrials / 2:
                isCongruent = True
            t = tau # start the accumulation process at non-decision time tau
            evidence = beta*alpha/2 - (1-beta)*alpha/2
            random.seed(iter)
            iter += 1
            while (evidence < alpha/2 and evidence > -alpha/2): # keep accumulating evidence until you reach a threshold
                evidence += self.calculateDelta(shape, characteristic_time, peak_amplitude, t, mu_c, isCongruent)*dt + random.choice(update_jitter)
                t += dt # increment time by the unit dt
                if evidence > alpha/2:
                    choicelist.append(1) # choose the upper threshold action
                    rtlist.append(t)
                elif evidence < -alpha/2:
                    choicelist.append(0) # choose the lower threshold action
                    rtlist.append(t)

        return (range(1, nTrials+1), choicelist, rtlist, congruence_list)
    
    def calculateDelta (self, shape, characteristic_time, peak_amplitude, automatic_time, mu_c, congruence):
        """
        Calculates the delta in accordance to the time. Necessary for DMC model simulations.
        @shape (float): shape parameter of the automatic process drift rate
        @characteristic_time (float): characteristic time parameter of the automatic process drift rate
        @peak_amplitude (float): peak amplitude parameter of the automatic process drift rate
        @automatic_time (float): automatic time parameter of the automatic process drift rate
        @mu_c (float): controlled process drift rate
        """
        if not congruence:
            return (-peak_amplitude * math.exp(-(automatic_time / characteristic_time)) *
            math.pow(((automatic_time * math.exp(1)) / ((shape - 1) * characteristic_time)), (shape - 1)) * (((shape - 1) / automatic_time) - (1 / characteristic_time))) + mu_c
        return (peak_amplitude * math.exp(-(automatic_time / characteristic_time)) *
        math.pow(((automatic_time * math.exp(1)) / ((shape - 1) * characteristic_time)), (shape - 1)) * (((shape - 1) / automatic_time) - (1 / characteristic_time))) + mu_c