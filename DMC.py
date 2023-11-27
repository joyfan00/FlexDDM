# packages 
# import pandas as pd
import numpy as np
import random
import math
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
        self.bounds = [(0,1),(0,1),(1,20),(1,10),(0.001,10),(0,1),(0,min(self.data['rt']))]
        super().__init__(self.param_number, self.bounds, self.parameter_names)

    @staticmethod
    def model_simulation(alpha, beta, tau, shape, characteristic_time, peak_amplitude, mu_c, dt=0.001, var=.1, nTrials=5000, noiseseed=0):
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
        # if (len(parameters) != self.param_number):
        #     print('Dictionary input is not correct.')
        
        # # define variables 
        # alpha = parameters['alpha']
        # beta = parameters['beta']
        # tau = parameters['tau']
        # shape = parameters['shape']
        # characteristic_time = parameters['characteristic_time']
        # peak_amplitude = parameters['peak_amplitude']
        # mu_c = parameters['mu_c']
        print()

        choicelist = []
        rtlist = []
        np.random.seed(Variables.NOISESEED)
        update_jitter = np.random.normal(loc=0, scale=Variables.VAR, size=10000)

        ### Add congruent list (make first half congruent)
        # it assumes an even number of trials within each job -> make better 
        # randomly decide the last element to be congruent or incongruent 
        congruence_list = ['congruent'] * (Variables.NTRIALS // 2) + ['incongruent'] * (Variables.NTRIALS // 2)
        iter = 0
        for n in range(0, Variables.NTRIALS):
            # congruence
            isCongruent = False
            if n < Variables.NTRIALS / 2:
                isCongruent = True
            t = tau # start the accumulation process at non-decision time tau
            evidence = beta*alpha/2 - (1-beta)*alpha/2
            random.seed(iter)
            iter += 1
            print(shape)
            print(characteristic_time)
            print(peak_amplitude)
            while (evidence < alpha/2 and evidence > -alpha/2): # keep accumulating evidence until you reach a threshold
                print("hello 2")
                print(shape, characteristic_time, peak_amplitude, t, mu_c, isCongruent)
                evidence += DMC.calculateDelta(shape, characteristic_time, peak_amplitude, t, mu_c, isCongruent)*Variables.DT + random.choice(update_jitter)
                t += Variables.DT # increment time by the unit dt
                if evidence > alpha/2:
                    choicelist.append(1) # choose the upper threshold action
                    rtlist.append(t)
                elif evidence < -alpha/2:
                    choicelist.append(0) # choose the lower threshold action
                    rtlist.append(t)

        return (range(1, Variables.NTRIALS+1), choicelist, rtlist, congruence_list)
    
    
    @staticmethod
    def calculateDelta(shape, characteristic_time, peak_amplitude, automatic_time, mu_c, is_congruent):
        if not is_congruent:
            return (-peak_amplitude * math.exp(-(automatic_time / characteristic_time)) *
            math.pow(((automatic_time * math.exp(1)) / ((shape - 1) * characteristic_time)), (shape - 1)) * (((shape - 1) / automatic_time) - (1 / characteristic_time))) + mu_c
        return (peak_amplitude * math.exp(-(automatic_time / characteristic_time)) *
        math.pow(((automatic_time * math.exp(1)) / ((shape - 1) * characteristic_time)), (shape - 1)) * (((shape - 1) / automatic_time) - (1 / characteristic_time))) + mu_c
    
    
    