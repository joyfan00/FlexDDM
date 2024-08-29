# packages 
import numpy as np
import numba as nb
from .Model import Model
from flexddm import _utilities as util

"""
Class to simulate data according to the Diffusion Model for Conlict (DMC) 
"""

class DMC (Model):

    global bounds
    global data
    parameter_names = ['alpha', 'beta', 'mu_c', 'shape', 'characteristic_time', 'peak_amplitude', 'tau']
    param_number = len(parameter_names)

    DT = 0.01
    VAR = 0.01
    NTRIALS = 100
    NOISESEED = 50

    def __init__(self, data=None, input_data_id="PPT", input_data_congruency="Condition", input_data_rt="RT", input_data_accuracy="Correct"):
        """
        Initializes a DMC model object. 
        """
        self.modelsimulationfunction = DMC.model_simulation

        if data != None:
            if isinstance(data, str): 
                self.data = util.getRTData(data, input_data_id, input_data_congruency, input_data_rt, input_data_accuracy)
            else:
                self.data = data
            self.bounds = [(0.07,0.38),(0,1),(0.2,0.8),(1.5,4.5),(0.01,1),(0.015,0.4),(0.15,min(self.data['rt']))]
        else: 
            self.bounds = [(0.07,0.38),(0,1),(0.2,0.8),(1.5,4.5),(0.01,1),(0.015,0.4),(0.15,0.45)]

        super().__init__(self.param_number, self.bounds, self.parameter_names)


    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(alpha, beta, mu_c, shape, characteristic_time, peak_amplitude, tau, dt=DT, var=VAR, nTrials=NTRIALS, noiseseed=NOISESEED,):
        """
        Performs simulations for DMC model. 
        @alpha (float): boundary separation
        @beta (float): initial bias
        @mu_c (float): drift rate of the controlled process
        @shape (float): shape parameter of gamma distribution function used to model the time-course of automatic activation 
        @characteristic_time (float): duration of the automatic process
        @peak_amplitude (float): amplitude of automatic activation
        @tau (float): non-decision time
        @dt (float): change in time 
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """

        choicelist = [np.nan]*nTrials
        rtlist = [np.nan]*nTrials
        np.random.seed(noiseseed)
        update_jitter = np.random.normal(loc=0, scale=var, size=1000)

        # beta = 0.5

        # Creates congruency list with first half of trials being congruent and the following being incongruent
        congruencylist = ['congruent'] * (nTrials // 2) + ['incongruent'] * (nTrials // 2)

        for n in np.arange(0, nTrials):
            t = tau # start the accumulation process at non-decision time tau
            evidence = beta*alpha/2 - (1-beta)*alpha/2
            np.random.seed(n)
            while (evidence < alpha/2 and evidence > -alpha/2): # keep accumulating evidence until you reach a threshold
                if congruencylist[n] == 'congruent':
                    delta = (peak_amplitude * np.exp(-(t / characteristic_time)) *
                            np.power(((t * np.exp(1)) / ((shape - 1) * characteristic_time)), (shape - 1)) * (((shape - 1) / t) - (1 / characteristic_time))) + mu_c
                else:
                    delta = (-peak_amplitude * np.exp(-(t / characteristic_time)) *
                            np.power(((t * np.exp(1)) / ((shape - 1) * characteristic_time)), (shape - 1)) * (((shape - 1) / t) - (1 / characteristic_time))) + mu_c
                evidence += delta*dt + np.random.choice(update_jitter)
                t += dt # increment time by the unit dt
                if evidence > alpha/2:
                    choicelist[n] = 1
                    rtlist[n] = t
                elif evidence < -alpha/2:
                    choicelist[n] = 0
                    rtlist[n] = t

        return (np.arange(1, nTrials+1), choicelist, rtlist, congruencylist)
    
    