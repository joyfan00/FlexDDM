# packages 
import numpy as np
import numba as nb
from .Model import Model

"""
Class to simulate data according to the standard flanker drift diffusion model  
"""

class StandardDDM(Model):
    param_number = 6
    global bounds
    global data
    parameter_names = ['alpha_c', 'alpha_i', 'beta', 'delta_c', 'delta_i', 'tau']

    DT = 0.01
    VAR = 0.1
    NTRIALS = 100
    NOISESEED = 50

    def __init__(self, path="S1FlankerData.csv"):
        """
        Initializes a standard diffusion model object. 
        """
        self.data = self.getRTData(path)
        self.bounds = [(0,20),(0,20),(0,1),(-10,10),(-10,10),(0,min(self.data['rt']))]
        super().__init__(self.param_number, self.bounds, self.parameter_names)

    # @staticmethod
    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation (alpha_c, alpha_i, beta, delta_c, delta_i, tau, dt=DT, var=VAR, nTrials=NTRIALS, noiseseed=NOISESEED):
        """
        Performs simulations for standard flanker diffusion model.
        @alpha_c (float): boundary separation for congruent trials
        @alpha_i (float): boundary separation for incongruent trials 
        @beta (float): initial bias
        @delta_c (float): drift rate for incongruent trials
        @delta_i (float): drift rate for incongruent trials
        @tau (float): non-decision time
        @dt (float): change in time 
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """
        choicelist = [np.nan]*nTrials
        rtlist = [np.nan]*nTrials
        congruencylist = ['congruent']*int(nTrials//2) + ['incongruent']*int(nTrials//2) 
        np.random.seed(noiseseed)
        updates = np.random.normal(loc=0, scale=var, size=10000)
        for n in np.arange(0, nTrials):
            if congruencylist[n] == 'congruent':
                alpha = alpha_c
                delta = delta_c
            else:
                alpha = alpha_i
                delta = delta_i
            t = tau # start the accumulation process at non-decision time tau
            evidence = beta # start our evidence at initial-bias beta
            np.random.seed(n)
            while evidence < alpha and evidence > -alpha: # keep accumulating evidence until you reach a threshold
                evidence += delta*dt + np.random.choice(updates) # add one of the many possible updates to evidence
                t += dt # increment time by the unit dt
            if evidence > alpha:
                choicelist[n] = 1 # choose the upper threshold action
            else:
                choicelist[n] = 0  # choose the lower threshold action
            rtlist[n] = t
        return (np.arange(1, nTrials+1), choicelist, rtlist, congruencylist)
congruencylist = 0




