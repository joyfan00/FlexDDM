# packages 
from .Model import Model
import numba as nb
import numpy as np
import math
from flexddm import _utilities as util

"""
Class to simulate data according to the Shrinking Spotlight Model with interference time (SSPit) 
"""

class SSPit(Model):

    global bounds
    global data
    global parameter_names
    global param_number
    
    DT = 0.01  # NORMAL: 0.01
    VAR = 0.01
    NTRIALS = 100
    NOISESEED = 50

    def __init__(self, data=None, input_data_id="PPT", input_data_congruency="Condition", input_data_rt="RT", input_data_accuracy="Correct"):
        """
        Initializes an SSPit model object.
        """
        self.modelsimulationfunction = SSPit.model_simulation

        if data is not None:
            if isinstance(data, str):
                self.data = util.getRTData(data, input_data_id, input_data_congruency, input_data_rt, input_data_accuracy)
            else:
                self.data = data
            self.bounds = {
                "alpha": (0.07, 0.19),
                "beta": (0, 1),
                "p": (0.2, 0.55),
                "sd_0_sd_r_ratio": (0.01, 100),
                "tau": (0.15, min(self.data['rt']))
            }
        else:
            self.bounds = {
                "alpha": (0.07, 0.19),
                "beta": (0, 1),
                "p": (0.2, 0.55),
                "sd_0_sd_r_ratio": (0.01, 100),
                "tau": (0.15, 0.45)
            }
        
        self.parameter_names = list(self.bounds.keys())
        self.param_number = len(self.parameter_names)

        super().__init__(self.param_number, list(self.bounds.values()), self.parameter_names)

    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(alpha, beta, p, sd_0_sd_r_ratio, tau, dt=DT, var=VAR, nTrials=NTRIALS, noiseseed=NOISESEED):
        """
        Performs simulations for SSP model.
        @alpha (float): boundary separation
        @beta (float): initial bias
        @p (float): perceptual input of the stimulus
        @sd_0_sd_r_ratio (float): ratio of the initial standard deviation of the Gaussian distribution describing the attentional spotlight and shrinking rate (i.e. interference time) 
        @tau (float): non-decision time
        @dt (float): change in time 
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """
        choicelist = [np.nan]*nTrials
        rtlist = [np.nan]*nTrials

        # beta = 0.5

        # Creates congruency list with first half of trials being congruent and the following being incongruent
        congruencylist = ['congruent']*int(nTrials//2) + ['incongruent']*int(nTrials//2) 
        np.random.seed(noiseseed)

        noise = np.random.normal(loc=0, scale=var, size=10000)
        for n in np.arange(0, nTrials):
            t = tau # start the accumulation process at non-decision time tau
            evidence = beta*alpha - (1-beta)*alpha # start our evidence at initial-bias beta
            np.random.seed(n)
            while evidence < alpha/2 and evidence > -alpha/2: # keep accumulating evidence until you reach a threshold
                sd = (sd_0_sd_r_ratio * 0.018) - (0.018 * (t-tau))
                if sd <= 0.001:
                    sd = 0.001
                s_ta = ((1 + math.erf((.5 - 0) / sd / np.sqrt(2))) / 2) - ((1 + math.erf((-.5 - 0) / sd / np.sqrt(2))) / 2)
                s_fl = 1 - s_ta
                if congruencylist[n] == 'incongruent':
                    delta = s_ta*p - s_fl*p
                else:
                    delta = s_ta*p + s_fl*p
                evidence += (delta*dt + np.random.choice(noise)) # add one of the many possible updates to evidence
                t += dt 
            if evidence > alpha/2:
                choicelist[n] = 1 # choose the upper threshold action
            else:
                choicelist[n] = 0 # choose the lower threshold action
            rtlist[n] = t
        return (np.arange(1, nTrials+1), choicelist, rtlist, congruencylist)