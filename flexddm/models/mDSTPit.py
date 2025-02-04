# import pandas as pd
import numpy as np
import pandas as pd
import numba as nb
from .Model import Model
from flexddm import _utilities as util

"""
Class to simulate data according to the Dual Stage Two Phase model (DSTP) 
"""

class mDSTP(Model):

    global data
    global bounds
    global parameter_names
    global param_number
    
    DT = 0.01
    VAR = 0.01
    NTRIALS = 100
    NOISESEED = 50

    def __init__(self, data=None, input_data_id="PPT", input_data_congruency="Condition", input_data_rt="RT", input_data_accuracy="Correct"):
        """
        Initializes an mDSTP model object.
        """
        self.modelsimulationfunction = mDSTP.model_simulation

        if data is not None:
            if isinstance(data, str):
                self.data = util.getRTData(data, input_data_id, input_data_congruency, input_data_rt, input_data_accuracy)
            else:
                self.data = data
            self.bounds = {
                "alphaSS": (0, 14),
                "betaSS": (0, 1),
                "etaSS": (0.25, 0.55),
                "alphaRS": (0.14, 0.38),
                "betaRS": (0, 1),
                "etaS1": (0.05, 0.15),
                "etaS2": (0.05, 0.25),
                "eta_r": (0.4, 1.2),
                "tau": (0.15, min(self.data['rt']))
            }
        else:
            self.bounds = {
                "alphaSS": (0.14, 0.38),
                "betaSS": (0, 1),
                "etaSS": (0.25, 0.55),
                "alphaRS": (0.14, 0.38),
                "betaRS": (0, 1),
                "etaS1": (0.05, 0.15),
                "etaS2": (0.05, 0.25),
                "eta_r": (0.4, 1.2),
                "tau": (0.15, 0.45)
            }
        
        self.parameter_names = list(self.bounds.keys())
        self.param_number = len(self.parameter_names)

        super().__init__(self.param_number, list(self.bounds.values()), self.parameter_names)


    @nb.jit(nopython=False, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(alphaSS, betaSS, etaSS, alphaRS, betaRS, etaS1, etaS2, eta_r, tau, dt=DT, var=VAR, nTrials=NTRIALS, noiseseed=NOISESEED):
        """
        Performs simulations for DSTP model.
        @alphaSS (float): boundary separation for stimulus selection phase
        @betaSS (float): initial bias for stimulus selection phase
        @deltaSS (float): drift rate for stimulus selection phase
        @alphaRS (float): boundary separation for response selection phase 
        @betaRS (float): inital bias for response selection phase 
        @delta_target (float): drift rate for target arrow during response selection BEFORE stimulus is selected 
        @delta_flanker (float): drift rate for flanker arrows during response selection BEFORE stimulus is selected
        @deltaRS (float): drift rate for the reponse selection phase after a stimulus (either flanker or target) has been selected
        @tau (float): non-decision time
        @dt (float): change in time 
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """

        deltaSS = 0.4
        delta_target = 0.1
        delta_flanker = 0.15
        deltaRS = 0.8

        choicelist = [np.nan]*nTrials
        rtlist = [np.nan]*nTrials
        np.random.seed(noiseseed)
        update_jitter = np.random.normal(loc=0, scale=var, size=10000)
        update_jitter_SS = update_jitter * (10**etaSS)
        update_jitter_RS1 = update_jitter * (10**etaS1)
        update_jitter_RS2 = update_jitter * (10**etaS2)

        congruencylist = ['congruent']*int(nTrials//2) + ['incongruent']*int(nTrials//2) 
        for n in np.arange(0, nTrials):
            if congruencylist[n] == 'congruent':
                deltaRS1 = delta_target + delta_flanker
            else:
                deltaRS1 = delta_target - delta_flanker
            t = tau # start the accumulation process at non-decision time tau
            evidenceSS = betaSS*alphaSS - (1-betaSS)*alphaSS # start our evidence at initial-bias beta (Kyle: I modified this so beta is always between 0 and 1, and alpha is the total distance between bounds)
            evidenceRS1 = betaRS*alphaRS - (1-betaRS)*alphaRS
            np.random.seed(n)
            while (evidenceSS < alphaSS/2 and evidenceSS > -alphaSS/2) or (evidenceRS1 < alphaRS/2 and evidenceRS1 > -alphaRS/2): # keep accumulating evidence until you reach a threshold
                delta_noise_SS = np.random.choice(update_jitter_SS)
                delta_noise_RS1 = np.random.choice(update_jitter_RS1)
                delta_noise_SS = delta_noise_SS*(np.exp(-1*(eta_r/2)*((t-tau))))
                delta_noise_RS1 = delta_noise_RS1*(np.exp(-1*(eta_r/2)*((t-tau))))
                evidenceSS += deltaSS*dt + delta_noise_SS # add one of the many possible updates to evidence
                evidenceRS1 += deltaRS1*dt + delta_noise_RS1
                t += dt # increment time by the unit dt
            if evidenceRS1 > alphaRS/2:
                choicelist[n] = 1 # choose the upper threshold action
                rtlist[n] = t
            elif evidenceRS1 < -alphaRS/2:
                choicelist[n] = 0 # choose the lower threshold action
                rtlist[n] = t

            # If Stimulus Selection (SS) completes before Response Selection 1 (RS1), begin Response Selection 2 (RS2) from where RS1 left off 
            else:
                if evidenceSS > alphaSS/2:
                    deltaRS = abs(deltaRS)
                elif evidenceSS < -alphaSS/2:
                    deltaRS = -1 * deltaRS
                evidenceRS2 = evidenceRS1 # start where you left off from RS1
                while (evidenceRS2 < alphaRS/2 and evidenceRS2 > -alphaRS/2): # keep accumulating evidence until you reach a threshold
                    delta_noise_RS2 = np.random.choice(update_jitter_RS2)
                    delta_noise_RS2 = delta_noise_RS2*(np.exp(-1*(eta_r/2)*((t-tau))))
                    evidenceRS2 += deltaRS*dt + delta_noise_RS2
                    t += dt # increment time by the unit dt
                if evidenceRS2 > alphaRS/2:
                    choicelist[n] = 1 # choose the upper threshold action
                    rtlist[n] = t
                elif evidenceRS2 < -alphaRS/2:
                    choicelist[n] = 0 # choose the lower threshold action
                    rtlist[n] = t
        return (np.arange(1, nTrials+1), choicelist, rtlist, congruencylist)
    




