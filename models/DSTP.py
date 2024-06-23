# import pandas as pd
import numpy as np
import pandas as pd
import numba as nb
from .Model import Model

"""
Class to simulate data according to the Dual Stage Two Phase model (DSTP) 
"""

class DSTP(Model):

    global data
    param_number = 9
    global bounds
    parameter_names = ['alphaSS', 'betaSS', 'deltaSS', 'alphaRS', 'betaRS', 'delta_target', 'delta_flanker', 'deltaRS', 'tau']
    DT = 0.001
    VAR = 0.1
    NTRIALS = 1000
    NOISESEED = 50

    def __init__(self, data=None):
        """
        Initializes a DSTP model object. 
        """
        self.modelsimulationfunction = DSTP.model_simulation

        if data != None:
            if isinstance(data, str): 
                self.data = self.getRTData(data)
            else:
                self.data = data
            self.bounds = [(0,20),(0,1),(-1,1),(0,20),(0,1),(-1,1),(-1,1),(-3,3),(0,min(self.data['rt']))]
        else: 
            self.bounds = [(0,20),(0,1),(-1,1),(0,20),(0,1),(-1,1),(-1,1),(-3,3),(0,5)]
            
        super().__init__(self.param_number, self.bounds, self.parameter_names)

    @nb.jit(nopython=False, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(alphaSS, betaSS, deltaSS, alphaRS, betaRS, delta_target, delta_flanker, deltaRS, tau, dt=DT, var=VAR, nTrials=NTRIALS, noiseseed=NOISESEED):
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

        choicelist = [np.nan]*nTrials
        rtlist = [np.nan]*nTrials
        np.random.seed(noiseseed)
        update_jitter = np.random.normal(loc=0, scale=var, size=10000)
        congruencylist = ['congruent']*int(nTrials//2) + ['incongruent']*int(nTrials//2) 
        for n in np.arange(0, nTrials):
            if congruencylist[n] == 'congruent':
                deltaRS1 = delta_target + delta_flanker
            else:
                deltaRS1 = delta_target - delta_flanker
            t = tau # start the accumulation process at non-decision time tau
            evidenceSS = betaSS*alphaSS/2 - (1-betaSS)*alphaSS/2 # start our evidence at initial-bias beta (Kyle: I modified this so beta is always between 0 and 1, and alpha is the total distance between bounds)
            evidenceRS1 = betaRS*alphaRS/2 - (1-betaRS)*alphaRS/2
            np.random.seed(n)
            while (evidenceSS < alphaSS/2 and evidenceSS > -alphaSS/2) or (evidenceRS1 < alphaRS/2 and evidenceRS1 > -alphaRS/2): # keep accumulating evidence until you reach a threshold
                evidenceSS += deltaSS*dt + np.random.choice(update_jitter) # add one of the many possible updates to evidence
                evidenceRS1 += deltaRS1*dt + np.random.choice(update_jitter)
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
                iter = 0
                while (evidenceRS2 < alphaRS/2 and evidenceRS2 > -alphaRS/2): # keep accumulating evidence until you reach a threshold
                    np.random.seed(100 + iter)
                    evidenceRS2 += deltaRS*dt + np.random.choice(update_jitter)
                    t += dt # increment time by the unit dt
                    iter += 1
                if evidenceRS2 > alphaRS/2:
                    choicelist[n] = 1 # choose the upper threshold action
                    rtlist[n] = t
                elif evidenceRS2 < -alphaRS/2:
                    choicelist[n] = 0 # choose the lower threshold action
                    rtlist[n] = t
        return (np.arange(1, nTrials+1), choicelist, rtlist, congruencylist)
    




