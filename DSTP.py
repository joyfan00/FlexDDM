# import pandas as pd
import numpy as np
import random
import pandas as pd
from file_input import *
from Model import Model

"""
This class is a specific DSTP model class. 
"""

class DSTP(Model):

    data = pd.DataFrame()
    param_number = 9
    global bounds
    
    def __init__(self):
        """
        Initializes a DSTP model object. 
        """
        data = getRTData()
        self.bounds = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,2),(0,min(data['rt']))]
        super().__init__(self.param_number, self.bounds)

    def model_simulation(self, parameters, dt, var, nTrials, noiseseed):
        """
        Performs simulations for DMC model.
        @parameters (dict): contains all variables and associated values for DSTP models- 
        keys include: alphaSS, betaSS, tauSS, alphaRS, betaRS1, delta_target, delta_flanker, deltaRS2, tau
        @dt (float): change in time 
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """
        if (len(parameters) != self.param_number):
            print('Dictionary input is not correct.')
        # define variables 
        alphaSS = parameters['alphaSS']
        betaSS = parameters['betaSS']
        deltaSS = parameters['deltaSS']
        alphaRS = parameters['alphaRS']
        betaRS1 = parameters['betaRS1']
        delta_target = parameters['delta_target']
        delta_flanker = parameters['delta_flanker']
        deltaRS2 = parameters['deltaRS2']
        tau = parameters['tau']

        choicelist = []
        rtlist = []
        # sample a bunch of possible updates to evidence with each unit of time dt. Updates are centered
        # around drift rate delta (scaled to the units of time), with variance var (because updates are noisy)

        np.random.seed(noiseseed)
        update_jitter = np.random.normal(loc=0, scale=var, size=10000)

        # updatesRS1 = np.random.normal(loc=deltaRS1*dt, scale=var, size=10000)
        ### Add congruent list (make first half congruent)
        congruencylist = ['congruent']*int(nTrials/2) + ['incongruent']*int(nTrials/2) 
        #deltaRS2 is always positive. Later in the model, you should either keep it positive
        ##### if the target bound is selected by SS, or multiply it by -1 if the flanker bound is selected by SS.
        iter=0
        for n in range(0, nTrials):
            # Assuming first half of trials are congruent
            if congruencylist[n] == 'congruent':
                deltaRS1 = delta_target + delta_flanker
            else:
                deltaRS1 = delta_target - delta_flanker
            t = tau # start the accumulation process at non-decision time tau
            evidenceSS = betaSS*alphaSS/2 - (1-betaSS)*alphaSS/2 # start our evidence at initial-bias beta (Kyle: I modified this so beta is always between 0 and 1, and alpha is the total distance between bounds)
            evidenceRS1 = betaRS1*alphaRS/2 - (1-betaRS1)*alphaRS/2
            while (evidenceSS < alphaSS/2 and evidenceSS > -alphaSS/2) or (evidenceRS1 < alphaRS/2 and evidenceRS1 > -alphaRS/2): # keep accumulating evidence until you reach a threshold
                random.seed(100+iter)
                evidenceSS += deltaSS*dt + random.choice(update_jitter) # add one of the many possible updates to evidence
                evidenceRS1 += deltaRS1*dt + random.choice(update_jitter)
                t += dt # increment time by the unit dt
                iter += 1
            if evidenceRS1 > alphaRS/2:
                choicelist.append(1) # choose the upper threshold action
                rtlist.append(t)
            elif evidenceRS1 < -alphaRS/2:
                choicelist.append(0) # choose the lower threshold action
                rtlist.append(t)

            # If SS completes before RS1, begin RS2 from where RS1 left off (but update drift rate/other params)
            else:
                if evidenceSS > alphaSS/2:
                    deltaRS2 = abs(deltaRS2)
                elif evidenceSS < -alphaSS/2:
                    deltaRS2 = -1 * deltaRS2
                evidenceRS2 = evidenceRS1 # start where you left off from RS1
                ## We weren't sure if the decision threshold is the same across both response selection phases
                while (evidenceRS2 < alphaRS/2 and evidenceRS2 > -alphaRS/2): # keep accumulating evidence until you reach a threshold
                    random.seed(100+iter)
                    evidenceRS2 += deltaRS2*dt + random.choice(update_jitter)
                    t += dt # increment time by the unit dt
                    iter += 1
                if evidenceRS2 > alphaRS/2:
                    choicelist.append(1) # choose the upper threshold action
                    rtlist.append(t)
                elif evidenceRS2 < -alphaRS/2:
                    choicelist.append(0) # choose the lower threshold action
                    rtlist.append(t)
        return (range(1, nTrials+1), choicelist, rtlist, congruencylist)
    
