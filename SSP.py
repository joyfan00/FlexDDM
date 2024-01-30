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
    # different file that processes the flanker data 
    # create documentation for what the csv, pkl, or json should look like before inputting 
    param_number = 6
    global bounds
    global data
    parameter_names = ['alpha', 'beta', 'p', 'sd_0', 'sd_r', 'tau']
    
    def __init__(self):
        """
        Initializes a DSTP model object. 
        """
        self.data = getRTData()
        self.bounds = [(0,1),(0,1),(0,1),(0,3),(0,1),(0,min(self.data['rt']))]
        super().__init__(self.param_number, self.bounds, self.parameter_names)

    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation (alpha, beta, p, sd_0, sd_r, tau, dt=Variables.DT, var=Variables.VAR, nTrials=Variables.NTRIALS, noiseseed=Variables.NOISESEED):
        choicelist = [np.nan]*nTrials
        rtlist = [np.nan]*nTrials

        congruencylist = ['congruent']*int(nTrials//2) + ['incongruent']*int(nTrials//2) 
        np.random.seed(noiseseed)

        noise = np.random.normal(loc=0, scale=.01, size=10000)
        iter=0
        for n in np.arange(0, nTrials):
            t = tau # start the accumulation process at non-decision time tau
            evidence = beta*alpha/2 - (1-beta)*alpha/2 # start our evidence at initial-bias beta
            while evidence < alpha/2 and evidence > -alpha/2: # keep accumulating evidence until you reach a threshold
                sd = sd_0 - (sd_r * (t-tau))
                if sd <= 0.001:
                    sd = 0.001
                s_ta = np.sum(np.random.normal(0, sd, size=1000) <= 0.5) / 1000 - np.sum(np.random.normal(0, sd, size=1000) <= -0.5) / 1000
                # s_ta = np.random.normal(0, sd).cdf(.5) - np.random.normal(0, sd).cdf(-.5)
                s_fl = 1 - s_ta
                if congruencylist[n] == 'incongruent':
                    delta = s_ta*p - s_fl*p
                else:
                    delta = s_ta*p + s_fl*p
                np.random.seed(100+iter)
                evidence += (delta*dt + np.random.choice(noise)) # add one of the many possible updates to evidence
                t += dt # increment time by the unit dt
                iter += 1
            if evidence > alpha/2:
                choicelist[n] = 1 # choose the upper threshold action
            else:
                choicelist[n] = 0 # choose the lower threshold action
            rtlist[n] = t
        return (np.arange(1, nTrials+1), choicelist, rtlist, congruencylist)
    
    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation2(alpha, beta, p, sd_0, sd_r, tau, dt=Variables.DT, var=Variables.VAR, nTrials=Variables.NTRIALS, noiseseed=Variables.NOISESEED):
        scale = 10000
        choicelist = np.empty(nTrials)
        rtlist = np.empty(nTrials)

        congruencylist = np.array(['congruent']*int(nTrials//2) + ['incongruent']*int(nTrials//2))
        np.random.seed(noiseseed)
        noise = np.random.normal(loc=0, scale=var, size=scale)

        tlist = np.arange(0, dt*scale, dt)
        s_ta = np.apply_along_axis(lambda x: SSP.sdfunc(x, sd_0, sd_r), 0, tlist)  # Replace with Numba-compatible function
        s_fl = np.apply_along_axis(lambda x: SSP.fastsub(x, s_ta), 0, np.ones(scale))  # Replace with Numba-compatible function

        for n in range(nTrials):
            tind = 0
            np.random.shuffle(noise)
            delta_ta = np.apply_along_axis(lambda x: SSP.fastmult(x, [p]*scale), 0, s_ta)  # Replace with Numba-compatible function
            delta_fl = np.apply_along_axis(lambda x: SSP.fastmult(x, [p]*scale), 0, s_fl)  # Replace with Numba-compatible function

            if congruencylist[n] == 'incongruent':
                delta = np.apply_along_axis(lambda x: SSP.fastsub(x, delta_fl), 0, delta_ta)  # Replace with Numba-compatible function
            else:
                delta = np.apply_along_axis(lambda x: SSP.fastadd(x, delta_fl), 0, delta_ta)  # Replace with Numba-compatible function

            changelist = np.apply_along_axis(lambda x: SSP.fastmult(x, [dt]*scale), 0, delta)  # Replace with Numba-compatible function
            evidencelist = changelist + noise
            evidencelist = np.insert(evidencelist, 0, beta*alpha/2 - (1-beta)*alpha/2)

            cumulative_evidence = np.cumsum(evidencelist)
            abovelist = cumulative_evidence > alpha/2
            belowlist = cumulative_evidence < -alpha/2

            while (np.sum(abovelist) < 1 and np.sum(belowlist) < 1):
                tind += scale
                oldevidence = cumulative_evidence[-1]
                np.random.shuffle(noise)

                tlist = np.arange(dt*tind, dt*(tind+scale), dt)
                if (sd_0 - (sd_r * (dt*tind))) <= 0.001:
                    s_ta = np.array([norm(0, 0.001).cdf(0.5) - norm(0, 0.001).cdf(-0.5)]*scale)
                else:
                    s_ta = np.apply_along_axis(lambda x: SSP.sdfunc(x, sd_0, sd_r), 0, tlist)  # Replace with Numba-compatible function
                s_fl = np.apply_along_axis(lambda x: SSP.fastsub(x, s_ta), 0, np.ones(scale))  # Replace with Numba-compatible function

                delta_ta = np.apply_along_axis(lambda x: SSP.fastmult(x, [p]*scale), 0, s_ta)  
                delta_fl = np.apply_along_axis(lambda x: SSP.fastmult(x, [p]*scale), 0, s_fl)  

                if congruencylist[n] == 'incongruent':
                    delta = np.apply_along_axis(lambda x: SSP.fastsub(x, delta_fl), 0, delta_ta)  
                else:
                    delta = np.apply_along_axis(lambda x: SSP.fastadd(x, delta_fl), 0, delta_ta)  

                changelist = np.apply_along_axis(lambda x: SSP.fastmult(x, [dt]*scale), 0, delta)  
                evidencelist = changelist + noise
                evidencelist = np.insert(evidencelist, 0, oldevidence)

                cumulative_evidence = np.cumsum(evidencelist)
                abovelist = cumulative_evidence > alpha/2
                belowlist = cumulative_evidence < -alpha/2

            above = scale
            below = scale

            if np.sum(abovelist) > 0:
                above = np.argmax(abovelist)
            if np.sum(belowlist) > 0:
                below = np.argmax(belowlist)

            if min(above, below) == above:
                choicelist[n] = 1
            else:
                choicelist[n] = 0

            t = (min(above, below) + tind)*dt + tau
            rtlist[n] = t

        return (np.arange(1, nTrials+1), choicelist, rtlist, congruencylist)
    
    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation_1(alpha, beta, p, sd_0, sd_r, tau, dt=Variables.DT, var=Variables.VAR, nTrials=Variables.NTRIALS, noiseseed=Variables.NOISESEED):
        """
        Performs simulations for DMC model.
        @parameters (dict): contains all variables and associated values for DMC models- 
        keys include: alpha, beta, p, sd_0, sd_r, tau
        @dt (float): change in time 
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """

        scale = 10000
        choicelist = [np.nan]*nTrials
        rtlist = [np.nan]*nTrials

        congruencylist = ['congruent']*int(nTrials//2) + ['incongruent']*int(nTrials//2) 
        np.random.seed(noiseseed)
        noise = np.random.normal(loc=0, scale=var, size=scale)

        tlist = np.array(np.arange(0, dt*scale, dt))
        s_ta = np.apply_along_axis(lambda x: SSP.sdfunc(x, sd_0, sd_r), 0, tlist)
        s_fl = np.apply_along_axis(lambda x: SSP.fastsub(x, s_ta), 0, [1]*scale)
        for n in np.arange(0, nTrials):
            tind = 0
            np.random.shuffle(noise)
            delta_ta = np.apply_along_axis(lambda x: SSP.fastmult(x, [p]*scale), 0, s_ta)
            delta_fl = np.apply_along_axis(lambda x: SSP.fastmult(x, [p]*scale), 0, s_fl)
            if congruencylist[n] == 'incongruent':
                delta = np.apply_along_axis(lambda x: SSP.fastsub(x, delta_fl), 0, delta_ta)
            else:
                delta = np.apply_along_axis(lambda x: SSP.fastadd(x, delta_fl), 0, delta_ta)
            changelist = np.apply_along_axis(lambda x: SSP.fastmult(x, [dt]*scale), 0, delta)
            evidencelist = np.array(np.apply_along_axis(lambda x: SSP.fastadd(x, noise), 0, changelist))
            evidencelist.insert(0, beta*alpha/2 - (1-beta)*alpha/2)
            cumulative_evidence = np.cumsum(evidencelist)
            abovelist = np.array(np.apply_along_axis(lambda i: i > alpha/2, 0, cumulative_evidence))
            belowlist = np.array(np.apply_along_axis(lambda i: i < -alpha/2, 0, cumulative_evidence))
            while (abovelist.count(True) < 1 and belowlist.count(True) < 1):
                tind += scale
                oldevidence = cumulative_evidence[-1]
                np.random.shuffle(noise)
                tlist = np.arange(dt*tind, dt*(tind+scale), dt)
                if (sd_0 - (sd_r * (dt*tind))) <= 0.001:
                    s_ta = [norm(0, 0.001).cdf(.5) - norm(0, 0.001).cdf(-.5)]*scale
                else:
                    s_ta = np.apply_along_axis(lambda x: SSP.sdfunc(x, sd_0, sd_r), 0, tlist)
                s_fl = np.apply_along_axis(lambda x: SSP.fastsub(x, s_ta), 0, [1]*scale)
                delta_ta = np.apply_along_axis(lambda x: SSP.fastmult(x, [p]*scale), 0, s_ta)
                delta_fl = np.apply_along_axis(lambda x: SSP.fastmult(x, [p]*scale), 0, s_fl)
                if congruencylist[n] == 'incongruent':
                    delta = np.apply_along_axis(lambda x: SSP.fastsub(x, delta_fl), 0, delta_ta)
                else:
                    delta = np.apply_along_axis(lambda x: SSP.fastadd(x, delta_fl), 0, delta_ta)
                changelist = np.apply_along_axis(lambda x: SSP.fastmult(x, [dt]*scale), 0, delta)
                evidencelist = np.array(np.apply_along_axis(lambda x: SSP.fastadd(x, noise), 0, changelist))
                np.insert(evidencelist, 0, oldevidence)
                cumulative_evidence = np.cumsum(evidencelist)
                abovelist = np.array(np.apply_along_axis(lambda i: i > alpha/2, 0, cumulative_evidence))
                belowlist = np.array(np.apply_along_axis(lambda i: i < -alpha/2, 0, cumulative_evidence))
            above = scale; below = scale
            if abovelist.count(True) > 0:
                above = abovelist.index(True)
            if belowlist.count(True) > 0:
                below = belowlist.index(True)
            if min(above, below) == above:
                choicelist[n] = 1
            else:
                choicelist[n] = 0
            t = (min(above, below) + tind)*dt + tau
            rtlist[n] = t

        return (np.arange(1, nTrials+1), choicelist, rtlist, congruencylist)
    
    @staticmethod
    @nb.vectorize(nopython=True)
    def sdfunc(x, sd_0, sd_r):
        sd = sd = sd_0 - (sd_r * (x))
        sd = np.where(sd < 0.001, 0.001, sd)
        s_ta = norm(0, sd).cdf(.5) - norm(0, sd).cdf(-.5)
        return s_ta

    @staticmethod
    @nb.vectorize(nopython=True)
    def fastmult(x, list2):
        return x * list2

    @staticmethod
    @nb.vectorize(nopython=True)
    def fastsub(x, list2):
        return x - list2

    @staticmethod
    @nb.vectorize(nopython=True)
    def fastadd(x, list2):
        return x + list2