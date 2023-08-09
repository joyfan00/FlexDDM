# packages 
from Model import Model
import pandas as pd
import numpy as np
import math
import random

"""
This class is a specific DSTP model class. 
"""

class SSP (Model):
    data = pd.DataFrame();
    param_number = 6
    bounds = [(0,1),(0,1),(0,1),(0,3),(0,1),(0,min(data['rt']))]
    
    def __init__(self):
        """
        Initializes a DSTP model object. 
        """
        super().__init__(self.param_number, self.bounds)
    
    def model_simulation(self, parameters, dt, var, nTrials, noiseseed):
        """
        Performs simulations for DMC model.
        @parameters (dict): contains all variables and associated values for DMC models- 
        keys include: alpha, beta, p, sd_0, sd_r, tau
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
        p = parameters['p']
        sd_0 = parameters['sd_0']
        sd_r = parameters['sd_r']
        tau = parameters['tau']

        scale = 10000
        choicelist = []
        rtlist = []
        congruencylist = ['congruent']*int(nTrials/2) + ['incongruent']*int(nTrials/2) 
        np.random.seed(noiseseed)
        noise = np.random.normal(loc=0, scale=.01, size=scale)
        tlist = np.array(np.arange(0, dt*scale, dt))
        s_ta = np.apply_along_axis(lambda x: self.sdfunc(x, sd_0, sd_r), 0, tlist)
        s_fl = np.apply_along_axis(lambda x: self.fastsub(x, s_ta), 0, [1]*scale)
        for n in range(0, nTrials):
            tind = 0
            np.random.shuffle(noise)
            delta_ta = np.apply_along_axis(lambda x: self.fastmult(x, [p]*scale), 0, s_ta)
            delta_fl = np.apply_along_axis(lambda x: self.fastmult(x, [p]*scale), 0, s_fl)
            if congruencylist[n] == 'incongruent':
                delta = np.apply_along_axis(lambda x: self.fastsub(x, delta_fl), 0, delta_ta)
            else:
                delta = np.apply_along_axis(lambda x: self.fastadd(x, delta_fl), 0, delta_ta)
            changelist = np.apply_along_axis(lambda x: self.fastmult(x, [dt]*scale), 0, delta)
            evidencelist = list(np.apply_along_axis(lambda x: self.fastadd(x, noise), 0, changelist))
            evidencelist.insert(0, beta*alpha/2 - (1-beta)*alpha/2)
            cumulative_evidence = np.cumsum(evidencelist)
            abovelist = list(np.apply_along_axis(lambda i: i > alpha/2, 0, cumulative_evidence))
            belowlist = list(np.apply_along_axis(lambda i: i < -alpha/2, 0, cumulative_evidence))
            while (abovelist.count(True) < 1 and belowlist.count(True) < 1):
                tind += scale
                oldevidence = cumulative_evidence[-1]
                np.random.shuffle(noise)
                tlist = np.arange(dt*tind, dt*(tind+scale), dt)
                if (sd_0 - (sd_r * (dt*tind))) <= 0.001:
                    s_ta = [norm(0, 0.001).cdf(.5) - norm(0, 0.001).cdf(-.5)]*scale
                else:
                    s_ta = np.apply_along_axis(lambda x: self.sdfunc(x, sd_0, sd_r), 0, tlist)
                s_fl = np.apply_along_axis(lambda x: self.fastsub(x, s_ta), 0, [1]*scale)
                delta_ta = np.apply_along_axis(lambda x: self.fastmult(x, [p]*scale), 0, s_ta)
                delta_fl = np.apply_along_axis(lambda x: self.fastmult(x, [p]*scale), 0, s_fl)
                if congruencylist[n] == 'incongruent':
                    delta = np.apply_along_axis(lambda x: self.fastsub(x, delta_fl), 0, delta_ta)
                else:
                    delta = np.apply_along_axis(lambda x: self.fastadd(x, delta_fl), 0, delta_ta)
                changelist = np.apply_along_axis(lambda x: self.fastmult(x, [dt]*scale), 0, delta)
                evidencelist = list(np.apply_along_axis(lambda x: self.fastadd(x, noise), 0, changelist))
                evidencelist.insert(0, oldevidence)
                cumulative_evidence = np.cumsum(evidencelist)
                abovelist = list(np.apply_along_axis(lambda i: i > alpha/2, 0, cumulative_evidence))
                belowlist = list(np.apply_along_axis(lambda i: i < -alpha/2, 0, cumulative_evidence))
            above = scale; below = scale
            if abovelist.count(True) > 0:
                above = abovelist.index(True)
            if belowlist.count(True) > 0:
                below = belowlist.index(True)
            if min(above, below) == above:
                choicelist.append(1)
            else:
                choicelist.append(0)
            t = (min(above, below) + tind)*dt + tau
            rtlist.append(t)

        return (range(1, nTrials+1), choicelist, rtlist, congruencylist)
    
    def sdfunc(x, sd_0, sd_r):
        sd = sd = sd_0 - (sd_r * (x))
        sd = np.where(sd < 0.001, 0.001, sd)
        s_ta = norm(0, sd).cdf(.5) - norm(0, sd).cdf(-.5)
        return s_ta

    def fastmult(x, list2):
        return x * list2

    def fastsub(x, list2):
        return x - list2

    def fastadd(x, list2):
        return x + list2