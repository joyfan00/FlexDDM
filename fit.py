from Model import Model
from SSP import SSP
from DMC import DMC
from DSTP import DSTP
import multiprocessing.pool as mpp
import sys
import json
import pandas as pd
import numpy as np
from variables import Variables
from file_input import *

# convert to jupyter notebook 
shrinking_spotlight = DMC()

# updated Pool with istarmap function 
# needs to be run before istarmap function 
mpp.Pool.istarmap = Model.istarmap

# mydata = getRTData()
# print(data)
# data = pd.DataFrame({'id': data['N_ind'], 'congruency': data['condition'],'rt': [x/1000 for x in data['RT']], 'accuracy': [x-1 for x in data['choice']]})
# data['congruency'] = ['congruent' if x == 1 else 'incongruent' for x in data['congruency']]
# mydata = data

# s = 24
# mynTrials = 1
# mycores = 1
# mybins = 1

## GOOD OUTPUT 
## excel file with fit function with chi squared, bic (shows fit), parameter values, get it to make a csv

# pars = [1, .5, .4, 1.5, .04, .3] #ssp
# pars = [1, .5, .4, 1, .5, .05, .05, 1.5, .3] #dstp
# pars = [.5, .5, .5, .5, .5, .5, .5]
# alpha = parameters['alpha']
#         beta = parameters['beta']
#         p = parameters['p']
#         sd_0 = parameters['sd_0']
#         sd_r = parameters['sd_r']
#         tau = parameters['tau']

#SSP PARS
# pars = {'alpha':0.5, 'beta':0.5, 'p':0.5, 'sd_0':0.5, 'sd_r':0.5, 'tau':0.5}

#DSTP PARS
# pars = {'alphaSS': 1, "betaSS": 0.5, "deltaSS": 0.4, "alphaRS": 1, "betaRS1": 0.5, "delta_target": .05, "delta_flanker": .05, "deltaRS2": 1.5, "tau": .3}

#DMC PARS
# pars = {'alpha':0.5, 'beta':0.5, 'tau':0.5, 'shape':5, 'characteristic_time':0.5, 'peak_amplitude':0.5, 'mu_c':0.5}
pars = [0.5, 0.5, 0.5, 5, 0.5, 0.5, 0.5]

for s in range(36, 110):
    print(s)
    with open('output_dmc_%s.txt' % s, 'w') as output:
        print('Model fitting ID %s' % s)
        fitstat = sys.maxsize-1; fitstat2 = sys.maxsize
        runint=1
        while fitstat != fitstat2:
            print('run %s' % runint)
            fitstat2 = fitstat
            print(runint)
            pars, fitstat = shrinking_spotlight.fit(DMC.model_simulation, shrinking_spotlight.data[shrinking_spotlight.data['id']==s], pars, run=runint)
            print(", ".join(str(x) for x in pars))
            print(" X^2 = %s" % fitstat)
            runint += 1
        # make quantiles caf and cdf changeable 
        quantiles_caf = [0.25, 0.5, 0.75]
        quantiles_cdf = [0.1, 0.3, 0.5, 0.7, 0.9]
        myprops = shrinking_spotlight.proportions(shrinking_spotlight.data[shrinking_spotlight.data['id']==s], quantiles_cdf, quantiles_caf)
        predictions = shrinking_spotlight.model_predict(DMC.model_simulation, pars, myprops)
        # x, props, predictions, param_number
        # print(pars)
        print(myprops)
        # print(shrinking_spotlight.param_number)
        bic = shrinking_spotlight.model_function(pars, myprops, predictions, shrinking_spotlight.param_number, final=True)
        output.write(", ".join(str(x) for x in pars))
        output.write(" X^2 = %s" % fitstat)
        output.write(" bic = %s" % bic)