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
shrinking_spotlight = SSP()

# updated Pool with istarmap function 
# needs to be run before istarmap function 
mpp.Pool.istarmap = Model.istarmap

# mydata = getRTData()
# print(data)
# data = pd.DataFrame({'id': data['N_ind'], 'congruency': data['condition'],'rt': [x/1000 for x in data['RT']], 'accuracy': [x-1 for x in data['choice']]})
# data['congruency'] = ['congruent' if x == 1 else 'incongruent' for x in data['congruency']]
# mydata = data

# s = 24
mynTrials = 1
mycores = 4
mybins = 4

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
pars = {'alpha':0.5, 'beta':0.5, 'p':0.5, 'sd_0':0.5, 'sd_r':0.5, 'tau':0.5}
for s in range(36, 110):
    with open('output_dmc_%s.txt' % s, 'w') as output:
        print('Model fitting ID %s' % s)
        fitstat = sys.maxsize-1; fitstat2 = sys.maxsize
        runint=1
        while fitstat != fitstat2:
            print('run %s' % runint)
            fitstat2 = fitstat
            pars, fitstat = shrinking_spotlight.fit(shrinking_spotlight.data[shrinking_spotlight.data['id']==s], pars, mynTrials, mycores, mybins, run=runint)
            print(", ".join(str(x) for x in pars))
            print(" X^2 = %s" % fitstat)
            runint += 1
        # make quantiles caf and cdf changeable 
        quantiles_caf = [0.25, 0.5, 0.75]
        quantiles_cdf = [0.1, 0.3, 0.5, 0.7, 0.9]
        myprops = shrinking_spotlight.proportions(shrinking_spotlight.data[shrinking_spotlight.data['id']==s], quantiles_cdf, quantiles_caf)
        bic = shrinking_spotlight.model_function(pars, myprops, mynTrials, mycores, mybins, final=True)
        output.write(", ".join(str(x) for x in pars))
        output.write(" X^2 = %s" % fitstat)
        output.write(" bic = %s" % bic)