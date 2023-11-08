from Model import Model
from DMC import DMC
import multiprocessing.pool as mpp
import sys
import json
import pandas as pd
import numpy as np
from variables import Variables
from file_input import *

# convert to jupyter notebook 
dmc = DMC()

# updated Pool with istarmap function 
# needs to be run before istarmap function 
mpp.Pool.istarmap = Model.istarmap

#'sd_r':0.5, 'tau':0.5}

#DSTP PARS
# pars = {'alphaSS': 1, "betaSS": 0.5, "deltaSS": 0.4, "alphaRS": 1, "betaRS1": 0.5, "delta_target": .05, "delta_flanker": .05, "deltaRS2": 1.5, "tau": .3}

#DMC PARS
# pars = {'alpha':0.5, 'beta':0.5, 'tau':0.5, 'shape':5, 'characteristic_time':0.5, 'peak_amplitude':0.5, 'mu_c':0.5}
pars = [0.5, 0.5, 10, 5, 5, 0.5, 0.05]
dmc.runSimulations(pars, 1, 2, fileName='output.csv')

# df = pd.DataFrame(columns=['alpha', 'beta', 'tau', 'shape', 'characteristic_time', 'peak_amplitude', 'mu_c', 'X^2', 'bic'])

# for s in range(37, 38):
#     print(s)
#     # with open('output_dmc_%s.txt' % s, 'w') as output:
#     print('Model fitting ID %s' % s)
#     fitstat = sys.maxsize-1; fitstat2 = sys.maxsize
#     runint=1
#     while fitstat != fitstat2:
#         print('run %s' % runint)
#         fitstat2 = fitstat
#         print(runint)
#         pars, fitstat = shrinking_spotlight.fit(DMC.model_simulation, shrinking_spotlight.data[shrinking_spotlight.data['id']==s], pars, run=runint)
#         print(", ".join(str(x) for x in pars))
#         print(" X^2 = %s" % fitstat)
#         runint += 1
#     # make quantiles caf and cdf changeable 
#     quantiles_caf = [0.25, 0.5, 0.75]
#     quantiles_cdf = [0.1, 0.3, 0.5, 0.7, 0.9]
#     myprops = shrinking_spotlight.proportions(shrinking_spotlight.data[shrinking_spotlight.data['id']==s], quantiles_cdf, quantiles_caf)
#     predictions = shrinking_spotlight.model_predict(DMC.model_simulation, pars, myprops)
#     # x, props, predictions, param_number
#     # print(pars)
#     print(myprops)
#     # print(shrinking_spotlight.param_number)
#     bic = shrinking_spotlight.model_function(pars, myprops, predictions, shrinking_spotlight.param_number, final=True)
#     df = df.append([pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], fitstat, bic])
#     df.to_csv('output.csv')