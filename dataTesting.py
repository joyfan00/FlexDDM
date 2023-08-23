# %%
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



# %%


# %%
data = getRTData()
print("hi")
print(data)
data = pd.DataFrame({'id': data['N_ind'], 'congruency': data['condition'],'rt': [x/1000 for x in data['RT']], 'accuracy': [x-1 for x in data['choice']]})
data['congruency'] = ['congruent' if x == 1 else 'incongruent' for x in data['congruency']]
mydata = data


