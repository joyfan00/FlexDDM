import json;
# from file_input import *
# from file_input import getRTData
import pandas as pd
import numpy as np

# static variables 
class Variables:

    #Have all of these as arguments to the one function the user runs and provide defaults
    DATA_FILE = 'flanker.json'
    QUANTILES_CDF = [0.1, 0.3, 0.5, 0.7, 0.9]
    QUANTILES_CAF = [0.25, 0.5, 0.75]

    DT = 0.001
    VAR = 0.1
    NTRIALS = 100
    NOISESEED = 50
    CORES = 4
    BINS = 4

    def __init__(self):
        pass

    # @staticmethod
    def getRTData(self):
        data = json.load(open(self.DATA_FILE))
        data = pd.DataFrame({'id': data['N_ind'], 'congruency': data['condition'],'rt': [x/1000 for x in data['RT']], 'accuracy': [x-1 for x in data['choice']]})
        data['congruency'] = ['congruent' if x == 1 else 'incongruent' for x in data['congruency']]
        return data

    DATA = self.getRTData()
    BOUNDS = [(0,1),(0,1),(1,20),(0,10),(0,10),(0,1),(0,min(DATA['rt']))]
