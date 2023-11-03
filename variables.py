import json;
from file_input import *

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

    DATA = getRTData()
    BOUNDS = [(0,1),(0,1),(1,20),(0,10),(0,10),(0,1),(0,min(DATA['rt']))]

    def __init__(self) -> None:
        pass