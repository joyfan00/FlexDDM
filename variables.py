# static variables 
class Variables:

    #Have all of these as arguments to the one function the user runs and provide defaults
    DATA_FILE = 'flanker.json'
    QUANTILES_CDF = [0.1, 0.3, 0.5, 0.7, 0.9]
    QUANTILES_CAF = [0.25, 0.5, 0.75]

    DT = 0.001
    VAR = 0.1
    NTRIALS = 1
    NOISESEED = 50
    CORES = 4
    BINS = 4

    def __init__(self) -> None:
        pass