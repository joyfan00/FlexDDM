from Model import Model
from DMC import DMC
from SSP import SSP
from DSTP import DSTP
import multiprocessing.pool as mpp
from file_input import *

#DMC
#dmc = DMC()
#mpp.Pool.istarmap = Model.istarmap
#pars = [0.5, 0.5, 10, 5, 5, 0.5, 0.05]
#dmc.runSimulations(pars, DMC.model_function, 1, 2, fileName='output.csv')

#SSP
#ssp = SSP()
#mpp.Pool.istarmap = Model.istarmap
#pars = [1, 0.5, 0.4, 1.5, 0.04, 0.3]
#ssp.runSimulations(pars, 1, 2, SSP.model_simulation, fileName='output_ssp.csv')

#DSTP
dstp = DSTP()
mpp.Pool.istarmap = Model.istarmap
pars = [1, 0.5, 0.4, 1, 0.5, 0.05, 0.05, 1.5, 0.3]
dstp.runSimulations(pars, 1, 2, DSTP.model_simulation, fileName='output_dstp.csv')

