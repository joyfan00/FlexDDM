import pandas as pd
from .Model import Model
import numpy as np
import sys
from models import runsimulations


def model_recovery(simulation_model, comparing_models):
    initial_params = []
    for lower_bound, upper_bound in simulation_model.bounds:
        initial_params.append(np.random.uniform(lower_bound, upper_bound))

    data = simulation_model.modelsimulationfunction(*initial_params)
    
    listofmodels = comparing_models.append(simulation_model)
    runsimulations.run_simulations(listofmodels)

   #runsimulations(comparing_models, data)