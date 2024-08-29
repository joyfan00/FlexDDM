from flexddm import validationtools
from .models.Model import Model
from flexddm import modelfit
import matplotlib.pyplot as plt
import sys
import pandas as pd
from ._utilities import convertToDF
import seaborn as sns

# function that performs model recovery and parameter recovery 
def validation(models, model_recovery=True, model_recovery_simulations=100, parameter_recovery=True, param_recovery_simulations=100):
    if model_recovery:
        validationtools.model_recovery(models, model_recovery_simulations)
    if parameter_recovery:
        validationtools.param_recovery(models, param_recovery_simulations)

# function that performs fitting and posterior predictive checks 
def fit(models,  input_data, startingParticipants=None, endingParticipants=None, input_data_id="PPT", input_data_congruency="Condition", input_data_rt="RT", input_data_accuracy="Correct", output_fileName='output.csv', return_dataframes=False, posterior_predictive_check=True):
    modelfit.fit(models, input_data, startingParticipants, endingParticipants, input_data_id, input_data_congruency, input_data_rt, input_data_accuracy, output_fileName, return_dataframes, posterior_predictive_check)