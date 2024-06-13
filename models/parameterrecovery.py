import pandas as pd
from .Model import Model
import numpy as np
import sys
from models import runsimulations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from .convertdf import convertToDF

# one set of parameters 
# we'd take those parameters, simulate the data according to that set of parameters, 
# then fit the model to the simulated data to see the comparison btw the found params and initial set
# then use heatmap to show the comparisons between the parameter values 
def param_recovery(models):
    counter = 0
    for model in models:
        generated_params = []
        fit_params_list = []
        for x in range(200): ######50
            np.random.seed(x)
            broken = True
            while broken:
                simulation_data = pd.DataFrame()
                try: 
                    initial_params = []
                    # randomly generating parameters 
                    for lower_bound, upper_bound in model.bounds:
                        initial_params.append(np.random.uniform(lower_bound, upper_bound))
                    generated_params.append(initial_params)
                    print("init params: ", initial_params)
                    # creating a giant dataframe with the data from one singular model 
                    simulation_data = convertToDF(model.modelsimulationfunction(*initial_params, nTrials=300), 0)
                    print("sim data: \n", simulation_data)
                    fit_data = runsimulations.run_simulations(models, 0, 0, simulation_data, return_dataframes = True, fileName='output' + str(counter) + '.csv')
                    fit_data = fit_data[0]
                    print(fit_data)
                    print(type(fit_data))
                    fit_data = fit_data.drop(columns=['id', 'X^2', 'bic'])
                    fit_data = fit_data.reset_index(drop=True)
                    fit_data = fit_data.iloc[0].tolist()
                    fit_params_list.append(fit_data)
                    broken = False
                except Exception as e:
                    print(e)
                    print("FITTING CANNOT OCCUR")
                    broken = True
        
        print(fit_params_list)
        print(generated_params)
        correlation_values = []
        sig_values = []

        # Calculate the correlations for each pair of sublists
        for i, param_name in enumerate(model.parameter_names):
            generated_i = [sublist[i] for sublist in generated_params]
            fit_params_i = [sublist[i] for sublist in fit_params_list]
            df = pd.DataFrame({'simulated %s' % param_name : generated_i, 'fit %s' % param_name : fit_params_i})
            correlation, sig_value = pearsonr(generated_i, fit_params_i)
            correlation_values.append(correlation)
            sig_values.append(sig_value)
            sns.lmplot(df, x='simulated %s' % param_name, y='fit %s' % param_name)
            plt.show()
        
        print(correlation_values)
        print(sig_values)