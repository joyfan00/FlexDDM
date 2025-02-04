import pandas as pd
from .models.Model import Model
import numpy as np
import sys
from flexddm import modelfit
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from ._utilities import convertToDF
import os

def model_recovery(models, num_simulations=50):
    output_dir = "validation"
    os.makedirs(output_dir, exist_ok=True)
    dfs_list = []
    counter = 0
    for model in models:
        simulation_data = pd.DataFrame()
        for x in range(num_simulations): 
            broken = True
            while broken:
                try: 
                    initial_params = []
                    for lower_bound, upper_bound in model.bounds:
                        initial_params.append(np.random.uniform(lower_bound, upper_bound))
                    # print("init params: ", initial_params)
                    simulation_data = pd.concat([simulation_data, convertToDF(model.modelsimulationfunction(*initial_params, nTrials=100), x)])
                    # print("sim data: \n", simulation_data)
                    broken = False
                except:
                    broken = True

        dfs_list.append(modelfit.fit(models, simulation_data, startingParticipants=0, endingParticipants=num_simulations - 1, posterior_predictive_check=False, return_dataframes = True, output_fileName='output' + str(counter) + '.csv'))
    # Iterate over the list of dataframes
    min_BIC_model_counts = []
    for dfs in dfs_list:
        counter = 0
        BIC_df = pd.DataFrame()
        for df in dfs:
            BIC_df[models[counter].__class__.__name__] = df['bic']
            counter += 1
        print("!!!!!", BIC_df)
        BIC_mins = BIC_df.idxmin(axis=1)  # finds the minimum value in the row, returns column name where min was found
        print("#####", BIC_mins) 
        # print("BIC MIN COLUMN NAME: ", BIC_mins.column)
        min_BIC_model_counts.append(BIC_mins.value_counts().reindex([model.__class__.__name__ for model in models]).reset_index())
    
    # Initialize a list to store the probabilities for each model_BIC_df
    probabilities = []

    # Iterate over each model_BIC_df
    for model_BIC_df in min_BIC_model_counts:
        # Extract the probabilities and append them to the list
        probabilities.append(model_BIC_df['count']/num_simulations)

    probabilities_df = pd.DataFrame(probabilities)

    probabilities_df = probabilities_df.fillna(0)

    sns.set(font_scale=1.2)  
    plt.figure(figsize=(10, 8))  
    heatmap = sns.heatmap(probabilities_df, cmap='crest', annot=True, fmt=".2f", linewidths=.5,
                        xticklabels=[model.__class__.__name__ for model in models],
                        yticklabels=[model.__class__.__name__ for model in models],
                        vmin=0, vmax=1)

    heatmap.xaxis.tick_bottom()

    plt.xlabel('Fit Model')
    plt.ylabel('Synthetic Data')

    plt.show()

    figure = heatmap.get_figure()    
    plot_path = os.path.join(output_dir, "model_validation.png")  
    figure.savefig(plot_path, dpi=400)

# one set of parameters 
# we'd take those parameters, simulate the data according to that set of parameters, 
# then fit the model to the simulated data to see the comparison btw the found params and initial set
# then use heatmap to show the comparisons between the parameter values 
def param_recovery(models, num_simulations=50):
    output_dir = "validation"
    os.makedirs(output_dir, exist_ok=True)
    param_validation_dir = os.path.join(output_dir, "parameter_validation")
    os.makedirs(param_validation_dir, exist_ok=True) 
    counter = 0
    for model in models:
        model_dir = os.path.join(param_validation_dir, model.__class__.__name__)
        os.makedirs(model_dir, exist_ok=True) 
        generated_params = []
        fit_params_list = []
        pbar = tqdm(range(num_simulations))
        pbar.set_description("Simulating Parameter Values")
        for x in pbar: 
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
                    # print("init params: ", initial_params)
                    # creating a giant dataframe with the data from one singular model 
                    simulation_data = convertToDF(model.modelsimulationfunction(*initial_params, nTrials=100), 0)
                    # print("sim data: \n", simulation_data)
                    fit_data = modelfit.fit([model], simulation_data, startingParticipants=0, endingParticipants=0, return_dataframes = True, posterior_predictive_check=False, output_fileName='output' + str(counter) + '.csv')
                    fit_data = fit_data[0]
                    # print(fit_data)
                    # print(type(fit_data))
                    fit_data = fit_data.drop(columns=['id', 'X^2', 'bic'])
                    fit_data = fit_data.reset_index(drop=True)
                    fit_data = fit_data.iloc[0].tolist()
                    fit_params_list.append(fit_data)
                    broken = False
                except Exception as e:
                    print(e)
                    print("FITTING CANNOT OCCUR")
                    broken = True
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
            param_recovery_plot = sns.lmplot(df, x='simulated %s' % param_name, y='fit %s' % param_name)
            plt.show()
            plot_path = os.path.join(model_dir, f"{param_name}.png")  
            param_recovery_plot.savefig(plot_path, dpi=400)
        
        print(correlation_values)
        print(sig_values)

