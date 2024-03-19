import pandas as pd
from .Model import Model
import numpy as np
import sys
from models import runsimulations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def convertToDF(tuple_data, participant_id):
    return pd.DataFrame({
        'id': [participant_id] * 100,
        'trial': tuple_data[0],
        'accuracy': tuple_data[1], # previously choice
        'rt': tuple_data[2],
        'congruency': tuple_data[3],
    })

def model_recovery(models):
    dfs_list = []
    counter = 0
    simulation_data = pd.DataFrame()
    for model in models:
        initial_params = []
        for x in range(50):
            for lower_bound, upper_bound in model.bounds:
                initial_params.append(np.random.uniform(lower_bound, upper_bound))

                simulation_data = pd.concat([simulation_data, convertToDF(model.modelsimulationfunction(*initial_params, nTrials=300), x)])
            
        dfs_list.append(runsimulations.run_simulations(models, 1, simulation_data['id'].astype('int').max(), simulation_data, return_dataframes = True, fileName='output' + str(counter) + '.csv'))

    # the length of models list determines how many dataframes are in df_list. if there are 3 models
    # there will be 9 dataframes, as all 3 models will be fit to data simulated by all 3 models

    # we want to find the best fit model for the simulation data from each model and record the fraction of 
    # participants who are best fit by each model. this is the probability of the model being the best fit for a 
    # participant given the simulation data

    # to determine the best fit model we need to find the minimum BIC value from each of the dataframes that contian 
    # the data fit to a specific model (this is where you need to only compare the groupings of fitting stats corresponding to a specific set of simulated data)

     # create a heatmap that has the probability values


        bics = [df['bic'].mean() for df in df_list]
        print(bics)
        average_bics.append(bics)

    # Create a DataFrame from the average BIC values
    average_bics_df = pd.DataFrame(average_bics)
    print(average_bics_df)

    # Create heatmap using seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(average_bics_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Average BIC'})
    plt.xlabel('Dataframe Index')
    plt.ylabel('List Index')
    plt.title('Heatmap of Average BIC Values')
    plt.show()


   #runsimulations(comparing_models, data)