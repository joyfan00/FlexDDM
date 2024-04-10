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
        'id': [participant_id] * 300,
        'trial': tuple_data[0],
        'accuracy': tuple_data[1], # previously choice
        'rt': tuple_data[2],
        'congruency': tuple_data[3],
    })

def model_recovery(models):
    dfs_list = []
    counter = 0
    for model in models:
        simulation_data = pd.DataFrame()
        for x in range(2): ######50
            initial_params = []
            for lower_bound, upper_bound in model.bounds:
                initial_params.append(np.random.uniform(lower_bound, upper_bound))
            print(initial_params)
            # simulate for each participant separately in a try catch, resample the params if not good
            # if they are all incorrect, you have a problem --> resample
            simulation_data = pd.concat([simulation_data, convertToDF(model.modelsimulationfunction(*initial_params, nTrials=300), x)])
            
        dfs_list.append(runsimulations.run_simulations(models, 1, simulation_data['id'].astype('int').max(), simulation_data, return_dataframes = True, fileName='output' + str(counter) + '.csv'))
    # Initialize lists to store counts of best fit models for each simulation ---> simulation_data['id'].astype('int').max()
    best_fit_counts = []

    # Iterate over the list of dataframes
    min_BIC_model_counts = []
    for dfs in dfs_list:
        BIC_df = pd.DataFrame()
        for df in dfs:
            counter += 1
            BIC_df[str(counter), ' BIC'] = df['BIC']
        BIC_mins = BIC_df.idxmin(axis=1) 
        min_BIC_model_counts.append(BIC_mins.value_counts())
    # for each list inside the minbicmodellist:
        # find percentage of participants best fit by each model by doing the 
            # number of occurences over total (50)

    

            
                 # compare the BICs in each row to find the minimum BIC for each participant. 
        # keep a tally of which column has the most minimum BICs total
        # 
        
        
        #BIC_list=[0.2,0.1,0.3]
        #depending on what index the lowest one is, increment the counter for that model



            



            print("###########")
            print(type(df))
            print(df)
            # Initialize counters for each model
            model_counts = {i: 0 for i in range(len(models))}
            # Iterate over each row (participant) in the dataframe
            for index, row in df.iterrows():
                # Find the index of the model with the minimum BIC for the current participant
                best_model_index = np.argmin(row.values)
                # Increment the count for the best fit model index
                model_counts[best_model_index] += 1
            # Append the counts for the current simulation to the list
            best_fit_counts.append(model_counts)

    # Calculate the percentage of participants for each model
    total_participants = simulation_data['id'].nunique()
    percentages = [[counts[i] / total_participants * 100 for i in range(len(models))] for counts in best_fit_counts]

    # Create a heatmap to visualize the percentage of participants best fit by each model for each simulation
    plt.figure(figsize=(10, 6))
    sns.heatmap(percentages, annot=True, cmap="YlGnBu", xticklabels=[f'Model {i}' for i in range(len(models))], yticklabels=[f'Simulation {i}' for i in range(len(dfs_list))])
    plt.title("Percentage of Participants Best Fit by Each Model for Each Simulation")
    plt.xlabel("Model")
    plt.ylabel("Simulation")
    plt.show()