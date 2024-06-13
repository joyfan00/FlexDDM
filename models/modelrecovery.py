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


def model_recovery(models):
    dfs_list = []
    counter = 0
    for model in models:
        simulation_data = pd.DataFrame()
        for x in range(5): ######50
            broken = True
            while broken:
                try: 
                    initial_params = []
                    for lower_bound, upper_bound in model.bounds:
                        initial_params.append(np.random.uniform(lower_bound, upper_bound))
                    print("init params: ", initial_params)
                    simulation_data = pd.concat([simulation_data, convertToDF(model.modelsimulationfunction(*initial_params, nTrials=300), x)])
                    print("sim data: \n", simulation_data)
                    broken = False
                except:
                    broken = True

        dfs_list.append(runsimulations.run_simulations(models, 0, 4, simulation_data, return_dataframes = True, fileName='output' + str(counter) + '.csv'))
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
        probabilities.append(model_BIC_df['count']/5)
    
    # Convert probabilities to DataFrame
    probabilities_df = pd.DataFrame(probabilities)

    # Fill NaN values with 0
    probabilities_df = probabilities_df.fillna(0)

    # Create a heatmap using seaborn
    sns.set(font_scale=1.2)  # Adjust font size if needed
    plt.figure(figsize=(10, 8))  # Adjust figure size if needed
    heatmap = sns.heatmap(probabilities_df, cmap='crest', annot=True, fmt=".2f", linewidths=.5,
                        xticklabels=[model.__class__.__name__ for model in models],
                        yticklabels=[model.__class__.__name__ for model in models])

    # Rotate x-labels and set their position to top
    heatmap.xaxis.tick_top()

    # Set labels and title
    plt.xlabel('Fit Model')
    plt.ylabel('Synthetic Data')

    plt.show()

    figure = heatmap.get_figure()    
    figure.savefig('model_validation.png', dpi=400)

