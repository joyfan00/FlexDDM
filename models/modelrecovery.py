import pandas as pd
from .Model import Model
import numpy as np
import sys
from models import runsimulations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def model_recovery(models):
    dfs_list = []
    for model in models:
        initial_params = []
        for lower_bound, upper_bound in model.bounds:
            initial_params.append(np.random.uniform(lower_bound, upper_bound))

        simulation_data = model.modelsimulationfunction(*initial_params)
        ## create helper to convert the simulation_data to a dataframe

        print(type(simulation_data))
        print(simulation_data)

        # for model in models:
        #     model.data = simulation_data

        dfs_list.append(runsimulations.run_simulations(models, 1, simulation_data['id'].astype('int').max(), simulation_data, return_dataframes = True))

    average_bics = []
    for df_list in dfs_list:
        bics = [df['bic'].mean() for df in df_list]
        average_bics.append(bics)

    # Create a DataFrame from the average BIC values
    average_bics_df = pd.DataFrame(average_bics)

    # Create heatmap using seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(average_bics_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Average BIC'})
    plt.xlabel('Dataframe Index')
    plt.ylabel('List Index')
    plt.title('Heatmap of Average BIC Values')
    plt.show()


   #runsimulations(comparing_models, data)