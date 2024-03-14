import pandas as pd
from .Model import Model
import numpy as np
import sys
from models import runsimulations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def convertToDF(tuple_data):
    return pd.DataFrame({
        'id': [1] * 100,
        'trial': tuple_data[0],
        'accuracy': tuple_data[1], # previously choice
        'rt': tuple_data[2],
        'congruency': tuple_data[3],
    })

def model_recovery(models):
    dfs_list = []
    counter = 0
    for model in models:
        counter+=1
        initial_params = []
        for lower_bound, upper_bound in model.bounds:
            initial_params.append(np.random.uniform(lower_bound, upper_bound))

        simulation_data = convertToDF(model.modelsimulationfunction(*initial_params))
        simulation_data.to_csv("simdata" + str(counter) + '.csv')
        dfs_list.append(runsimulations.run_simulations(models, 1, simulation_data['id'].astype('int').max(), simulation_data, return_dataframes = True, fileName='output' + str(counter) + '.csv'))

    average_bics = []
    for df_list in dfs_list:
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