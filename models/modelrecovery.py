import pandas as pd
from .Model import Model
import numpy as np
import sys
from models import runsimulations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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

# one set of parameters 
# we'd take those parameters, simulate the data according to that set of parameters, 
# then fit the model to the simulated data to see the comparison btw the found params and initial set
# then use heatmap to show the comparisons between the parameter values 
def param_recovery(models):
    counter = 0
    for model in models:
        generated_params = []
        fit_params_list = []
        # for x in range(1): ######50
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
                fit_data = fit_data.drop(columns=['id', 'X^2', 'bic'])
                fit_data = fit_data.reset_index(drop=True)
                fit_data = fit_data.iloc[0].tolist()
                fit_params_list.append(fit_data)
                broken = False
            except:
                print("FITTING CANNOT OCCUR")
                broken = True
                
        # check the correlation 
        # Convert the nested lists into numpy arrays for easier manipulation
        array1 = np.array(generated_params)
        array2 = np.array(fit_params_list)

        # Initialize a matrix to store the average correlation coefficients
        num_sublists = array1.shape[0]
        average_correlation_matrix = np.zeros((num_sublists, num_sublists))

        # Calculate the correlations for each pair of sublists
        for i in range(num_sublists):
            for j in range(num_sublists):
                correlation_values = []
                for k in range(array1.shape[1]):
                    correlation, _ = pearsonr(array1[i], array2[j])
                    correlation_values.append(correlation)
                average_correlation_matrix[i, j] = np.mean(correlation_values)

        # Create a DataFrame for the averaged correlation matrix
        sublists_labels = [f'Sublist{i+1}' for i in range(num_sublists)]
        average_correlation_df = pd.DataFrame(average_correlation_matrix, index=sublists_labels, columns=sublists_labels)

        # Create a heatmap using seaborn
        sns.heatmap(average_correlation_df, annot=True, cmap='crest', vmin=-1, vmax=1)

        # Display the heatmap
        plt.title('Averaged Correlation Matrix Heatmap')
        plt.show()