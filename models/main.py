from models import validationtools
from .Model import Model
import matplotlib.pyplot as plt
import sys
import pandas as pd
from .convertdf import convertToDF

# function that performs model recovery and parameter recovery 
def validation(models, model_recovery=True, model_recovery_simulations=50, parameter_recovery=True, param_recovery_simulations=50):
    if model_recovery:
        validationtools.model_recovery(models, model_recovery_simulations)
    if parameter_recovery:
        validationtools.param_recovery(models, param_recovery_simulations)

# function that performs fitting and posterior predictive checks 
def fit(models, startingParticipants, endingParticipants, input_data, fileName='output.csv', return_dataframes=False, posterior_predictive_check=True):
    if isinstance(input_data, str): 
        print("in path instance")
        input_data = Model.getRTData(input_data)
    dflist = []
    for model in models:
        df = pd.DataFrame(columns=['id'] + model.parameter_names + ['X^2', 'bic'])
        for id in range(startingParticipants, endingParticipants + 1):
            print("PARTICIPANT " + str(id))
            fitstat = sys.maxsize - 1
            fitstat2 = sys.maxsize
            pars = None
            runint = 1
            while fitstat != fitstat2:
                print('run %s' % runint)
                fitstat2 = fitstat
                pars, fitstat = model.fit(model.modelsimulationfunction, input_data[input_data['id'] == id], pars, run=runint)
                print(", ".join(str(x) for x in pars))
                print(" X^2 = %s" % fitstat)
                runint += 1
            quantiles_caf = [0.25, 0.5, 0.75]
            quantiles_cdf = [0.1, 0.3, 0.5, 0.7, 0.9]
            myprops = model.proportions(input_data[input_data['id'] == id], quantiles_cdf, quantiles_caf)
            print(myprops)
            bic = Model.model_function(pars, myprops, model.param_number, model.parameter_names, model.modelsimulationfunction, input_data[input_data['id'] == id], model.bounds, final=True)
            df.loc[len(df)] = [id] + list(pars) + [fitstat, bic]
            
            if posterior_predictive_check:
                pars = list(pars)
                print("PARS TYPE: ", type(pars))
                print(pars)
                res = model.modelsimulationfunction(*pars, nTrials=300)
                print(res)
                simulated_rts = convertToDF(res, id)['rt'].tolist()
                print("sim: ", simulated_rts)

                # Plot the raw values
                plt.plot(simulated_rts, label='Simulated', marker='o')
                plt.plot(input_data["rt"].to_list(), label='Experimental', marker='x')

                # Add labels and title
                plt.xlabel('Index')
                plt.ylabel('Response Time')
                plt.title('Response Times')
                plt.legend()

                # Show the plot
                plt.show()

        if return_dataframes:
            dflist.append(df)
        else:
            df.to_csv(model.__class__.__name__ + '_' + fileName, index=False)
    if return_dataframes:
        return dflist
