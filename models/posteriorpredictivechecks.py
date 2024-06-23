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

'''
after you fit a model to data, 
you simulate more data using those 
FOUND parameters and you compare the
SIMULATED DIST OF RTS TO THE OG RAW RTS
'''
def posterior_predictive(models, startingParticipants, endingParticipants, input_data, fileName='output.csv'):
    if isinstance(input_data, str): 
        print("in path instance")
        input_data = Model.getRTData(input_data)
    dflist = []
    for model in models:
        df = pd.DataFrame(columns = ['id'] + model.parameter_names + ['X^2', 'bic'])
        for id in range(startingParticipants, endingParticipants + 1):
            print("PARTICIPANT " + str(id))
            fitstat = sys.maxsize-1; fitstat2 = sys.maxsize
            pars = None
            runint=1
            while fitstat != fitstat2:
                print('run %s' % runint)
                fitstat2 = fitstat
                print(runint)
                pars, fitstat = model.fit(model.modelsimulationfunction, input_data[input_data['id']==id], pars, run=runint)
                print(", ".join(str(x) for x in pars))
                print(" X^2 = %s" % fitstat)
                runint += 1
            quantiles_caf = [0.25, 0.5, 0.75]
            quantiles_cdf = [0.1, 0.3, 0.5, 0.7, 0.9]
            myprops = model.proportions(input_data[input_data['id']==id], quantiles_cdf, quantiles_caf)
            print(myprops)
            bic = Model.model_function(pars, myprops, model.param_number, model.parameter_names, model.modelsimulationfunction, input_data[input_data['id']==id], model.bounds, final=True)
            df.loc[len(df)] = [id] + list(pars) + [fitstat, bic]
            pars = list(pars)
            print("PARS TYPE: ", type(pars))
            print(pars)
        
            res = model.modelsimulationfunction(*pars, nTrials = 1000)
            print(res)
            simulated_rts = convertToDF(res, id)['rt'].tolist()
            print("sim: ",simulated_rts)

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
