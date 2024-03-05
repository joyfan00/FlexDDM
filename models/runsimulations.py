import pandas as pd
from .Model import Model
import sys

def run_simulations(models, startingParticipants, endingParticipants, input_data, fileName='output.csv', return_dataframes = False):
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
        if return_dataframes:
            dflist.append(df)
        else:
            df.to_csv(model.__class__.__name__ + '_' + fileName, index=False)
    if return_dataframes:
        return dflist