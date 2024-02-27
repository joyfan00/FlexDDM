import pandas as pd
from .Model import Model
import sys

def runSimulations(models, startingParticipants, endingParticipants, fileName='output.csv'):
    for model in models:
        df = pd.DataFrame(columns=model.parameter_names + ['X^2', 'bic'])
        for s in range(startingParticipants, endingParticipants):
            print("PARTICIPANT " + str(s))
            fitstat = sys.maxsize-1; fitstat2 = sys.maxsize
            pars = None
            runint=1
            while fitstat != fitstat2:
                print('run %s' % runint)
                fitstat2 = fitstat
                print(runint)
                pars, fitstat = model.fit(model.modelsimulationfunction, model.data[model.data['id']==s], pars, run=runint)
                print(", ".join(str(x) for x in pars))
                print(" X^2 = %s" % fitstat)
                runint += 1
            quantiles_caf = [0.25, 0.5, 0.75]
            quantiles_cdf = [0.1, 0.3, 0.5, 0.7, 0.9]
            myprops = model.proportions(model.data[model.data['id']==s], quantiles_cdf, quantiles_caf)
            print(myprops)
            bic = Model.model_function(pars, myprops, model.param_number, model.parameter_names, model.modelsimulationfunction, model.data[model.data['id']==s], model.bounds, final=True)
            df.loc[len(df)] = list(pars) + [fitstat, bic]
        df.to_csv(model.__class__.__name__ + '_' + fileName, index=False)