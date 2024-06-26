from models import validationtools
from .Model import Model
import matplotlib.pyplot as plt
import sys
import pandas as pd
from .convertdf import convertToDF
import seaborn as sns

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
            current_input = input_data[input_data['id'] == id]
            myprops = model.proportions(current_input, quantiles_cdf, quantiles_caf)
            print(myprops)
            bic = Model.model_function(pars, myprops, model.param_number, model.parameter_names, model.modelsimulationfunction, current_input, model.bounds, final=True)
            df.loc[len(df)] = [id] + list(pars) + [fitstat, bic]
                        
            if posterior_predictive_check:
                pars = list(pars)
                res = model.modelsimulationfunction(*pars, nTrials=len(current_input))
                simulated_rts = convertToDF(res, id)

                # experimental congruency, experimental accuracy, simulated congruency, simulated accuracy 
                # loop through 4 times, first pass find only the KDE simulated & experimental RTs correct and congruent, incorrect and congruent, correct and incongruent, incorrect and incongruent 
                rt_data = pd.DataFrame({
                    'experimental_rts': current_input["rt"].tolist(), 
                    'experimental_congruency': current_input['congruency'].tolist(),
                    'experimental_accuracy': current_input['accuracy'].tolist(),
                    'simulated_rts': simulated_rts['rt'].tolist(),
                    'simulated_congruency': simulated_rts['congruency'].tolist(),
                    'simulated_accuracy': simulated_rts['accuracy'].tolist()
                }) # have multiple columns that specify the congruency and accuracy 

                fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
                labels_added = set()  # To keep track of added labels

                for x, i in zip([['congruent', 1], ['congruent', 0], ['incongruent', 1], ['incongruent', 0]], [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]):
                    experimental_rt_data = rt_data[(rt_data['experimental_congruency'] == x[0]) & (rt_data['experimental_accuracy'] == x[1])]
                    simulated_rt_data = rt_data[(rt_data['simulated_congruency'] == x[0]) & (rt_data['simulated_accuracy'] == x[1])]

                    if 'simulated reaction times' not in labels_added:
                        sns.kdeplot(simulated_rt_data, x='simulated_rts', label='simulated reaction times', ax=i)
                        labels_added.add('simulated reaction times')
                    else:
                        sns.kdeplot(simulated_rt_data, x='simulated_rts', label='_nolegend_', ax=i)

                    if 'experimental reaction times' not in labels_added:
                        sns.kdeplot(experimental_rt_data, x='experimental_rts', label='experimental reaction times', ax=i)
                        labels_added.add('experimental reaction times')
                    else:
                        sns.kdeplot(experimental_rt_data, x='experimental_rts', label='_nolegend_', ax=i)

                    i.annotate(f"{x[0].capitalize()}, {'Correct' if x[1] else 'Incorrect'}", xy=(0.96, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                            textcoords='offset fontsize', fontsize='small', horizontalalignment='right', verticalalignment='top', fontfamily='arial', bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
                    i.annotate(f"N = {len(experimental_rt_data)}", xy=(0.96, 0.9), xycoords='axes fraction', xytext=(+0.5, -0.5),
                            textcoords='offset fontsize', fontsize='small', horizontalalignment='right', verticalalignment='top', fontfamily='arial', bbox=dict(facecolor='white', edgecolor='none', pad=3.0))

                fig.suptitle('Posterior Predictive Check Participant ' + str(id))
                fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=1)
                plt.subplots_adjust(bottom=0.3)  # Adjust bottom to make more space for the legend
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust the layout to make space for the title and legend
                plt.show()

        if return_dataframes:
            dflist.append(df)
        else:
            df.to_csv(model.__class__.__name__ + '_' + fileName, index=False)
    if return_dataframes:
        return dflist
