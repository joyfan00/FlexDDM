from .models.Model import Model
import matplotlib.pyplot as plt
import sys
import pandas as pd
from tqdm.notebook import tqdm
from ._utilities import convertToDF, getRTData
import seaborn as sns
import os

def fit(models, input_data, startingParticipants=None, endingParticipants=None, input_data_id="PPT", input_data_congruency="Condition", input_data_rt="RT", input_data_accuracy="Correct", output_fileName='output.csv', return_dataframes=False, posterior_predictive_check=True): 
    output_dir = "fit"
    os.makedirs(output_dir, exist_ok=True) 
    fit_parameters_dir = os.path.join(output_dir, "fitted_parameters")
    os.makedirs(fit_parameters_dir, exist_ok=True)
    posterior_predictive_check_dir = ""
    if posterior_predictive_check:
        posterior_predictive_check_dir = os.path.join(output_dir, "posterior_predictive_check")
        os.makedirs(posterior_predictive_check_dir, exist_ok=True)   
    if isinstance(input_data, str): 
        # print("in path instance")
        input_data = getRTData(path=input_data, id=input_data_id, congruency=input_data_congruency, rt=input_data_rt, accuracy=input_data_accuracy)
   
    if startingParticipants==None and endingParticipants==None:
        startingParticipants = input_data['id'].min()
        endingParticipants = input_data['id'].max()

    dflist = []
    for model in models:
        df = pd.DataFrame(columns=['id'] + model.parameter_names + ['X^2', 'bic'])
        if endingParticipants - startingParticipants > 1:
            pbar = tqdm(range(startingParticipants, endingParticipants + 1))
            pbar.set_description("Fitting Model to Data")
        else:
            pbar = range(startingParticipants, endingParticipants + 1)
        for id in pbar:
            if input_data[input_data['id'] == id].empty:
                continue
            # print("PARTICIPANT " + str(id))
            fitstat = sys.maxsize - 1
            fitstat2 = sys.maxsize
            pars = None
            runint = 1
            while fitstat != fitstat2:
                # print('run %s' % runint)
                fitstat2 = fitstat
                pars, fitstat = model.fit(model.modelsimulationfunction, input_data[input_data['id'] == id], pars, run=runint)
                # print(", ".join(str(x) for x in pars))
                # print(" X^2 = %s" % fitstat)
                runint += 1
            quantiles_caf = [0.25, 0.5, 0.75]
            quantiles_cdf = [0.1, 0.3, 0.5, 0.7, 0.9]
            current_input = input_data[input_data['id'] == id]
            myprops = model.proportions(current_input, quantiles_cdf, quantiles_caf)
            # print(myprops)
            bic = Model.model_function(pars, myprops, model.param_number, model.parameter_names, model.modelsimulationfunction, current_input, model.bounds, final=True)
            print('BIC = ',bic)
            df.loc[len(df)] = [id] + list(pars) + [fitstat, bic]

            if posterior_predictive_check:
                posterior_predictive_check_model_dir = os.path.join(posterior_predictive_check_dir, model.__class__.__name__)
                os.makedirs(posterior_predictive_check_model_dir, exist_ok=True)   
                pars = list(pars)
                res = model.modelsimulationfunction(*pars, nTrials=len(current_input))
                simulated_rts = convertToDF(res, id)

                # Prepare the combined DataFrame
                rt_data = pd.DataFrame({
                    'experimental_rts': current_input["rt"].tolist(),
                    'experimental_congruency': current_input['congruency'].tolist(),
                    'experimental_accuracy': current_input['accuracy'].tolist(),
                    'simulated_rts': simulated_rts['rt'].tolist(),
                    'simulated_congruency': simulated_rts['congruency'].tolist(),
                    'simulated_accuracy': simulated_rts['accuracy'].tolist()
                })

                fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
                labels_added = set()

                conditions = [
                    (['congruent', 1], axes[0, 0]),
                    (['congruent', 0], axes[0, 1]),
                    (['incongruent', 1], axes[1, 0]),
                    (['incongruent', 0], axes[1, 1])
                ]

                for (condition, ax) in conditions:
                    congruency, accuracy = condition
                    experimental_rt_data = rt_data[(rt_data['experimental_congruency'] == congruency) & (rt_data['experimental_accuracy'] == accuracy)]
                    simulated_rt_data = rt_data[(rt_data['simulated_congruency'] == congruency) & (rt_data['simulated_accuracy'] == accuracy)]

                    sns.kdeplot(simulated_rt_data['simulated_rts'], label='simulated reaction times' if 'simulated reaction times' not in labels_added else '_nolegend_', ax=ax, color='#CC79A7')
                    sns.kdeplot(experimental_rt_data['experimental_rts'], label='experimental reaction times' if 'experimental reaction times' not in labels_added else '_nolegend_', ax=ax, color='#0072B2')
                    labels_added.update(['simulated reaction times', 'experimental reaction times'])

                    ax.annotate(f"{congruency.capitalize()}, {'Correct' if accuracy else 'Incorrect'}", xy=(0.96, 1), xycoords='axes fraction', xytext=(0, -5),
                                textcoords='offset points', fontsize='small', ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
                    ax.annotate(f"Experimental N = {len(experimental_rt_data)}", xy=(0.96, 0.9), xycoords='axes fraction', xytext=(0, -5),
                                textcoords='offset points', fontsize='small', ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
                    ax.annotate(f"Simulated N = {len(simulated_rt_data)}", xy=(0.96, 0.8), xycoords='axes fraction', xytext=(0, -5),
                                textcoords='offset points', fontsize='small', ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
                    ax.set_xlabel("Response Time (s)")

                fig.suptitle('Posterior Predictive Check Participant ' + str(id))
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make space for the title
                fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=1, bbox_transform=fig.transFigure, fontsize='x-small', frameon=False)
                plt.show()
                plot_path = os.path.join(posterior_predictive_check_model_dir, f"participant_{id}.png")
                # Save the figure
                fig.savefig(plot_path, dpi=400)
            if not return_dataframes:
                df.to_csv(fit_parameters_dir + "/" + model.__class__.__name__ + '_' + output_fileName, index=False)

        if return_dataframes:
            dflist.append(df)
        df.to_csv(fit_parameters_dir + "/" + model.__class__.__name__ + '_' + output_fileName, index=False)
    if return_dataframes:
        return dflist
