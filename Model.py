# packages 
import pandas as pd
import numpy as np
import sys
import math
from multiprocessing.pool import Pool
from variables import Variables
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import math
import multiprocessing.pool as mpp
import sys
import warnings
warnings.filterwarnings("ignore")

"""
This class is a parent class that provides the structure of what functions 
each specific model class should contain.
"""

class Model:

    global param_number
    global bounds
    global cdfs 
    global cafs
    global parameter_names

    def __init__(self, param_number, bounds, parameter_names):
        """
        Initializes a model object. 
        @param_number (int): the number of parameters (also known as variables) necessary for model 
        @bounds (list of tuples): bounds necessary for model fitting
        """
        self.param_number = param_number
        self.bounds = bounds
        self.parameter_names = parameter_names

    def istarmap(self, func, iterable, chunksize=1):
        """
        Runs a specific function using a set of arguments. Uses them across different threads. Is the starmap-version of imap.

        @func: the function being applied to the arguments 
        @iterable: the arguments for the function 
        @chunksize: 
        """

        self._check_running()
        if chunksize < 1:
            raise ValueError(
                "Chunksize must be 1+, not {0:n}".format(
                    chunksize))

        task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
        result = mpp.IMapIterator(self)
        self._taskqueue.put(
            (
                self._guarded_task_generation(result._job,
                                            mpp.starmapstar,
                                            task_batches),
                result._set_length
            ))
        return (item for chunk in result for item in chunk)

    def fit(self, function, data, params, run=1):
        """
        Fits the data according to the model. 

        @data (): 
        @params (dict): contains a dictionary of parameters 
        @nTrials (int): number of trials to try to fit data 
        @cores (int): number of cores 
        @bins (int): number of bins 
        @run (int): counter for what run number 
        """
        props = self.proportions(data, Variables.QUANTILES_CDF, Variables.QUANTILES_CAF)
        bounds_var = self.bounds
        if run > 1:
            fit = minimize(Model.model_function, x0=params, args=(props,self.param_number,self.parameter_names,function, data, bounds_var), options={'maxiter': 100},
                        method='Nelder-Mead')
        else:
            fit = differential_evolution(Model.model_function, bounds=bounds_var, 
                                    args=(props,self.param_number,self.parameter_names, function, data, bounds_var), maxiter=1, seed=10,
                                    disp=True, popsize=100, polish=True)
            
        bestparams = fit.x
        fitstat = fit.fun
        return bestparams, fitstat

    @staticmethod
    def model_function(x, props, param_number, parameter_names, function, data, bounds, final=False):

        """
        Runs the model function. 

        @x (): 
        @prop (): 
        @nTrials (int): number of trials
        @cores (int): number of cores 
        @bins (int): number of bins 
        @final (bool): if this is the final trial or not  
        """
        
        if min(x) < 0:
            return sys.maxsize
        m = Model(bounds=bounds, param_number=param_number, parameter_names=parameter_names)
        predictions = m.model_predict(function, x, props)
        empirical_proportions = [props['cdf_props_congruent'], props['cdf_props_incongruent'],
                                props['caf_props_congruent'], props['caf_props_incongruent']]
        model_proportions = [predictions['cdf_props_congruent'], predictions['cdf_props_incongruent'],
                                predictions['caf_props_congruent'], predictions['caf_props_incongruent']]
        looplist = [empirical_proportions, model_proportions]
        for i, x in enumerate(looplist):
            for j, y in enumerate(x):
                for k, z in enumerate(y):
                    if z == 0:
                        looplist[i][j][k] = 0.0001
        empirical_proportions = [item for sublist in empirical_proportions for item in sublist]
        model_proportions = [item for sublist in model_proportions for item in sublist]
        chisquare = 0; finalsum = 0
        if final == True:
            for i, j in enumerate(empirical_proportions):
                finalsum += 250 * j * np.log(model_proportions[i])
            return -2 * finalsum + param_number * np.log(250)
        for i, j in enumerate(empirical_proportions):
            chisquare += 250 * j * np.log(j / model_proportions[i])
        chisquare = chisquare * 2
        if math.isinf(chisquare) == True:
            return sys.maxsize
        else:
            return chisquare
    
    def parallel_sim(self, function, parameters):
        """
        Runs parallel simulations for a specific model. 

        @function (str): function being run for a specific model 
        @parameters (dict): all of the variables necessary for a particular model in dictionary form (name of variable is key, value of variable is value)
        @nTrials (int): number of trials 
        @cores (int): number of cores 
        @bins (int): number of bins 
        """
        jobs=[]
        results = []
 
        values_list = list(parameters)
        values_tuple = tuple(values_list)
        jobs = [values_tuple]*Variables.BINS
        
        for x in range(len(jobs)):
            jobs[x] = jobs[x] + (0.001, 0.01, int(Variables.NTRIALS/Variables.BINS)) + (x,)
        
        print("JOBS")
        print(jobs)

        with Pool(Variables.CORES) as pool:
            for x in pool.istarmap(function, jobs):
                results.append(x)

        acclist = [results[x][1] for x in range(len(jobs))]
        rtlist = [results[x][2] for x in range(len(jobs))]
        congruencylist = [results[x][3] for x in range(len(jobs))]

        sim_data = pd.DataFrame({'accuracy': [item for sublist in acclist for item in sublist],
                                'rt': [item for sublist in rtlist for item in sublist],
                                'congruency': [item for sublist in congruencylist for item in sublist]})

        return sim_data

    def proportions(self, data, cdfs=[0.1, 0.3, 0.5, 0.7, 0.9], cafs=[0.25, 0.5, 0.75]):
        """
        Calculate proportion of how percentage of RTs in quantiles. 
        @data (): simulated data
        @cdfs (): percentiles used (0.1, 0.3, 0.5, 0.7, 0.9)
        @cafs (): percentiles used 
        """
        props = self.cdf_binsize(cdfs)
        data_congruent = data[data['congruency']=='congruent']
        data_incongruent = data[data['congruency']=='incongruent']

        cdf_props_congruent, caf_props_congruent = self.cdf_caf_proportions(data_congruent, props, cafs)
        cdf_props_incongruent, caf_props_incongruent = self.cdf_caf_proportions(data_incongruent, props, cafs)

        caf_congruent, cdf_congruent, caf_cutoff_congruent = self.caf_cdf(data_congruent)
        caf_incongruent, cdf_incongruent, caf_cutoff_incongruent = self.caf_cdf(data_incongruent)
        # add the word cutoff to cdf_congruent and cdf_incongruent (so cdf_cutoff_congruent and cdf_cutoff_incongruent)
        return {'cdfs': cdfs, 'cafs': cafs, 'cdf_props_congruent': cdf_props_congruent,
                'cdf_props_incongruent': cdf_props_incongruent, 'caf_props_congruent': caf_props_congruent,
                'caf_props_incongruent': caf_props_incongruent, 'cdf_congruent': cdf_congruent,
                'cdf_incongruent': cdf_incongruent, 'caf_cutoff_congruent': caf_cutoff_congruent,
                'caf_cutoff_incongruent': caf_cutoff_incongruent, 'caf_congruent_rt': list(caf_congruent['rt']),
                'caf_congruent_acc': list(caf_congruent['acc']), 'caf_incongruent_rt': list(caf_incongruent['rt']),
                'caf_incongruent_acc': list(caf_incongruent['acc'])}

    def cdf_binsize(self, cdfs=[0.1, 0.3, 0.5, 0.7, 0.9]):
        """
        Calculates the distance between each of the cdfs. 
        ex. [0.1, 0.3, 0.5, 0.7, 0.9] would result in [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
        @cdfs (list of floats): list of percentiles 
        """
        proportionslist = []
        for i, c in enumerate(cdfs):
            if i == 0:
                proportionslist.append(c - 0)
            else:
                proportionslist.append(c - cdfs[i-1])
        proportionslist.append(1 - c)
        return proportionslist

    def cdf_caf_proportions(self, data, cdfs=[0.1, 0.3, 0.5, 0.7, 0.9], cafs=[0.25, 0.50, 0.75]):
        """
        @data (): 
        @cdfs ():
        @cafs ():
        """
        subs = data['id'].unique()
        caf_propslist = []
        cdf_propslist = []
        for j, s in enumerate(subs):
            temp_s = data[data['id']==s]
            temp_s_acc = temp_s[temp_s['accuracy']==1]
            acc = len(temp_s_acc)/len(temp_s)
            cdf_propslist.append([x*acc for x in cdfs])
            temp_quantiles = np.quantile(temp_s['rt'], cafs)
            caf_propslist.append([])
            for i, q in enumerate(temp_quantiles):
                if i == 0:
                    temp = temp_s[temp_s['rt'] <= q]
                    caf_propslist[j].append(len(temp[temp['accuracy']==0])/len(temp_s))
                else:
                    temp = temp_s[temp_s['rt'] <= q].loc[temp_s['rt'] > temp_quantiles[i-1]]
                    caf_propslist[j].append(len(temp[temp['accuracy']==0])/len(temp_s))
            temp = temp_s[temp_s['rt'] > temp_quantiles[-1]]
            caf_propslist[j].append(len(temp[temp['accuracy']==0])/len(temp_s))
        return list(pd.DataFrame(cdf_propslist).mean()), list(pd.DataFrame(caf_propslist).mean())

    def caf_cdf(self, data):
        """
        @data: 
        """
        subs = data['id'].unique()
        meanrtlist = []
        acclist = []
        errorproplist = []
        cdfslist = []
        cafcutofflist = []

        for k, s in enumerate(subs):
            temp = data[data['id']==s]
            cafs = np.quantile(temp['rt'], Variables.QUANTILES_CAF)
            cdfslist.append(np.quantile(temp[temp['accuracy']==1]['rt'], Variables.QUANTILES_CDF))
            cafcutofflist.append(np.quantile(temp['rt'], Variables.QUANTILES_CAF))
            for i, q in enumerate(Variables.QUANTILES_CAF):
                if i == 0:
                    temp_q = temp[temp['rt'] <= cafs[i]]
                    meanrtlist.append([np.mean(temp_q['rt'])])
                    acclist.append([np.mean(temp_q['accuracy'])])
                else:
                    temp_q = temp[temp['rt'] <= cafs[i]].loc[temp['rt'] > cafs[i-1]]
                    meanrtlist[k].append(np.mean(temp_q['rt']))
                    acclist[k].append(np.mean(temp_q['accuracy']))
            temp_q = temp[temp['rt'] > cafs[-1]]
            meanrtlist[k].append(np.mean(temp_q['rt']))
            acclist[k].append(np.mean(temp_q['accuracy']))
        group_rt = list(pd.DataFrame(meanrtlist).mean())
        group_accuracy = list(pd.DataFrame(acclist).mean())
        group_caf_quantiles = pd.DataFrame({'rt': group_rt, 'acc': group_accuracy})
        group_cdf_quantiles = list(pd.DataFrame(cdfslist).mean())
        group_caf_cutoffs = list(pd.DataFrame(cafcutofflist).mean())
        return group_caf_quantiles, group_cdf_quantiles, group_caf_cutoffs
    
  
    def model_predict(self, function, params, props):
        """
        Predicts using the behavioral model. 
        @params (list):
        @nTrials (int):
        @props ():
        @cores (int):
        @bins (int):
        @dt (float): change in time
        @var (float): variance
        """
        np.random.seed(100)
        sim_data = self.parallel_sim(function, params)

        sim_data_congruent = sim_data[sim_data['congruency']=='congruent']
        sim_data_incongruent = sim_data[sim_data['congruency']=='incongruent']
        cdfs_congruent, cafs_congruent = self.model_cdf_caf_proportions(sim_data_congruent, props['cdf_congruent'], props['caf_cutoff_congruent'])
        cdfs_incongruent, cafs_incongruent = self.model_cdf_caf_proportions(sim_data_incongruent, props['cdf_incongruent'], props['caf_cutoff_incongruent'])

        modelprops = {'cdf_props_congruent': cdfs_congruent, 'caf_props_congruent': cafs_congruent,
                    'cdf_props_incongruent': cdfs_incongruent, 'caf_props_incongruent': cafs_incongruent}
        return modelprops

    def model_cdf_caf_proportions(self, data, cdfs, cafcutoffs):
        """

        @data (): simulated data
        @cdfs (list of floats): accurate percentiles
        @cafcutoffs (list of floats): inaccurate percentiles 
        """
        temp_acc = data[data['accuracy']==1]
        props_cdf = []
        props_caf = []
        for i, q in enumerate(cdfs):
            if i == 0:
                temp = temp_acc[temp_acc['rt'] <= q]
            else:
                temp = temp_acc[temp_acc['rt'] <= q].loc[temp_acc['rt'] > cdfs[i-1]]
            props_cdf.append(len(temp)/len(data))
        temp = temp_acc[temp_acc['rt'] > cdfs[-1]]
        props_cdf.append(len(temp)/len(data))

        for i, q in enumerate(cafcutoffs):
            if i == 0:
                temp = data[data['rt'] <= q]
            else:
                temp = data[data['rt'] <= q].loc[data['rt'] > cafcutoffs[i-1]]
            if len(temp) > 0:
                props_caf.append(len(temp[temp['accuracy']==0])/len(data))
            else:
                props_caf.append(0)
        temp = data[data['rt'] > cafcutoffs[-1]]
        if len(temp) > 0:
            props_caf.append(len(temp[temp['accuracy']==0])/len(data))
        else:
            props_caf.append(0)
        return props_cdf, props_caf
    
    def runSimulations(self, pars, startingParticipants, endingParticipants, function, fileName='output.csv'):
        df = pd.DataFrame(columns=self.parameter_names + ['X^2', 'bic'])

        for s in range(startingParticipants, endingParticipants):
            print("PARTICIPANT " + str(s))
            fitstat = sys.maxsize-1; fitstat2 = sys.maxsize
            runint=1
            while fitstat != fitstat2:
                print('run %s' % runint)
                fitstat2 = fitstat
                print(runint)
                pars, fitstat = self.fit(function, self.data[self.data['id']==s], pars, run=runint)
                print(", ".join(str(x) for x in pars))
                print(" X^2 = %s" % fitstat)
                runint += 1
            quantiles_caf = [0.25, 0.5, 0.75]
            quantiles_cdf = [0.1, 0.3, 0.5, 0.7, 0.9]
            myprops = self.proportions(self.data[self.data['id']==s], quantiles_cdf, quantiles_caf)
            predictions = self.model_predict(function, pars, myprops)
            print(myprops)
            bic = Model.model_function(pars, myprops, self.param_number, DMC.model_simulation, self.data[self.data['id']==s], self.bounds, final=True)
            df.loc[len(df)] = pars + [fitstat, bic]
        df.to_csv(fileName, index=False)



        