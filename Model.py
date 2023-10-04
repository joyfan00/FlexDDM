# packages 
# import pandas as pd
import numpy as np
import random
import sys
import math

from multiprocessing.pool import Pool
from variables import Variables
# import tqdm
from scipy.optimize import minimize
from scipy.optimize import differential_evolution, shgo
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import json
import multiprocessing.pool as mpp
from scipy.stats import norm
import random
from operator import add, sub, mul
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

    def __init__(self, param_number, bounds):
        """
        Initializes a model object. 
        @param_number (int): the number of parameters (also known as variables) necessary for model 
        @bounds (list of tuples): bounds necessary for model fitting
        """
        self.param_number = param_number
        self.bounds = bounds

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

    @staticmethod
    def model_simulation(alpha, beta, tau, shape, characteristic_time, peak_amplitude, mu_c):
        pass

    # def model_simulation(self, parameters, dt, var, nTrials):
    #     """
    #     Base code for a model simulation. 

    #     @parameters (dict): contains a dictionary of variables to their associated values  
    #     @dt (float): time difference
    #     @var (float): variance 
    #     @nTrials (int): number of trials running simulations on
    #     """
    #     # define variables 
    #     alpha = parameters['alpha']
    #     delta = parameters['delta']
    #     tau = parameters['tau']
    #     beta = parameters['beta']

    #     choicelist = []
    #     rtlist = []
    #     #sample a bunch of possible updates to evidence with each unit of time dt. Updates are centered
    #     #around drift rate delta (scaled to the units of time), with variance var (because updates are noisy)
    #     updates = np.random.normal(loc=delta*dt, scale=var, size=10000) 
    #     congruencylist = ['congruent']*int(nTrials/2) + ['incongruent']*int(nTrials/2) 
    #     for n in range(0, nTrials):
    #         t = tau # start the accumulation process at non-decision time tau
    #         evidence = beta # start our evidence at initial-bias beta
    #         while evidence < alpha and evidence > -alpha: # keep accumulating evidence until you reach a threshold
    #             evidence += random.choice(updates) # add one of the many possible updates to evidence
    #             t += dt # increment time by the unit dt
    #         if evidence > alpha:
    #             choicelist.append(1) # choose the upper threshold action
    #         else:
    #             choicelist.append(0) # choose the lower threshold action
    #         rtlist.append(t)
    #     return (range(1, nTrials+1), choicelist, rtlist, congruencylist)

    # def parallel_sim(self, function, parameters):
    #     """
    #     Runs a parallel simulation on a set of parameters given a specific function. 

    #     @function (str): the model being simulated
    #     @parameters (dict): the parameters that will be input into the function being simulated
    #     @nTrials (int): total number of trials to be distributed across the cores
    #     @cores (int): the number of cores to distribute the simulation trials across
    #     @bins (int): the number of groups that the trials are broken into to be distributed across the cores
    #     """
    #     print("HELLO ONE")
    #     jobs=[]
    #     results = []
    #     print(type(parameters))

    #     # Create a list that contains all of the parameter values (input params + how many trials should go in each bin)
    #     # Each extending model class has default dt, var, nTrial, and noiseSeed values for their model_simulation() 
    #     values_list = [parameters]
        
    #     print(values_list)
        
    #     ## preferably have dt, var, etc be defined by user in the beginning in the one file they run. only have default values for the fn the user is 
    #     # using, not within the methods themselves

    #     #remove defaults from extending model classes for dt var etc, and add them to tuple here

    #     # Turn the params list into a tuple
    #     values_tuple = tuple(values_list)
    #     print(values_tuple)
    #     # Create a list of tuples (one tuple per bin)

    #     ###important:
    #     ### if we have self going into model_simulation -- we need to have it as an argument in the jobs tuple -
    #         #otherwise istarmap wont
    #     jobs = [values_tuple]*Variables.BINS

    #     # Label each tuple with its index #
    #     for x in range(len(jobs)):
    #         jobs[x] = jobs[x] + (x,)

    #     # Pool is the number of threads 
    #     with Pool(Variables.CORES) as pool:

    #         for x in pool.istarmap(function, jobs):
    #             results.append(x)

    #     acclist = [results[x][1] for x in range(len(jobs))]
    #     rtlist = [results[x][2] for x in range(len(jobs))]
    #     congruencylist = [results[x][3] for x in range(len(jobs))]

    #     sim_data = pd.DataFrame({'accuracy': [item for sublist in acclist for item in sublist],
    #                             'rt': [item for sublist in rtlist for item in sublist],
    #                             'congruency': [item for sublist in congruencylist for item in sublist]})

    #     return sim_data
    

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
        predictions = self.model_predict(function, params, props)
        if run != 1:
            fit = minimize(Model.model_function, x0=params, args=(props,predictions), options={'maxiter': 100},
                        method='Nelder-Mead')
        else:
            # print(props)
            fit = differential_evolution(Model.model_function, bounds=bounds_var, 
                                    args=(props,predictions), maxiter=1, seed=100,
                                    disp=True, popsize=100, polish=True)
                
        bestparams = fit.x
        fitstat = fit.fun
        return bestparams, fitstat
    
    def model_function_calculations(self):
        return self.model_fuction()

    @staticmethod
    def model_function(x, props, predictions, final=False):
        ####
        #### important 
        ####
        # cant take self in the start bc diff_evolution and minimize HAVE TO start with x as a parameter. 
        # original versions of the fns were written with the contraints of diff_ev and minimize in mind

        """
        Runs the model function. 

        @x (): 
        @prop (): 
        @nTrials (int): number of trials
        @cores (int): number of cores 
        @bins (int): number of bins 
        @final (bool): if this is the final trial or not  
        """
        # print(x)
        # x = np.divide(x, np.array([1, 1, 10, 100, 1, 10]))
        if min(x) < 0:
            return sys.maxsize
        # cdf_props_congruent:
        # what percent of RTs fall within those buckets
        # cdf_props_congruents: list of percentages that fall within quantiles, percentage of RTs that are congruent
        # compare to the data simulated 
        # keep adjusting until get to those percentages within quantiles with the simulated data 
        empirical_proportions = [props['cdf_props_congruent'], props['cdf_props_incongruent'],
                                props['caf_props_congruent'], props['caf_props_incongruent']]
        model_proportions = [predictions['cdf_props_congruent'], predictions['cdf_props_incongruent'],
                                predictions['caf_props_congruent'], predictions['caf_props_incongruent']]
        # see if value is ever changed in emp_prop or model_prop
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
            #calc approx bayesian info -- want to penalize models that have more parameters
            # refactor so not limited to these 3 models --> have free param for # of model params
            for i, j in enumerate(empirical_proportions):
                finalsum += 250 * j * np.log(model_proportions[i])
            return -2 * finalsum + self.param_number * np.log(250)
        for i, j in enumerate(empirical_proportions):
            chisquare += 250 * j * np.log(j / model_proportions[i])
        chisquare = chisquare * 2
        # empirical_proportions = np.array([np.array(xi) for xi in empirical_proportions])
        # model_proportions = np.array([np.array(xi) for xi in model_proportions])
        # div = np.divide(empirical_proportions, model_proportions)
        # loglist = [np.log(x) for x in div]
        # mult = np.multiply(empirical_proportions, np.array(loglist))
        # chisquare = 2 * np.sum([np.sum(x*250) for x in mult])
        if math.isinf(chisquare) == True:
            return sys.maxsize
        else:
            return chisquare
    
    # make parameters a dictionary and loop over keys 
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

        # Create a list that contains all of the parameter values (input params + how many trials should go in each bin)
        # Each extending model class has default dt, var, nTrial, and noiseSeed values for their model_simulation() 
        values_list = list(parameters.values())
        
        print(values_list)
        
        ## preferably have dt, var, etc be defined by user in the beginning in the one file they run. only have default values for the fn the user is 
        # using, not within the methods themselves

        #remove defaults from extending model classes for dt var etc, and add them to tuple here

        # Turn the params list into a tuple
        values_tuple = tuple(values_list)
        print(values_tuple)
        # Create a list of tuples (one tuple per bin)

        ###important:
        ### if we have self going into model_simulation -- we need to have it as an argument in the jobs tuple -
            #otherwise istarmap wont
        jobs = [values_tuple]*Variables.BINS
        
        for x in range(len(jobs)):
            jobs[x] = jobs[x] + (x,)
            print("1 " + str(jobs[x]))

        with Pool(Variables.CORES) as pool:
            # appends for each list, unpacking results into lists 
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

        # enumerate keeps track of the index and element 
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
                    # errorproplist.append([len(temp_q[temp_q['accuracy']==0]) / len(temp_q)])
                else:
                    temp_q = temp[temp['rt'] <= cafs[i]].loc[temp['rt'] > cafs[i-1]]
                    meanrtlist[k].append(np.mean(temp_q['rt']))
                    acclist[k].append(np.mean(temp_q['accuracy']))
                    # errorproplist[k].append(len(temp_q[temp_q['accuracy']==0]) / len(temp_q))
            temp_q = temp[temp['rt'] > cafs[-1]]
            meanrtlist[k].append(np.mean(temp_q['rt']))
            acclist[k].append(np.mean(temp_q['accuracy']))
            # errorproplist[k].append(len(temp_q[temp_q['accuracy']==0]) / len(temp_q))
            # for i, q in enumerate(quantiles_cdf):
            #     if i == 0:
        
        # takes the mean out of all of those columns 
        group_rt = list(pd.DataFrame(meanrtlist).mean())
        group_accuracy = list(pd.DataFrame(acclist).mean())
        # group_errorprop = list(pd.DataFrame(errorproplist).mean())
        group_caf_quantiles = pd.DataFrame({'rt': group_rt, 'acc': group_accuracy})
        group_cdf_quantiles = list(pd.DataFrame(cdfslist).mean())
        group_caf_cutoffs = list(pd.DataFrame(cafcutofflist).mean())
        return group_caf_quantiles, group_cdf_quantiles, group_caf_cutoffs
    
  
    def model_predict(self, function, params, props):
        """
        Predicts using the behavioral model. 
        @params (dict): 
        @nTrials (int):
        @props ():
        @cores (int):
        @bins (int):
        @dt (float): change in time
        @var (float): variance
        """
        np.random.seed(100)
        print('hi')
        print(params)
        print(type(params))
        # THIS IS WHERE MODEL SIMULATION IS CALLED
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



        