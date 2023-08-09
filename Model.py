# packages 
import pandas as pd
import numpy as np
import random
import sys
import math

"""
This class is a parent class that provides the structure of what functions 
each specific model class should contain.
"""

class Model:

    global param_number
    global bounds

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
        @iterable: 
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

    def model_simulation(self, parameters, dt, var, nTrials):
        """
        Base code for a model simulation. 
        @parameters (dict): contains a dictionary of variables to their associated values  
        @dt (float): time difference
        @var (float): variance 
        @nTrials (float): number of trials running simulations on
        """
        # define variables 
        alpha = parameters['alpha']
        delta = parameters['delta']
        tau = parameters['tau']
        beta = parameters['beta']

        choicelist = []
        rtlist = []
        #sample a bunch of possible updates to evidence with each unit of time dt. Updates are centered
        #around drift rate delta (scaled to the units of time), with variance var (because updates are noisy)
        updates = np.random.normal(loc=delta*dt, scale=var, size=10000) 
        congruencylist = ['congruent']*int(nTrials/2) + ['incongruent']*int(nTrials/2) 
        for n in range(0, nTrials):
            t = tau # start the accumulation process at non-decision time tau
            evidence = beta # start our evidence at initial-bias beta
            while evidence < alpha and evidence > -alpha: # keep accumulating evidence until you reach a threshold
                evidence += random.choice(updates) # add one of the many possible updates to evidence
                t += dt # increment time by the unit dt
            if evidence > alpha:
                choicelist.append(1) # choose the upper threshold action
            else:
                choicelist.append(0) # choose the lower threshold action
            rtlist.append(t)
        return (range(1, nTrials+1), choicelist, rtlist, congruencylist)

    def parallel_sim(self, function, parameters, nTrials=5000, cores=4, bins=4):
        """
        Runs a parallel simulation on a set of parameters given a specific function. 

        @function (str): 
        @parameters (dict): 
        @nTrials (int):
        @cores (int): 
        @bins (int): 
        """
        jobs=[]
        results = []

        values_list = list(parameters.values()) + list(int(nTrials/bins))
        values_tuple = tuple(values_list)
        jobs = [values_tuple]*bins

        for x in range(len(jobs)):
            jobs[x] = jobs[x] + (x,)

        with Pool(cores) as pool:
            for x in pool.istarmap(function, jobs):
                results.append(x)

        acclist = [results[x][1] for x in range(len(jobs))]
        rtlist = [results[x][2] for x in range(len(jobs))]
        congruencylist = [results[x][3] for x in range(len(jobs))]

        sim_data = pd.DataFrame({'accuracy': [item for sublist in acclist for item in sublist],
                                'rt': [item for sublist in rtlist for item in sublist],
                                'congruency': [item for sublist in congruencylist for item in sublist]})

        return sim_data
    

    def fit(self, data, params, nTrials=1000, cores=4, bins=100, run=1):
        """
        Fits the data according to the model. 
        @data (): 
        @params (dict): contains a dictionary of parameters 
        @nTrials (int): number of trials to try to fit data 
        @cores (int): number of cores 
        @bins (int): number of bins 
        @run (int): counter for what run number the 
        """
        quantiles_caf = [0.25, 0.5, 0.75]
        quantiles_cdf = [0.1, 0.3, 0.5, 0.7, 0.9]
        props = self.proportions(data, quantiles_cdf, quantiles_caf)
        if run != 1:
            fit = self.minimize(self.model_function, x0=params, args=(props, nTrials, cores, bins), options={'maxiter': 100},
                        method='Nelder-Mead')
        else:
            fit = self.differential_evolution(self.model_function, bounds=self.bounds, 
                                    args=(props, nTrials, cores, bins), maxiter=1, seed=100,
                                    disp=True, popsize=100, polish=True)
        bestparams = fit.x
        fitstat = fit.fun
        return bestparams, fitstat
    
    def model_function(self, x, props, nTrials, cores, bins, final=False):
        """
        Runs the model function. 

        @x:
        @prop:
        @nTrials: number of trials
        @cores: number of cores 
        @bins: number of bins 
        @final:
        """
        # print(x)
        # x = np.divide(x, np.array([1, 1, 10, 100, 1, 10]))
        if min(x) < 0:
            return sys.maxsize
        predictions = self.model_predict(x, nTrials, props, cores, bins, model)
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
    def parallel_simulations(self, function, parameters, nTrials=5000, cores=4, bins=4):
        """
        Runs parallel simulations for a specific model. 

        @function (str): function being run for a specific model 
        @parameters (dict): all of the variables necessary for a particular model in dictionary form (name of variable is key, value of variable is value)
        @nTrials: number of trials 
        @cores: number of cores 
        @bins: number of bins 
        """
        jobs=[]
        results = []

        param_list = list(parameters.values()) + list(int(nTrials/bins))
        param_tuple = tuple(param_list)
        jobs.append(param_tuple * bins)
        jobs = jobs*bins
        
        for x in range(len(jobs)):
            jobs[x] = jobs[x] + (x,)

        with Pool(cores) as pool:
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

    def proportions(self, data, cdfs, cafs):
        """
        @data: 
        @cdfs: 
        @cafs: 
        """
        props = self.cdf_binsize(cdfs)
        data_congruent = data[data['congruency']=='congruent']
        data_incongruent = data[data['congruency']=='incongruent']

        cdf_props_congruent, caf_props_congruent = cdf_caf_proportions(data_congruent, props, cafs)
        cdf_props_incongruent, caf_props_incongruent = cdf_caf_proportions(data_incongruent, props, cafs)

        caf_congruent, cdf_congruent, caf_cutoff_congruent = caf_cdf(data_congruent, cdfs, cafs)
        caf_incongruent, cdf_incongruent, caf_cutoff_incongruent = caf_cdf(data_incongruent, cdfs, cafs)
        return {'cdfs': cdfs, 'cafs': cafs, 'cdf_props_congruent': cdf_props_congruent,
                'cdf_props_incongruent': cdf_props_incongruent, 'caf_props_congruent': caf_props_congruent,
                'caf_props_incongruent': caf_props_incongruent, 'cdf_congruent': cdf_congruent,
                'cdf_incongruent': cdf_incongruent, 'caf_cutoff_congruent': caf_cutoff_congruent,
                'caf_cutoff_incongruent': caf_cutoff_incongruent, 'caf_congruent_rt': list(caf_congruent['rt']),
                'caf_congruent_acc': list(caf_congruent['acc']), 'caf_incongruent_rt': list(caf_incongruent['rt']),
                'caf_incongruent_acc': list(caf_incongruent['acc'])}

    def cdf_binsize(self, cdfs):
        """

        @cdfs: 
        """
        proportionslist = []
        for i, c in enumerate(cdfs):
            if i == 0:
                proportionslist.append(c - 0)
            else:
                proportionslist.append(c - cdfs[i-1])
        proportionslist.append(1 - c)
        return proportionslist

    def cdf_caf_proportions(self, data, cdfs, cafs):
        """
        @data: 
        @cdfs:
        @cafs:
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

    def caf_cdf(self, data, quantiles_cdf, quantiles_caf):
        """
        @data: 
        @quantiles_cdf: 
        @quantiles_caf: 
        """
        subs = data['id'].unique()
        meanrtlist = []
        acclist = []
        errorproplist = []
        cdfslist = []
        cafcutofflist = []

        for k, s in enumerate(subs):
            temp = data[data['id']==s]
            cafs = np.quantile(temp['rt'], quantiles_caf)
            cdfslist.append(np.quantile(temp[temp['accuracy']==1]['rt'], quantiles_cdf))
            cafcutofflist.append(np.quantile(temp['rt'], quantiles_caf))
            for i, q in enumerate(quantiles_caf):
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

        group_rt = list(pd.DataFrame(meanrtlist).mean())
        group_accuracy = list(pd.DataFrame(acclist).mean())
        # group_errorprop = list(pd.DataFrame(errorproplist).mean())
        group_caf_quantiles = pd.DataFrame({'rt': group_rt, 'acc': group_accuracy})
        group_cdf_quantiles = list(pd.DataFrame(cdfslist).mean())
        group_caf_cutoffs = list(pd.DataFrame(cafcutofflist).mean())
        return group_caf_quantiles, group_cdf_quantiles, group_caf_cutoffs
    
    def model_predict(self, params, nTrials, props, cores, bins, dt=0.001, var=0.1):
        """
        Predicts using the behavioral model. 
        @params: 
        @nTrials:
        @props:
        @cores (int):
        @bins (int):
        @dt (float): change in time
        @var (float): variance
        """
        np.random.seed(100)
        sim_data = self.parallel_sim(self.model_simulation, params, nTrials, cores, bins)

        sim_data_congruent = sim_data[sim_data['congruency']=='congruent']
        sim_data_incongruent = sim_data[sim_data['congruency']=='incongruent']
        cdfs_congruent, cafs_congruent = model_cdf_caf_proportions(sim_data_congruent, props['cdf_congruent'], props['caf_cutoff_congruent'])
        cdfs_incongruent, cafs_incongruent = model_cdf_caf_proportions(sim_data_incongruent, props['cdf_incongruent'], props['caf_cutoff_incongruent'])

        modelprops = {'cdf_props_congruent': cdfs_congruent, 'caf_props_congruent': cafs_congruent,
                    'cdf_props_incongruent': cdfs_incongruent, 'caf_props_incongruent': cafs_incongruent}
        return modelprops

    def model_cdf_caf_proportions(self, data, cdfs, cafcutoffs):
        """

        @data: 
        @cdfs:
        @cafcutoffs: 
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

# mydata = data
# # s = 24
# mynTrials = 1600
# mycores = 16
# mybins = 16
# # pars = [1, .5, .4, 1.5, .04, .3] #ssp
# # pars = [1, .5, .4, 1, .5, .05, .05, 1.5, .3] #dstp
# pars = [.5, .5, .5, .5, .5, .5, .5]
# for s in range(36, 110):
#     with open('output_dmc_%s.txt' % s, 'w') as output:
#         print('Model fitting ID %s' % s)
#         fitstat = sys.maxsize-1; fitstat2 = sys.maxsize
#         runint=1
#         while fitstat != fitstat2:
#             print('run %s' % runint)
#             fitstat2 = fitstat
#             pars, fitstat = dmc_fit(mydata[mydata['id']==s], np.array(pars), mynTrials, mycores, mybins, run=runint)
#             print(", ".join(str(x) for x in pars))
#             print(" X^2 = %s" % fitstat)
#             runint += 1
#         quantiles_caf = [0.25, 0.5, 0.75]
#         quantiles_cdf = [0.1, 0.3, 0.5, 0.7, 0.9]
#         myprops = proportions(mydata[mydata['id']==s], quantiles_cdf, quantiles_caf)
#         bic = model_function(pars, myprops, mynTrials, mycores, mybins, 'dmc', final=True)
#         output.write(", ".join(str(x) for x in pars))
#         output.write(" X^2 = %s" % fitstat)
#         output.write(" bic = %s" % bic)

        