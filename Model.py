# packages 
from abc import ABC

"""
This class is an abstract class that provides the structure of what functions 
each model class should contain.
"""

class Model (ABC):
    def istarmap(self, func, iterable, chunksize=1):
        """starmap-version of imap
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

    @abstractmethod
    def model_simulation():
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
    
    @abstractmethod
    def parallel_simulations():
        pass
    
    def fit(data, params, bounds, nTrials=1000, cores=4, bins=100, run=1):
        quantiles_caf = [0.25, 0.5, 0.75]
        quantiles_cdf = [0.1, 0.3, 0.5, 0.7, 0.9]
        props = proportions(data, quantiles_cdf, quantiles_caf)
        if run != 1:
            fit = minimize(model_function, x0=params, args=(props, nTrials, cores, bins), options={'maxiter': 100},
                        method='Nelder-Mead')
        else:
            fit = differential_evolution(model_function, bounds=bounds, 
                                    args=(props, nTrials, cores, bins), maxiter=1, seed=100,
                                    disp=True, popsize=100, polish=True)
        bestparams = fit.x
        fitstat = fit.fun
        return bestparams, fitstat
    
    def model_function(x, props, nTrials, cores, bins, param_number, final=False):
        # print(x)
        # x = np.divide(x, np.array([1, 1, 10, 100, 1, 10]))
        if min(x) < 0:
            return sys.maxsize
        predictions = model_predict(x, nTrials, props, cores, bins, model)
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
            return -2 * finalsum + param_number * np.log(250)
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
    
    def sdfunc(x, sd_0, sd_r):
        sd = sd = sd_0 - (sd_r * (x))
        sd = np.where(sd < 0.001, 0.001, sd)
        s_ta = norm(0, sd).cdf(.5) - norm(0, sd).cdf(-.5)
        return s_ta

    def fastmult(x, list2):
        return x * list2

    def fastsub(x, list2):
        return x - list2

    def fastadd(x, list2):
        return x + list2

    def proportions(data, cdfs, cafs):
        props = cdf_binsize(cdfs)
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

    def cdf_binsize(cdfs):
        proportionslist = []
        for i, c in enumerate(cdfs):
            if i == 0:
                proportionslist.append(c - 0)
            else:
                proportionslist.append(c - cdfs[i-1])
        proportionslist.append(1 - c)
        return proportionslist

    def cdf_caf_proportions(data, cdfs, cafs):
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

    def caf_cdf(data, quantiles_cdf, quantiles_caf):
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
    
    def model_predict(params, nTrials, props, cores, bins, model='dstp', dt=0.001, var=0.1):
        np.random.seed(100)
        if model == 'dstp':
            sim_data = parallel_sim_dstp(dstp_sim, params, nTrials, cores, bins)
        elif model == 'ssp':
            sim_data = parallel_sim(ssp_sim_new, params, nTrials, cores, bins)
        elif model == 'dmc':
            sim_data = parallel_sim_dmc(dmc_sim, params, nTrials, cores, bins)

        sim_data_congruent = sim_data[sim_data['congruency']=='congruent']
        sim_data_incongruent = sim_data[sim_data['congruency']=='incongruent']
        cdfs_congruent, cafs_congruent = model_cdf_caf_proportions(sim_data_congruent, props['cdf_congruent'], props['caf_cutoff_congruent'])
        cdfs_incongruent, cafs_incongruent = model_cdf_caf_proportions(sim_data_incongruent, props['cdf_incongruent'], props['caf_cutoff_incongruent'])

        modelprops = {'cdf_props_congruent': cdfs_congruent, 'caf_props_congruent': cafs_congruent,
                    'cdf_props_incongruent': cdfs_incongruent, 'caf_props_incongruent': cafs_incongruent}
        return modelprops

    def model_cdf_caf_proportions(data, cdfs, cafcutoffs):
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

mydata = data
# s = 24
mynTrials = 1600
mycores = 16
mybins = 16
# pars = [1, .5, .4, 1.5, .04, .3] #ssp
# pars = [1, .5, .4, 1, .5, .05, .05, 1.5, .3] #dstp
pars = [.5, .5, .5, .5, .5, .5, .5]
for s in range(36, 110):
    with open('output_dmc_%s.txt' % s, 'w') as output:
        print('Model fitting ID %s' % s)
        fitstat = sys.maxsize-1; fitstat2 = sys.maxsize
        runint=1
        while fitstat != fitstat2:
            print('run %s' % runint)
            fitstat2 = fitstat
            pars, fitstat = dmc_fit(mydata[mydata['id']==s], np.array(pars), mynTrials, mycores, mybins, run=runint)
            print(", ".join(str(x) for x in pars))
            print(" X^2 = %s" % fitstat)
            runint += 1
        quantiles_caf = [0.25, 0.5, 0.75]
        quantiles_cdf = [0.1, 0.3, 0.5, 0.7, 0.9]
        myprops = proportions(mydata[mydata['id']==s], quantiles_cdf, quantiles_caf)
        bic = model_function(pars, myprops, mynTrials, mycores, mybins, 'dmc', final=True)
        output.write(", ".join(str(x) for x in pars))
        output.write(" X^2 = %s" % fitstat)
        output.write(" bic = %s" % bic)

        