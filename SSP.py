class SSP (Model):
    def parallel_sim(function, parameters, nTrials=5000, cores=4, bins=4):
        jobs=[]
        results = []
        alpha = parameters[0]
        beta = parameters[1]
        p = parameters[2]
        sd_0 = parameters[3]
        sd_r = parameters[4]
        tau = parameters[5]
    
        jobs = [(alpha, beta, p, sd_0, sd_r, tau, .001, .01, int(nTrials/bins))]*bins
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
    
    def model_function(x, props, nTrials, cores, bins, final=False):
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
            return -2 * finalsum + 6 * np.log(250)
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