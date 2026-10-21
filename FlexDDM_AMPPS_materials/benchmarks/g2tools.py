import numpy as np, pandas as pd, pyddm
from scipy.optimize import differential_evolution, minimize
from flexddm.stats import empirical_targets, g2_statistic
from flexddm.fokker_planck import _cumulative_mass, _interpolated_mass_bins
from models import pyddm_dmct, pyddm_set
BUILD = {'dmct': pyddm_dmct}

def pyddm_props(model, values: dict, targets, T):
    """Bin masses from a PyDDM solution, using FlexDDM's G2 bins and binning rule."""
    pyddm_set(model, values); props = {}
    for cond, tg in targets.conditions.items():
        sol = model.solve(conditions={'cong': int(cond)})
        # PyDDM pdfs already include the nondecision time, so T is on the response time scale
        t = T; cu = _cumulative_mass(sol.pdf('correct')); cl = _cumulative_mass(sol.pdf('error')); tot = cu[-1] + cl[-1]
        props[f'cdf_props_{cond}'] = _interpolated_mass_bins(t, cu, tg.cdf_cutoffs, tot)
        props[f'caf_props_{cond}'] = _interpolated_mass_bins(t, cl, tg.caf_cutoffs, tot)
    return props

def pyddm_g2(kind, values: dict, df, dt=0.005, dx=0.005, T_dur=2.5):
    m = BUILD[kind](float(df.rt.min()), dt=dt, dx=dx)
    targets = empirical_targets(df)
    return g2_statistic(targets, pyddm_props(m, values, targets, m.t_domain()))

def fit_with_pyddm_solver(kind, names, bounds, df, maxiter, popsize, seed=10, dt=0.005, dx=0.005):
    """Same objective, optimizer, and settings as FlexDDM's fit_fokker_planck_model; only the solver differs."""
    m = BUILD[kind](float(df.rt.min()), dt=dt, dx=dx)
    T = m.t_domain(); targets = empirical_targets(df)
    def obj(x):
        try:
            return g2_statistic(targets, pyddm_props(m, dict(zip(names, x)), targets, T))
        except Exception:
            return float(np.iinfo(np.int64).max)
    r = differential_evolution(obj, bounds=bounds, maxiter=maxiter, popsize=popsize, seed=seed, polish=False, updating='immediate')
    x, f = r.x, float(r.fun)
    rr = minimize(obj, x0=x, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
    if rr.fun < f: x, f = rr.x, float(rr.fun)
    return dict(zip(names, map(float, x))), f
