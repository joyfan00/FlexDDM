# V1: accuracy of FlexDDM's Fokker-Planck predictions for the standard DDM against
# (a) closed-form choice probabilities and (b) PyDDM's analytical solver.
import numpy as np, pandas as pd, pyddm, time, json
from flexddm.builtin_fp_models import BUILTIN_FP_MODELS
from flexddm.fokker_planck import FokkerPlanckConfig
m = BUILTIN_FP_MODELS['StandardDDM']
rng = np.random.default_rng(2026)
S = 0.1
sets = []
for i in range(50):
    sets.append(dict(alpha=rng.uniform(0.08,0.20), beta=rng.uniform(0.35,0.65),
                     v=rng.uniform(0.2,0.8), tau=rng.uniform(0.2,0.35)))
def p_upper(v,a,z,s=S):  # boundaries 0,a ; start z ; prob absorb at a
    k=2*v/s**2
    return (1-np.exp(-k*z))/(1-np.exp(-k*a))
configs = {'GUI default (dt=20 ms, dx=.01)':FokkerPlanckConfig(dt=0.02,dx=0.01),
           'dt=5 ms, dx=.005':FokkerPlanckConfig(dt=0.005,dx=0.005),
           'dt=1 ms, dx=.002':FokkerPlanckConfig(dt=0.001,dx=0.002)}
qs=[0.1,0.5,0.9]
def stats_from(times, up, lo):
    dt = times[1]-times[0]
    pu = up.sum()*dt; pl = lo.sum()*dt
    tot=pu+pl
    w = up/up.sum()
    cdf = np.cumsum(w)
    qv = [np.interp(q, cdf, times) for q in qs]
    mean = (w*times).sum()
    return pu/tot, mean, qv
rows=[]
for k,p in enumerate(sets):
    a,b,v,tau=p['alpha'],p['beta'],p['v'],p['tau']
    exact_pu = p_upper(v,a,b*a)
    ref = pyddm.gddm(drift=v, noise=S, bound=a/2, starting_position=2*b-1, nondecision=0, mixture_coef=0, T_dur=3.0, dt=0.0005, dx=0.0005)
    sol = ref.solve(force_analytical=True) if 'force_analytical' in ref.solve.__code__.co_varnames else ref.solve()
    t = ref.t_domain()
    r_pu, r_mean, r_q = stats_from(t, sol.pdf('correct'), sol.pdf('error'))
    for name,cfg in configs.items():
        vals=[a,b,v,v,tau]
        t0=time.time(); s=m.solve_condition(vals,'0',cfg); el=time.time()-t0
        f_pu,f_mean,f_q = stats_from(s.times, s.upper_pdf, s.lower_pdf)
        rows.append(dict(set=k, config=name, secs=el, dP_exact=abs(f_pu-exact_pu), dP_pyddm=abs(f_pu-r_pu),
                         dPy_exact=abs(r_pu-exact_pu),
                         dMean_ms=abs(f_mean-r_mean)*1000, **{f'dQ{int(q*100)}_ms':abs(f_q[i]-r_q[i])*1000 for i,q in enumerate(qs)}))
df=pd.DataFrame(rows); df.to_csv('benchmarks/v1_solver.csv',index=False)
summ = df.groupby('config').agg(['mean','max']).drop(columns=['set'])
pd.set_option('display.width',250); print(summ.round(4))
