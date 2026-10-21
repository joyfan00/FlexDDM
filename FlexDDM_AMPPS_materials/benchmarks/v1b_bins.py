import numpy as np, pandas as pd, pyddm
from flexddm.builtin_fp_models import BUILTIN_FP_MODELS
from flexddm.fokker_planck import FokkerPlanckConfig, _cumulative_mass, _interpolated_mass_bins
m = BUILTIN_FP_MODELS['StandardDDM']; S=0.1
rng = np.random.default_rng(2026)
sets=[dict(alpha=rng.uniform(0.08,0.20), beta=rng.uniform(0.35,0.65), v=rng.uniform(0.2,0.8), tau=rng.uniform(0.2,0.35)) for _ in range(50)]
configs = {'default':FokkerPlanckConfig(dt=0.02,dx=0.01),'fine5':FokkerPlanckConfig(dt=0.005,dx=0.005),'fine1':FokkerPlanckConfig(dt=0.001,dx=0.002)}
Q=[.1,.3,.5,.7,.9]
rows=[]
for k,p in enumerate(sets):
    a,b,v,tau=p['alpha'],p['beta'],p['v'],p['tau']
    ref = pyddm.gddm(drift=v, noise=S, bound=a/2, starting_position=2*b-1, mixture_coef=0, T_dur=3.0, dt=0.0002, dx=0.0005)
    sol = ref.solve_analytical(); t=ref.t_domain(); dt=t[1]-t[0]
    up=sol.pdf('correct')*dt; lo=sol.pdf('error')*dt; tot=up.sum()+lo.sum()
    # exact cutoffs: quantiles of correct-RT distribution (continuous interpolation on very fine grid)
    cdf=np.cumsum(up)/up.sum(); cut=[np.interp(q,cdf,t)+tau for q in Q]
    cu=_cumulative_mass(up/dt); ref_bins=np.array(_interpolated_mass_bins(t+tau, cu, cut, cu[-1]+_cumulative_mass(lo/dt)[-1]))
    for name,cfg in configs.items():
        s=m.solve_condition([a,b,v,v,tau],'0',cfg)
        cu=_cumulative_mass(s.upper_pdf); cl=_cumulative_mass(s.lower_pdf); fb=np.array(_interpolated_mass_bins(s.times+tau, cu, cut, cu[-1]+cl[-1]))
        rows.append(dict(set=k,config=name,max_bin_err=np.abs(fb-ref_bins).max(), mean_bin_err=np.abs(fb-ref_bins).mean()))
df=pd.DataFrame(rows); df.to_csv('benchmarks/v1b_bins.csv',index=False)
print(df.groupby('config')[['mean_bin_err','max_bin_err']].agg(['mean','max']).round(4))
