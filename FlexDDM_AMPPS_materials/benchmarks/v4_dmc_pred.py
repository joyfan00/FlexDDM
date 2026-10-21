# FlexDDM DMC predictions (both variants) vs. a fine-grid PyDDM solution (0.5-ms steps).
import sys, warnings; sys.path.insert(0, 'benchmarks'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from models import *
from flexddm.fokker_planck import FokkerPlanckConfig
rng = np.random.default_rng(11); Q = [.1, .3, .5, .7, .9]
B = dict(alpha=(0.08,0.2), beta=(0.3,0.7), mu_c=(0.3,0.7), shape=(2.0,4.0), characteristic_time=(0.03,0.2), peak_amplitude=(0.02,0.1), tau=(0.2,0.3))
sets = [{k: rng.uniform(*b) for k, b in B.items()} for _ in range(40)]
cfgs = {'5 ms': FokkerPlanckConfig(dt=0.005, dx=0.005, t_max=1.5), '2 ms': FokkerPlanckConfig(dt=0.002, dx=0.004, t_max=1.5)}
def stats(t, up, lo):
    pc = up.sum() / (up.sum() + lo.sum()); cdf = np.cumsum(up) / up.sum()
    return pc, np.array([np.interp(q, cdf, t) for q in Q])
rows = []
for kind, spec, build in (('dmc', FLEX_DMC, pyddm_dmc), ('dmct', FLEX_DMCT, pyddm_dmct)):
    ref = build(1.0, dt=0.0005, dx=0.0005)
    for i, v in enumerate(sets):
        pyddm_set(ref, v)
        for cond in '01':
            sol = ref.solve(conditions={'cong': int(cond)}); T = ref.t_domain()
            rp, rq = stats(T, sol.pdf('correct'), sol.pdf('error'))
            for name, cfg in cfgs.items():
                s = spec.solve_condition([v[n] for n in DMC_NAMES], cond, cfg)
                fp, fq = stats(s.times + v['tau'], s.upper_pdf, s.lower_pdf)
                rows.append(dict(kind=kind, set=i, cond=cond, cfg=name, dP=abs(fp - rp)*100, dQ=np.abs(fq - rq).mean()*1000, dQmax=np.abs(fq - rq).max()*1000))
df = pd.DataFrame(rows); df.to_csv('benchmarks/v4_dmc_pred.csv', index=False)
print(df.groupby(['kind','cfg'])[['dP','dQ','dQmax']].agg(['mean','max']).round(2))
