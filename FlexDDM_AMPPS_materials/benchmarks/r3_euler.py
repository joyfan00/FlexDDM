import sys; sys.path.insert(0, 'benchmarks')
import numpy as np, pandas as pd
from common import gp, euler_sim
P=pd.read_csv('benchmarks/r3_params.csv'); C=pd.read_csv('benchmarks/r3_compare.csv'); C=C[C.config=='fine']
c=0.1/(4*np.sqrt(1000)); rows=[]
for s in [0,2,4,8,10,12,14,17]:
    p=P[P.set==s].iloc[0]
    alpha=2*p.bnds*c; beta=0.5+p.spBias/(2*p.bnds); mu=p.drc*1000*c; A=p.amp*c; ct=p.tau/1000; a=p.aaShape; nd=p.resMean/1000
    for cond,comp,d in [('0','comp',1),('1','incomp',-1)]:
        out={}
        for dt in (1e-3, 1e-4):
            rt,ch=euler_sim(lambda t: mu+d*gp(max(t,dt), a, ct, A), alpha, beta, 20000, dt=dt, seed=s*10+d+5)
            out[dt]=(np.nanmean(rt[ch==1])+nd)*1000
        r=C[(C.set==s)&(C.comp==comp)].iloc[0]
        rows.append(dict(set=s,shape=round(a,2),comp=comp,flex=r.rtCor_flex,dmcfun=r.rtCor_dmcfun,euler_1ms=out[1e-3],euler_01ms=out[1e-4]))
df=pd.DataFrame(rows); df.to_csv('benchmarks/r3_euler.csv',index=False); print(df.round(1).to_string())
