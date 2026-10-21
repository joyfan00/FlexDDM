import sys; sys.path.insert(0, 'benchmarks')
import numpy as np, pandas as pd
from models import FLEX_DMC
from flexddm.fokker_planck import FokkerPlanckConfig
P=pd.read_csv('benchmarks/r3_params.csv'); D=pd.read_csv('benchmarks/r3_dmcfun_out.csv')
c=0.1/(4*np.sqrt(1000))
Q=[.1,.3,.5,.7,.9]
rows=[]
for cfgname,cfg in [('fine',FokkerPlanckConfig(dt=0.001,dx=0.002)),('fit',FokkerPlanckConfig(dt=0.005,dx=0.005))]:
  for _,p in P.iterrows():
    alpha=2*p.bnds*c; beta=0.5+p.spBias/(2*p.bnds)
    vals=[alpha,beta,p.drc*1000*c,p.aaShape,p.tau/1000,p.amp*c,p.resMean/1000]
    for cond,comp in [('0','comp'),('1','incomp')]:
        try: s=FLEX_DMC.solve_condition(vals,cond,cfg)
        except Exception as e:
            print('FAIL', cfgname, p.set, cond, round(p.aaShape,2)); continue
        t=s.times+vals[-1]
        up,lo=s.upper_pdf,s.lower_pdf; tot=up.sum()+lo.sum()
        cdf=np.cumsum(up)/up.sum()
        # quantiles by linear interpolation of the CDF (mass at t_i spread over (t_{i-1}, t_i])
        qv=[np.interp(q, cdf, t)*1000 for q in Q]
        f=dict(rtCor=(t*up).sum()/up.sum()*1000, perErr=lo.sum()/tot*100, rtErr=(t*lo).sum()/max(lo.sum(),1e-300)*1000, **{f'q{int(q*100)}':v for q,v in zip(Q,qv)})
        r=D[(D.set==p.set)&(D.comp==comp)].iloc[0]
        rows.append(dict(config=cfgname,set=p.set,comp=comp, **{k+'_flex':f[k] for k in f}, **{k+'_dmcfun':r[k] for k in f}))
df=pd.DataFrame(rows); df.to_csv('benchmarks/r3_compare.csv',index=False)
for cfgname,g in df.groupby('config'):
    print(cfgname)
    for k in ['rtCor','perErr','q10','q30','q50','q70','q90']:
        d=g[k+'_flex']-g[k+'_dmcfun']; print(f'  {k}: mean diff {d.mean():.2f}, mean |diff| {d.abs().mean():.2f}, max |diff| {d.abs().max():.2f}, r={np.corrcoef(g[k+"_flex"],g[k+"_dmcfun"])[0,1]:.4f}')
    # CAF-style: congruency effects
    e=g.pivot(index='set',columns='comp',values='rtCor_flex'); e2=g.pivot(index='set',columns='comp',values='rtCor_dmcfun')
    d=(e.incomp-e.comp)-(e2.incomp-e2.comp); print('  congruency effect diff mean|d| %.2f max %.2f'%(d.abs().mean(), d.abs().max()))
