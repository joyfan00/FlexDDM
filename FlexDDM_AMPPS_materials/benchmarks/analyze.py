# Summarize the accuracy checks reported in Appendix B and Table B1. Writes benchmarks/summary.json.
import sys, json, warnings; sys.path.insert(0, 'benchmarks'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from models import DMC_NAMES, load_ulrich
from g2tools import pyddm_g2
S = {}

# Standard DDM: choice probabilities (closed form) and predicted bin proportions (PyDDM analytical solution)
v1 = pd.read_csv('benchmarks/v1_solver.csv'); v1b = pd.read_csv('benchmarks/v1b_bins.csv')
S['v1_sets'] = int(v1.set.nunique())
S['v1_dP_max_fine'] = float(v1[v1.config.str.startswith('dt=1 ms')].dP_exact.max())
for k in ('default', 'fine5', 'fine1'):
    S[f'v1_binerr_{k}'] = float(v1b[v1b.config == k].max_bin_err.mean())

# DMC: DMCfun simulations and an independent simulator (decision-onset DMC, impulse shape >= 2)
P = pd.read_csv('benchmarks/r3_params.csv'); C = pd.read_csv('benchmarks/r3_compare.csv').merge(P[['set', 'aaShape']], on='set')
C2 = C[(C.config == 'fine') & (C.aaShape >= 2)]
S['v2_sets'] = int(C2.set.nunique())
S['v2_rt_meanabs'] = float((C2.rtCor_flex - C2.rtCor_dmcfun).abs().mean())
S['v2_err_meanabs'] = float((C2.perErr_flex - C2.perErr_dmcfun).abs().mean())
E = pd.read_csv('benchmarks/r3_euler.csv'); E = E[E['shape'] >= 2]
S['v2_euler_flex_meanabs'] = float((E.flex - E.euler_01ms).abs().mean())
S['v2_euler1_dmcfun_meanabs'] = float((E.dmcfun - E.euler_1ms).abs().mean())

# DMC fits to the flanker data: the FlexDDM fits from the walkthrough vs. PyDDM fits on a 0.5-ms grid
D = load_ulrich()
F = pd.read_csv('flexddm_outputs/06_fit_DMC/fit_result.csv').set_index('participant')
Pf = pd.read_csv('benchmarks/fits_dmc_pyddm.csv').set_index('participant')
ids = sorted(set(F.index) & set(Pf.index)); S['v5_n'] = len(ids)
fine = np.array([pyddm_g2('dmct', {n: F.loc[p, n] for n in DMC_NAMES}, D[D.id == p], dt=0.0005, dx=0.0005) for p in ids])
S['v5_own_vs_fine_meanabs'] = float(np.abs(fine - F.loc[ids, 'g2'].values).mean())
d = fine - Pf.loc[ids, 'g2'].values
S['v5_flex_better'] = int((d < 0).sum()); S['v5_dG2_median'] = float(np.median(d))
for n in DMC_NAMES:
    S[f'v5_r_{n}'] = float(np.corrcoef(F.loc[ids, n], Pf.loc[ids, n])[0, 1])

# DMC predictions vs. a fine-grid PyDDM solution (decision-onset DMC)
v4 = pd.read_csv('benchmarks/v4_dmc_pred.csv')
S['v4_sets'] = int(v4.set.nunique())
for cfg, lab in (('2 ms', '2'), ('5 ms', '5')):
    S[f'v4_dmc_{lab}_dQ_mean'] = float(v4[(v4.kind == 'dmc') & (v4.cfg == cfg)].dQ.mean())

json.dump(S, open('benchmarks/summary.json', 'w'), indent=1)
print(json.dumps(S, indent=1))
