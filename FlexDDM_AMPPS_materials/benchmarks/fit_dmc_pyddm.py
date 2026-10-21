# Reference fits for Table B1 (last row): fit the DMC template to every participant of Ulrich et al. (2015) with PyDDM's
# solver on a fine grid (0.5-ms time steps), using the same G2 objective, bins, parameter bounds, and optimizer settings
# as the FlexDDM fits in flexddm_outputs/06_fit_DMC.
# Usage: python benchmarks/fit_dmc_pyddm.py   (writes benchmarks/fits_dmc_pyddm.csv)
import sys, time, warnings
sys.path.insert(0, 'benchmarks'); warnings.filterwarnings('ignore')
import pandas as pd
from models import FLEX_DMCT, DMC_NAMES, load_ulrich
from g2tools import fit_with_pyddm_solver
from flexddm.stats import bic_from_g2
ids = [int(a) for a in sys.argv[1:]] or None
out = 'benchmarks/fits_dmc_pyddm.csv' if ids is None else f'benchmarks/fits_dmc_pyddm_part_{ids[0]}.csv'
rows = []
for pid, df in load_ulrich().groupby('id'):
    if ids is not None and pid not in ids: continue
    t0 = time.time()
    est, g2 = fit_with_pyddm_solver('dmct', DMC_NAMES, FLEX_DMCT.bounds_for_data(df), df, maxiter=30, popsize=5, seed=10, dt=0.0005, dx=0.0005)
    rows.append({'participant': pid, **{n: est[n] for n in DMC_NAMES}, 'g2': g2, 'bic': bic_from_g2(g2, len(DMC_NAMES), len(df))})
    print(pid, round(time.time() - t0, 1), round(g2, 2), flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
