import numpy as np, pandas as pd, pyddm
from flexddm.fokker_planck import FokkerPlanckSpec
from flexddm.model import Parameter
from flexddm.builtin_fp_models import BUILTIN_FP_MODELS, gamma_pulse, symmetric_bound, constant_noise, tau_nondecision, beta_start
from common import gp

def load_ulrich():
    d = pd.read_csv('data/ulrich2015.csv')
    return pd.DataFrame({'id': d.PPT, 'condition': np.where(d.Condition == 1, '0', '1'), 'rt': d.RT, 'accuracy': d.Correct.astype(int)})

# ---- FlexDDM specs ----

def dmc_onset_drift(x, t, condition, alpha, beta, mu_c, shape, characteristic_time, peak_amplitude, tau):
    # DMC as in Ulrich et al. (2015): automatic pulse aligned to decision onset (t = 0)
    direction = 1 if condition == "0" else -1
    return mu_c + direction * gamma_pulse(max(t, 1e-6), shape, characteristic_time, peak_amplitude)

_t = BUILTIN_FP_MODELS['DMC']
DMC_PARAMS = tuple(Parameter('shape', 2.0, 4.5) if p.name == 'shape' else p for p in _t.parameters)
FLEX_DMC = FokkerPlanckSpec(name='DMC', parameters=DMC_PARAMS, conditions=('0','1'), drift=dmc_onset_drift,
                            bound=symmetric_bound, noise=constant_noise, nondecision=tau_nondecision, starting_position=beta_start)
DMC_NAMES = ['alpha','beta','mu_c','shape','characteristic_time','peak_amplitude','tau']

# ---- PyDDM equivalents (same parameterization, noise = 0.1) ----
def pyddm_dmc(min_rt, dt=0.005, dx=0.005):
    return pyddm.gddm(drift=lambda t, cong, mu_c, shape, characteristic_time, peak_amplitude:
                          mu_c + (1 if cong == 0 else -1) * gp(t, shape, characteristic_time, peak_amplitude),
        noise=0.1, bound=lambda alpha: alpha/2, starting_position=lambda beta: 2*beta-1, nondecision='tau',
        parameters={'alpha':(0.07,0.38),'beta':(0.,1.),'mu_c':(0.2,0.8),'shape':(2.0,4.5),'characteristic_time':(0.01,1.0),
                    'peak_amplitude':(0.015,0.4),'tau':(0.15,min_rt)},
        conditions=['cong'], mixture_coef=0, T_dur=3.0, dt=dt, dx=dx)

def pyddm_set(model, values: dict):
    names = model.get_model_parameter_names()
    model.set_model_parameters([values['tau' if (n.startswith('tau') or n=='nondectime') else n] for n in names])

# ---- DMC template exactly as shipped in FlexDDM (automatic pulse evaluated at t + tau) ----
FLEX_DMCT = BUILTIN_FP_MODELS['DMC']
def pyddm_dmct(min_rt, dt=0.005, dx=0.005):
    return pyddm.gddm(drift=lambda t, cong, mu_c, shape, characteristic_time, peak_amplitude, tau:
                          mu_c + (1 if cong == 0 else -1) * gp(t + tau, shape, characteristic_time, peak_amplitude),
        noise=0.1, bound=lambda alpha: alpha/2, starting_position=lambda beta: 2*beta-1, nondecision='tau',
        parameters={'alpha':(0.07,0.38),'beta':(0.,1.),'mu_c':(0.2,0.8),'shape':(1.5,4.5),'characteristic_time':(0.01,1.0),
                    'peak_amplitude':(0.015,0.4),'tau':(0.15,min_rt)},
        conditions=['cong'], mixture_coef=0, T_dur=3.0, dt=dt, dx=dx)
