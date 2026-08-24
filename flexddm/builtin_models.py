from __future__ import annotations

import math

import numpy as np

from .model import ModelSpec, Parameter, SimulationResult


def standard_ddm(alpha, beta, delta_c, delta_i, tau, nTrials=800, dt=0.01, var=0.01, noiseseed=50):
    rng = np.random.default_rng(noiseseed)
    trial = np.arange(1, nTrials + 1)
    accuracy = np.full(nTrials, np.nan)
    rt = np.full(nTrials, np.nan)
    condition = np.zeros((nTrials, 1), dtype=int)
    condition[nTrials // 2 :] = 1

    for index in range(nTrials):
        drift = delta_c if condition[index, 0] == 0 else delta_i
        evidence = (beta - 0.5) * alpha
        time = tau
        while -alpha / 2 < evidence < alpha / 2 and time < 10:
            evidence += drift * dt + rng.normal(0, var)
            time += dt
        accuracy[index] = 1 if evidence > alpha / 2 else 0
        rt[index] = time

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


def dmc(alpha, beta, mu_c, shape, characteristic_time, peak_amplitude, tau, nTrials=800, dt=0.01, var=0.01, noiseseed=50):
    rng = np.random.default_rng(noiseseed)
    trial = np.arange(1, nTrials + 1)
    accuracy = np.full(nTrials, np.nan)
    rt = np.full(nTrials, np.nan)
    condition = np.zeros((nTrials, 1), dtype=int)
    condition[nTrials // 2 :] = 1

    for index in range(nTrials):
        evidence = (beta - 0.5) * alpha
        time = tau
        while -alpha / 2 < evidence < alpha / 2 and time < 10:
            direction = 1 if condition[index, 0] == 0 else -1
            automatic = direction * _gamma_pulse(time, shape, characteristic_time, peak_amplitude)
            evidence += (automatic + mu_c) * dt + rng.normal(0, var)
            time += dt
        accuracy[index] = 1 if evidence > alpha / 2 else 0
        rt[index] = time

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


def mdmc(alpha, beta, eta, eta_r, shape, characteristic_time, peak_amplitude, tau, nTrials=800, dt=0.01, var=0.01, noiseseed=50):
    rng = np.random.default_rng(noiseseed)
    trial = np.arange(1, nTrials + 1)
    accuracy = np.full(nTrials, np.nan)
    rt = np.full(nTrials, np.nan)
    condition = np.zeros((nTrials, 1), dtype=int)
    condition[nTrials // 2 :] = 1

    for index in range(nTrials):
        evidence = (beta - 0.5) * alpha
        time = tau
        while -alpha / 2 < evidence < alpha / 2 and time < 10:
            direction = 1 if condition[index, 0] == 0 else -1
            automatic = direction * _gamma_pulse(time, shape, characteristic_time, peak_amplitude)
            noise = var * (10**eta) * math.exp(-0.5 * eta_r * max(time - tau, 0.0))
            evidence += (automatic + 0.5) * dt + rng.normal(0, noise)
            time += dt
        accuracy[index] = 1 if evidence > alpha / 2 else 0
        rt[index] = time

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


def dstp(alpha_ss, beta_ss, delta_ss, alpha_rs, beta_rs, delta_target, delta_flanker, delta_rs, tau, nTrials=800, dt=0.01, var=0.01, noiseseed=50):
    rng = np.random.default_rng(noiseseed)
    trial = np.arange(1, nTrials + 1)
    accuracy = np.full(nTrials, np.nan)
    rt = np.full(nTrials, np.nan)
    condition = np.zeros((nTrials, 1), dtype=int)
    condition[nTrials // 2 :] = 1

    for index in range(nTrials):
        evidence_ss = (beta_ss - 0.5) * alpha_ss
        evidence_rs = (beta_rs - 0.5) * alpha_rs
        early_rs = delta_target + delta_flanker if condition[index, 0] == 0 else delta_target - delta_flanker
        time = tau
        while (-alpha_ss / 2 < evidence_ss < alpha_ss / 2 or -alpha_rs / 2 < evidence_rs < alpha_rs / 2) and time < 10:
            evidence_ss += delta_ss * dt + rng.normal(0, var)
            evidence_rs += early_rs * dt + rng.normal(0, var)
            time += dt
        if evidence_rs > alpha_rs / 2 or evidence_rs < -alpha_rs / 2:
            accuracy[index] = 1 if evidence_rs > alpha_rs / 2 else 0
            rt[index] = time
            continue
        late_rs = abs(delta_rs) if evidence_ss > alpha_ss / 2 else -abs(delta_rs)
        while -alpha_rs / 2 < evidence_rs < alpha_rs / 2 and time < 10:
            evidence_rs += late_rs * dt + rng.normal(0, var)
            time += dt
        accuracy[index] = 1 if evidence_rs > alpha_rs / 2 else 0
        rt[index] = time

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


def ssp(alpha, beta, p, sd_0, sd_r, tau, nTrials=800, dt=0.01, var=0.01, noiseseed=50):
    rng = np.random.default_rng(noiseseed)
    trial = np.arange(1, nTrials + 1)
    accuracy = np.full(nTrials, np.nan)
    rt = np.full(nTrials, np.nan)
    condition = np.zeros((nTrials, 1), dtype=int)
    condition[nTrials // 2 :] = 1

    for index in range(nTrials):
        evidence = (beta - 0.5) * alpha
        time = tau
        while -alpha / 2 < evidence < alpha / 2 and time < 10:
            spotlight_width = max(sd_0 - sd_r * max(time - tau, 0), 1e-3)
            target_weight = _normal_pdf(0.0, spotlight_width)
            flanker_weight = 2 * _normal_pdf(1.0, spotlight_width) + 2 * _normal_pdf(2.0, spotlight_width)
            flank_direction = 1 if condition[index, 0] == 0 else -1
            drift = p * (target_weight + flank_direction * flanker_weight)
            evidence += drift * dt + rng.normal(0, var)
            time += dt
        accuracy[index] = 1 if evidence > alpha / 2 else 0
        rt[index] = time

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


def _gamma_pulse(time: float, shape: float, characteristic_time: float, peak_amplitude: float) -> float:
    safe_time = max(time, 1e-6)
    safe_shape = max(shape, 1.0001)
    safe_characteristic_time = max(characteristic_time, 1e-6)
    base = (safe_time * math.e) / ((safe_shape - 1) * safe_characteristic_time)
    return (
        peak_amplitude
        * math.exp(-(safe_time / safe_characteristic_time))
        * math.pow(base, safe_shape - 1)
        * (((safe_shape - 1) / safe_time) - (1 / safe_characteristic_time))
    )


def _normal_pdf(x: float, sd: float) -> float:
    return math.exp(-(x * x) / (2 * sd * sd)) / (sd * math.sqrt(2 * math.pi))


BUILTIN_MODELS: dict[str, ModelSpec] = {
    "StandardDDM": ModelSpec(
        name="StandardDDM",
        parameters=(
            Parameter("alpha", 0.07, 0.38),
            Parameter("beta", 0.0, 1.0),
            Parameter("delta_c", 0.2, 0.8),
            Parameter("delta_i", 0.2, 0.8),
            Parameter("tau", 0.15, "min_rt"),
        ),
        simulate=standard_ddm,
        default_trials=800,
    ),
    "DMC": ModelSpec(
        name="DMC",
        parameters=(
            Parameter("alpha", 0.07, 0.38),
            Parameter("beta", 0.0, 1.0),
            Parameter("mu_c", 0.2, 0.8),
            Parameter("shape", 1.5, 4.5),
            Parameter("characteristic_time", 0.01, 1.0),
            Parameter("peak_amplitude", 0.015, 0.4),
            Parameter("tau", 0.15, "min_rt"),
        ),
        simulate=dmc,
        default_trials=800,
    ),
    "mDMC": ModelSpec(
        name="mDMC",
        parameters=(
            Parameter("alpha", 0.07, 0.38),
            Parameter("beta", 0.0, 1.0),
            Parameter("eta", -1.0, 1.0),
            Parameter("eta_r", -5.0, 5.0),
            Parameter("shape", 1.5, 4.5),
            Parameter("characteristic_time", 0.01, 1.0),
            Parameter("peak_amplitude", 0.015, 0.4),
            Parameter("tau", 0.15, "min_rt"),
        ),
        simulate=mdmc,
        default_trials=800,
    ),
    "DSTP": ModelSpec(
        name="DSTP",
        parameters=(
            Parameter("alpha_ss", 0.14, 0.38),
            Parameter("beta_ss", 0.0, 1.0),
            Parameter("delta_ss", 0.25, 0.55),
            Parameter("alpha_rs", 0.14, 0.38),
            Parameter("beta_rs", 0.0, 1.0),
            Parameter("delta_target", 0.05, 0.15),
            Parameter("delta_flanker", 0.05, 0.25),
            Parameter("delta_rs", 0.4, 1.2),
            Parameter("tau", 0.15, "min_rt"),
        ),
        simulate=dstp,
        default_trials=800,
    ),
    "SSP": ModelSpec(
        name="SSP",
        parameters=(
            Parameter("alpha", 0.14, 0.38),
            Parameter("beta", 0.0, 1.0),
            Parameter("p", 0.5, 1.5),
            Parameter("sd_0", 0.5, 3.0),
            Parameter("sd_r", 0.1, 2.0),
            Parameter("tau", 0.15, "min_rt"),
        ),
        simulate=ssp,
        default_trials=800,
    ),
}


def describe_models() -> list[dict[str, object]]:
    return [
        {
            "name": model.name,
            "parameters": [
                {"name": parameter.name, "lower": parameter.lower, "upper": parameter.upper}
                for parameter in model.parameters
            ],
            "defaultTrials": model.default_trials,
        }
        for model in BUILTIN_MODELS.values()
    ]
