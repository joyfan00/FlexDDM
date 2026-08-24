from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

from .fokker_planck import FokkerPlanckSpec
from .model import Parameter
from .stats import FitTargets


def constant_noise(x, t, condition, *params):
    return 0.1


def standard_drift(x, t, condition, alpha, beta, delta_c, delta_i, tau):
    return delta_c if condition == "0" else delta_i


def dmc_drift(x, t, condition, alpha, beta, mu_c, shape, characteristic_time, peak_amplitude, tau):
    direction = 1 if condition == "0" else -1
    return mu_c + direction * gamma_pulse(max(t + tau, 1e-6), shape, characteristic_time, peak_amplitude)


def mdmc_drift(x, t, condition, alpha, beta, eta, eta_r, shape, characteristic_time, peak_amplitude, tau):
    direction = 1 if condition == "0" else -1
    return 0.5 + direction * gamma_pulse(max(t + tau, 1e-6), shape, characteristic_time, peak_amplitude)


def mdmc_noise(x, t, condition, alpha, beta, eta, eta_r, shape, characteristic_time, peak_amplitude, tau):
    return 0.1 * (10**eta) * math.exp(-0.5 * eta_r * max(t, 0.0))


def dstp_drift(
    x,
    t,
    condition,
    alpha_ss,
    beta_ss,
    delta_ss,
    alpha_rs,
    beta_rs,
    delta_target,
    delta_flanker,
    delta_rs,
    tau,
):
    early = delta_target + delta_flanker if condition == "0" else delta_target - delta_flanker
    ss_start = (beta_ss - 0.5) * alpha_ss
    ss_target = alpha_ss / 2 if delta_ss >= 0 else -alpha_ss / 2
    switch_time = max((ss_target - ss_start) / max(abs(delta_ss), 1e-9), 0.0)
    if t < switch_time:
        return early
    return abs(delta_rs) if delta_ss >= 0 else -abs(delta_rs)


def dstp_bound(t, condition, alpha_ss, beta_ss, delta_ss, alpha_rs, beta_rs, delta_target, delta_flanker, delta_rs, tau):
    return alpha_rs / 2


def dstp_start(t, condition, alpha_ss, beta_ss, delta_ss, alpha_rs, beta_rs, delta_target, delta_flanker, delta_rs, tau):
    return (beta_rs - 0.5) * alpha_rs


def ssp_drift(x, t, condition, alpha, beta, p, sd_0, sd_r, tau):
    spotlight_width = max(sd_0 - sd_r * max(t, 0.0), 1e-3)
    target_weight = normal_pdf(0.0, spotlight_width)
    flanker_weight = 2 * normal_pdf(1.0, spotlight_width) + 2 * normal_pdf(2.0, spotlight_width)
    flank_direction = 1 if condition == "0" else -1
    return p * (target_weight + flank_direction * flanker_weight)


def symmetric_bound(t, condition, *params):
    return params[0] / 2


def tau_nondecision(t, condition, *params):
    return params[-1]


def beta_start(t, condition, *params):
    alpha = params[0]
    beta = params[1]
    return (beta - 0.5) * alpha


def rlddm_update_values(q_left, q_right, choice, reward, learning_rate):
    if choice == 1:
        q_left = q_left + learning_rate * (reward - q_left)
    else:
        q_right = q_right + learning_rate * (reward - q_right)
    return q_left, q_right


def gamma_pulse(time, shape, characteristic_time, peak_amplitude):
    time = max(time, 1e-6)
    shape = max(shape, 1.0001)
    characteristic_time = max(characteristic_time, 1e-6)
    base = (time * math.e) / ((shape - 1) * characteristic_time)
    return (
        peak_amplitude
        * math.exp(-(time / characteristic_time))
        * math.pow(base, shape - 1)
        * (((shape - 1) / time) - (1 / characteristic_time))
    )


def normal_pdf(x, sd):
    return math.exp(-(x * x) / (2 * sd * sd)) / (sd * math.sqrt(2 * math.pi))


class RLDDMFokkerPlanckSpec:
    name = "RLDDM"
    parameters = (
        Parameter("alpha", 0.07, 0.58),
        Parameter("beta", 0.0, 1.0),
        Parameter("learning_rate", 0.01, 0.99),
        Parameter("value_scale", 0.05, 4.0),
        Parameter("drift_bias", -1.0, 1.0),
        Parameter("tau", 0.15, "min_rt"),
    )
    conditions = ("early", "middle", "late")
    default_trials = 120

    @property
    def parameter_names(self) -> list[str]:
        return [parameter.name for parameter in self.parameters]

    def bounds_for_data(self, data: pd.DataFrame | None = None) -> list[tuple[float, float]]:
        return [parameter.bounds_for_data(data) for parameter in self.parameters]

    def neg2_log_likelihood(self, values, data: pd.DataFrame, config) -> float:
        alpha, beta, learning_rate, value_scale, drift_bias, tau = map(float, values)
        total = 0.0
        epsilon = 1e-12
        for _, group in data.groupby("id", dropna=False, sort=False):
            q_left = 0.5
            q_right = 0.5
            for _, row in group.reset_index(drop=True).iterrows():
                drift = drift_bias + value_scale * (q_left - q_right)
                solution = self._solve_trial(alpha, beta, drift, config)
                rt_grid = solution.times + tau
                rt = float(row["rt"])
                choice = int(round(float(row["accuracy"])))
                density = np.interp(rt, rt_grid, solution.upper_pdf if choice == 1 else solution.lower_pdf, left=epsilon, right=epsilon)
                total += math.log(max(float(density), epsilon))
                reward_left = float(row.get("reward_left", 0.5))
                reward_right = float(row.get("reward_right", 0.5))
                reward = reward_left if choice == 1 else reward_right
                q_left, q_right = rlddm_update_values(q_left, q_right, choice, reward, learning_rate)
        if not np.isfinite(total):
            return float(np.iinfo(np.int64).max)
        return -2.0 * total

    def predict_props(self, values, targets: FitTargets, config, data: pd.DataFrame | None = None) -> dict[str, list[float]]:
        if data is None:
            raise ValueError("RLDDM Fokker-Planck predictions require the trial-wise reward schedule.")
        simulated = self.sample_dataframe_like(values, data, "fp_pred", config)
        from .stats import simulated_props

        return simulated_props(simulated, targets)

    def sample_dataframe_like(self, values, observed: pd.DataFrame, participant_id: object = "fp", config=None) -> pd.DataFrame:
        from .fokker_planck import FokkerPlanckConfig

        config = config or FokkerPlanckConfig()
        alpha, beta, learning_rate, value_scale, drift_bias, tau = map(float, values)
        rng = np.random.default_rng(config.seed)
        rows: list[dict[str, object]] = []
        for original_id, group in observed.groupby("id", dropna=False, sort=False):
            q_left = 0.5
            q_right = 0.5
            for _, row in group.reset_index(drop=True).iterrows():
                condition = str(row.get("condition", "all"))
                drift = drift_bias + value_scale * (q_left - q_right)
                solution = self._solve_trial(alpha, beta, drift, config)
                weights = np.concatenate([solution.upper_pdf, solution.lower_pdf])
                weight_sum = float(np.nansum(weights))
                if not np.isfinite(weight_sum) or weight_sum <= 0:
                    raise ValueError("RLDDM produced no finite boundary-crossing density.")
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0) / weight_sum
                draw = int(rng.choice(np.arange(len(weights)), p=weights))
                choice = 1 if draw < len(solution.times) else 0
                time_index = draw if choice == 1 else draw - len(solution.times)
                reward_left = float(row.get("reward_left", 0.5))
                reward_right = float(row.get("reward_right", 0.5))
                reward = reward_left if choice == 1 else reward_right
                q_left, q_right = rlddm_update_values(q_left, q_right, choice, reward, learning_rate)
                rows.append(
                    {
                        "id": original_id if participant_id == "fp_pred" else participant_id,
                        "trial": int(row.get("trial", len(rows) + 1)),
                        "accuracy": choice,
                        "rt": float(solution.times[time_index] + tau),
                        "condition": condition,
                    }
                )
        return pd.DataFrame(rows)

    def _solve_trial(self, alpha: float, beta: float, drift: float, config):
        return _solve_rlddm_trial(
            round(float(alpha), 6),
            round(float(beta), 6),
            round(float(drift), 6),
            round(float(config.dt), 6),
            round(float(config.dx), 6),
            round(float(config.t_max), 6),
        )


@lru_cache(maxsize=8192)
def _solve_rlddm_trial(alpha: float, beta: float, drift: float, dt: float, dx: float, t_max: float):
    drift_function = lambda t, condition: drift
    bound_function = lambda t, condition: alpha / 2
    start_function = lambda t, condition: (beta - 0.5) * alpha
    trial_model = FokkerPlanckSpec(
        name="RLDDM_trial",
        parameters=(),
        conditions=("trial",),
        drift=drift_function,
        bound=bound_function,
        noise=lambda t, condition: 0.1,
        nondecision=0.0,
        starting_position=start_function,
    )
    from .fokker_planck import FokkerPlanckConfig

    return trial_model.solve_condition((), "trial", FokkerPlanckConfig(dt=dt, dx=dx, t_max=t_max))


BUILTIN_FP_MODELS = {
    "StandardDDM": FokkerPlanckSpec(
        name="StandardDDM",
        parameters=(
            Parameter("alpha", 0.07, 0.38),
            Parameter("beta", 0.0, 1.0),
            Parameter("delta_c", 0.2, 0.8),
            Parameter("delta_i", 0.2, 0.8),
            Parameter("tau", 0.15, "min_rt"),
        ),
        conditions=("0", "1"),
        drift=standard_drift,
        bound=symmetric_bound,
        noise=constant_noise,
        nondecision=tau_nondecision,
        starting_position=beta_start,
    ),
    "DMC": FokkerPlanckSpec(
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
        conditions=("0", "1"),
        drift=dmc_drift,
        bound=symmetric_bound,
        noise=constant_noise,
        nondecision=tau_nondecision,
        starting_position=beta_start,
    ),
    "mDMC": FokkerPlanckSpec(
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
        conditions=("0", "1"),
        drift=mdmc_drift,
        bound=symmetric_bound,
        noise=mdmc_noise,
        nondecision=tau_nondecision,
        starting_position=beta_start,
    ),
    "DSTP": FokkerPlanckSpec(
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
        conditions=("0", "1"),
        drift=dstp_drift,
        bound=dstp_bound,
        noise=constant_noise,
        nondecision=tau_nondecision,
        starting_position=dstp_start,
    ),
    "SSP": FokkerPlanckSpec(
        name="SSP",
        parameters=(
            Parameter("alpha", 0.14, 0.38),
            Parameter("beta", 0.0, 1.0),
            Parameter("p", 0.5, 1.5),
            Parameter("sd_0", 0.5, 3.0),
            Parameter("sd_r", 0.1, 2.0),
            Parameter("tau", 0.15, "min_rt"),
        ),
        conditions=("0", "1"),
        drift=ssp_drift,
        bound=symmetric_bound,
        noise=constant_noise,
        nondecision=tau_nondecision,
        starting_position=beta_start,
    ),
    "RLDDM": RLDDMFokkerPlanckSpec(),
}
