from __future__ import annotations


BLANK_BOUNDS = {
    "alpha": (0.07, 0.38),
    "beta": (0.0, 1.0),
    "drift_rate": (-1.0, 1.0),
    "tau": (0.15, "min_rt"),
}

STANDARD_DDM_BOUNDS = {
    "alpha": (0.07, 0.38),
    "beta": (0.0, 1.0),
    "delta_c": (0.2, 0.8),
    "delta_i": (0.2, 0.8),
    "tau": (0.15, "min_rt"),
}

DMC_BOUNDS = {
    "alpha": (0.07, 0.38),
    "beta": (0.0, 1.0),
    "mu_c": (0.2, 0.8),
    "shape": (1.5, 4.5),
    "characteristic_time": (0.01, 1.0),
    "peak_amplitude": (0.015, 0.4),
    "tau": (0.15, "min_rt"),
}

MDMC_BOUNDS = {
    "alpha": (0.07, 0.38),
    "beta": (0.0, 1.0),
    "eta": (-1.0, 1.0),
    "eta_r": (-5.0, 5.0),
    "shape": (1.5, 4.5),
    "characteristic_time": (0.01, 1.0),
    "peak_amplitude": (0.015, 0.4),
    "tau": (0.15, "min_rt"),
}

SSP_BOUNDS = {
    "alpha": (0.14, 0.38),
    "beta": (0.0, 1.0),
    "p": (0.5, 1.5),
    "sd_0": (0.5, 3.0),
    "sd_r": (0.1, 2.0),
    "tau": (0.15, "min_rt"),
}

WEIBULL_BOUND_BOUNDS = {
    "alpha": (0.07, 0.50),
    "beta": (0.0, 1.0),
    "delta_c": (0.2, 0.8),
    "delta_i": (0.2, 0.8),
    "tau": (0.15, "min_rt"),
    "collapse_rate": (0.05, 2.0),
    "collapse_shape": (0.5, 5.0),
    "collapse_amount": (0.0, 0.95),
}

DSTP_BOUNDS = {
    "alpha_ss": (0.14, 0.38),
    "beta_ss": (0.0, 1.0),
    "delta_ss": (0.25, 0.55),
    "alpha_rs": (0.14, 0.38),
    "beta_rs": (0.0, 1.0),
    "delta_target": (0.05, 0.15),
    "delta_flanker": (0.05, 0.25),
    "delta_rs": (0.4, 1.2),
    "tau": (0.15, "min_rt"),
}

CUSTOM_SIMULATOR_BOUNDS = {
    "alpha": (0.07, 0.58),
    "beta": (0.0, 1.0),
    "drift_rate": (-1.0, 1.0),
    "tau": (0.15, "min_rt"),
}

CAST_M_BOUNDS = {
    "alpha": (0.07, 0.58),
    "mu_c_start": (-0.8, 0.8),
    "mu_c_end": (-0.8, 0.8),
    "mu_v": (-2.0, 2.0),
    "math_characteristic_time_start": (0.01, 10.0),
    "word_characteristic_time_start": (0.01, 10.0),
    "math_characteristic_time_end": (0.01, 10.0),
    "word_characteristic_time_end": (0.01, 10.0),
    "math_bias_start": (-0.38, 0.38),
    "word_bias_start": (-0.38, 0.38),
    "math_bias_end": (-0.38, 0.38),
    "word_bias_end": (-0.38, 0.38),
    "tau": (0.15, "min_rt"),
    "math_hard_prob_start": (0.0, 1.0),
    "word_hard_prob_start": (0.0, 1.0),
    "math_hard_prob_end": (0.0, 1.0),
    "word_hard_prob_end": (0.0, 1.0),
}

RACING_DIFFUSION_BOUNDS = {
    "alpha": (0.07, 0.58),
    "beta_a": (0.0, 1.0),
    "beta_b": (0.0, 1.0),
    "drift_a_0": (-1.0, 1.0),
    "drift_b_0": (-1.0, 1.0),
    "drift_a_1": (-1.0, 1.0),
    "drift_b_1": (-1.0, 1.0),
    "tau": (0.15, "min_rt"),
}

RLDDM_BOUNDS = {
    "alpha": (0.07, 0.58),
    "beta": (0.0, 1.0),
    "learning_rate": (0.01, 0.99),
    "value_scale": (0.05, 4.0),
    "drift_bias": (-1.0, 1.0),
    "tau": (0.15, "min_rt"),
}


BLANK_TEMPLATE = """from __future__ import annotations

from flexddm import define_model


def drift(t, condition, alpha, beta, drift_rate, tau):
    return drift_rate


process_model = define_model(
    name="CustomModel",
    conditions=("0", "1"),
    drift=drift,
    nondecision="tau",
)

model = process_model.to_simulation_model()
"""


STANDARD_DDM_TEMPLATE = """from __future__ import annotations

from flexddm import define_model


def drift(t, condition, alpha, beta, delta_c, delta_i, tau):
    return delta_c if condition == "0" else delta_i


process_model = define_model(
    name="StandardDDM_Custom",
    conditions=("0", "1"),
    drift=drift,
    nondecision="tau",
)

model = process_model.to_simulation_model()
"""


DMC_TEMPLATE = """from __future__ import annotations

import math

from flexddm import define_model


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


def drift(t, condition, alpha, beta, mu_c, shape, characteristic_time, peak_amplitude, tau):
    direction = 1 if condition == "0" else -1
    return mu_c + direction * gamma_pulse(t + tau, shape, characteristic_time, peak_amplitude)


process_model = define_model(
    name="DMC_Custom",
    conditions=("0", "1"),
    drift=drift,
    nondecision="tau",
)

model = process_model.to_simulation_model()
"""


MDMC_TEMPLATE = """from __future__ import annotations

import math

from flexddm import define_model


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


def drift(t, condition, alpha, beta, eta, eta_r, shape, characteristic_time, peak_amplitude, tau):
    direction = 1 if condition == "0" else -1
    return 0.5 + direction * gamma_pulse(t + tau, shape, characteristic_time, peak_amplitude)


def noise(t, condition, alpha, beta, eta, eta_r, shape, characteristic_time, peak_amplitude, tau):
    base_noise = 0.1 * (10 ** eta)
    return base_noise * math.exp(-0.5 * eta_r * max(t, 0.0))


process_model = define_model(
    name="mDMC_Custom",
    conditions=("0", "1"),
    drift=drift,
    noise=noise,
    nondecision="tau",
)

model = process_model.to_simulation_model()
"""


SSP_TEMPLATE = """from __future__ import annotations

import math

from flexddm import define_model


def normal_pdf(x, sd):
    return math.exp(-(x * x) / (2 * sd * sd)) / (sd * math.sqrt(2 * math.pi))


def drift(t, condition, alpha, beta, p, sd_0, sd_r, tau):
    spotlight_width = max(sd_0 - sd_r * max(t, 0.0), 1e-3)
    target_weight = normal_pdf(0.0, spotlight_width)
    flanker_weight = 2 * normal_pdf(1.0, spotlight_width) + 2 * normal_pdf(2.0, spotlight_width)
    flank_direction = 1 if condition == "0" else -1
    return p * (target_weight + flank_direction * flanker_weight)


process_model = define_model(
    name="SSP_Custom",
    conditions=("0", "1"),
    drift=drift,
    nondecision="tau",
)

model = process_model.to_simulation_model()
"""


WEIBULL_BOUND_TEMPLATE = """from __future__ import annotations

import math

from flexddm import define_model


def drift(t, condition, alpha, beta, delta_c, delta_i, tau, collapse_rate, collapse_shape, collapse_amount):
    return delta_c if condition == "0" else delta_i


def bound(t, condition, alpha, beta, delta_c, delta_i, tau, collapse_rate, collapse_shape, collapse_amount):
    scaled_time = max(t, 0.0) / max(collapse_rate, 1e-6)
    collapse = collapse_amount * (1.0 - math.exp(-(scaled_time ** collapse_shape)))
    return max((alpha / 2) * (1.0 - collapse), 0.02)


process_model = define_model(
    name="WeibullBoundaryDDM_Custom",
    conditions=("0", "1"),
    drift=drift,
    bound=bound,
    nondecision="tau",
)

model = process_model.to_simulation_model()
"""


DSTP_TEMPLATE = """from __future__ import annotations

import numpy as np

from flexddm import ModelSpec, SimulationResult, define_model, parameters_from_bounds


def simulate(alpha_ss, beta_ss, delta_ss, alpha_rs, beta_rs, delta_target, delta_flanker, delta_rs, tau, runtime):
    rng = np.random.default_rng(runtime.seed)
    trial = np.arange(1, runtime.trials + 1)
    accuracy = np.full(runtime.trials, np.nan)
    rt = np.full(runtime.trials, np.nan)
    condition = np.zeros(runtime.trials, dtype=object)

    for index in range(runtime.trials):
        condition_value = "0" if index < runtime.trials // 2 else "1"
        condition[index] = condition_value
        evidence_ss = (beta_ss - 0.5) * alpha_ss
        evidence_rs = (beta_rs - 0.5) * alpha_rs
        early_rs = delta_target + delta_flanker if condition_value == "0" else delta_target - delta_flanker
        time = tau

        while (-alpha_ss / 2 < evidence_ss < alpha_ss / 2 or -alpha_rs / 2 < evidence_rs < alpha_rs / 2) and time < runtime.max_time:
            evidence_ss += delta_ss * runtime.dt + rng.normal(0.0, runtime.noise_sd)
            evidence_rs += early_rs * runtime.dt + rng.normal(0.0, runtime.noise_sd)
            time += runtime.dt

        if -alpha_rs / 2 < evidence_rs < alpha_rs / 2:
            late_rs = abs(delta_rs) if evidence_ss > alpha_ss / 2 else -abs(delta_rs)
            while -alpha_rs / 2 < evidence_rs < alpha_rs / 2 and time < runtime.max_time:
                evidence_rs += late_rs * runtime.dt + rng.normal(0.0, runtime.noise_sd)
                time += runtime.dt

        accuracy[index] = 1 if evidence_rs > alpha_rs / 2 else 0
        rt[index] = time

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


model = ModelSpec(
    name="DSTP_Custom",
    parameters=parameters_from_bounds(),
    simulate=simulate,
    conditions=("0", "1"),
)


def drift(t, condition, alpha_ss, beta_ss, delta_ss, alpha_rs, beta_rs, delta_target, delta_flanker, delta_rs, tau):
    early = delta_target + delta_flanker if condition == "0" else delta_target - delta_flanker
    ss_start = (beta_ss - 0.5) * alpha_ss
    ss_target = alpha_ss / 2 if delta_ss >= 0 else -alpha_ss / 2
    switch_time = max((ss_target - ss_start) / max(abs(delta_ss), 1e-9), 0.0)
    return early if t < switch_time else (abs(delta_rs) if delta_ss >= 0 else -abs(delta_rs))


def bound(t, condition, alpha_ss, beta_ss, delta_ss, alpha_rs, beta_rs, delta_target, delta_flanker, delta_rs, tau):
    return alpha_rs / 2


def starting_position(t, condition, alpha_ss, beta_ss, delta_ss, alpha_rs, beta_rs, delta_target, delta_flanker, delta_rs, tau):
    return (beta_rs - 0.5) * alpha_rs


process_model = define_model(
    name="DSTP_Custom",
    conditions=("0", "1"),
    drift=drift,
    bound=bound,
    starting_position=starting_position,
    nondecision="tau",
)
"""


CUSTOM_SIMULATOR_TEMPLATE = """from __future__ import annotations

import numpy as np

from flexddm import ModelSpec, SimulationResult, define_model, parameters_from_bounds


def simulate(alpha, beta, drift_rate, tau, runtime):
    rng = np.random.default_rng(runtime.seed)
    trial = np.arange(1, runtime.trials + 1)
    accuracy = np.full(runtime.trials, np.nan)
    rt = np.full(runtime.trials, np.nan)
    condition = np.zeros(runtime.trials, dtype=object)

    for index in range(runtime.trials):
        condition_value = "0" if index < runtime.trials // 2 else "1"
        condition[index] = condition_value
        evidence = (beta - 0.5) * alpha
        time = tau
        while -alpha / 2 < evidence < alpha / 2 and time < runtime.max_time:
            evidence += drift_rate * runtime.dt + rng.normal(0.0, runtime.noise_sd)
            time += runtime.dt
        accuracy[index] = 1 if evidence >= alpha / 2 else 0
        rt[index] = time

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


model = ModelSpec(
    name="CustomSimulator",
    parameters=parameters_from_bounds(),
    simulate=simulate,
    conditions=("0", "1"),
)


def drift(t, condition, alpha, beta, drift_rate, tau):
    return drift_rate


process_model = define_model(
    name="CustomSimulator",
    conditions=("0", "1"),
    drift=drift,
    nondecision="tau",
)
"""


CAST_M_TEMPLATE = """from __future__ import annotations

import math

from flexddm import define_model


def gamma_pulse(time, shape, characteristic_time, peak_amplitude):
    time = max(time, 1e-6)
    characteristic_time = max(characteristic_time, 1e-6)
    base = (time * math.e) / ((shape - 1) * characteristic_time)
    return (
        peak_amplitude
        * math.exp(-(time / characteristic_time))
        * math.pow(base, shape - 1)
        * (((shape - 1) / time) - (1 / characteristic_time))
    )


def cast_condition(condition):
    domain, difficulty, phase = [int(part) for part in str(condition).split("-")]
    return domain, difficulty, phase


def drift(
    t,
    condition,
    alpha,
    mu_c_start,
    mu_c_end,
    mu_v,
    math_characteristic_time_start,
    word_characteristic_time_start,
    math_characteristic_time_end,
    word_characteristic_time_end,
    math_bias_start,
    word_bias_start,
    math_bias_end,
    word_bias_end,
    tau,
    math_hard_prob_start,
    word_hard_prob_start,
    math_hard_prob_end,
    word_hard_prob_end,
):
    domain, difficulty, phase = cast_condition(condition)
    is_math = 1 if domain == 1 else 0
    is_word = 1 - is_math
    if phase == 3:
        controlled = (mu_c_end + mu_v * ((math_hard_prob_end * (difficulty + 2)) - 2)) if is_math else (mu_c_end + mu_v * ((word_hard_prob_end * (difficulty + 2)) - 2))
        peak_amplitude = -math_bias_end * is_math - word_bias_end * is_word
        characteristic_time = math_characteristic_time_end * is_math + word_characteristic_time_end * is_word
    else:
        controlled = (mu_c_start + mu_v * ((math_hard_prob_start * (difficulty + 2)) - 2)) if is_math else (mu_c_start + mu_v * ((word_hard_prob_start * (difficulty + 2)) - 2))
        peak_amplitude = -math_bias_start * is_math - word_bias_start * is_word
        characteristic_time = math_characteristic_time_start * is_math + word_characteristic_time_start * is_word
    return controlled + gamma_pulse(t + tau, 2.0, characteristic_time, peak_amplitude)


process_model = define_model(
    name="CastM_Custom",
    conditions=(
        "0-0-0", "1-0-0", "0-1-0", "1-1-0", "0-2-0", "1-2-0", "0-3-0", "1-3-0", "0-4-0", "1-4-0",
        "0-0-3", "1-0-3", "0-1-3", "1-1-3", "0-2-3", "1-2-3", "0-3-3", "1-3-3", "0-4-3", "1-4-3",
    ),
    drift=drift,
    nondecision="tau",
)

model = process_model.to_simulation_model()
"""


RACING_DIFFUSION_TEMPLATE = """from __future__ import annotations

import numpy as np

from flexddm import ModelSpec, SimulationResult, parameters_from_bounds
from flexddm.builtin_fp_models import RLDDMFokkerPlanckSpec


def simulate(alpha, beta_a, beta_b, drift_a_0, drift_b_0, drift_a_1, drift_b_1, tau, runtime):
    rng = np.random.default_rng(runtime.seed)
    trial = np.arange(1, runtime.trials + 1)
    accuracy = np.full(runtime.trials, np.nan)
    rt = np.full(runtime.trials, np.nan)
    condition = np.zeros(runtime.trials, dtype=object)

    for index in range(runtime.trials):
        condition_value = "0" if index < runtime.trials // 2 else "1"
        condition[index] = condition_value
        drift_a = drift_a_0 if condition_value == "0" else drift_a_1
        drift_b = drift_b_0 if condition_value == "0" else drift_b_1
        evidence_a = beta_a * alpha
        evidence_b = beta_b * alpha
        time = tau

        while evidence_a < alpha and evidence_b < alpha and time < runtime.max_time:
            evidence_a += drift_a * runtime.dt + rng.normal(0.0, runtime.noise_sd)
            evidence_b += drift_b * runtime.dt + rng.normal(0.0, runtime.noise_sd)
            time += runtime.dt

        accuracy[index] = 1 if evidence_a >= alpha and evidence_a >= evidence_b else 0
        rt[index] = time

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


model = ModelSpec(
    name="RacingDiffusion_Custom",
    parameters=parameters_from_bounds(),
    simulate=simulate,
    conditions=("0", "1"),
)
"""


RLDDM_TEMPLATE = """from __future__ import annotations

import numpy as np

from flexddm import ModelSpec, SimulationResult, parameters_from_bounds
from flexddm.builtin_fp_models import RLDDMFokkerPlanckSpec


def update_values(q_left, q_right, choice, reward, learning_rate):
    if choice == 1:
        q_left = q_left + learning_rate * (reward - q_left)
    else:
        q_right = q_right + learning_rate * (reward - q_right)
    return q_left, q_right


def simulate(alpha, beta, learning_rate, value_scale, drift_bias, tau, runtime):
    rng = np.random.default_rng(runtime.seed)
    design = runtime.design
    if design is None or "reward_left" not in design or "reward_right" not in design:
        n_trials = runtime.trials
        design = {
            "reward_left": np.where(np.arange(n_trials) % 40 < 20, 0.75, 0.25),
            "reward_right": np.where(np.arange(n_trials) % 40 < 20, 0.25, 0.75),
            "condition": np.where(np.arange(n_trials) < n_trials / 3, "early", np.where(np.arange(n_trials) < 2 * n_trials / 3, "middle", "late")),
        }
    else:
        design = design.reset_index(drop=True)
        n_trials = min(runtime.trials, len(design))

    trial = np.arange(1, n_trials + 1)
    accuracy = np.full(n_trials, np.nan)
    rt = np.full(n_trials, np.nan)
    condition = np.full(n_trials, "all", dtype=object)
    q_left = 0.5
    q_right = 0.5

    for index in range(n_trials):
        if hasattr(design, "iloc"):
            row = design.iloc[index]
            reward_left = float(row.get("reward_left", 0.5))
            reward_right = float(row.get("reward_right", 0.5))
            condition_value = str(row.get("condition", "all"))
        else:
            reward_left = float(design["reward_left"][index])
            reward_right = float(design["reward_right"][index])
            condition_value = str(design["condition"][index])

        condition[index] = condition_value
        drift = drift_bias + value_scale * (q_left - q_right)
        evidence = (beta - 0.5) * alpha
        time = 0.0
        while -alpha / 2 < evidence < alpha / 2 and time < runtime.max_time:
            evidence += drift * runtime.dt + rng.normal(0.0, runtime.noise_sd)
            time += runtime.dt

        choice_left = 1 if evidence >= alpha / 2 else 0
        reward = reward_left if choice_left else reward_right
        q_left, q_right = update_values(q_left, q_right, choice_left, reward, learning_rate)
        accuracy[index] = choice_left
        rt[index] = time + tau

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


model = ModelSpec(
    name="RLDDM_Custom",
    parameters=parameters_from_bounds(),
    simulate=simulate,
    conditions=("early", "middle", "late"),
)

fp_model = RLDDMFokkerPlanckSpec()
"""


STANDARD_DDM_FP_TEMPLATE = STANDARD_DDM_TEMPLATE
DMC_FP_TEMPLATE = DMC_TEMPLATE
SSP_FP_TEMPLATE = SSP_TEMPLATE

DSTP_FP_TEMPLATE = """from __future__ import annotations

from flexddm import define_model


def drift(t, condition, alpha_ss, beta_ss, delta_ss, alpha_rs, beta_rs, delta_target, delta_flanker, delta_rs, tau):
    early = delta_target + delta_flanker if condition == "0" else delta_target - delta_flanker
    ss_start = (beta_ss - 0.5) * alpha_ss
    ss_target = alpha_ss / 2 if delta_ss >= 0 else -alpha_ss / 2
    switch_time = max((ss_target - ss_start) / max(abs(delta_ss), 1e-9), 0.0)
    return early if t < switch_time else (abs(delta_rs) if delta_ss >= 0 else -abs(delta_rs))


def bound(t, condition, alpha_ss, beta_ss, delta_ss, alpha_rs, beta_rs, delta_target, delta_flanker, delta_rs, tau):
    return alpha_rs / 2


def starting_position(t, condition, alpha_ss, beta_ss, delta_ss, alpha_rs, beta_rs, delta_target, delta_flanker, delta_rs, tau):
    return (beta_rs - 0.5) * alpha_rs


process_model = define_model(
    name="DSTP_Custom",
    conditions=("0", "1"),
    drift=drift,
    bound=bound,
    starting_position=starting_position,
    nondecision="tau",
)

model = process_model.to_simulation_model()
"""

STANDARD_DDM_SIM_TEMPLATE = """from __future__ import annotations

import numpy as np

from flexddm import ModelSpec, SimulationResult, parameters_from_bounds


def simulate(alpha, beta, delta_c, delta_i, tau, runtime):
    rng = np.random.default_rng(runtime.seed)
    trial = np.arange(1, runtime.trials + 1)
    accuracy = np.full(runtime.trials, np.nan)
    rt = np.full(runtime.trials, np.nan)
    condition = np.zeros(runtime.trials, dtype=object)

    for index in range(runtime.trials):
        condition_value = "0" if index < runtime.trials // 2 else "1"
        condition[index] = condition_value
        drift = delta_c if condition_value == "0" else delta_i
        evidence = (beta - 0.5) * alpha
        time = tau
        while -alpha / 2 < evidence < alpha / 2 and time < runtime.max_time:
            evidence += drift * runtime.dt + rng.normal(0.0, runtime.noise_sd)
            time += runtime.dt
        accuracy[index] = 1 if evidence >= alpha / 2 else 0
        rt[index] = time

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


model = ModelSpec(
    name="StandardDDM_Custom",
    parameters=parameters_from_bounds(),
    simulate=simulate,
    conditions=("0", "1"),
)
"""

DMC_SIM_TEMPLATE = """from __future__ import annotations

import math
import numpy as np

from flexddm import ModelSpec, SimulationResult, parameters_from_bounds


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


def simulate(alpha, beta, mu_c, shape, characteristic_time, peak_amplitude, tau, runtime):
    rng = np.random.default_rng(runtime.seed)
    trial = np.arange(1, runtime.trials + 1)
    accuracy = np.full(runtime.trials, np.nan)
    rt = np.full(runtime.trials, np.nan)
    condition = np.zeros(runtime.trials, dtype=object)

    for index in range(runtime.trials):
        condition_value = "0" if index < runtime.trials // 2 else "1"
        condition[index] = condition_value
        evidence = (beta - 0.5) * alpha
        time = tau
        while -alpha / 2 < evidence < alpha / 2 and time < runtime.max_time:
            direction = 1 if condition_value == "0" else -1
            drift = mu_c + direction * gamma_pulse(time, shape, characteristic_time, peak_amplitude)
            evidence += drift * runtime.dt + rng.normal(0.0, runtime.noise_sd)
            time += runtime.dt
        accuracy[index] = 1 if evidence >= alpha / 2 else 0
        rt[index] = time

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


model = ModelSpec(
    name="DMC_Custom",
    parameters=parameters_from_bounds(),
    simulate=simulate,
    conditions=("0", "1"),
)
"""

DSTP_SIM_TEMPLATE = """from __future__ import annotations

import numpy as np

from flexddm import ModelSpec, SimulationResult, parameters_from_bounds


def simulate(alpha_ss, beta_ss, delta_ss, alpha_rs, beta_rs, delta_target, delta_flanker, delta_rs, tau, runtime):
    rng = np.random.default_rng(runtime.seed)
    trial = np.arange(1, runtime.trials + 1)
    accuracy = np.full(runtime.trials, np.nan)
    rt = np.full(runtime.trials, np.nan)
    condition = np.zeros(runtime.trials, dtype=object)

    for index in range(runtime.trials):
        condition_value = "0" if index < runtime.trials // 2 else "1"
        condition[index] = condition_value
        evidence_ss = (beta_ss - 0.5) * alpha_ss
        evidence_rs = (beta_rs - 0.5) * alpha_rs
        early_rs = delta_target + delta_flanker if condition_value == "0" else delta_target - delta_flanker
        time = tau

        while (-alpha_ss / 2 < evidence_ss < alpha_ss / 2 or -alpha_rs / 2 < evidence_rs < alpha_rs / 2) and time < runtime.max_time:
            evidence_ss += delta_ss * runtime.dt + rng.normal(0.0, runtime.noise_sd)
            evidence_rs += early_rs * runtime.dt + rng.normal(0.0, runtime.noise_sd)
            time += runtime.dt

        if -alpha_rs / 2 < evidence_rs < alpha_rs / 2:
            late_rs = abs(delta_rs) if evidence_ss > alpha_ss / 2 else -abs(delta_rs)
            while -alpha_rs / 2 < evidence_rs < alpha_rs / 2 and time < runtime.max_time:
                evidence_rs += late_rs * runtime.dt + rng.normal(0.0, runtime.noise_sd)
                time += runtime.dt

        accuracy[index] = 1 if evidence_rs >= alpha_rs / 2 else 0
        rt[index] = time

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


model = ModelSpec(
    name="DSTP_Custom",
    parameters=parameters_from_bounds(),
    simulate=simulate,
    conditions=("0", "1"),
)
"""

SSP_SIM_TEMPLATE = """from __future__ import annotations

import math
import numpy as np

from flexddm import ModelSpec, SimulationResult, parameters_from_bounds


def normal_pdf(x, sd):
    return math.exp(-(x * x) / (2 * sd * sd)) / (sd * math.sqrt(2 * math.pi))


def simulate(alpha, beta, p, sd_0, sd_r, tau, runtime):
    rng = np.random.default_rng(runtime.seed)
    trial = np.arange(1, runtime.trials + 1)
    accuracy = np.full(runtime.trials, np.nan)
    rt = np.full(runtime.trials, np.nan)
    condition = np.zeros(runtime.trials, dtype=object)

    for index in range(runtime.trials):
        condition_value = "0" if index < runtime.trials // 2 else "1"
        condition[index] = condition_value
        evidence = (beta - 0.5) * alpha
        time = tau
        while -alpha / 2 < evidence < alpha / 2 and time < runtime.max_time:
            spotlight_width = max(sd_0 - sd_r * max(time - tau, 0.0), 1e-3)
            target_weight = normal_pdf(0.0, spotlight_width)
            flanker_weight = 2 * normal_pdf(1.0, spotlight_width) + 2 * normal_pdf(2.0, spotlight_width)
            flank_direction = 1 if condition_value == "0" else -1
            drift = p * (target_weight + flank_direction * flanker_weight)
            evidence += drift * runtime.dt + rng.normal(0.0, runtime.noise_sd)
            time += runtime.dt
        accuracy[index] = 1 if evidence >= alpha / 2 else 0
        rt[index] = time

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


model = ModelSpec(
    name="SSP_Custom",
    parameters=parameters_from_bounds(),
    simulate=simulate,
    conditions=("0", "1"),
)
"""

FOKKER_PLANCK_TEMPLATES = {
    "StandardDDM - Fokker-Planck": STANDARD_DDM_FP_TEMPLATE,
    "DMC - Fokker-Planck": DMC_FP_TEMPLATE,
    "DSTP - Fokker-Planck": DSTP_FP_TEMPLATE,
    "SSP - Fokker-Planck": SSP_FP_TEMPLATE,
}

SIMULATION_TEMPLATES = {
    "StandardDDM - Simulation": STANDARD_DDM_SIM_TEMPLATE,
    "DMC - Simulation": DMC_SIM_TEMPLATE,
    "DSTP - Simulation": DSTP_SIM_TEMPLATE,
    "SSP - Simulation": SSP_SIM_TEMPLATE,
}

SEEDED_CUSTOM_MODELS = {}

MODEL_TEMPLATES = {
    **FOKKER_PLANCK_TEMPLATES,
    **SIMULATION_TEMPLATES,
    **SEEDED_CUSTOM_MODELS,
    "Blank": BLANK_TEMPLATE,
}

MODEL_STARTER_TEMPLATES = {
    **FOKKER_PLANCK_TEMPLATES,
    **SIMULATION_TEMPLATES,
}

CUSTOM_EXAMPLE_MODELS = {}

MODEL_TEMPLATE_BOUNDS = {
    "Blank": BLANK_BOUNDS,
    "StandardDDM - Fokker-Planck": STANDARD_DDM_BOUNDS,
    "DMC - Fokker-Planck": DMC_BOUNDS,
    "DSTP - Fokker-Planck": DSTP_BOUNDS,
    "SSP - Fokker-Planck": SSP_BOUNDS,
    "StandardDDM - Simulation": STANDARD_DDM_BOUNDS,
    "DMC - Simulation": DMC_BOUNDS,
    "DSTP - Simulation": DSTP_BOUNDS,
    "SSP - Simulation": SSP_BOUNDS,
    "mDMC varying noise": MDMC_BOUNDS,
    "Weibull boundary": WEIBULL_BOUND_BOUNDS,
    "CastM": CAST_M_BOUNDS,
    "Racing diffusion": RACING_DIFFUSION_BOUNDS,
    "RLDDM": RLDDM_BOUNDS,
}
