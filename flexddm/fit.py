from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .fokker_planck import FokkerPlanckConfig, FokkerPlanckSpec
from .model import ModelSpec
from .plotting import plot_posterior_predictive
from .posterior_predictive import sample_like_observed
from .stats import bic_from_g2, empirical_targets, g2_statistic, simulated_props


@dataclass(frozen=True)
class FitConfig:
    maxiter: int = 200
    popsize: int = 8
    seed: int = 10
    polish: bool = True
    workers: int = 1
    local_refine: bool = True
    simulated_trials: int | None = None
    progress_interval: int = 25
    simulation_dt: float = 0.01
    simulation_noise_sd: float = 0.01
    cdf_quantiles: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
    caf_quantiles: tuple[float, ...] = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class FitResult:
    model_name: str
    parameters: dict[str, float]
    g2: float
    bic: float
    objective_name: str = "G2"
    objective_score: float | None = None
    objective_bic: float | None = None


def fit_model(
    model: ModelSpec,
    data: pd.DataFrame,
    config: FitConfig = FitConfig(),
    ppc_output_path: str | Path | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> FitResult:
    targets = empirical_targets(data, list(config.cdf_quantiles), list(config.caf_quantiles))
    bounds = model.bounds_for_data(data)
    evaluations = 0
    generations = 0
    expected_evaluations = max(config.maxiter + 1, 1) * max(config.popsize, 1) * max(len(bounds), 1)

    def bump_generation() -> int:
        nonlocal generations
        generations += 1
        return generations

    def objective(values: np.ndarray) -> float:
        nonlocal evaluations
        _check_cancel(cancel_callback)
        evaluations += 1
        if progress_callback is not None and evaluations % max(config.progress_interval, 1) == 0:
            progress_callback(
                {
                    "phase": "objective",
                    "evaluations": evaluations,
                    "expectedEvaluations": expected_evaluations,
                    "parameters": len(bounds),
                    "maxiter": config.maxiter,
                    "popsize": config.popsize,
                    "polish": config.polish,
                    "localRefine": config.local_refine,
                }
            )
        try:
            objective_trials = len(data) if {"reward_left", "reward_right"}.issubset(data.columns) else (config.simulated_trials or max(len(data), model.default_trials))
            simulated = model.simulate_dataframe(
                values,
                n_trials=objective_trials,
                dt=config.simulation_dt,
                noise_sd=config.simulation_noise_sd,
                seed=config.seed,
                data_like=data,
            )
            return g2_statistic(targets, simulated_props(simulated, targets))
        except Exception:
            return float(np.iinfo(np.int64).max)

    fit = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=config.maxiter,
        popsize=config.popsize,
        seed=config.seed,
        polish=config.polish,
        workers=1,
        updating="immediate",
        callback=lambda xk, convergence: _report_generation(
            progress_callback,
            config,
            len(bounds),
            expected_evaluations,
            evaluations,
            convergence,
            bump_generation(),
        ),
    )

    values = fit.x
    score = float(fit.fun)
    _check_cancel(cancel_callback)
    if config.local_refine:
        if progress_callback is not None:
            progress_callback({"phase": "local_refine", "evaluations": evaluations, "expectedEvaluations": expected_evaluations})
        refined = minimize(objective, x0=values, method="Nelder-Mead", options={"maxiter": 1000})
        if refined.fun < score:
            values = refined.x
            score = float(refined.fun)

    _check_cancel(cancel_callback)
    simulated = sample_like_observed(model, values, data)
    if ppc_output_path is not None:
        plot_posterior_predictive(data, simulated, ppc_output_path, title=f"{model.name} posterior predictive check")

    bic = bic_from_g2(score, len(model.parameters), len(data))
    return FitResult(
        model_name=model.name,
        parameters=dict(zip(model.parameter_names, map(float, values))),
        g2=score,
        bic=bic,
        objective_name="G2",
        objective_score=score,
        objective_bic=bic,
    )


def fit_fokker_planck_model(
    model: FokkerPlanckSpec,
    data: pd.DataFrame,
    config: FitConfig = FitConfig(),
    fp_config: FokkerPlanckConfig = FokkerPlanckConfig(),
    ppc_output_path: str | Path | None = None,
    objective_name: str = "g2",
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> FitResult:
    targets = empirical_targets(data, list(config.cdf_quantiles), list(config.caf_quantiles))
    bounds = model.bounds_for_data(data)
    objective_key = objective_name.strip().lower()
    evaluations = 0
    generations = 0
    expected_evaluations = max(config.maxiter + 1, 1) * max(config.popsize, 1) * max(len(bounds), 1)
    evaluation_lock = threading.Lock()

    def bump_generation() -> int:
        nonlocal generations
        generations += 1
        return generations

    def objective(values: np.ndarray) -> float:
        nonlocal evaluations
        _check_cancel(cancel_callback)
        with evaluation_lock:
            evaluations += 1
            current_evaluations = evaluations
        if progress_callback is not None and current_evaluations % max(config.progress_interval, 1) == 0:
            progress_callback(
                {
                    "phase": "objective",
                    "evaluations": current_evaluations,
                    "expectedEvaluations": expected_evaluations,
                    "parameters": len(bounds),
                    "maxiter": config.maxiter,
                    "popsize": config.popsize,
                    "polish": config.polish,
                    "localRefine": config.local_refine,
                }
            )
        try:
            if objective_key == "g2":
                return g2_statistic(targets, _fp_predict_props(model, values, targets, fp_config, data))
            return model.neg2_log_likelihood(values, data, fp_config)
        except Exception:
            return float(np.iinfo(np.int64).max)

    worker_count = max(int(config.workers), 1)
    executor = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    try:
        fit = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=config.maxiter,
            popsize=config.popsize,
            seed=config.seed,
            polish=config.polish,
            workers=executor.map if executor is not None else 1,
            updating="deferred" if executor is not None else "immediate",
            callback=lambda xk, convergence: _report_generation(
                progress_callback,
                config,
                len(bounds),
                expected_evaluations,
                evaluations,
                convergence,
                bump_generation(),
            ),
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    values = fit.x
    score = float(fit.fun)
    _check_cancel(cancel_callback)
    if config.local_refine:
        if progress_callback is not None:
            progress_callback({"phase": "local_refine", "evaluations": evaluations, "expectedEvaluations": expected_evaluations})
        refined = minimize(objective, x0=values, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
        if refined.fun < score:
            values = refined.x
            score = float(refined.fun)

    _check_cancel(cancel_callback)
    simulated = sample_like_observed(model, values, data, fp_config=fp_config)
    if ppc_output_path is not None:
        plot_posterior_predictive(data, simulated, ppc_output_path, title=f"{model.name} Fokker-Planck posterior predictive check")

    comparison_g2 = g2_statistic(targets, _fp_predict_props(model, values, targets, fp_config, data))
    comparison_bic = bic_from_g2(comparison_g2, len(model.parameters), len(data))
    optimized_name = "G2" if objective_key == "g2" else "-2LL"
    return FitResult(
        model_name=f"{model.name} (Fokker-Planck)",
        parameters=dict(zip(model.parameter_names, map(float, values))),
        g2=comparison_g2,
        bic=comparison_bic,
        objective_name=optimized_name,
        objective_score=score,
        objective_bic=bic_from_g2(score, len(model.parameters), len(data)),
    )


def _fp_predict_props(model: FokkerPlanckSpec, values: np.ndarray, targets, fp_config: FokkerPlanckConfig, data: pd.DataFrame):
    try:
        return model.predict_props(values, targets, fp_config, data=data)
    except TypeError:
        return model.predict_props(values, targets, fp_config)


def _report_generation(
    progress_callback: Callable[[dict[str, object]], None] | None,
    config: FitConfig,
    n_parameters: int,
    expected_evaluations: int,
    evaluations: int,
    convergence: float,
    generation: int,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "phase": "generation",
            "generation": generation,
            "maxiter": config.maxiter,
            "evaluations": evaluations,
            "expectedEvaluations": expected_evaluations,
            "parameters": n_parameters,
            "popsize": config.popsize,
            "convergence": float(convergence),
            "polish": config.polish,
            "localRefine": config.local_refine,
        }
    )


def _check_cancel(cancel_callback: Callable[[], bool] | None) -> None:
    if cancel_callback is not None and cancel_callback():
        raise InterruptedError("Fit cancelled.")
