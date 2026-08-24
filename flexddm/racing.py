from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .fokker_planck import FokkerPlanckConfig, FokkerPlanckSolution, FokkerPlanckSpec
from .model import Parameter
from .stats import FitTargets


@dataclass(frozen=True)
class RacingFokkerPlanckSpec:
    name: str
    parameters: tuple[Parameter, ...]
    conditions: tuple[str, ...]
    drifts: tuple[Callable[..., float], ...]
    bound: Callable[..., float]
    noise: Callable[..., float] | float = 0.1
    nondecision: Callable[..., float] | float = 0.0
    default_trials: int = 800

    @property
    def parameter_names(self) -> list[str]:
        return [parameter.name for parameter in self.parameters]

    @property
    def uses_trial_context(self) -> bool:
        return any(_needs_trial_context(component) for component in (*self.drifts, self.bound, self.noise, self.nondecision))

    def bounds_for_data(self, data: pd.DataFrame | None = None) -> list[tuple[float, float]]:
        return [parameter.bounds_for_data(data) for parameter in self.parameters]

    def _parameter_dict(self, values: Iterable[float]) -> dict[str, float]:
        return dict(zip(self.parameter_names, map(float, values)))

    def _component_params(self, component: Callable[..., float], params: dict[str, float]) -> dict[str, float]:
        try:
            signature = inspect.signature(component).parameters
        except (TypeError, ValueError):
            return {}
        return {name: params[name] for name in signature if name in params}

    def _call(self, component, t: float, condition: str, params: dict[str, float], trial_row: dict[str, object] | None = None, x: float | None = None) -> float:
        if not callable(component):
            return float(component)
        kwargs = self._component_params(component, params)
        signature = inspect.signature(component).parameters
        if "trial_row" in signature:
            kwargs["trial_row"] = trial_row or {}
        if "covariates" in signature:
            kwargs["covariates"] = trial_row or {}
        names = list(signature)
        if x is not None and names and names[0] == "x":
            return float(component(x, t, condition, **kwargs))
        return float(component(t, condition, **kwargs))

    def _one_accumulator_solution(
        self,
        drift: Callable[..., float],
        values: Iterable[float],
        condition: str,
        config: FokkerPlanckConfig,
        trial_row: dict[str, object] | None = None,
    ) -> FokkerPlanckSolution:
        params = self._parameter_dict(values)

        def drift_component(t, condition, *unused, trial_row=None):
            return self._call(drift, t, condition, params, trial_row=trial_row)

        def bound_component(t, condition, *unused, trial_row=None):
            return abs(self._call(self.bound, t, condition, params, trial_row=trial_row))

        def noise_component(t, condition, *unused, trial_row=None):
            return abs(self._call(self.noise, t, condition, params, trial_row=trial_row))

        return FokkerPlanckSpec(
            name=f"{self.name}_accumulator",
            parameters=(),
            conditions=self.conditions,
            drift=drift_component,
            bound=bound_component,
            noise=noise_component,
            nondecision=0.0,
            starting_position=0.0,
            default_trials=self.default_trials,
        ).solve_condition((), condition, config, trial_row=trial_row)

    def _race_densities(
        self,
        values: Iterable[float],
        condition: str,
        config: FokkerPlanckConfig,
        trial_row: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if len(self.drifts) != 2:
            raise ValueError("Current FlexDDM objectives support Fokker-Planck racing fits with exactly two response accumulators.")
        params = self._parameter_dict(values)
        solutions = [self._one_accumulator_solution(drift, values, condition, config, trial_row=trial_row) for drift in self.drifts]
        times = solutions[0].times
        upper_pdfs = [np.nan_to_num(solution.upper_pdf, nan=0.0, posinf=0.0, neginf=0.0) for solution in solutions]
        all_pdfs = [np.nan_to_num(solution.upper_pdf + solution.lower_pdf, nan=0.0, posinf=0.0, neginf=0.0) for solution in solutions]
        survivals = [np.clip(1.0 - np.cumsum(pdf) * config.dt, 0.0, 1.0) for pdf in all_pdfs]
        response_one = upper_pdfs[0] * survivals[1]
        response_zero = upper_pdfs[1] * survivals[0]
        nondecision = self._call(self.nondecision, 0.0, condition, params, trial_row=trial_row)
        return times, response_one, response_zero, nondecision

    def neg2_log_likelihood(self, values: Iterable[float], data: pd.DataFrame, config: FokkerPlanckConfig = FokkerPlanckConfig()) -> float:
        total = 0.0
        epsilon = 1e-12
        if self.uses_trial_context:
            for _, row in data.reset_index(drop=True).iterrows():
                condition = str(row["condition"])
                times, response_one, response_zero, nondecision = self._race_densities(values, condition, config, trial_row=row.to_dict())
                rt = times + nondecision
                response = int(round(float(row["accuracy"])))
                density = np.interp(float(row["rt"]), rt, response_one if response == 1 else response_zero, left=epsilon, right=epsilon)
                total += float(np.log(max(float(density), epsilon)))
        else:
            for condition, group in data.groupby("condition", dropna=False):
                condition = str(condition)
                times, response_one, response_zero, nondecision = self._race_densities(values, condition, config)
                rt = times + nondecision
                one_rt = group.loc[group["accuracy"].round().astype(int) == 1, "rt"].to_numpy(dtype=float)
                zero_rt = group.loc[group["accuracy"].round().astype(int) == 0, "rt"].to_numpy(dtype=float)
                if one_rt.size:
                    total += float(np.log(np.maximum(np.interp(one_rt, rt, response_one, left=epsilon, right=epsilon), epsilon)).sum())
                if zero_rt.size:
                    total += float(np.log(np.maximum(np.interp(zero_rt, rt, response_zero, left=epsilon, right=epsilon), epsilon)).sum())
        if not np.isfinite(total):
            return float(np.iinfo(np.int64).max)
        return -2.0 * total

    def predict_props(self, values: Iterable[float], targets: FitTargets, config: FokkerPlanckConfig = FokkerPlanckConfig(), data: pd.DataFrame | None = None) -> dict[str, list[float]]:
        props: dict[str, list[float]] = {}
        if self.uses_trial_context:
            if data is None:
                raise ValueError(f"{self.name} racing FP predictions require trial-level data.")
            from .stats import simulated_props

            return simulated_props(self.sample_dataframe_like(values, data, "fp_pred", config), targets)
        for condition, target in targets.conditions.items():
            times, response_one, response_zero, nondecision = self._race_densities(values, condition, config)
            rt = times + nondecision
            total_mass = float((response_one.sum() + response_zero.sum()) * config.dt)
            if not np.isfinite(total_mass) or total_mass <= 0:
                raise ValueError(f"{self.name} produced no finite racing density for condition {condition}.")
            props[f"cdf_props_{condition}"] = _mass_bins(rt, response_one, target.cdf_cutoffs, total_mass, config.dt)
            props[f"caf_props_{condition}"] = _mass_bins(rt, response_zero, target.caf_cutoffs, total_mass, config.dt)
        return props

    def sample_dataframe_like(
        self,
        values: Iterable[float],
        observed: pd.DataFrame,
        participant_id: object = "fp",
        config: FokkerPlanckConfig = FokkerPlanckConfig(),
    ) -> pd.DataFrame:
        rng = np.random.default_rng(config.seed)
        rows: list[dict[str, object]] = []
        solution_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
        for index, row in observed.reset_index(drop=True).iterrows():
            trial_row = row.to_dict()
            condition = str(row["condition"])
            cache_key = condition if not self.uses_trial_context else f"{condition}_{index}"
            if cache_key not in solution_cache:
                solution_cache[cache_key] = self._race_densities(values, condition, config, trial_row=trial_row if self.uses_trial_context else None)
            times, response_one, response_zero, nondecision = solution_cache[cache_key]
            weights = np.concatenate([response_one, response_zero])
            weight_sum = float(np.nansum(weights))
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                raise ValueError(f"{self.name} produced no finite racing density for condition {condition}.")
            draw = int(rng.choice(np.arange(len(weights)), p=np.nan_to_num(weights, nan=0.0) / weight_sum))
            response = 1 if draw < len(times) else 0
            time_index = draw if response == 1 else draw - len(times)
            rows.append(
                {
                    "id": row.get("id", participant_id) if participant_id == "fp_pred" else participant_id,
                    "trial": int(row.get("trial", index + 1)),
                    "accuracy": response,
                    "rt": float(times[time_index] + nondecision),
                    "condition": condition,
                }
            )
        return pd.DataFrame(rows)


def _mass_bins(rt: np.ndarray, density: np.ndarray, cutoffs: list[float], denominator: float, dt: float) -> list[float]:
    values: list[float] = []
    previous = -np.inf
    for cutoff in cutoffs:
        mask = (rt > previous) & (rt <= cutoff)
        values.append(float(density[mask].sum() * dt) / denominator if denominator else 0.0)
        previous = cutoff
    values.append(float(density[rt > previous].sum() * dt) / denominator if denominator else 0.0)
    return values


def _needs_trial_context(component) -> bool:
    if not callable(component):
        return False
    try:
        parameters = inspect.signature(component).parameters
    except (TypeError, ValueError):
        return False
    return "trial_row" in parameters or "covariates" in parameters
