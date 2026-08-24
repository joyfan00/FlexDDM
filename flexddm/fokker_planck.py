from __future__ import annotations

import ast
from functools import lru_cache
from dataclasses import dataclass
import inspect
import textwrap
from typing import Callable, Iterable, Protocol

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.special import ndtr

from .compiled_kfe import HAS_NUMBA, solve_kfe_arrays
from .model import Parameter
from .stats import FitTargets

try:
    from ._cpp import solve_kfe as _cpp_solve_kfe
    from ._cpp import solve_transition as _cpp_solve_transition

    HAS_CPP_TRANSITION = True
    HAS_CPP_KFE = True
except Exception:  # pragma: no cover - exercised when the extension is not built
    _cpp_solve_kfe = None
    _cpp_solve_transition = None
    HAS_CPP_TRANSITION = False
    HAS_CPP_KFE = False


@dataclass(frozen=True)
class FokkerPlanckConfig:
    dt: float = 0.02
    dx: float = 0.01
    t_max: float = 3.0
    seed: int = 10


@dataclass(frozen=True)
class FokkerPlanckSolution:
    times: np.ndarray
    upper_pdf: np.ndarray
    lower_pdf: np.ndarray


@dataclass(frozen=True)
class FokkerPlanckSpec:
    name: str
    parameters: tuple[Parameter, ...]
    conditions: tuple[str, ...]
    drift: Callable[..., float]
    bound: Callable[..., float]
    noise: Callable[..., float]
    nondecision: Callable[..., float] | float = 0.0
    starting_position: Callable[..., float] | float = 0.0
    default_trials: int = 800

    @property
    def parameter_names(self) -> list[str]:
        return [parameter.name for parameter in self.parameters]

    def bounds_for_data(self, data: pd.DataFrame | None = None) -> list[tuple[float, float]]:
        return [parameter.bounds_for_data(data) for parameter in self.parameters]

    def solve_condition(
        self,
        values: Iterable[float],
        condition: str,
        config: FokkerPlanckConfig = FokkerPlanckConfig(),
        trial_row: dict[str, object] | None = None,
    ) -> FokkerPlanckSolution:
        params = tuple(values)
        times = np.arange(0.0, config.t_max + config.dt, config.dt)
        bounds = np.array([abs(float(_call_or_value(self.bound, time, condition, *params, trial_row=trial_row))) for time in times], dtype=float)
        if not np.all(np.isfinite(bounds)) or np.any(bounds <= 0):
            raise ValueError(f"{self.name} produced invalid boundary values for condition {condition}.")
        start = _call_or_value(self.starting_position, 0.0, condition, *params, trial_row=trial_row)
        if not np.isfinite(start):
            raise ValueError(f"{self.name} produced a non-finite starting position for condition {condition}.")
        if abs(start) >= bounds[0]:
            raise ValueError(f"{self.name} starting position is outside the boundary for condition {condition}.")
        kfe_solution = _try_solve_kfe(self, times, bounds, start, condition, params, config, trial_row=trial_row)
        if kfe_solution is not None:
            return FokkerPlanckSolution(times=times, upper_pdf=kfe_solution[0], lower_pdf=kfe_solution[1])

        max_bound = max(float(bounds.max()), config.dx * 4)
        grid = np.arange(-max_bound, max_bound + config.dx, config.dx)
        edges = np.concatenate(([grid[0] - config.dx / 2], grid[:-1] + config.dx / 2, [grid[-1] + config.dx / 2]))
        drift_grid, noise_grid = _transition_components(self, grid, times[:-1], condition, params, trial_row=trial_row)
        if HAS_CPP_TRANSITION and _cpp_solve_transition is not None:
            upper, lower = _cpp_solve_transition(
                np.ascontiguousarray(grid, dtype=float),
                np.ascontiguousarray(edges, dtype=float),
                np.ascontiguousarray(bounds, dtype=float),
                np.ascontiguousarray(drift_grid, dtype=float),
                np.ascontiguousarray(noise_grid, dtype=float),
                float(start),
                float(config.dt),
            )
        else:
            upper, lower = _solve_transition_python(grid, edges, bounds, drift_grid, noise_grid, start, config.dt)

        return FokkerPlanckSolution(times=times, upper_pdf=upper, lower_pdf=lower)

    def neg2_log_likelihood(
        self,
        values: Iterable[float],
        data: pd.DataFrame,
        config: FokkerPlanckConfig = FokkerPlanckConfig(),
    ) -> float:
        params = tuple(values)
        total = 0.0
        epsilon = 1e-12
        if self.uses_trial_context:
            for _, row in data.reset_index(drop=True).iterrows():
                condition = str(row["condition"])
                if condition not in self.conditions:
                    return float(np.iinfo(np.int64).max)
                trial_row = row.to_dict()
                solution = self.solve_condition(params, condition, config, trial_row=trial_row)
                nondecision = _call_or_value(self.nondecision, 0.0, condition, *params, trial_row=trial_row)
                rt_grid = solution.times + nondecision
                rt = float(row["rt"])
                response = int(round(float(row["accuracy"])))
                density = np.interp(rt, rt_grid, solution.upper_pdf if response == 1 else solution.lower_pdf, left=epsilon, right=epsilon)
                total += float(np.log(max(float(density), epsilon)))
            if not np.isfinite(total):
                return float(np.iinfo(np.int64).max)
            return -2.0 * total

        for condition, group in data.groupby("condition", dropna=False):
            condition = str(condition)
            if condition not in self.conditions:
                return float(np.iinfo(np.int64).max)
            solution = self.solve_condition(params, condition, config)
            nondecision = _call_or_value(self.nondecision, 0.0, condition, *params)
            rt_grid = solution.times + nondecision
            upper_rt = group.loc[group["accuracy"].round().astype(int) == 1, "rt"].to_numpy(dtype=float)
            lower_rt = group.loc[group["accuracy"].round().astype(int) == 0, "rt"].to_numpy(dtype=float)
            if upper_rt.size:
                upper_density = np.interp(upper_rt, rt_grid, solution.upper_pdf, left=epsilon, right=epsilon)
                total += float(np.log(np.maximum(upper_density, epsilon)).sum())
            if lower_rt.size:
                lower_density = np.interp(lower_rt, rt_grid, solution.lower_pdf, left=epsilon, right=epsilon)
                total += float(np.log(np.maximum(lower_density, epsilon)).sum())
        if not np.isfinite(total):
            return float(np.iinfo(np.int64).max)
        return -2.0 * total

    def predict_props(
        self,
        values: Iterable[float],
        targets: FitTargets,
        config: FokkerPlanckConfig = FokkerPlanckConfig(),
        data: pd.DataFrame | None = None,
    ) -> dict[str, list[float]]:
        if self.uses_trial_context:
            if data is None:
                raise ValueError(f"{self.name} Fokker-Planck predictions require trial-level data for covariate-aware components.")
            from .stats import simulated_props

            simulated = self.sample_dataframe_like(values, data, "fp_pred", config)
            return simulated_props(simulated, targets)

        params = tuple(values)
        props: dict[str, list[float]] = {}
        for condition, target in targets.conditions.items():
            if condition not in self.conditions:
                props[f"cdf_props_{condition}"] = [0.0 for _ in range(len(target.cdf_cutoffs) + 1)]
                props[f"caf_props_{condition}"] = [0.0 for _ in range(len(target.caf_cutoffs) + 1)]
                continue
            solution = self.solve_condition(params, condition, config)
            nondecision = _call_or_value(self.nondecision, 0.0, condition, *params)
            if not np.isfinite(nondecision):
                raise ValueError(f"{self.name} produced a non-finite non-decision time for condition {condition}.")
            rt = solution.times + nondecision
            total_mass = float(solution.upper_pdf.sum() + solution.lower_pdf.sum())
            if not np.isfinite(total_mass) or total_mass <= 0:
                raise ValueError(f"{self.name} produced no finite boundary-crossing density for condition {condition}.")
            props[f"cdf_props_{condition}"] = _mass_bins(rt, solution.upper_pdf, target.cdf_cutoffs, total_mass)
            props[f"caf_props_{condition}"] = _mass_bins(rt, solution.lower_pdf, target.caf_cutoffs, total_mass)
        return props

    def sample_dataframe(
        self,
        values: Iterable[float],
        n_trials: int | None = None,
        participant_id: object = "fp",
        config: FokkerPlanckConfig = FokkerPlanckConfig(),
    ) -> pd.DataFrame:
        params = tuple(values)
        n_trials = n_trials or self.default_trials
        rng = np.random.default_rng(config.seed)
        per_condition = max(n_trials // max(len(self.conditions), 1), 1)
        rows: list[dict[str, object]] = []

        for condition in self.conditions:
            solution = self.solve_condition(params, condition, config)
            weights = np.concatenate([solution.upper_pdf, solution.lower_pdf])
            weight_sum = float(np.nansum(weights))
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                raise ValueError(f"{self.name} produced no finite boundary-crossing density for condition {condition}.")
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0) / weight_sum
            draws = rng.choice(np.arange(len(weights)), size=per_condition, p=weights)
            nondecision = _call_or_value(self.nondecision, 0.0, condition, *params)
            for draw in draws:
                upper = draw < len(solution.times)
                time_index = draw if upper else draw - len(solution.times)
                rows.append(
                    {
                        "id": participant_id,
                        "trial": len(rows) + 1,
                        "accuracy": 1 if upper else 0,
                        "rt": float(solution.times[time_index] + nondecision),
                        "condition": condition,
                    }
                )

        return pd.DataFrame(rows)

    def sample_dataframe_like(
        self,
        values: Iterable[float],
        observed: pd.DataFrame,
        participant_id: object = "fp",
        config: FokkerPlanckConfig = FokkerPlanckConfig(),
    ) -> pd.DataFrame:
        if not self.uses_trial_context:
            counts = {str(condition): int(count) for condition, count in observed["condition"].astype(str).value_counts().items()}
            return self.sample_dataframe(values, n_trials=sum(counts.values()), participant_id=participant_id, config=config)

        params = tuple(values)
        rng = np.random.default_rng(config.seed)
        rows: list[dict[str, object]] = []
        for index, row in observed.reset_index(drop=True).iterrows():
            trial_row = row.to_dict()
            condition = str(row["condition"])
            if condition not in self.conditions:
                raise ValueError(f"Observed condition {condition!r} is not produced by model {self.name!r}.")
            solution = self.solve_condition(params, condition, config, trial_row=trial_row)
            weights = np.concatenate([solution.upper_pdf, solution.lower_pdf])
            weight_sum = float(np.nansum(weights))
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                raise ValueError(f"{self.name} produced no finite boundary-crossing density for condition {condition}.")
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0) / weight_sum
            draw = int(rng.choice(np.arange(len(weights)), p=weights))
            upper = draw < len(solution.times)
            time_index = draw if upper else draw - len(solution.times)
            nondecision = _call_or_value(self.nondecision, 0.0, condition, *params, trial_row=trial_row)
            rows.append(
                {
                    "id": row.get("id", participant_id) if participant_id == "fp_pred" else participant_id,
                    "trial": int(row.get("trial", index + 1)),
                    "accuracy": 1 if upper else 0,
                    "rt": float(solution.times[time_index] + nondecision),
                    "condition": condition,
                }
            )
        return pd.DataFrame(rows)

    @property
    def uses_trial_context(self) -> bool:
        return any(_component_needs_trial_context(component) for component in (self.drift, self.bound, self.noise, self.nondecision, self.starting_position))


class TrialwiseFokkerPlanckProtocol(Protocol):
    name: str
    parameters: tuple[Parameter, ...]
    conditions: tuple[str, ...]

    @property
    def parameter_names(self) -> list[str]: ...

    def bounds_for_data(self, data: pd.DataFrame | None = None) -> list[tuple[float, float]]: ...

    def neg2_log_likelihood(self, values: Iterable[float], data: pd.DataFrame, config: FokkerPlanckConfig = FokkerPlanckConfig()) -> float: ...

    def predict_props(self, values: Iterable[float], targets: FitTargets, config: FokkerPlanckConfig = FokkerPlanckConfig(), data: pd.DataFrame | None = None) -> dict[str, list[float]]: ...

    def sample_dataframe_like(self, values: Iterable[float], observed: pd.DataFrame, participant_id: object = "fp", config: FokkerPlanckConfig = FokkerPlanckConfig()) -> pd.DataFrame: ...


def _call_or_value(value: Callable[..., float] | float, *args: object, trial_row: dict[str, object] | None = None) -> float:
    return float(_call_with_optional_trial_context(value, *args, trial_row=trial_row) if callable(value) else value)


def _call_with_optional_trial_context(value: Callable[..., float], *args: object, trial_row: dict[str, object] | None = None) -> float:
    try:
        parameters = inspect.signature(value).parameters
    except (TypeError, ValueError):
        return value(*args)
    kwargs = {}
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if "trial_row" in parameters or accepts_kwargs:
        kwargs["trial_row"] = trial_row or {}
    if "covariates" in parameters or accepts_kwargs:
        kwargs["covariates"] = trial_row or {}
    return value(*args, **kwargs)


def _physical_drift(
    model: FokkerPlanckSpec,
    x_values: np.ndarray,
    time: float,
    condition: str,
    params: tuple[float, ...],
    trial_row: dict[str, object] | None = None,
) -> np.ndarray:
    if _component_ignores_x(model.drift):
        return np.full_like(x_values, float(_call_or_value(model.drift, time, condition, *params, trial_row=trial_row)), dtype=float)
    return np.array([_call_or_value(model.drift, x, time, condition, *params, trial_row=trial_row) for x in x_values], dtype=float)


def _physical_noise(
    model: FokkerPlanckSpec,
    x_values: np.ndarray,
    time: float,
    condition: str,
    params: tuple[float, ...],
    trial_row: dict[str, object] | None = None,
) -> np.ndarray:
    if _component_ignores_x(model.noise):
        return np.full_like(x_values, abs(float(_call_or_value(model.noise, time, condition, *params, trial_row=trial_row))), dtype=float)
    return np.array([abs(_call_or_value(model.noise, x, time, condition, *params, trial_row=trial_row)) for x in x_values], dtype=float)


def _try_solve_kfe(
    model: FokkerPlanckSpec,
    times: np.ndarray,
    bounds: np.ndarray,
    start: float,
    condition: str,
    params: tuple[float, ...],
    config: FokkerPlanckConfig,
    trial_row: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not HAS_CPP_KFE or _cpp_solve_kfe is None:
        return None

    nx = max(int(round(2.0 / config.dx)), 12)
    dx = 2.0 / nx
    x_vec = np.linspace(-1.0, 1.0, nx + 1)
    normalized_start = start / bounds[0]
    if not np.isfinite(normalized_start) or abs(normalized_start) >= 1.0:
        return None

    drift_values = np.empty_like(times, dtype=float)
    noise_values = np.empty_like(times, dtype=float)
    probe = np.array([-0.5, 0.0, 0.5], dtype=float)
    drift_static = _component_has_unused_x(model.drift)
    noise_static = _component_has_unused_x(model.noise)
    for index, time in enumerate(times):
        if _component_ignores_x(model.drift):
            drift_values[index] = float(_call_or_value(model.drift, float(time), condition, *params, trial_row=trial_row))
        elif drift_static:
            drift_values[index] = float(_call_or_value(model.drift, 0.0, float(time), condition, *params, trial_row=trial_row))
        else:
            physical_probe = probe * bounds[index]
            drift_probe = _physical_drift(model, physical_probe, float(time), condition, params, trial_row=trial_row)
            if not np.all(np.isfinite(drift_probe)):
                raise ValueError(f"{model.name} produced non-finite drift at t={time}.")
            if np.nanmax(drift_probe) - np.nanmin(drift_probe) > 1e-8:
                return None
            drift_values[index] = float(np.nanmean(drift_probe))

        if _component_ignores_x(model.noise):
            noise_values[index] = abs(float(_call_or_value(model.noise, float(time), condition, *params, trial_row=trial_row)))
        elif noise_static:
            noise_values[index] = abs(float(_call_or_value(model.noise, 0.0, float(time), condition, *params, trial_row=trial_row)))
        else:
            physical_probe = probe * bounds[index]
            noise_probe = _physical_noise(model, physical_probe, float(time), condition, params, trial_row=trial_row)
            if not np.all(np.isfinite(noise_probe)):
                raise ValueError(f"{model.name} produced non-finite noise at t={time}.")
            if np.nanmax(noise_probe) - np.nanmin(noise_probe) > 1e-8:
                return None
            noise_values[index] = float(np.nanmean(np.abs(noise_probe)))

    noise_values = np.maximum(np.asarray(noise_values, dtype=float), 1e-12)
    if not np.all(np.isfinite(noise_values)) or np.any(noise_values <= 0):
        raise ValueError(f"{model.name} produced invalid noise for condition {condition}.")
    if float(np.nanmax(noise_values) - np.nanmin(noise_values)) > 1e-8:
        return None

    initial = np.zeros_like(x_vec, dtype=float)
    initial[np.argmin(np.abs(x_vec - normalized_start))] = 1.0 / dx
    bound_derivative = np.gradient(bounds, config.dt)
    upper, lower = _cpp_solve_kfe(
        np.ascontiguousarray(initial, dtype=float),
        np.ascontiguousarray(x_vec, dtype=float),
        np.ascontiguousarray(bounds, dtype=float),
        np.ascontiguousarray(drift_values, dtype=float),
        np.ascontiguousarray(bound_derivative, dtype=float),
        float(np.nanmean(noise_values)),
        float(config.dt),
        float(dx),
    )
    if not np.all(np.isfinite(upper)) or not np.all(np.isfinite(lower)) or (float(np.sum(upper)) + float(np.sum(lower))) <= 0:
        raise ValueError(f"{model.name} KFE solver produced invalid density for condition {condition}.")
    return upper, lower


def _transition_components(
    model: FokkerPlanckSpec,
    grid: np.ndarray,
    times: np.ndarray,
    condition: str,
    params: tuple[float, ...],
    trial_row: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    drift_grid = np.empty((len(times), len(grid)), dtype=float)
    noise_grid = np.empty((len(times), len(grid)), dtype=float)
    for index, time in enumerate(times):
        drift = _physical_drift(model, grid, float(time), condition, params, trial_row=trial_row)
        noise = _physical_noise(model, grid, float(time), condition, params, trial_row=trial_row)
        if not np.all(np.isfinite(drift)) or not np.all(np.isfinite(noise)):
            raise ValueError(f"{model.name} produced non-finite drift/noise at t={time}.")
        drift_grid[index] = drift
        noise_grid[index] = np.maximum(np.abs(noise), 1e-12)
    return drift_grid, noise_grid


def _solve_transition_python(
    grid: np.ndarray,
    edges: np.ndarray,
    bounds: np.ndarray,
    drift_grid: np.ndarray,
    noise_grid: np.ndarray,
    start: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    state_mass = np.zeros_like(grid, dtype=float)
    state_mass[np.argmin(np.abs(grid - start))] = 1.0

    upper_mass = np.zeros_like(bounds)
    lower_mass = np.zeros_like(bounds)
    for time_index in range(1, len(bounds)):
        current_bound = bounds[time_index - 1]
        next_bound = bounds[time_index]
        inside_current = np.abs(grid) < current_bound
        state_mass[~inside_current] = 0.0
        state_mass = np.nan_to_num(state_mass, nan=0.0, posinf=0.0, neginf=0.0)
        if state_mass.sum() <= 0:
            break

        active = inside_current & (state_mass > 0)
        active_grid = grid[active]
        active_mass = state_mass[active]
        drift = drift_grid[time_index - 1, active]
        noise = noise_grid[time_index - 1, active]

        sigmas = np.maximum(np.abs(noise) * np.sqrt(dt), 1e-12)
        means = active_grid + drift * dt
        lower_cross = ndtr((-next_bound - means) / sigmas)
        upper_cross = 1.0 - ndtr((next_bound - means) / sigmas)
        lower_mass[time_index] = float(np.dot(active_mass, lower_cross))
        upper_mass[time_index] = float(np.dot(active_mass, upper_cross))

        inside_next = np.abs(grid) < next_bound
        next_mass = np.zeros_like(state_mass)
        if np.any(inside_next):
            lower_edges = np.maximum(edges[:-1], -next_bound)
            upper_edges = np.minimum(edges[1:], next_bound)
            transition = ndtr((upper_edges[:, None] - means) / sigmas) - ndtr((lower_edges[:, None] - means) / sigmas)
            transition[upper_edges <= lower_edges, :] = 0.0
            transition = np.nan_to_num(transition, nan=0.0, posinf=0.0, neginf=0.0)
            transition = np.clip(transition, 0.0, None)
            next_mass = transition @ active_mass
            next_mass[~inside_next] = 0.0
        state_mass = next_mass
        if upper_mass[: time_index + 1].sum() + lower_mass[: time_index + 1].sum() > 0.999:
            break

    upper = upper_mass / dt
    lower = lower_mass / dt
    scale = (float(upper.sum()) + float(lower.sum())) * dt
    if np.isfinite(scale) and scale > 0:
        upper = upper / scale
        lower = lower / scale
    return upper, lower


def _scaled_drift(
    model: FokkerPlanckSpec,
    grid: np.ndarray,
    time: float,
    condition: str,
    bound: float,
    bound_derivative: float,
    params: tuple[float, ...],
) -> np.ndarray:
    physical_x = grid * bound
    if _component_ignores_x(model.drift):
        drift = np.full_like(grid, float(model.drift(time, condition, *params)), dtype=float)
    else:
        drift = np.array([model.drift(x, time, condition, *params) for x in physical_x], dtype=float)
    return (drift - bound_derivative * grid) / bound


def _scaled_noise(
    model: FokkerPlanckSpec,
    grid: np.ndarray,
    time: float,
    condition: str,
    bound: float,
    params: tuple[float, ...],
) -> float:
    physical_x = grid * bound
    if _component_ignores_x(model.noise):
        return max(abs(float(model.noise(time, condition, *params))) / bound, 1e-9)
    noise = np.array([abs(model.noise(x, time, condition, *params)) for x in physical_x], dtype=float)
    if np.nanmax(noise) - np.nanmin(noise) > 1e-8:
        raise ValueError("The current KFE solver supports time-varying scalar noise, but not x-dependent noise.")
    return max(float(np.nanmean(noise)) / bound, 1e-9)


def _component_ignores_x(component: Callable[..., float]) -> bool:
    try:
        parameters = list(inspect.signature(component).parameters)
    except (TypeError, ValueError):
        return False
    return not parameters or parameters[0] != "x"


def _component_has_unused_x(component: Callable[..., float]) -> bool:
    try:
        parameters = list(inspect.signature(component).parameters)
    except (TypeError, ValueError):
        return False
    if not parameters or parameters[0] != "x":
        return False
    return _source_omits_name(component, parameters[0])


def _component_needs_trial_context(component: Callable[..., float] | float) -> bool:
    if not callable(component):
        return False
    try:
        parameters = inspect.signature(component).parameters
    except (TypeError, ValueError):
        return False
    return "trial_row" in parameters or "covariates" in parameters


@lru_cache(maxsize=256)
def _source_omits_name(component: Callable[..., float], name: str) -> bool:
    try:
        source = textwrap.dedent(inspect.getsource(component))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return False

    bodies: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            bodies.extend(node.body)
            break
        if isinstance(node, ast.Lambda):
            bodies.append(node.body)
            break
    if not bodies:
        return False

    return not any(isinstance(node, ast.Name) and node.id == name for body in bodies for node in ast.walk(body))


def _crank_nicolson_step(
    density: np.ndarray,
    mu_old: np.ndarray,
    mu_new: np.ndarray,
    sigma_old: float,
    sigma_new: float,
    dtbase: float,
    dx: float,
) -> np.ndarray:
    tol = 1e-11
    accepted = density.copy()
    elapsed = 0.0
    dt = dtbase
    nsteps = 0
    while elapsed < dtbase - tol:
        candidate = _crank_nicolson_substep(accepted, mu_old, mu_new, sigma_old, sigma_new, dt, dx)
        if np.nanmin(candidate) >= -tol:
            accepted = np.clip(candidate, 0.0, None)
            accepted[0] = 0.0
            accepted[-1] = 0.0
            elapsed += dt
            dt = min(dtbase, 2.0 * dt, dtbase - elapsed)
        else:
            dt *= 0.125
        nsteps += 1
        if nsteps > 1000:
            raise ValueError("Extreme parameters: number of adaptive KFE steps exceeded.")
    return accepted


def _solve_kfe_arrays_python(
    initial_density: np.ndarray,
    mu_grid: np.ndarray,
    sigma_values: np.ndarray,
    dt: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    density = initial_density.copy()
    upper = np.zeros(len(sigma_values), dtype=float)
    lower = np.zeros(len(sigma_values), dtype=float)
    for time_index in range(1, len(sigma_values)):
        density[0] = 0.0
        density[-1] = 0.0
        if density.sum() * dx <= 0:
            break
        density = _crank_nicolson_step(
            density,
            mu_grid[time_index - 1],
            mu_grid[time_index],
            float(sigma_values[time_index - 1]),
            float(sigma_values[time_index]),
            dt,
            dx,
        )
        sigma_new = float(sigma_values[time_index])
        upper_density = 0.5 * sigma_new * sigma_new / (dx * dx) * (
            3.0 * density[-2] - 1.5 * density[-3] + (1.0 / 3.0) * density[-4]
        )
        lower_density = 0.5 * sigma_new * sigma_new / (dx * dx) * (
            3.0 * density[1] - 1.5 * density[2] + (1.0 / 3.0) * density[3]
        )
        upper[time_index] = max(float(upper_density), 0.0)
        lower[time_index] = max(float(lower_density), 0.0)
        if (upper[: time_index + 1].sum() + lower[: time_index + 1].sum()) * dt > 0.999:
            break

    scale = (float(upper.sum()) + float(lower.sum())) * dt
    if np.isfinite(scale) and scale > 0:
        upper = upper / scale
        lower = lower / scale
    return upper, lower


def _crank_nicolson_substep(
    density: np.ndarray,
    mu_old: np.ndarray,
    mu_new: np.ndarray,
    sigma_old: float,
    sigma_new: float,
    dt: float,
    dx: float,
) -> np.ndarray:
    n = len(density) - 1
    l_old = 0.5 * sigma_old * sigma_old / dx
    l_new = 0.5 * sigma_new * sigma_new / dx
    rhs = np.zeros_like(density)
    rhs[1:n] = dx * density[1:n] - 0.5 * dt * (
        (-l_old - 0.5 * mu_old[1:n]) * density[0 : n - 1]
        + (2.0 * l_old) * density[1:n]
        + (-l_old + 0.5 * mu_old[1:n]) * density[2 : n + 1]
    )

    lower = np.zeros_like(density)
    main = np.ones_like(density)
    upper = np.zeros_like(density)
    lower[1:n] = 0.5 * dt * (-l_new - 0.5 * mu_new[1:n])
    main[1:n] = dx + 0.5 * dt * (2.0 * l_new)
    upper[1:n] = 0.5 * dt * (-l_new + 0.5 * mu_new[1:n])

    banded = np.zeros((3, n + 1), dtype=float)
    banded[0, 1:] = upper[:-1]
    banded[1] = main
    banded[2, :-1] = lower[1:]
    return solve_banded((1, 1), banded, rhs, check_finite=False)


def _mass_bins(rt: np.ndarray, mass: np.ndarray, cutoffs: list[float], denominator: float) -> list[float]:
    bins: list[float] = []
    previous = None
    for cutoff in cutoffs:
        mask = rt <= cutoff if previous is None else (rt > previous) & (rt <= cutoff)
        bins.append(float(mass[mask].sum()) / denominator)
        previous = cutoff
    if previous is None:
        bins.append(float(mass.sum()) / denominator)
    else:
        bins.append(float(mass[rt > previous].sum()) / denominator)
    return bins
