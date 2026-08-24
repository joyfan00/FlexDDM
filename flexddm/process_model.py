from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable, Iterable

import numpy as np

from .fokker_planck import FokkerPlanckSpec, _component_has_unused_x, _component_ignores_x, _physical_drift, _physical_noise
from .model import ModelSpec, Parameter, SimulationResult

try:
    from ._cpp import simulate_scalar_process as _cpp_simulate_scalar_process

    HAS_CPP_SIMULATION = True
except Exception:  # pragma: no cover - exercised when the extension is not built
    _cpp_simulate_scalar_process = None
    HAS_CPP_SIMULATION = False


@dataclass(frozen=True)
class ProcessModelSpec:
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

    def to_simulation_model(self) -> ModelSpec:
        return ModelSpec(
            name=self.name,
            parameters=self.parameters,
            simulate=self.simulate,
            default_trials=self.default_trials,
            conditions=self.conditions,
        )

    def to_fokker_planck_model(self) -> FokkerPlanckSpec:
        return FokkerPlanckSpec(
            name=self.name,
            parameters=self.parameters,
            conditions=self.conditions,
            drift=self.drift,
            bound=self.bound,
            noise=self.noise,
            nondecision=self.nondecision,
            starting_position=self.starting_position,
            default_trials=self.default_trials,
        )

    def simulate(self, *values: float, nTrials: int | None = None, dt: float = 0.01, noiseseed: int = 50, runtime=None) -> SimulationResult:
        params = tuple(values)
        if runtime is not None:
            n_trials = runtime.trials
            dt = runtime.dt
            noiseseed = runtime.seed
            design = runtime.design
        else:
            n_trials = nTrials or self.default_trials
            design = None
        compiled = _try_compiled_scalar_simulation(self, params, n_trials, dt, noiseseed)
        if compiled is not None:
            return compiled

        rng = np.random.default_rng(noiseseed)
        trial = np.arange(1, n_trials + 1)
        accuracy = np.full(n_trials, np.nan)
        rt = np.full(n_trials, np.nan)
        condition = np.empty(n_trials, dtype=object)

        for index in range(n_trials):
            trial_row = _trial_row(design, index)
            condition_value = str(trial_row.get("condition", self.conditions[index % len(self.conditions)])) if trial_row is not None else self.conditions[index % len(self.conditions)]
            condition[index] = condition_value
            time = 0.0
            evidence = _call_or_value(self.starting_position, time, condition_value, *params, trial_row=trial_row)
            nondecision = _call_or_value(self.nondecision, time, condition_value, *params, trial_row=trial_row)

            while time < 10:
                boundary = abs(float(_call_or_value(self.bound, time, condition_value, *params, trial_row=trial_row)))
                if evidence >= boundary:
                    accuracy[index] = 1
                    rt[index] = time + nondecision
                    break
                if evidence <= -boundary:
                    accuracy[index] = 0
                    rt[index] = time + nondecision
                    break
                evidence += _call_process(self.drift, evidence, time, condition_value, *params, trial_row=trial_row) * dt
                evidence += abs(_call_process(self.noise, evidence, time, condition_value, *params, trial_row=trial_row)) * np.sqrt(dt) * rng.normal()
                time += dt

            if np.isnan(rt[index]):
                boundary = abs(float(_call_or_value(self.bound, time, condition_value, *params, trial_row=trial_row)))
                accuracy[index] = 1 if evidence >= 0 else 0
                rt[index] = time + nondecision

        return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


def _trial_row(design, index: int) -> dict[str, object] | None:
    if design is None or len(design) == 0:
        return None
    row = design.iloc[index % len(design)]
    return row.to_dict() if hasattr(row, "to_dict") else dict(row)


def _call_or_value(value: Callable[..., float] | float, *args: object, trial_row: dict[str, object] | None = None) -> float:
    return float(_call_with_optional_trial_context(value, *args, trial_row=trial_row) if callable(value) else value)


def _call_process(value: Callable[..., float], x: float, t: float, condition: str, *params: float, trial_row: dict[str, object] | None = None) -> float:
    try:
        parameter_names = list(inspect.signature(value).parameters)
    except (TypeError, ValueError):
        parameter_names = ["x"]
    if parameter_names and parameter_names[0] == "x":
        return float(_call_with_optional_trial_context(value, x, t, condition, *params, trial_row=trial_row))
    return float(_call_with_optional_trial_context(value, t, condition, *params, trial_row=trial_row))


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


def _try_compiled_scalar_simulation(
    model: ProcessModelSpec,
    params: tuple[float, ...],
    n_trials: int,
    dt: float,
    seed: int,
) -> SimulationResult | None:
    if not HAS_CPP_SIMULATION or _cpp_simulate_scalar_process is None:
        return None
    if any(_component_needs_trial_context(component) for component in (model.drift, model.bound, model.noise, model.starting_position, model.nondecision)):
        return None

    times = np.arange(0.0, 10.0 + dt, dt)
    drift = np.empty((len(model.conditions), len(times)), dtype=float)
    bound = np.empty_like(drift)
    noise = np.empty_like(drift)
    start = np.empty(len(model.conditions), dtype=float)
    nondecision = np.empty(len(model.conditions), dtype=float)
    probe = np.array([-0.5, 0.0, 0.5], dtype=float)
    drift_static = _component_has_unused_x(model.drift)
    noise_static = _component_has_unused_x(model.noise)

    for condition_index, condition in enumerate(model.conditions):
        start[condition_index] = _call_or_value(model.starting_position, 0.0, condition, *params)
        nondecision[condition_index] = _call_or_value(model.nondecision, 0.0, condition, *params)
        for time_index, time in enumerate(times):
            boundary = abs(float(model.bound(float(time), condition, *params)))
            if not np.isfinite(boundary) or boundary <= 0:
                return None
            bound[condition_index, time_index] = boundary

            if _component_ignores_x(model.drift):
                drift[condition_index, time_index] = float(model.drift(float(time), condition, *params))
            elif drift_static:
                drift[condition_index, time_index] = float(model.drift(0.0, float(time), condition, *params))
            else:
                drift_probe = _physical_drift(model.to_fokker_planck_model(), probe * boundary, float(time), condition, params)
                if np.nanmax(drift_probe) - np.nanmin(drift_probe) > 1e-8:
                    return None
                drift[condition_index, time_index] = float(np.nanmean(drift_probe))

            if _component_ignores_x(model.noise):
                noise[condition_index, time_index] = abs(float(model.noise(float(time), condition, *params)))
            elif noise_static:
                noise[condition_index, time_index] = abs(float(model.noise(0.0, float(time), condition, *params)))
            else:
                noise_probe = _physical_noise(model.to_fokker_planck_model(), probe * boundary, float(time), condition, params)
                if np.nanmax(noise_probe) - np.nanmin(noise_probe) > 1e-8:
                    return None
                noise[condition_index, time_index] = float(np.nanmean(np.abs(noise_probe)))

    if (
        not np.all(np.isfinite(drift))
        or not np.all(np.isfinite(bound))
        or not np.all(np.isfinite(noise))
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(nondecision))
    ):
        return None

    trial, accuracy, rt, condition_index = _cpp_simulate_scalar_process(
        np.ascontiguousarray(drift, dtype=float),
        np.ascontiguousarray(bound, dtype=float),
        np.ascontiguousarray(noise, dtype=float),
        np.ascontiguousarray(start, dtype=float),
        np.ascontiguousarray(nondecision, dtype=float),
        int(n_trials),
        float(dt),
        int(seed),
        10.0,
    )
    conditions = np.array([model.conditions[int(index)] for index in condition_index], dtype=object)
    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=conditions)


def _component_needs_trial_context(component: Callable[..., float] | float) -> bool:
    if not callable(component):
        return False
    try:
        parameters = inspect.signature(component).parameters
    except (TypeError, ValueError):
        return False
    return "trial_row" in parameters or "covariates" in parameters
