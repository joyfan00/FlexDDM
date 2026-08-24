from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Parameter:
    name: str
    lower: float
    upper: float | str

    def bounds_for_data(self, data: pd.DataFrame | None = None) -> tuple[float, float]:
        if isinstance(self.upper, str):
            if data is None or data.empty:
                raise ValueError(f"Parameter {self.name!r} uses upper={self.upper!r} but no data were supplied.")
            constants = data.attrs.get("bound_constants", {})
            if self.upper in constants:
                upper = float(constants[self.upper])
            elif self.upper == "min_rt":
                upper = float(data["rt"].min())
            else:
                available = ", ".join(sorted(set(constants).union({"min_rt"})))
                raise ValueError(f"Unknown upper-bound symbol {self.upper!r} for {self.name!r}. Available symbols: {available}.")
        else:
            upper = float(self.upper)
        lower = float(self.lower)
        if upper <= lower:
            raise ValueError(f"Invalid bounds for {self.name!r}: lower={lower}, upper={upper}")
        return lower, upper


@dataclass(frozen=True)
class SimulationResult:
    trial: Sequence[int]
    accuracy: Sequence[float]
    rt: Sequence[float]
    condition: Sequence[object]

    @classmethod
    def from_legacy_tuple(cls, result: tuple[Sequence[int], Sequence[float], Sequence[float], Sequence[object]]) -> "SimulationResult":
        return cls(trial=result[0], accuracy=result[1], rt=result[2], condition=result[3])

    def to_dataframe(self, participant_id: object = "simulated") -> pd.DataFrame:
        condition = [_condition_to_string(value) for value in self.condition]
        return pd.DataFrame(
            {
                "id": participant_id,
                "trial": list(self.trial),
                "accuracy": np.asarray(self.accuracy, dtype=float).round().astype(int),
                "rt": np.asarray(self.rt, dtype=float),
                "condition": condition,
            }
        )


@dataclass(frozen=True)
class SimulationRuntime:
    trials: int
    dt: float = 0.01
    noise_sd: float = 0.01
    seed: int = 50
    max_time: float = 10.0
    design: pd.DataFrame | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameters: tuple[Parameter, ...]
    simulate: Callable[..., SimulationResult | tuple[Sequence[int], Sequence[float], Sequence[float], Sequence[object]]]
    default_trials: int = 800
    conditions: tuple[str, ...] | None = None

    @property
    def parameter_names(self) -> list[str]:
        return [parameter.name for parameter in self.parameters]

    def bounds_for_data(self, data: pd.DataFrame | None = None) -> list[tuple[float, float]]:
        return [parameter.bounds_for_data(data) for parameter in self.parameters]

    def simulate_dataframe(
        self,
        values: Iterable[float],
        n_trials: int | None = None,
        participant_id: object = "simulated",
        dt: float = 0.01,
        noise_sd: float = 0.01,
        seed: int = 50,
        data_like: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        trials = n_trials or (len(data_like) if data_like is not None else self.default_trials)
        runtime = SimulationRuntime(trials=trials, dt=dt, noise_sd=noise_sd, seed=seed, design=data_like)
        result = _call_simulate(self.simulate, tuple(values), runtime)
        if not isinstance(result, SimulationResult):
            result = SimulationResult.from_legacy_tuple(result)
        return result.to_dataframe(participant_id=participant_id)


def _call_simulate(simulate: Callable, values: tuple[float, ...], runtime: SimulationRuntime):
    try:
        parameters = inspect.signature(simulate).parameters
    except (TypeError, ValueError):
        return simulate(*values, nTrials=runtime.trials)
    if "runtime" in parameters:
        return simulate(*values, runtime=runtime)
    kwargs = {}
    if "nTrials" in parameters:
        kwargs["nTrials"] = runtime.trials
    elif "n_trials" in parameters:
        kwargs["n_trials"] = runtime.trials
    if "dt" in parameters:
        kwargs["dt"] = runtime.dt
    if "var" in parameters:
        kwargs["var"] = runtime.noise_sd
    elif "noise_sd" in parameters:
        kwargs["noise_sd"] = runtime.noise_sd
    if "noiseseed" in parameters:
        kwargs["noiseseed"] = runtime.seed
    elif "seed" in parameters:
        kwargs["seed"] = runtime.seed
    return simulate(*values, **kwargs)


def _condition_to_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    array = np.asarray(value)
    if array.ndim > 0 and array.size > 1:
        return "-".join(str(int(x)) if float(x).is_integer() else str(x) for x in array.ravel())
    if array.ndim > 0 and array.size == 1:
        item = array.item()
        return str(int(item)) if isinstance(item, (int, float, np.integer, np.floating)) and float(item).is_integer() else str(item)
    return str(value)
