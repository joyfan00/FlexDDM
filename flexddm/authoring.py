from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Sequence

from .model import Parameter
from .process_model import ProcessModelSpec


def param(name: str, lower: float, upper: float | str) -> Parameter:
    return Parameter(name, lower, upper)


def parameters_from_bounds(parameter_bounds: dict[str, tuple[float, float | str] | list[float | str]] | None = None) -> tuple[Parameter, ...]:
    if parameter_bounds is None:
        caller_namespace = _caller_namespace()
        parameter_bounds = caller_namespace.get("PARAMETER_BOUNDS") or caller_namespace.get("parameter_bounds")
        if not isinstance(parameter_bounds, dict) or not parameter_bounds:
            simulate = caller_namespace.get("simulate")
            names = _component_parameter_names(simulate)
            parameter_bounds = {name: _default_bounds_for_name(name) for name in names}
    return tuple(Parameter(name, bounds[0], bounds[1]) for name, bounds in parameter_bounds.items())


def define_model(
    name: str,
    conditions: Iterable[str],
    parameters: Sequence[Parameter | tuple[str, float, float | str]] | None = None,
    drift: Callable[..., float] | None = None,
    bound: Callable[..., float] | str | float | None = None,
    noise: Callable[..., float] | str | float = 0.1,
    nondecision: Callable[..., float] | str | float = 0.0,
    starting_position: Callable[..., float] | str | float | None = None,
    default_trials: int = 800,
) -> ProcessModelSpec:
    caller_namespace = _caller_namespace()

    drift_component = drift or caller_namespace.get("drift")
    if drift_component is None:
        raise ValueError("define_model() needs a drift function or a function named drift(...) in the model file.")

    bound_component = bound if bound is not None else caller_namespace.get("bound")
    start_component = starting_position if starting_position is not None else caller_namespace.get("starting_position")

    if parameters is None:
        parameter_bounds = caller_namespace.get("PARAMETER_BOUNDS") or caller_namespace.get("parameter_bounds")
        if not isinstance(parameter_bounds, dict) or not parameter_bounds:
            parameter_names = _infer_parameter_names(
                drift_component,
                bound_component,
                noise,
                nondecision,
                start_component,
                known_bounds=None,
            )
            parameter_bounds = {name: _default_bounds_for_name(name) for name in parameter_names}
        else:
            parameter_names = _infer_parameter_names(
                drift_component,
                bound_component,
                noise,
                nondecision,
                start_component,
                known_bounds=parameter_bounds,
            )
        parameters = [(name, parameter_bounds[name][0], parameter_bounds[name][1]) for name in parameter_names]

    parameter_tuple = tuple(item if isinstance(item, Parameter) else Parameter(*item) for item in parameters)
    parameter_index = {parameter.name: index for index, parameter in enumerate(parameter_tuple)}

    if bound_component is None:
        bound_component = _default_bound(parameter_index)
    if start_component is None:
        start_component = _default_start(parameter_index)

    return ProcessModelSpec(
        name=name,
        parameters=parameter_tuple,
        conditions=tuple(str(condition) for condition in conditions),
        drift=drift_component,
        bound=_component_function(bound_component, parameter_index),
        noise=_component_function(noise, parameter_index),
        nondecision=_component_function(nondecision, parameter_index, needs_x=False),
        starting_position=_component_function(start_component, parameter_index, needs_x=False),
        default_trials=default_trials,
    )


def _caller_namespace() -> dict[str, object]:
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return {}
    caller = frame.f_back.f_back
    return {**caller.f_globals, **caller.f_locals}


def _infer_parameter_names(*components, known_bounds: dict[str, object] | None = None) -> list[str]:
    names: list[str] = []
    for component in components:
        for name in _component_parameter_names(component):
            if (known_bounds is None or name in known_bounds) and name not in names:
                names.append(name)
    if known_bounds is not None:
        for name in known_bounds:
            if name not in names:
                names.append(name)
    return names


def _default_bounds_for_name(name: str) -> tuple[float, float | str]:
    if name == "alpha" or name.endswith("_alpha"):
        return (0.07, 0.38)
    if name == "beta" or name.endswith("_beta"):
        return (0.0, 1.0)
    if name == "tau" or "nondecision" in name:
        return (0.15, "min_rt")
    if "shape" in name:
        return (0.5, 5.0)
    if "time" in name or "rate" in name:
        return (0.01, 2.0)
    if "noise" in name or "sd" in name:
        return (0.01, 2.0)
    return (-1.0, 1.0)


def _component_parameter_names(component) -> list[str]:
    if isinstance(component, str):
        return [component]
    if not callable(component):
        return []
    try:
        parameters = inspect.signature(component).parameters
    except (TypeError, ValueError):
        return []
    ignored = {
        "self",
        "x",
        "t",
        "time",
        "condition",
        "conditions",
        "runtime",
        "trial_row",
        "covariates",
        "trial_index",
        "q_left",
        "q_right",
        "choice",
        "reward",
    }
    names: list[str] = []
    for name, parameter in parameters.items():
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if name not in ignored:
            names.append(name)
    return names


def _component_function(component, parameter_index: dict[str, int], needs_x: bool = True):
    if callable(component):
        return component
    if isinstance(component, str):
        if component not in parameter_index:
            raise ValueError(f"Unknown parameter name {component!r}.")
        index = parameter_index[component]

        if needs_x:
            def by_parameter(x, t, condition, *params):
                return params[index]
        else:
            def by_parameter(t, condition, *params):
                return params[index]

        return by_parameter

    value = float(component)

    if needs_x:
        def constant(x, t, condition, *params):
            return value
    else:
        def constant(t, condition, *params):
            return value

    return constant


def _default_bound(parameter_index: dict[str, int]):
    if "alpha" not in parameter_index:
        raise ValueError("No bound(...) function was provided, and no alpha parameter exists for the default alpha/2 boundary.")
    alpha_index = parameter_index["alpha"]

    def bound(t, condition, *params):
        return params[alpha_index] / 2

    return bound


def _default_start(parameter_index: dict[str, int]):
    if "alpha" in parameter_index and "beta" in parameter_index:
        alpha_index = parameter_index["alpha"]
        beta_index = parameter_index["beta"]

        def beta_start(t, condition, *params):
            alpha = params[alpha_index]
            beta = params[beta_index]
            return (beta - 0.5) * alpha

        return beta_start

    def zero_start(t, condition, *params):
        return 0.0

    return zero_start
