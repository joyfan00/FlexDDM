from __future__ import annotations

from pathlib import Path
import inspect
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from .model import ModelSpec
from .process_model import ProcessModelSpec, _call_or_value, _call_process


def plot_posterior_predictive(
    observed: pd.DataFrame,
    simulated: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Posterior Predictive Check",
) -> plt.Figure:
    observed = observed.copy()
    simulated = simulated.copy()
    observed["condition"] = observed["condition"].astype(str)
    simulated["condition"] = simulated["condition"].astype(str)
    response_kind = str(observed.attrs.get("response_kind", "accuracy"))
    response_column = str(observed.attrs.get("response_column", "response"))

    conditions = sorted(set(observed["condition"]).union(set(simulated["condition"])))
    accuracies = sorted(set(observed["accuracy"].dropna().astype(int)).union(set(simulated["accuracy"].dropna().astype(int))))
    if not accuracies:
        accuracies = [0, 1]

    fig, axes = plt.subplots(
        max(len(conditions), 1),
        max(len(accuracies), 1),
        figsize=(11.0, 7.5),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    palette = {"Observed": "#0072B2", "Simulated": "#CC79A7"}

    for row_index, condition in enumerate(conditions or ["all"]):
        for col_index, accuracy in enumerate(accuracies):
            ax = axes[row_index, col_index]
            observed_values = observed.loc[(observed["condition"] == condition) & (observed["accuracy"].astype(int) == accuracy), "rt"]
            simulated_values = simulated.loc[(simulated["condition"] == condition) & (simulated["accuracy"].astype(int) == accuracy), "rt"]
            _draw_distribution(ax, observed_values, "Observed", palette["Observed"])
            _draw_distribution(ax, simulated_values, "Simulated", palette["Simulated"])
            ax.set_title(_wrap_label(f"{condition} | {_response_value_label(accuracy, response_kind, response_column)}", 36), fontsize=8)
            ax.set_xlabel("Response time (s)")
            ax.set_ylabel("Density")
            ax.text(
                0.98,
                0.96,
                f"Observed N = {len(observed_values)}\nSimulated N = {len(simulated_values)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#d0d7de", "pad": 3},
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    if title:
        fig.suptitle(title, y=0.985, fontsize=13)
        fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.93))
    else:
        fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.98))

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    return fig


def _response_value_label(value: int, response_kind: str, response_column: str) -> str:
    if response_kind == "accuracy":
        return "Correct" if int(value) == 1 else "Incorrect"
    return f"{response_column} = {int(value)}"


def _wrap_label(value: object, width: int = 24, max_chars: int = 80) -> str:
    text = str(value)
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "..."
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def _draw_distribution(ax: plt.Axes, values: pd.Series, label: str, color: str) -> None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) >= 2 and np.nanstd(values) > 0:
        sns.kdeplot(values, ax=ax, color=color, label=label, warn_singular=False)
    elif len(values) == 1:
        ax.axvline(float(values.iloc[0]), color=color, linewidth=1.8, label=label)
        ax.plot([float(values.iloc[0])], [0], marker="o", color=color, markersize=4)
    else:
        ax.plot([], [], color=color, label=label)


def plot_model_schematic(
    model: object,
    output_path: str | Path | None = None,
    title: str = "Model Schematic",
    trial_row: dict[str, object] | None = None,
    parameter_values: dict[str, float] | None = None,
) -> plt.Figure:
    """Draw a compact teaching schematic from process functions or a simulator."""
    if hasattr(model, "drift") and hasattr(model, "bound"):
        fig = _plot_process_schematic(model, title=title, trial_row=trial_row, parameter_values=parameter_values)
    elif isinstance(model, ModelSpec):
        fig = _plot_simulator_schematic(model, title=title)
    else:
        raise ValueError("Model schematic needs a ProcessModelSpec, FokkerPlanckSpec, or ModelSpec.")

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        plt.close(fig)
    return fig


def model_schematic_data(
    model: object,
    trial_row: dict[str, object] | None = None,
    parameter_values: dict[str, float] | None = None,
    t_max: float = 1.5,
) -> dict[str, object]:
    if hasattr(model, "racing_drifts"):
        return _racing_schematic_data(model, trial_row=trial_row, parameter_values=parameter_values, t_max=t_max)
    if not hasattr(model, "drift") or not hasattr(model, "bound"):
        return {"mode": "simulation", "parameters": _parameter_controls(model, parameter_values), "conditions": []}

    params, controls = _parameter_values_and_controls(model, parameter_values)
    conditions = tuple(str(condition) for condition in getattr(model, "conditions", ("0",))) or ("0",)
    times = np.linspace(0.0, t_max, 130)
    first_condition = conditions[0]
    first_bounds = _boundary_series(model, times, first_condition, params, trial_row=trial_row)
    if not np.any(np.isfinite(first_bounds)):
        first_bounds = np.full_like(times, 0.15)
    first_bounds = np.maximum(np.abs(first_bounds), 1e-4)
    lower_bounds = -first_bounds
    start = _safe_component_value(model, "starting_position", 0.0, first_condition, params, trial_row=trial_row, default=0.0)
    paths = []
    all_y = [float(np.nanmax(first_bounds)), float(np.nanmin(lower_bounds)), float(start)]
    for condition in conditions[:4]:
        path = _evidence_path(model, times, condition, params, trial_row=trial_row, noisy=False)
        x_values = times[: len(path)]
        all_y.extend(path)
        paths.append({"condition": condition, "x": x_values.tolist(), "y": path})
    y_limit = max(max(abs(value) for value in all_y if np.isfinite(value)), 0.18) * 1.28
    drift_label = _drift_label_position(np.asarray(paths[0]["x"], dtype=float), paths[0]["y"] if paths else [start], y_limit)
    return {
        "mode": "process",
        "parameters": controls,
        "conditions": list(conditions[:4]),
        "times": times.tolist(),
        "upper": first_bounds.tolist(),
        "lower": lower_bounds.tolist(),
        "start": float(start),
        "paths": paths,
        "yMin": -float(y_limit),
        "yMax": float(y_limit),
        "xMin": float(times[0]),
        "xMax": float(times[-1]),
        "driftLabel": drift_label,
    }


def model_trajectory_data(
    model: object,
    trial_row: dict[str, object] | None = None,
    parameter_values: dict[str, float] | None = None,
    condition: str | None = None,
    seed: int = 1,
    t_max: float = 1.5,
) -> dict[str, object]:
    if not hasattr(model, "drift") or not hasattr(model, "bound"):
        raise ValueError("Animated evidence trajectories require a process-readable model with drift and boundary functions.")
    params, _ = _parameter_values_and_controls(model, parameter_values)
    conditions = tuple(str(item) for item in getattr(model, "conditions", ("0",))) or ("0",)
    selected = str(condition) if condition is not None and str(condition) in conditions else conditions[0]
    times = np.linspace(0.0, t_max, 130)
    path = _evidence_path(model, times, selected, params, trial_row=trial_row, noisy=True, seed=seed)
    return {"condition": selected, "x": times[: len(path)].tolist(), "y": path}


def _racing_schematic_data(
    model: object,
    trial_row: dict[str, object] | None = None,
    parameter_values: dict[str, float] | None = None,
    t_max: float = 1.5,
) -> dict[str, object]:
    params, controls = _parameter_values_and_controls(model, parameter_values)
    conditions = tuple(str(condition) for condition in getattr(model, "conditions", ("0",))) or ("0",)
    condition = conditions[0]
    times = np.linspace(0.0, t_max, 130)
    bound_component = getattr(model, "racing_bound", None)
    nondecision_component = getattr(model, "racing_nondecision", None)
    nondecision = max(_component_or_default(nondecision_component, model, 0.0, condition, params, trial_row=trial_row, default=0.0), 0.0)
    upper = []
    paths = []
    all_y = [0.0]
    for time_value in times:
        decision_time = max(float(time_value) - nondecision, 0.0)
        upper.append(abs(_component_or_default(bound_component, model, decision_time, condition, params, trial_row=trial_row, default=_default_alpha_bound(model, params))))
    upper = np.maximum(np.asarray(upper, dtype=float), 1e-4)
    drifts = tuple(getattr(model, "racing_drifts", ()))
    for index, drift_function in enumerate(drifts[:4]):
        evidence = 0.0
        path = [evidence]
        for time_index, time_value in enumerate(times[:-1]):
            next_time = float(times[time_index + 1])
            if next_time <= nondecision:
                path.append(evidence)
                continue
            decision_time = max(float(time_value) - nondecision, 0.0)
            step_dt = max(next_time - max(float(time_value), nondecision), 0.0)
            drift = _call_racing_drift(drift_function, model, evidence, decision_time, condition, params, trial_row=trial_row)
            evidence = float(evidence + drift * step_dt)
            boundary = float(upper[time_index + 1])
            if evidence >= boundary:
                path.append(boundary)
                break
            path.append(evidence)
        all_y.extend(path)
        label = "response 1 accumulator" if index == 0 else f"response {index + 1} accumulator"
        paths.append({"condition": label, "x": times[: len(path)].tolist(), "y": path})
    y_limit = max(max(abs(value) for value in all_y if np.isfinite(value)), float(np.nanmax(upper)), 0.18) * 1.28
    drift_label = _drift_label_position(np.asarray(paths[0]["x"], dtype=float), paths[0]["y"] if paths else [0.0], y_limit)
    return {
        "mode": "process",
        "schemaKind": "racing",
        "parameters": controls,
        "conditions": [path["condition"] for path in paths],
        "times": times.tolist(),
        "upper": upper.tolist(),
        "lower": np.zeros_like(upper).tolist(),
        "start": 0.0,
        "paths": paths,
        "yMin": 0.0,
        "yMax": float(y_limit),
        "xMin": float(times[0]),
        "xMax": float(times[-1]),
        "driftLabel": drift_label,
    }


def _component_or_default(component, model: object, t: float, condition: str, params: list[float], trial_row: dict[str, object] | None, default: float) -> float:
    if component is None:
        return float(default)


def _call_racing_drift(component, model: object, x: float, t: float, condition: str, params: list[float], trial_row: dict[str, object] | None) -> float:
    try:
        signature = inspect.signature(component).parameters
        names = list(signature)
        parameter_names = [str(getattr(parameter, "name", "")) for parameter in getattr(model, "parameters", ())]
        named = {name: params[index] for index, name in enumerate(parameter_names) if index < len(params) and name in signature}
        if "trial_row" in signature:
            named["trial_row"] = trial_row or {}
        if "covariates" in signature:
            named["covariates"] = trial_row or {}
        if names and names[0] == "x":
            return float(component(x, t, condition, **named))
        return float(component(t, condition, **named))
    except Exception:
        return 0.0
    try:
        signature = inspect.signature(component).parameters
        parameter_names = [str(getattr(parameter, "name", "")) for parameter in getattr(model, "parameters", ())]
        named = {name: params[index] for index, name in enumerate(parameter_names) if index < len(params) and name in signature}
        if "trial_row" in signature:
            named["trial_row"] = trial_row or {}
        return float(component(t, condition, **named))
    except Exception:
        return float(default)


def _default_alpha_bound(model: object, params: list[float]) -> float:
    for index, parameter in enumerate(getattr(model, "parameters", ())):
        if str(getattr(parameter, "name", "")) == "alpha" and index < len(params):
            return abs(float(params[index])) / 2
    return 0.15


def _plot_process_schematic(model: object, title: str, trial_row: dict[str, object] | None = None, parameter_values: dict[str, float] | None = None) -> plt.Figure:
    data = model_schematic_data(model, trial_row=trial_row, parameter_values=parameter_values)
    times = np.asarray(data["times"], dtype=float)
    colors = ["#0072B2", "#CC79A7", "#0f766e"]

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(times, data["upper"], color="#18212f", linewidth=2.0)
    ax.plot(times, data["lower"], color="#18212f", linewidth=2.0)
    ax.axhline(0.0, color="#5c6674", linewidth=1.1, linestyle="--")
    for condition_index, path in enumerate(data["paths"]):
        ax.plot(path["x"], path["y"], color=colors[condition_index % len(colors)], linewidth=1.8, alpha=0.92, label=path["condition"])

    ax.set_ylim(data["yMin"], data["yMax"])
    ax.set_xlim(data["xMin"], data["xMax"])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Evidence")
    ax.text(times[-1], data["upper"][-1], " upper boundary", va="center", ha="left", fontsize=8)
    ax.text(times[-1], data["lower"][-1], " lower boundary", va="center", ha="left", fontsize=8)
    ax.text(times[4], data["start"], "z start", va="bottom", ha="left", fontsize=8, color="#5c6674")
    drift = data["driftLabel"]
    ax.annotate(
        "drift(t)",
        xy=(drift["x"], drift["y"]),
        xytext=(drift["labelX"], drift["labelY"]),
        arrowprops={"arrowstyle": "->", "color": "#5c6674", "linewidth": 1},
        fontsize=8,
        color="#18212f",
    )
    if len(data["paths"]) > 1:
        ax.legend(frameon=False, fontsize=7, title="Condition", loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=min(len(data["paths"]), 3))
    ax.grid(True, color="#eef1f5", linewidth=0.8)
    fig.tight_layout()
    return fig


def _evidence_path(
    model: object,
    times: np.ndarray,
    condition: str,
    params: list[float],
    trial_row: dict[str, object] | None,
    noisy: bool = False,
    seed: int = 1,
) -> list[float]:
    rng = np.random.default_rng(seed)
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.01
    evidence = _safe_component_value(model, "starting_position", 0.0, condition, params, trial_row=trial_row, default=0.0)
    nondecision = max(_safe_component_value(model, "nondecision", 0.0, condition, params, trial_row=trial_row, default=0.0), 0.0)
    path = [float(evidence)]
    drift_params = _params_without_tau(model, params)
    for index, time_value in enumerate(times[:-1]):
        current_total_time = float(time_value)
        next_total_time = float(times[index + 1])
        if next_total_time <= nondecision:
            path.append(float(evidence))
            continue
        decision_time = max(current_total_time - nondecision, 0.0)
        next_decision_time = max(next_total_time - nondecision, 0.0)
        step_dt = max(next_total_time - max(current_total_time, nondecision), 0.0)
        drift = _safe_process_value(model, "drift", evidence, decision_time, condition, drift_params, trial_row=trial_row, default=0.0)
        step = drift * step_dt
        if noisy:
            noise = abs(_safe_process_value(model, "noise", evidence, decision_time, condition, drift_params, trial_row=trial_row, default=0.1))
            step += rng.normal() * noise * np.sqrt(step_dt)
        evidence = float(evidence + step)
        boundary = abs(_safe_component_value(model, "bound", next_decision_time, condition, drift_params, trial_row=trial_row, default=max(abs(evidence), 0.15)))
        if evidence >= boundary:
            path.append(float(boundary))
            break
        if evidence <= -boundary:
            path.append(float(-boundary))
            break
        path.append(evidence)
    return path


def _boundary_series(model: object, times: np.ndarray, condition: str, params: list[float], trial_row: dict[str, object] | None) -> np.ndarray:
    nondecision = max(_safe_component_value(model, "nondecision", 0.0, condition, params, trial_row=trial_row, default=0.0), 0.0)
    component_params = _params_without_tau(model, params)
    values = []
    for time_value in times:
        decision_time = max(float(time_value) - nondecision, 0.0)
        values.append(abs(_safe_component_value(model, "bound", decision_time, condition, component_params, trial_row=trial_row, default=np.nan)))
    return np.asarray(values, dtype=float)


def _drift_label_position(times: np.ndarray, path: list[float], y_limit: float) -> dict[str, float]:
    if len(path) <= 2 or len(times) <= 2:
        return {"x": 0.0, "y": float(path[0] if path else 0.0), "labelX": 0.05, "labelY": 0.35 * y_limit}
    index = min(max(int(len(path) * 0.62), 1), len(path) - 2)
    y_value = float(path[index])
    direction = -1.0 if y_value > 0 else 1.0
    return {
        "x": float(times[index]),
        "y": y_value,
        "labelX": float(times[max(index - 18, 0)]),
        "labelY": float(np.clip(y_value + direction * 0.32 * y_limit, -0.85 * y_limit, 0.85 * y_limit)),
    }


def _params_without_tau(model: object, params: list[float]) -> list[float]:
    adjusted = list(params)
    for index, parameter in enumerate(getattr(model, "parameters", ())):
        if str(getattr(parameter, "name", "")) == "tau":
            adjusted[index] = 0.0
    return adjusted


def _plot_simulator_schematic(model: ModelSpec, title: str) -> plt.Figure:
    params = _midpoint_values(model)
    simulated = model.simulate_dataframe(params, n_trials=min(max(model.default_trials, 40), 120), seed=42)
    simulated["rt"] = pd.to_numeric(simulated["rt"], errors="coerce")
    simulated["accuracy"] = pd.to_numeric(simulated["accuracy"], errors="coerce")
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    response_colors = np.where(simulated["accuracy"].fillna(0).astype(int).to_numpy() == 1, "#0072B2", "#CC79A7")
    ax.scatter(simulated["rt"], simulated["accuracy"], c=response_colors, alpha=0.72, edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Simulated response time")
    ax.set_ylabel("Boundary / response")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["lower / response 0", "upper / response 1"])
    ax.annotate(
        "opaque custom simulator",
        xy=(0.65, 0.72),
        xycoords="axes fraction",
        xytext=(0.36, 0.88),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#5c6674", "linewidth": 1},
        fontsize=8,
    )
    ax.grid(True, color="#eef1f5", linewidth=0.8)
    fig.tight_layout()
    return fig


def _midpoint_values(model: object) -> list[float]:
    values, _ = _parameter_values_and_controls(model, None)
    return values


def _parameter_controls(model: object, parameter_values: dict[str, float] | None = None) -> list[dict[str, float | str]]:
    _, controls = _parameter_values_and_controls(model, parameter_values)
    return controls


def _parameter_values_and_controls(model: object, parameter_values: dict[str, float] | None = None) -> tuple[list[float], list[dict[str, float | str]]]:
    values: list[float] = []
    controls: list[dict[str, float | str]] = []
    fallback = pd.DataFrame({"rt": [0.45]})
    fallback.attrs["bound_constants"] = {"min_rt": 0.45}
    for parameter in getattr(model, "parameters", ()):
        try:
            lower, upper = parameter.bounds_for_data(fallback)
        except Exception:
            lower, upper = float(parameter.lower), float(parameter.lower) + 1.0
        candidate = None
        if parameter_values and parameter.name in parameter_values:
            try:
                candidate = float(parameter_values[parameter.name])
            except (TypeError, ValueError):
                candidate = None
        value = float((lower + upper) / 2.0) if candidate is None else float(np.clip(candidate, lower, upper))
        values.append(value)
        controls.append({"name": parameter.name, "lower": float(lower), "upper": float(upper), "value": value})
    return values, controls


def _component_series(model: object, component_name: str, times: np.ndarray, condition: str, params: list[float], trial_row: dict[str, object] | None) -> np.ndarray:
    return np.array(
        [
            _safe_component_value(model, component_name, float(time_value), condition, params, trial_row=trial_row, default=np.nan)
            for time_value in times
        ],
        dtype=float,
    )


def _safe_component_value(
    model: object,
    component_name: str,
    time_value: float,
    condition: str,
    params: list[float],
    trial_row: dict[str, object] | None,
    default: float,
) -> float:
    try:
        component = getattr(model, component_name)
        return float(_call_or_value(component, time_value, condition, *params, trial_row=trial_row))
    except Exception:
        return float(default)


def _safe_process_value(
    model: object,
    component_name: str,
    evidence: float,
    time_value: float,
    condition: str,
    params: list[float],
    trial_row: dict[str, object] | None,
    default: float,
) -> float:
    try:
        component = getattr(model, component_name)
        return float(_call_process(component, evidence, time_value, condition, *params, trial_row=trial_row))
    except Exception:
        return float(default)


def plot_data_overview(
    data: pd.DataFrame,
    output_path: str | Path | None = None,
    response_label: str = "Response",
    title: str = "Data Overview",
) -> plt.Figure:
    data = data.copy()
    data["condition"] = data["condition"].astype(str)
    data["accuracy"] = pd.to_numeric(data["accuracy"], errors="coerce")
    data["rt"] = pd.to_numeric(data["rt"], errors="coerce")
    conditions = sorted(data["condition"].dropna().astype(str).unique())
    palette = ["#0f766e", "#5b5bd6", "#9a5b05", "#0072B2", "#CC79A7"]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    ax = axes[0, 0]
    sns.histplot(data["rt"].dropna(), bins=24, kde=False, color=palette[0], ax=ax)
    ax.set_title("Response-time distribution")
    ax.set_xlabel("Response time (s)")
    ax.set_ylabel("Trials")

    ax = axes[0, 1]
    response_counts = data["accuracy"].dropna().astype(int).value_counts().sort_index()
    response_counts.plot(kind="bar", ax=ax, color=palette[1], rot=0)
    ax.set_title(f"{response_label} counts")
    ax.set_xlabel(response_label)
    ax.set_ylabel("Trials")
    for index, value in enumerate(response_counts.tolist()):
        ax.text(index, value, str(int(value)), ha="center", va="bottom", fontsize=9)

    ax = axes[1, 0]
    if conditions:
        means = []
        errors = []
        labels = []
        for condition in conditions:
            subset = data[data["condition"] == condition]["accuracy"].dropna()
            n = len(subset)
            mean = float(subset.mean()) if n else np.nan
            se = float(np.sqrt(mean * (1.0 - mean) / n)) if n and np.isfinite(mean) else 0.0
            labels.append(_wrap_label(condition, 16, 44))
            means.append(mean)
            errors.append(se)
        ax.bar(labels, means, yerr=errors, color=palette[0], capsize=4)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    ax.set_title(f"Mean {response_label.lower()} by condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel(f"Mean {response_label.lower()}")

    ax = axes[1, 1]
    if len(conditions) <= 8 and conditions:
        for index, condition in enumerate(conditions):
            values = data.loc[data["condition"] == condition, "rt"].dropna()
            if values.empty:
                continue
            sns.kdeplot(values, ax=ax, label=_wrap_label(condition, 18, 46), warn_singular=False, color=palette[index % len(palette)])
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=7, title="Condition", loc="best")
    else:
        sns.boxplot(data=data, x="accuracy", y="rt", ax=ax, color=palette[2])
    if len(conditions) <= 8:
        ax.set_title("RT distributions by condition")
        ax.set_xlabel("Response time (s)")
        ax.set_ylabel("Density")
    else:
        ax.set_title(f"RT by {response_label.lower()}")
        ax.set_xlabel(response_label)
        ax.set_ylabel("Response time (s)")

    if title:
        fig.suptitle(title, y=0.995, fontsize=13)
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    else:
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        plt.close(fig)
    return fig


def plot_internal_consistency(
    estimates: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Internal Consistency Reliability",
) -> plt.Figure:
    parameters = sorted(estimates["parameter"].astype(str).unique()) if not estimates.empty else []
    n_parameters = max(len(parameters), 1)
    n_cols = min(3, n_parameters)
    n_rows = int(np.ceil(n_parameters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)

    for axis in axes.ravel():
        axis.set_visible(False)

    for index, parameter in enumerate(parameters):
        ax = axes.ravel()[index]
        ax.set_visible(True)
        subset = estimates[estimates["parameter"].astype(str) == parameter]
        ax.scatter(subset["even"], subset["odd"], color="#0f766e", alpha=0.75, edgecolor="white", linewidth=0.4)
        if len(subset) >= 2:
            both = pd.concat([subset["even"], subset["odd"]])
            lo = float(both.min())
            hi = float(both.max())
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [lo, hi], color="#5c6674", linestyle="--", linewidth=1)
            corr = subset["even"].corr(subset["odd"]) if subset["even"].nunique() > 1 and subset["odd"].nunique() > 1 else np.nan
            reliability = spearman_brown(corr)
            corr_text = "SB r = n/a" if not np.isfinite(reliability) else f"SB r = {reliability:.2f}"
        else:
            corr_text = "r = n/a"
        ax.set_title(f"{parameter} ({corr_text})", fontsize=10)
        ax.set_xlabel("Even trials")
        ax.set_ylabel("Odd trials")

    fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    return fig


def spearman_brown(split_half_r: float | None) -> float:
    if split_half_r is None or not np.isfinite(split_half_r) or split_half_r <= -1.0:
        return float("nan")
    return float((2.0 * split_half_r) / (1.0 + split_half_r))


def plot_parameter_recovery(
    estimates: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Parameter Recovery",
) -> plt.Figure:
    parameters = sorted(estimates["parameter"].astype(str).unique()) if not estimates.empty else []
    n_parameters = max(len(parameters), 1)
    n_cols = min(3, n_parameters)
    n_rows = int(np.ceil(n_parameters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)
    for axis in axes.ravel():
        axis.set_visible(False)

    for index, parameter in enumerate(parameters):
        ax = axes.ravel()[index]
        ax.set_visible(True)
        subset = estimates[estimates["parameter"].astype(str) == parameter]
        ax.scatter(subset["generated"], subset["recovered"], color="#5b5bd6", alpha=0.75, edgecolor="white", linewidth=0.4)
        both = pd.concat([subset["generated"], subset["recovered"]])
        lo = float(both.min())
        hi = float(both.max())
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [lo, hi], color="#5c6674", linestyle="--", linewidth=1)
        if len(subset) >= 3 and subset["generated"].nunique() > 1 and subset["recovered"].nunique() > 1:
            r_value, p_value = pearsonr(subset["generated"], subset["recovered"])
            stat_text = f"r = {r_value:.2f}, p = {p_value:.3f}"
        else:
            stat_text = "r = n/a, p = n/a"
        ax.set_title(f"{parameter}\n{stat_text}", fontsize=10)
        ax.set_xlabel("Generated")
        ax.set_ylabel("Recovered")

    fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    return fig


def plot_model_recovery(
    rows: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Model Recovery",
) -> plt.Figure:
    matrix = rows.pivot_table(index="fitModel", columns="generatedBy", values="proportion", fill_value=0.0)
    fig_width = max(4.5, 1.2 * max(len(matrix.columns), 1))
    fig_height = max(3.8, 0.8 * max(len(matrix.index), 1))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(matrix, vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".0%", cbar_kws={"label": "Recovered proportion"}, ax=ax)
    ax.set_xlabel("Generated model")
    ax.set_ylabel("Fit model")
    ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    return fig
