from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .fokker_planck import FokkerPlanckConfig, FokkerPlanckSpec
from .model import ModelSpec


def sample_like_observed(
    model: ModelSpec | FokkerPlanckSpec,
    values: Iterable[float],
    observed: pd.DataFrame,
    participant_id: object = "simulated",
    fp_config: FokkerPlanckConfig | None = None,
) -> pd.DataFrame:
    counts = {str(condition): int(count) for condition, count in observed["condition"].astype(str).value_counts().items()}
    if isinstance(model, FokkerPlanckSpec):
        if model.uses_trial_context:
            return model.sample_dataframe_like(values, observed, participant_id, fp_config or FokkerPlanckConfig())
        return _sample_fp_like_observed(model, values, counts, participant_id, fp_config or FokkerPlanckConfig())
    if hasattr(model, "sample_dataframe_like"):
        return model.sample_dataframe_like(values, observed, participant_id, fp_config or FokkerPlanckConfig())
    return _sample_simulation_like_observed(model, values, observed, counts, participant_id)


def _sample_fp_like_observed(
    model: FokkerPlanckSpec,
    values: Iterable[float],
    counts: dict[str, int],
    participant_id: object,
    config: FokkerPlanckConfig,
) -> pd.DataFrame:
    params = tuple(values)
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []

    for condition, needed in counts.items():
        if condition not in model.conditions:
            raise ValueError(f"Observed condition {condition!r} is not produced by model {model.name!r}.")
        solution = model.solve_condition(params, condition, config)
        weights = np.concatenate([solution.upper_pdf, solution.lower_pdf])
        weight_sum = float(np.nansum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError(f"{model.name} produced no finite boundary-crossing density for condition {condition}.")
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0) / weight_sum
        draws = rng.choice(np.arange(len(weights)), size=needed, p=weights)
        nondecision = model.nondecision(0.0, condition, *params) if callable(model.nondecision) else float(model.nondecision)

        for draw in draws:
            upper = draw < len(solution.times)
            time_index = int(draw if upper else draw - len(solution.times))
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


def _sample_simulation_like_observed(
    model: ModelSpec,
    values: Iterable[float],
    observed: pd.DataFrame,
    counts: dict[str, int],
    participant_id: object,
) -> pd.DataFrame:
    needed = counts.copy()
    accepted: list[pd.DataFrame] = []
    total_needed = sum(needed.values())
    batch_size = max(total_needed * 2, model.default_trials, 100)

    for _ in range(40):
        if all(count <= 0 for count in needed.values()):
            break
        design = observed if len(observed) <= batch_size else observed.sample(n=batch_size, replace=True, random_state=17).reset_index(drop=True)
        batch = model.simulate_dataframe(values, n_trials=batch_size, participant_id=participant_id, data_like=design)
        batch = batch.copy()
        batch["condition"] = batch["condition"].astype(str)

        for condition in list(needed):
            if needed[condition] <= 0:
                continue
            condition_rows = batch[batch["condition"] == condition].head(needed[condition])
            if not condition_rows.empty:
                accepted.append(condition_rows)
                needed[condition] -= len(condition_rows)

    missing = {condition: count for condition, count in needed.items() if count > 0}
    if missing:
        missing_text = ", ".join(f"{condition}: {count}" for condition, count in sorted(missing.items()))
        raise ValueError(f"{model.name} could not generate enough PPC trials for observed condition counts ({missing_text}).")

    simulated = pd.concat(accepted, ignore_index=True) if accepted else pd.DataFrame(columns=["id", "trial", "accuracy", "rt", "condition"])
    simulated = simulated[["id", "trial", "accuracy", "rt", "condition"]].copy()
    simulated["id"] = participant_id
    simulated["trial"] = np.arange(1, len(simulated) + 1)
    return simulated
