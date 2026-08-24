from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .fit import FitConfig, fit_model
from .model import ModelSpec


@dataclass(frozen=True)
class ParameterRecoverySummary:
    parameter: str
    correlation: float | None


def parameter_recovery(
    model: ModelSpec,
    simulations: int = 3,
    trials_per_simulation: int = 80,
    fit_config: FitConfig | None = None,
) -> list[dict[str, object]]:
    fit_config = fit_config or FitConfig(maxiter=1, popsize=3, polish=False, local_refine=False, simulated_trials=trials_per_simulation)
    rng = np.random.default_rng(1)
    generated_rows: list[dict[str, float]] = []
    recovered_rows: list[dict[str, float]] = []

    for simulation_index in range(simulations):
        bounds = model.bounds_for_data(_fallback_data())
        values = np.array([rng.uniform(lower, upper) for lower, upper in bounds])
        synthetic = model.simulate_dataframe(values, n_trials=trials_per_simulation, participant_id=simulation_index)
        result = fit_model(model, synthetic, fit_config)
        generated_rows.append(dict(zip(model.parameter_names, map(float, values))))
        recovered_rows.append(result.parameters)

    generated = pd.DataFrame(generated_rows)
    recovered = pd.DataFrame(recovered_rows)
    summaries: list[ParameterRecoverySummary] = []
    for parameter in model.parameter_names:
        correlation = None
        if simulations >= 2 and generated[parameter].nunique() > 1 and recovered[parameter].nunique() > 1:
            correlation = float(generated[parameter].corr(recovered[parameter]))
        summaries.append(ParameterRecoverySummary(parameter=parameter, correlation=correlation))
    return [asdict(summary) for summary in summaries]


def model_recovery(
    generating_models: list[ModelSpec],
    candidate_models: list[ModelSpec],
    simulations: int = 2,
    trials_per_simulation: int = 80,
    fit_config: FitConfig | None = None,
) -> list[dict[str, object]]:
    fit_config = fit_config or FitConfig(maxiter=1, popsize=3, polish=False, local_refine=False, simulated_trials=trials_per_simulation)
    rng = np.random.default_rng(2)
    rows: list[dict[str, object]] = []

    for generating_model in generating_models:
        winners: dict[str, int] = {candidate.name: 0 for candidate in candidate_models}
        for simulation_index in range(simulations):
            bounds = generating_model.bounds_for_data(_fallback_data())
            values = np.array([rng.uniform(lower, upper) for lower, upper in bounds])
            synthetic = generating_model.simulate_dataframe(values, n_trials=trials_per_simulation, participant_id=simulation_index)
            fits = [fit_model(candidate, synthetic, fit_config) for candidate in candidate_models]
            winner = min(fits, key=lambda item: item.bic).model_name
            winners[winner] = winners.get(winner, 0) + 1
        for candidate_name, wins in winners.items():
            rows.append(
                {
                    "generatedBy": generating_model.name,
                    "fitModel": candidate_name,
                    "count": wins,
                    "proportion": wins / simulations if simulations else 0,
                }
            )
    return rows


def _fallback_data() -> pd.DataFrame:
    return pd.DataFrame({"id": [0], "condition": ["0"], "rt": [0.45], "accuracy": [1]})
