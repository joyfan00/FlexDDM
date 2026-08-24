from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPSILON = 1e-4


@dataclass(frozen=True)
class ConditionTargets:
    cdf_cutoffs: list[float]
    caf_cutoffs: list[float]
    empirical_cdf_props: list[float]
    empirical_caf_props: list[float]


@dataclass(frozen=True)
class FitTargets:
    conditions: dict[str, ConditionTargets]
    n_observations: int


def cdf_bin_sizes(quantiles: list[float]) -> list[float]:
    previous = 0.0
    sizes: list[float] = []
    for quantile in quantiles:
        sizes.append(float(quantile) - previous)
        previous = float(quantile)
    sizes.append(1.0 - previous)
    return sizes


def empirical_targets(data: pd.DataFrame, cdf_quantiles: list[float] | None = None, caf_quantiles: list[float] | None = None) -> FitTargets:
    cdf_quantiles = cdf_quantiles or [0.10, 0.30, 0.50, 0.70, 0.90]
    caf_quantiles = caf_quantiles or [0.25, 0.50, 0.75]
    conditions: dict[str, ConditionTargets] = {}

    for condition in sorted(map(str, data["condition"].unique())):
        condition_data = data[data["condition"].astype(str) == condition]
        cdf_cutoffs = _safe_percentiles(condition_data.loc[condition_data["accuracy"] == 1, "rt"], cdf_quantiles)
        caf_cutoffs = _safe_percentiles(condition_data["rt"], caf_quantiles)
        cdf_props, caf_props = _cdf_caf_props(condition_data, cdf_cutoffs, caf_cutoffs)
        conditions[condition] = ConditionTargets(cdf_cutoffs, caf_cutoffs, cdf_props, caf_props)

    return FitTargets(conditions=conditions, n_observations=len(data))


def simulated_props(simulated: pd.DataFrame, targets: FitTargets) -> dict[str, list[float]]:
    props: dict[str, list[float]] = {}
    simulated = simulated.copy()
    simulated["condition"] = simulated["condition"].astype(str)
    for condition, target in targets.conditions.items():
        condition_data = simulated[simulated["condition"] == condition]
        cdf_props, caf_props = _cdf_caf_props(condition_data, target.cdf_cutoffs, target.caf_cutoffs)
        props[f"cdf_props_{condition}"] = cdf_props
        props[f"caf_props_{condition}"] = caf_props
    return props


def g2_statistic(targets: FitTargets, predictions: dict[str, list[float]]) -> float:
    observed: list[float] = []
    expected: list[float] = []
    for condition, target in targets.conditions.items():
        observed.extend(target.empirical_cdf_props)
        observed.extend(target.empirical_caf_props)
        expected.extend(predictions.get(f"cdf_props_{condition}", []))
        expected.extend(predictions.get(f"caf_props_{condition}", []))

    observed_arr = np.clip(np.asarray(observed, dtype=float), EPSILON, None)
    expected_arr = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    if observed_arr.shape != expected_arr.shape:
        return float(np.iinfo(np.int64).max)
    return float(2 * targets.n_observations * np.sum(observed_arr * np.log(observed_arr / expected_arr)))


def bic_from_g2(g2: float, n_parameters: int, n_observations: int) -> float:
    return float(g2 + n_parameters * np.log(max(n_observations, 1)))


def _safe_percentiles(values: pd.Series, quantiles: list[float]) -> list[float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return [0.0 for _ in quantiles]
    return [float(x) for x in np.nanpercentile(clean, [q * 100 for q in quantiles])]


def _cdf_caf_props(data: pd.DataFrame, cdf_cutoffs: list[float], caf_cutoffs: list[float]) -> tuple[list[float], list[float]]:
    if data.empty:
        return [0.0 for _ in range(len(cdf_cutoffs) + 1)], [0.0 for _ in range(len(caf_cutoffs) + 1)]

    total = len(data)
    correct = data[data["accuracy"] == 1]
    cdf_props = _bin_proportions(correct, cdf_cutoffs, total)
    caf_props = []
    for chunk in _cutoff_chunks(data, caf_cutoffs):
        caf_props.append(float((chunk["accuracy"] == 0).sum()) / total if len(chunk) else 0.0)
    return cdf_props, caf_props


def _bin_proportions(data: pd.DataFrame, cutoffs: list[float], denominator: int) -> list[float]:
    return [float(len(chunk)) / denominator if denominator else 0.0 for chunk in _cutoff_chunks(data, cutoffs)]


def _cutoff_chunks(data: pd.DataFrame, cutoffs: list[float]) -> list[pd.DataFrame]:
    chunks: list[pd.DataFrame] = []
    previous = None
    for cutoff in cutoffs:
        mask = data["rt"] <= cutoff if previous is None else (data["rt"] > previous) & (data["rt"] <= cutoff)
        chunks.append(data[mask])
        previous = cutoff
    if previous is None:
        chunks.append(data)
    else:
        chunks.append(data[data["rt"] > previous])
    return chunks
