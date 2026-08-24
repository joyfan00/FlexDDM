from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnMap:
    participant: str = "PPT"
    condition: str = "Condition"
    rt: str = "RT"
    accuracy: str = "Correct"
    response_kind: str = "accuracy"


def load_rt_data(source: str | Path | IO[str] | pd.DataFrame, columns: ColumnMap = ColumnMap()) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        raw = pd.read_csv(source, encoding="utf-8-sig")
    raw_columns = list(raw.columns)

    required_columns = [columns.participant, columns.rt, columns.accuracy]
    if columns.condition:
        required_columns.append(columns.condition)
    missing = [name for name in required_columns if name not in raw]
    if missing:
        available = ", ".join(map(str, raw.columns))
        raise ValueError(f"Missing required columns {missing}. Available columns: {available}")

    response_numeric = pd.to_numeric(raw[columns.accuracy], errors="coerce")
    if response_numeric.notna().any():
        binary_response = response_numeric
        response_mapping = None
    else:
        labels = raw[columns.accuracy].astype(str).str.strip()
        unique_labels = sorted(label for label in labels.dropna().unique())
        mapping = {label: index for index, label in enumerate(unique_labels)}
        binary_response = labels.map(mapping)
        response_mapping = mapping

    data = pd.DataFrame(
        {
            "id": raw[columns.participant],
            "condition": raw[columns.condition].astype(str) if columns.condition else "all",
            "rt": pd.to_numeric(raw[columns.rt], errors="coerce"),
            "accuracy": binary_response,
        }
    )
    mapped = {columns.participant, columns.rt, columns.accuracy}
    if columns.condition:
        mapped.add(columns.condition)
    used = set(data.columns)
    for column in raw.columns:
        if column in mapped:
            continue
        safe_name = _safe_extra_column_name(str(column), used)
        used.add(safe_name)
        converted = pd.to_numeric(raw[column], errors="coerce")
        data[safe_name] = converted if converted.notna().any() else raw[column]
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["id", "condition", "rt", "accuracy"])
    data = data[data["rt"] > 0].copy()
    raw_rows = raw.loc[data.index].copy()
    data["accuracy"] = data["accuracy"].round().astype(int)
    response_values = data["accuracy"].dropna().unique()
    if len(response_values) > 2:
        raise ValueError(
            f"Response column {columns.accuracy!r} has {len(response_values)} levels after coding. "
            "Current FlexDDM objectives expect a binary response/accuracy column."
        )
    data["condition"] = data["condition"].str.strip()
    data.attrs["response_kind"] = columns.response_kind
    data.attrs["response_column"] = columns.accuracy
    data.attrs["raw_columns"] = raw_columns
    data.attrs["raw_rows"] = raw_rows.reset_index(drop=True)
    data.attrs["raw_unfiltered_rows"] = raw_rows.reset_index(drop=True)
    data.attrs["column_map"] = {
        "participant": columns.participant,
        "condition": columns.condition,
        "rt": columns.rt,
        "accuracy": columns.accuracy,
    }
    if response_mapping:
        data.attrs["response_mapping"] = response_mapping
    condition_parts = data["condition"].str.split("-", expand=True)
    if condition_parts.shape[1] > 1:
        for index in range(condition_parts.shape[1]):
            converted = pd.to_numeric(condition_parts[index], errors="coerce")
            data[f"condition_{index}"] = converted if converted.notna().all() else condition_parts[index]

    attrs = dict(data.attrs)
    data = data.reset_index(drop=True)
    data.attrs.update(attrs)
    return data


def _renumber_ids(values: pd.Series) -> list[str]:
    unique = list(dict.fromkeys(values.astype(str).tolist()))
    unique = sorted(unique, key=lambda value: (not _is_integerish(value), int(float(value)) if _is_integerish(value) else value))
    mapping = {value: str(index + 1) for index, value in enumerate(unique)}
    return [mapping[str(value)] for value in values]


def _is_integerish(value: str) -> bool:
    try:
        numeric = float(str(value).strip())
    except ValueError:
        return False
    return numeric.is_integer()


def _safe_extra_column_name(name: str, used: set[str]) -> str:
    safe = "".join(character.lower() if character.isalnum() else "_" for character in name.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    if not safe or safe[0].isdigit():
        safe = f"extra_{safe or 'column'}"
    candidate = safe
    index = 2
    while candidate in used:
        candidate = f"{safe}_{index}"
        index += 1
    return candidate
