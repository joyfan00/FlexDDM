from __future__ import annotations

import re
from dataclasses import dataclass

from .model import Parameter

PARAMETER_LINE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::|=|between)\s*"
    r"(?P<lower>-?\d+(?:\.\d+)?)\s*(?:\.\.|,|to|and)\s*(?P<upper>-?\d+(?:\.\d+)?|min_rt)",
    re.IGNORECASE,
)
BARE_PARAMETER = re.compile(r"\bparameter\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ParameterInference:
    parameters: tuple[Parameter, ...]
    unresolved_names: tuple[str, ...]


def infer_parameters(pseudocode: str) -> ParameterInference:
    parameters: dict[str, Parameter] = {}
    unresolved: set[str] = set()

    for match in PARAMETER_LINE.finditer(pseudocode):
        upper_text = match.group("upper")
        parameters[match.group("name")] = Parameter(
            name=match.group("name"),
            lower=float(match.group("lower")),
            upper=upper_text if upper_text == "min_rt" else float(upper_text),
        )

    for match in BARE_PARAMETER.finditer(pseudocode):
        name = match.group("name")
        if name not in parameters:
            unresolved.add(name)

    return ParameterInference(tuple(parameters.values()), tuple(sorted(unresolved)))


def model_skeleton_from_pseudocode(model_name: str, pseudocode: str) -> str:
    inference = infer_parameters(pseudocode)
    parameter_lines = "\n".join(
        f'        Parameter("{param.name}", {param.lower!r}, {param.upper!r}),'
        for param in inference.parameters
    )
    notes = "\n".join(f"# TODO: choose bounds for unresolved parameter {name}" for name in inference.unresolved_names)
    return f'''from __future__ import annotations

import numpy as np

from flexddm import ModelSpec, Parameter, SimulationResult

{notes}


def simulate({", ".join(param.name for param in inference.parameters)}, nTrials=800):
    """Generated from pseudocode. Replace the body with the model equations."""
    trial = np.arange(1, nTrials + 1)
    accuracy = np.full(nTrials, np.nan)
    rt = np.full(nTrials, np.nan)
    condition = np.zeros((nTrials, 1), dtype=int)

    # Original pseudocode:
{_comment_block(pseudocode)}

    return SimulationResult(trial=trial, accuracy=accuracy, rt=rt, condition=condition)


model = ModelSpec(
    name="{model_name}",
    parameters=(
{parameter_lines}
    ),
    simulate=simulate,
)
'''


def _comment_block(text: str) -> str:
    return "\n".join(f"    # {line}" for line in text.splitlines())
