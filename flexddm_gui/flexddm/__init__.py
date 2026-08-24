from .data import ColumnMap, load_rt_data
from .model import ModelSpec, Parameter, SimulationResult
from .pseudocode import infer_parameters, model_skeleton_from_pseudocode

__all__ = [
    "ColumnMap",
    "FitConfig",
    "FitResult",
    "ModelSpec",
    "Parameter",
    "SimulationResult",
    "fit_model",
    "infer_parameters",
    "load_rt_data",
    "model_skeleton_from_pseudocode",
    "plot_posterior_predictive",
    "FokkerPlanckConfig",
    "FokkerPlanckSpec",
    "ProcessModelSpec",
    "define_model",
    "param",
    "parameters_from_bounds",
]


def __getattr__(name: str):
    if name in {"FitConfig", "FitResult", "fit_model"}:
        from .fit import FitConfig, FitResult, fit_model

        return {"FitConfig": FitConfig, "FitResult": FitResult, "fit_model": fit_model}[name]
    if name == "plot_posterior_predictive":
        from .plotting import plot_posterior_predictive

        return plot_posterior_predictive
    if name in {"BUILTIN_MODELS", "describe_models"}:
        from .builtin_models import BUILTIN_MODELS, describe_models

        return {"BUILTIN_MODELS": BUILTIN_MODELS, "describe_models": describe_models}[name]
    if name in {"FokkerPlanckConfig", "FokkerPlanckSpec"}:
        from .fokker_planck import FokkerPlanckConfig, FokkerPlanckSpec

        return {"FokkerPlanckConfig": FokkerPlanckConfig, "FokkerPlanckSpec": FokkerPlanckSpec}[name]
    if name == "ProcessModelSpec":
        from .process_model import ProcessModelSpec

        return ProcessModelSpec
    if name in {"define_model", "param", "parameters_from_bounds"}:
        from .authoring import define_model, param, parameters_from_bounds

        return {"define_model": define_model, "param": param, "parameters_from_bounds": parameters_from_bounds}[name]
    if name in {"parameter_recovery", "model_recovery"}:
        from .validation import model_recovery, parameter_recovery

        return {"model_recovery": model_recovery, "parameter_recovery": parameter_recovery}[name]
    raise AttributeError(name)
