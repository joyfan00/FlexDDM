from __future__ import annotations

import importlib.util
import base64
import copy
import hashlib
import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import types
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from .builtin_models import BUILTIN_MODELS, describe_models
from .builtin_fp_models import BUILTIN_FP_MODELS
from .data import ColumnMap, load_rt_data
from .fit import FitConfig, FitResult, fit_fokker_planck_model, fit_model
from .fokker_planck import FokkerPlanckConfig, FokkerPlanckSpec, _component_has_unused_x, _component_ignores_x
from .model import ModelSpec, Parameter
from .model_templates import (
    FOKKER_PLANCK_TEMPLATES,
    MODEL_STARTER_TEMPLATES,
    MODEL_TEMPLATE_BOUNDS,
    MODEL_TEMPLATES,
    SIMULATION_TEMPLATES,
)
from .plotting import (
    model_schematic_data,
    model_trajectory_data,
    plot_data_overview,
    plot_internal_consistency,
    plot_model_recovery,
    plot_model_schematic,
    plot_parameter_recovery,
    spearman_brown,
)
from .posterior_predictive import sample_like_observed
from .pseudocode import infer_parameters, model_skeleton_from_pseudocode
from .process_model import ProcessModelSpec
from .stats import empirical_targets, g2_statistic, simulated_props
from .validation import model_recovery, parameter_recovery

SOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
PACKAGED_CLEAN_DEFAULTS = getattr(sys, "frozen", False) and os.environ.get("FLEXDDM_KEEP_BUNDLED_STATE") != "1"
WORKSPACE = (
    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "FlexDDM"
    if getattr(sys, "frozen", False)
    else SOURCE_ROOT
)
GENERATED_MODELS = WORKSPACE / "generated_models"
RUNS = Path(tempfile.gettempdir()) / "FlexDDM" / "runs"
CLUSTER_EXPORTS = RUNS / "cluster_exports"
DATA_SETTINGS_DIR = WORKSPACE / "data_settings"
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()
DATA_PREVIEW_PLOT_MAX_ROWS = 2500
APP_VERSION = "0.1.1"
APP_BUILD_DATE = "2026-08-21"
APP_CONTACT_NAME = "Kyle J. LaFollette"
APP_CONTACT_INSTITUTION = "University of Chicago Booth School of Business"
APP_CONTACT_EMAIL = "Kyle.Lafollette@ChicagoBooth.edu"
APP_DESCRIPTION = (
    "A flexible decision-diffusion modeling application for building, fitting, "
    "visualizing, and validating custom diffusion-style process models."
)
APP_CITATION = (
    "LaFollette, K. J., Fan, J., Puccio, A., & Demaree, H. A. (2024). "
    "FlexDDM: A flexible decision-diffusion Python package for the behavioral sciences. "
    "In L. K. Samuelson, S. Frank, M. Toneva, A. Mackey, & E. Hazeltine (Eds.), "
    "Proceedings of the Annual Meeting of the Cognitive Science Society, 46."
)
APP_COPYRIGHT_NOTICE = """Copyright 2024-2026, The University of Chicago, Chicago, IL.

All Rights Reserved

Permission to use, copy, modify, and distribute this software and
its documentation for any purpose other than its incorporation into a
commercial product or service is hereby granted without fee, provided
that the above copyright notice appear in all copies and that both
that copyright notice and this permission notice appear in supporting
documentation, and that the name of The University of Chicago or the
University of Chicago Booth School of Business not be used in
advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

THE UNIVERSITY OF CHICAGO DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR ANY PARTICULAR PURPOSE. IN NO EVENT SHALL THE UNIVERSITY OF CHICAGO
BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY
DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE."""


def _ensure_runtime_workspace() -> None:
    if not getattr(sys, "frozen", False):
        return
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if PACKAGED_CLEAN_DEFAULTS:
        reset_marker = WORKSPACE / ".packaged_clean_defaults_20260822_clean2"
        if not reset_marker.exists():
            for folder_name in ["generated_models", "data", "data_settings", "runs"]:
                target = WORKSPACE / folder_name
                if target.exists():
                    shutil.rmtree(target)
            reset_marker.write_text(str(time.time()), encoding="utf-8")
    for folder_name in ["generated_models", "data", "data_settings"]:
        (WORKSPACE / folder_name).mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    bundled_data = SOURCE_ROOT / "data"
    local_data = WORKSPACE / "data"
    if bundled_data.is_dir():
        for source in bundled_data.glob("*"):
            destination = local_data / source.name
            if source.is_file() and not destination.exists():
                shutil.copy2(source, destination)
    bundled_models = SOURCE_ROOT / "generated_models"
    if bundled_models.is_dir():
        GENERATED_MODELS.mkdir(parents=True, exist_ok=True)
        for source in bundled_models.glob("*"):
            destination = GENERATED_MODELS / source.name
            if source.is_file() and not destination.exists():
                shutil.copy2(source, destination)


def _available_seeded_custom_models() -> dict[str, str]:
    return {}


_ensure_runtime_workspace()

TEMPLATE_DESCRIPTIONS = {
    "StandardDDM": "A compact two-boundary diffusion model. Evidence drifts toward one response or the other, with separate drift rates for the two condition labels.",
    "DMC": "Diffusion model for conflict tasks. A controlled drift is combined with a brief automatic impulse that can push evidence toward or away from the correct response over time.",
    "DSTP": "Dual-stage two-phase model for flanker tasks. Evidence combines target selection and response selection processes so conflict can change across the decision.",
    "SSP": "Shrinking spotlight model for flanker tasks. Attention narrows over time, changing how strongly target and flanker information contribute to the response drift.",
    "mDMC varying noise": "A DMC variant where diffusion noise can change over time, useful for testing whether transient variability improves model fit.",
    "Weibull boundary": "A DDM variant with a time-varying decision boundary, useful for exploring urgency or collapsing-bound effects.",
    "CastM": "A motivation/conflict model with several task conditions and value-like modulation of controlled drift.",
    "Racing diffusion": "A simulation-only racing accumulator example where multiple accumulators compete and the first to reach its boundary determines the response.",
    "RLDDM": "A reinforcement-learning diffusion example where latent values update trial by trial and influence the drift process.",
}


class FlexDDMHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        from .routes import handle_get

        parsed = urlparse(self.path)
        handle_get(self, parsed.path, parse_qs(parsed.query))

    def do_POST(self) -> None:
        from .routes import handle_post

        route = urlparse(self.path).path
        try:
            handle_post(self, route, self._read_json())
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[FlexDDM app] {self.address_string()} - {fmt % args}")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(_json_safe(payload), indent=2, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, root: Path, cache: bool = False) -> None:
        resolved = path.resolve()
        root = root.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            self._send_json({"error": "Not found"}, status=404)
            return
        if not resolved.is_file():
            self._send_json({"error": "Not found"}, status=404)
            return
        body = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), FlexDDMHandler)
    print(f"FlexDDM app running at http://{host}:{port}")
    server.serve_forever()


def _worker_status_rows(worker_progress: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in sorted(worker_progress.values(), key=lambda row: int(row.get("slot", 0)) if isinstance(row, dict) else 0):
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if not label:
            participant_index = item.get("participantIndex")
            participant_id = item.get("participantId")
            participant = f"Participant {participant_index}"
            if participant_id not in (None, ""):
                participant = f"{participant} ({participant_id})"
            label = f"Worker {item.get('slot')} | {participant}"
        rows.append(
            {
                "slot": item.get("slot"),
                "label": label,
                "progress": item.get("participantProgress"),
                "status": item.get("status", "running"),
            }
        )
    return rows


def _start_job(kind: str, payload: dict, runner) -> dict[str, object]:
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "jobId": job_id,
            "kind": kind,
            "status": "running",
            "message": f"Starting {kind}...",
            "progress": 0.0,
            "startedAt": time.time(),
            "updatedAt": time.time(),
            "cancelRequested": False,
        }

    def progress(message: str, value: float | None = None, detail: dict[str, object] | None = None) -> None:
        with JOBS_LOCK:
            job = JOBS[job_id]
            if job.get("cancelRequested"):
                raise InterruptedError(f"{kind.title()} cancelled.")
            job["message"] = message
            job["updatedAt"] = time.time()
            job["elapsedSeconds"] = int(round(time.time() - float(job["startedAt"])))
            if value is not None:
                job["progress"] = max(0.0, min(1.0, float(value)))
                if 0.01 < float(job["progress"]) < 1.0:
                    elapsed = time.time() - float(job["startedAt"])
                    eta = elapsed * (1.0 - float(job["progress"])) / float(job["progress"])
                    job["etaSeconds"] = max(0, int(round(eta)))
            if detail is not None:
                if detail.get("etaSeconds") is not None:
                    job["etaSeconds"] = max(0, int(round(float(detail.get("etaSeconds") or 0))))
                job["detail"] = detail
                worker_slot = detail.get("workerSlot")
                if worker_slot is not None:
                    worker_progress = dict(job.get("workerProgress") or {})
                    slot_key = str(worker_slot)
                    worker_progress[slot_key] = {
                        "slot": int(worker_slot),
                        "label": detail.get("label"),
                        "message": message,
                        "participantIndex": detail.get("participantIndex"),
                        "participantId": detail.get("participantId"),
                        "participantProgress": detail.get("participantProgress"),
                        "status": detail.get("status", "running"),
                        "updatedAt": job["updatedAt"],
                    }
                    job["workerProgress"] = worker_progress
                    job["statusRows"] = _worker_status_rows(worker_progress)

    def cancel_requested() -> bool:
        with JOBS_LOCK:
            return bool(JOBS.get(job_id, {}).get("cancelRequested"))

    def target() -> None:
        try:
            result = runner(payload, progress, cancel_requested)
            with JOBS_LOCK:
                JOBS[job_id].update(
                    {
                        "status": "complete",
                        "message": f"{kind.title()} complete",
                        "progress": 1.0,
                        "result": result,
                        "updatedAt": time.time(),
                        "elapsedSeconds": int(round(time.time() - float(JOBS[job_id]["startedAt"]))),
                    }
                )
        except InterruptedError as exc:
            with JOBS_LOCK:
                JOBS[job_id].update(
                    {
                        "status": "cancelled",
                        "message": str(exc) or f"{kind.title()} cancelled.",
                        "progress": JOBS[job_id].get("progress", 0.0),
                        "updatedAt": time.time(),
                    }
                )
        except Exception as exc:
            with JOBS_LOCK:
                JOBS[job_id].update(
                    {
                        "status": "error",
                        "message": str(exc),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "updatedAt": time.time(),
                    }
                )

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return {"jobId": job_id, "status": "running"}


def _cancel_job(job_id: str) -> dict[str, object]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return {"status": "missing", "message": "Job not found"}
        if job.get("status") == "running":
            job["cancelRequested"] = True
            job["message"] = f"Cancelling {job.get('kind', 'job')}..."
            job["updatedAt"] = time.time()
        return {"jobId": job_id, "status": job.get("status"), "message": job.get("message", "")}


def _browse_csv_file() -> dict[str, object]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose a FlexDDM data file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(WORKSPACE),
        )
        root.destroy()
        return {"path": path}
    except Exception as exc:
        raise ValueError(f"Could not open file browser: {exc}") from exc


def _read_system_clipboard() -> str:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            value = root.clipboard_get()
        finally:
            root.destroy()
        return str(value or "")
    except Exception as exc:
        raise ValueError(f"Could not read clipboard: {exc}") from exc


def _save_artifact(payload: dict) -> dict[str, object]:
    kind = str(payload.get("kind", "")).strip().lower()
    suggested_name = _safe_export_filename(str(payload.get("filename", "flexddm_export")))
    extension = ".zip" if kind == "file-url" else (".csv" if kind == "csv" else ".png")
    if not suggested_name.lower().endswith(extension):
        suggested_name = f"{suggested_name}{extension}"
    if kind == "csv":
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
    elif kind == "file-url":
        filetypes = [("Zip archives", "*.zip"), ("All files", "*.*")]
    else:
        filetypes = [("PNG images", "*.png"), ("All files", "*.*")]
    destination = _ask_save_path(suggested_name, extension, filetypes)
    if not destination:
        return {"saved": False, "path": ""}

    if kind == "csv":
        content = str(payload.get("content", ""))
        Path(destination).write_text(content, encoding="utf-8", newline="")
    elif kind == "image-url":
        source = _artifact_source_path(str(payload.get("sourceUrl", "")))
        shutil.copyfile(source, destination)
    elif kind == "canvas-png":
        data_url = str(payload.get("dataUrl", ""))
        prefix = "data:image/png;base64,"
        if not data_url.startswith(prefix):
            raise ValueError("Canvas export did not provide a PNG data URL.")
        Path(destination).write_bytes(base64.b64decode(data_url[len(prefix) :]))
    elif kind == "file-url":
        source = _artifact_source_path(str(payload.get("sourceUrl", "")))
        shutil.copyfile(source, destination)
    else:
        raise ValueError("Unknown export type.")
    return {"saved": True, "path": str(destination)}


def _ask_save_path(suggested_name: str, extension: str, filetypes: list[tuple[str, str]]) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title="Save FlexDDM output",
            initialdir=str(WORKSPACE),
            initialfile=suggested_name,
            defaultextension=extension,
            filetypes=filetypes,
        )
        root.destroy()
        return path
    except Exception as exc:
        raise ValueError(f"Could not open save dialog: {exc}") from exc


def _safe_export_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return stem or "flexddm_export"


def _artifact_source_path(source_url: str) -> Path:
    path = urlparse(source_url).path
    if not path.startswith("/runs/"):
        raise ValueError("Only generated FlexDDM run images can be saved from a URL.")
    source = (RUNS / path.removeprefix("/runs/")).resolve()
    if not source.is_file() or not str(source).startswith(str(RUNS.resolve())):
        raise ValueError("The requested image is not available.")
    return source


def _default_data_path(name: str) -> Path:
    safe = Path(str(name)).name or "ulrich2015.csv"
    candidates = [WORKSPACE / "data" / safe, SOURCE_ROOT / "data" / safe]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _data_settings_path(path: str) -> Path:
    DATA_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]
    return DATA_SETTINGS_DIR / f"{digest}.json"


def _load_data_settings(payload: dict) -> dict[str, object]:
    path = str(payload.get("path", "")).strip()
    version = str(payload.get("version", "")).strip()
    if not path:
        return {"settings": None}
    settings_path = _data_settings_path(path)
    if not settings_path.is_file():
        return {"settings": None}
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return {"settings": None}
    if version and data.get("version") != version:
        return {"settings": None}
    return {"settings": data.get("settings")}


def _save_data_settings(payload: dict) -> dict[str, object]:
    path = str(payload.get("path", "")).strip()
    if not path:
        return {"saved": False}
    settings_path = _data_settings_path(path)
    settings_path.write_text(
        json.dumps(
            {
                "version": str(payload.get("version", "")),
                "path": path,
                "settings": payload.get("settings") or {},
                "savedAt": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"saved": True, "path": str(settings_path)}


def _run_simulation_request(payload: dict) -> dict[str, object]:
    model_name = str(payload.get("model", "DMC"))
    participants = max(1, int(float(payload.get("participants", 20))))
    trials = max(1, int(float(payload.get("trials", 120))))
    seed = int(float(payload.get("seed", 10)))
    rng = np.random.default_rng(seed)
    model = _load_model(model_name)
    param_specs = {str(row.get("name", "")): row for row in payload.get("parameters") or []}
    parameter_matrix = np.zeros((participants, len(model.parameters)), dtype=float)
    parameter_rows: list[dict[str, object]] = []
    latent_z: dict[str, np.ndarray] = {}
    for index, parameter in enumerate(model.parameters):
        spec = param_specs.get(parameter.name, {})
        try:
            lower = float(spec.get("lower", parameter.lower))
        except (TypeError, ValueError):
            lower = float(parameter.lower)
        try:
            upper = float(spec.get("upper", parameter.upper))
        except (TypeError, ValueError):
            try:
                lower, upper = parameter.bounds_for_data(None)
            except Exception:
                upper = lower + 1.0
        if upper <= lower:
            upper = lower + 1.0
        mean = float(spec.get("mean", (lower + upper) / 2))
        sd = float(spec.get("sd", max((upper - lower) / 6, 1e-6)))
        skew = float(spec.get("skew", 0.0))
        values, z = _draw_bounded_values(rng, participants, lower, upper, mean, sd, skew)
        parameter_matrix[:, index] = values
        latent_z[parameter.name] = z
    covariate_specs = [row for row in (payload.get("covariates") or []) if str(row.get("name", "")).strip()]
    frames: list[pd.DataFrame] = []
    for participant_index in range(participants):
        participant_id = str(participant_index + 1)
        participant_values = parameter_matrix[participant_index, :]
        participant_traits = {
            spec["name"]: _draw_correlated_covariate_value(rng, spec, latent_z, participant_index)
            for spec in covariate_specs
            if str(spec.get("level", "trial")) == "participant"
        }
        design = pd.DataFrame(
            {
                "condition": [model.conditions[index % len(model.conditions)] if model.conditions else str(index % 2) for index in range(trials)],
                "trial": np.arange(1, trials + 1),
            }
        )
        for name, value in participant_traits.items():
            design[name] = value
        for spec in covariate_specs:
            if str(spec.get("level", "trial")) != "trial":
                continue
            name = str(spec.get("name", "")).strip()
            z = _correlated_covariate_z(rng, spec, latent_z, participant_index, trials)
            mean = float(spec.get("mean", 0.0) or 0.0)
            sd = abs(float(spec.get("sd", 1.0) or 1.0))
            skew = float(spec.get("skew", 0.0) or 0.0)
            shaped = _skew_normal_standard(rng, z, skew)
            design[name] = mean + sd * shaped
        simulated = model.simulate_dataframe(
            participant_values,
            n_trials=trials,
            participant_id=participant_id,
            dt=float(payload.get("dt", 0.01) or 0.01),
            noise_sd=float(payload.get("noiseSd", 0.01) or 0.01),
            seed=seed + participant_index,
            data_like=design,
        )
        simulated = simulated.reset_index(drop=True)
        for column in design.columns:
            if column not in simulated.columns:
                simulated[column] = design[column].to_numpy()[: len(simulated)]
        for param_index, parameter in enumerate(model.parameters):
            simulated[f"param_{parameter.name}"] = float(participant_values[param_index])
        frames.append(simulated)
        row = {"id": participant_id}
        row.update({parameter.name: float(participant_values[index]) for index, parameter in enumerate(model.parameters)})
        row.update(participant_traits)
        parameter_rows.append(row)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["id", "trial", "accuracy", "rt", "condition"])
    data.attrs["response_kind"] = "choice"
    data.attrs["response_column"] = "accuracy"
    data.attrs["raw_rows"] = data.copy()
    preview = _data_preview(data, data, preview_limit=len(data))
    return {
        "model": model.name,
        "rows": len(data),
        "participants": participants,
        "trials": trials,
        "parameterValues": parameter_rows,
        "data": _json_safe(data.to_dict(orient="records")),
        "preview": preview,
    }


def _draw_bounded_values(
    rng: np.random.Generator,
    n: int,
    lower: float,
    upper: float,
    mean: float,
    sd: float,
    skew: float,
) -> tuple[np.ndarray, np.ndarray]:
    width = max(upper - lower, 1e-9)
    mean = float(np.clip(mean, lower, upper))
    sd = max(abs(sd), width / 1000)
    z = rng.normal(size=n)
    shaped = _skew_normal_standard(rng, z, skew)
    values = mean + sd * shaped
    return np.clip(values, lower, upper), z


def _draw_correlated_covariate_value(rng, spec: dict, latent_z: dict[str, np.ndarray], participant_index: int) -> float:
    z = float(_correlated_covariate_z(rng, spec, latent_z, participant_index, None))
    mean = float(spec.get("mean", 0.0) or 0.0)
    sd = abs(float(spec.get("sd", 1.0) or 1.0))
    skew = float(spec.get("skew", 0.0) or 0.0)
    z = float(_skew_normal_standard(rng, np.asarray([z], dtype=float), skew)[0])
    return float(mean + sd * z)


def _correlated_covariate_z(
    rng: np.random.Generator,
    spec: dict,
    latent_z: dict[str, np.ndarray],
    participant_index: int,
    trials: int | None,
) -> float | np.ndarray:
    weights = _covariate_target_weights(spec, latent_z)
    base = 0.0
    square_sum = 0.0
    for target, corr in weights.items():
        values = latent_z.get(target)
        if values is None or participant_index >= len(values):
            continue
        base += corr * float(values[participant_index])
        square_sum += corr * corr
    residual_sd = float(np.sqrt(max(0.0, 1.0 - square_sum)))
    if trials is None:
        return base + residual_sd * float(rng.normal())
    return base + residual_sd * rng.normal(size=trials)


def _covariate_target_weights(spec: dict, latent_z: dict[str, np.ndarray]) -> dict[str, float]:
    raw_targets = spec.get("targets")
    if not isinstance(raw_targets, dict):
        target = str(spec.get("target", "")).strip()
        raw_targets = {target: spec.get("correlation", 0.0)} if target else {}
    weights: dict[str, float] = {}
    for target, value in raw_targets.items():
        target_name = str(target).strip()
        if not target_name or target_name not in latent_z:
            continue
        try:
            corr = float(value)
        except (TypeError, ValueError):
            corr = 0.0
        if abs(corr) > 0:
            weights[target_name] = float(np.clip(corr, -0.99, 0.99))
    square_sum = sum(value * value for value in weights.values())
    if square_sum > 0.98:
        scale = float(np.sqrt(0.98 / square_sum))
        weights = {target: value * scale for target, value in weights.items()}
    return weights


def _skew_normal_standard(rng: np.random.Generator, z: np.ndarray, skew: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    alpha = float(skew) * 2.0
    if abs(alpha) < 1e-9:
        return z
    delta = alpha / np.sqrt(1.0 + alpha * alpha)
    paired = rng.normal(size=z.shape)
    shaped = delta * np.abs(z) + np.sqrt(max(0.0, 1.0 - delta * delta)) * paired
    mean = delta * np.sqrt(2.0 / np.pi)
    sd = np.sqrt(max(1.0 - (2.0 * delta * delta / np.pi), 1e-9))
    return (shaped - mean) / sd


def _simulated_preview_request(payload: dict) -> dict[str, object]:
    rows = payload.get("rows") or []
    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("Simulate data first.")
    for required in ["id", "condition", "rt", "accuracy"]:
        if required not in data.columns:
            raise ValueError(f"Simulated data is missing {required!r}.")
    data["id"] = data["id"].astype(str)
    data["condition"] = data["condition"].astype(str)
    data["rt"] = pd.to_numeric(data["rt"], errors="coerce")
    data["accuracy"] = pd.to_numeric(data["accuracy"], errors="coerce").fillna(0).round().astype(int)
    data = data.dropna(subset=["rt"]).reset_index(drop=True)
    data.attrs["response_kind"] = "choice"
    data.attrs["response_column"] = "accuracy"
    data.attrs["raw_rows"] = data.copy()
    participant_id = str(payload.get("previewParticipantId", "")).strip()
    selected = _select_preview_data(data, {"previewParticipantId": participant_id}) if participant_id else data
    return _data_preview(selected, data, preview_limit=len(selected))


def _export_cluster_job(payload: dict) -> dict[str, object]:
    kind = str(payload.get("kind", "fit")).strip().lower()
    if kind not in {"fit", "validation"}:
        raise ValueError("Cluster export kind must be fit or validation.")
    analysis_payload = dict(payload.get("payload") or {})
    if not analysis_payload:
        raise ValueError("Cluster export needs a fit or validation payload.")
    cluster_settings = _cluster_settings(payload.get("clusterSettings") or {}, analysis_payload)
    validation_mode = str(analysis_payload.get("mode", "parameter")).strip().lower()
    needs_data = kind == "fit" or (kind == "validation" and validation_mode == "internal")
    if needs_data and not str(analysis_payload.get("path", "")).strip():
        raise ValueError("Cluster export needs mapped data. Load a CSV and complete the data mapping before exporting.")

    export_id = f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    export_dir = CLUSTER_EXPORTS / export_id
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    packaged_payload = dict(analysis_payload)
    row_count = None
    participant_count = None
    if needs_data:
        analysis_data = _load_payload_data(analysis_payload)
        analysis_data.to_csv(export_dir / "analysis_data.csv", index=False)
        row_count = int(len(analysis_data))
        participant_count = int(analysis_data["id"].nunique()) if "id" in analysis_data else None
        packaged_payload.update(
            {
                "path": "analysis_data.csv",
                "participant": "id",
                "condition": "condition",
                "rt": "rt",
                "accuracy": "accuracy",
                "columnFilters": [],
            }
        )
    else:
        packaged_payload.pop("path", None)
        packaged_payload["columnFilters"] = []
    (export_dir / "payload.json").write_text(json.dumps(_json_safe(packaged_payload), indent=2), encoding="utf-8", newline="\n")
    (export_dir / "run_flexddm_cluster.py").write_text(_cluster_runner_script(kind), encoding="utf-8", newline="\n")
    (export_dir / "submit.slurm").write_text(_cluster_slurm_template(kind, cluster_settings), encoding="utf-8", newline="\n")
    (export_dir / "README_CLUSTER.md").write_text(_cluster_readme(kind, cluster_settings, export_id, f"{export_id}.zip"), encoding="utf-8", newline="\n")
    (export_dir / "requirements.txt").write_text(_cluster_requirements_text(), encoding="utf-8", newline="\n")
    (export_dir / "pyproject.toml").write_text(_cluster_pyproject_text(), encoding="utf-8", newline="\n")
    shutil.copytree(SOURCE_ROOT / "flexddm", export_dir / "flexddm", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _copy_selected_generated_models(_selected_models_for_export(kind, analysis_payload), export_dir / "generated_models")
    for filename in ["setup.py"]:
        source = SOURCE_ROOT / filename
        if source.is_file():
            shutil.copy2(source, export_dir / filename)

    archive_path = Path(shutil.make_archive(str(export_dir), "zip", root_dir=CLUSTER_EXPORTS, base_dir=export_id))
    return {
        "exportId": export_id,
        "folderName": export_id,
        "archiveName": archive_path.name,
        "path": str(archive_path),
        "url": f"/runs/cluster_exports/{archive_path.name}",
        "kind": kind,
        "rows": row_count,
        "participants": participant_count,
    }


def _cluster_settings(settings: dict[str, object], analysis_payload: dict[str, object]) -> dict[str, str]:
    workers = str(analysis_payload.get("workers", "4")).strip()
    return {
        "time": _safe_slurm_value(settings.get("time"), "12:00:00"),
        "cpus": _safe_slurm_value(settings.get("cpus"), workers or "4"),
        "mem": _safe_slurm_value(settings.get("mem"), "16G"),
        "partition": _safe_slurm_value(settings.get("partition"), ""),
        "account": _safe_slurm_value(settings.get("account"), ""),
        "python_module": _safe_slurm_value(settings.get("pythonModule"), ""),
    }


def _safe_slurm_value(value: object, default: str) -> str:
    text = str(value if value is not None else default).strip()
    text = re.sub(r"[^A-Za-z0-9_./:+-]+", "", text)
    return text or default


def _selected_models_for_export(kind: str, payload: dict[str, object]) -> list[str]:
    if kind == "fit":
        names = [str(payload.get("model", "")).strip()]
    else:
        names = [str(name).strip() for name in payload.get("models", []) if str(name).strip()]
    return [name for name in names if name]


def _copy_selected_generated_models(model_names: list[str], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if not GENERATED_MODELS.exists():
        return
    for model_name in model_names:
        safe_name = _safe_model_name(model_name)
        for suffix in [".py", ".bounds.json", ".meta.json"]:
            source = GENERATED_MODELS / f"{safe_name}{suffix}"
            if source.is_file():
                shutil.copy2(source, destination / source.name)


def _cluster_requirements_text() -> str:
    return "\n".join(
        [
            "matplotlib>=3.7,<3.10; python_version < '3.10'",
            "matplotlib>=3.7; python_version >= '3.10'",
            "numpy>=1.24,<2.1; python_version < '3.10'",
            "numpy>=1.24; python_version >= '3.10'",
            "numba>=0.60,<0.61; python_version < '3.10'",
            "numba>=0.61; python_version >= '3.10'",
            "pandas>=1.5",
            "pybind11>=2.12",
            "scipy>=1.10",
            "seaborn>=0.12",
            "",
        ]
    )


def _cluster_pyproject_text() -> str:
    return '''[project]
name = "flexddm"
version = "0.1.1"
description = "A cluster-exported FlexDDM analysis package."
requires-python = ">=3.9"
dependencies = []

[project.scripts]
flexddm = "flexddm.app:run"

[build-system]
requires = ["setuptools>=68", "pybind11>=2.12"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["flexddm*"]
'''


def _cluster_runner_script(kind: str) -> str:
    runner_name = "_run_fit_request" if kind == "fit" else "_run_validation_request"
    return f'''from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from flexddm.app import {runner_name}


def _format_seconds(seconds):
    seconds = int(max(0, round(float(seconds or 0))))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{{hours}}h {{minutes}}m {{secs}}s"
    if minutes:
        return f"{{minutes}}m {{secs}}s"
    return f"{{secs}}s"


def _copy_result_file(value, outputs, name=None):
    if not value:
        return None
    path = str(value)
    source = ROOT / path.removeprefix("/") if path.startswith("/runs/") else Path(path)
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        return None
    target = outputs / (name or source.name)
    shutil.copy2(source, target)
    return str(target)


def _write_table(rows, outputs, filename):
    if rows:
        pd.DataFrame(rows).to_csv(outputs / filename, index=False)


def main():
    payload = json.loads((ROOT / "payload.json").read_text(encoding="utf-8"))
    if payload.get("path"):
        payload["path"] = str(ROOT / payload["path"])
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    started = time.time()

    def progress(message, value=None, detail=None):
        pct = "" if value is None else f" {{float(value) * 100:5.1f}}%"
        elapsed = _format_seconds(time.time() - started)
        print(f"[{{elapsed}}]{{pct}} {{message}}", flush=True)

    result = {runner_name}(payload, progress=progress)
    elapsed = time.time() - started
    result["elapsedSeconds"] = int(round(elapsed))
    (outputs / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if "{kind}" == "fit":
        rows = result.get("parameterRows") or [result.get("parameters", {{}})]
        _write_table(rows, outputs, "fit_parameters.csv")
        _write_table(result.get("conditionSummary"), outputs, "condition_summary.csv")
        _write_table(result.get("fitCovariateSummary"), outputs, "participant_covariate_associations.csv")
        _copy_result_file(result.get("outputCsv"), outputs, "fit_result.csv")
        _copy_result_file(result.get("ppcPath"), outputs, "posterior_predictive.png")
        for option in result.get("ppcOptions") or []:
            url = option.get("url")
            if url:
                _copy_result_file(url, outputs)
    else:
        _write_table(result.get("rows"), outputs, "validation_table.csv")
        _copy_result_file(result.get("estimateCsv"), outputs)
        _copy_result_file(result.get("plotUrl"), outputs, "validation_diagnostics.png")

    print(f"FlexDDM {kind} complete in {{_format_seconds(elapsed)}}. Outputs are in {{outputs}}", flush=True)


if __name__ == "__main__":
    main()
'''


def _cluster_slurm_template(kind: str, settings: dict[str, str]) -> str:
    partition = f"#SBATCH --partition={settings['partition']}\n" if settings.get("partition") else ""
    account = f"#SBATCH --account={settings['account']}\n" if settings.get("account") else ""
    return f'''#!/bin/bash
#SBATCH --job-name=flexddm-{kind}
#SBATCH --output=flexddm-%j.out
#SBATCH --error=flexddm-%j.err
#SBATCH --time={settings['time']}
#SBATCH --cpus-per-task={settings['cpus']}
#SBATCH --mem={settings['mem']}
{partition}{account}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

set -e

REQUESTED_PYTHON_MODULE="{settings.get('python_module') or ''}"
PYTHON_BIN=""

if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
        . /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
        . /usr/share/Modules/init/bash
    fi
fi

python_version_ok() {{
    "$1" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 9) else 1)
PY
}}

find_python_bin() {{
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && python_version_ok "$candidate"; then
            PYTHON_BIN="$(command -v "$candidate")"
            return 0
        fi
    done
    return 1
}}

if [ -n "$REQUESTED_PYTHON_MODULE" ]; then
    if ! command -v module >/dev/null 2>&1; then
        echo "Requested Python module '$REQUESTED_PYTHON_MODULE', but the module command is not available." >&2
        exit 1
    fi
    echo "Loading requested Python module: $REQUESTED_PYTHON_MODULE"
    if ! module load "$REQUESTED_PYTHON_MODULE"; then
        echo "Could not load requested Python module '$REQUESTED_PYTHON_MODULE'. Use the exact module name from 'module avail' or leave the field blank for auto-detection." >&2
        exit 1
    fi
fi

if ! find_python_bin && [ -z "$REQUESTED_PYTHON_MODULE" ] && command -v module >/dev/null 2>&1; then
    echo "No Python module specified and current Python is too old or unavailable. Searching module avail for Python 3.9+..."
    for candidate in $(module -t avail 2>&1 | tr ' ' '\\n' | sed 's/(default)//g; s/(D)//g; s/:$//; s/^[[:space:]]*//; s/[[:space:]]*$//' | grep -Ei 'python' | grep -E '3\\.([9]|1[0-9])' | sort -Vr | awk '!seen[$0]++'); do
        [ -z "$candidate" ] && continue
        echo "Trying Python module: $candidate"
        if module load "$candidate" >/dev/null 2>&1; then
            if find_python_bin; then
                echo "Loaded Python module: $candidate"
                break
            fi
            module unload "$candidate" >/dev/null 2>&1 || true
        fi
    done
fi

if ! find_python_bin; then
    echo "FlexDDM cluster exports need Python 3.9 or newer." >&2
    echo "Run 'module avail python' or 'module avail Python', then enter the exact module name in the FlexDDM cluster export dialog." >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
print("Using Python", sys.version.replace("\\n", " "))
if sys.version_info < (3, 9):
    raise SystemExit("FlexDDM cluster exports need Python 3.9 or newer. Load a newer Python module, then resubmit.")
PY

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python run_flexddm_cluster.py
'''


def _cluster_readme(kind: str, settings: dict[str, str], folder_name: str, archive_name: str) -> str:
    return f'''# FlexDDM Cluster Export

This package reproduces the {kind} job exported from the FlexDDM GUI.

## What Is In This Folder

- `payload.json`: the model, backend, validation/fit, and option settings exported from the GUI.
- `analysis_data.csv`: the filtered analysis data, only included for fit jobs and internal consistency reliability.
- `run_flexddm_cluster.py`: the script Slurm runs.
- `submit.slurm`: a starter Slurm job script using the settings you chose.
- `outputs/`: created after the job runs. It will contain result tables, `result.json`, and diagnostic images.

## Requested Slurm Resources

- Time limit: `{settings['time']}`
- CPU cores: `{settings['cpus']}`
- Memory: `{settings['mem']}`
- Partition: `{settings.get('partition') or 'not specified'}`
- Account: `{settings.get('account') or 'not specified'}`
- Python module: `{settings.get('python_module') or 'auto-detect'}`

## Step By Step: Submit With Slurm

1. Copy `{archive_name}` to the cluster.
2. Open a terminal on the cluster, go to the directory containing the zip, and unpack it:

```bash
unzip {archive_name}
```

This creates a folder named `{folder_name}`.

3. Move into the folder:

```bash
cd {folder_name}
```

4. If your cluster uses a different Python module, partition, account, memory, or time limit, edit `submit.slurm`. FlexDDM cluster exports need Python 3.9 or newer. If no Python module was specified, `submit.slurm` tries the current `python3`/`python` first, then searches `module avail` for Python 3.9+ modules.
5. Submit the job:

```bash
sbatch submit.slurm
```

6. Check progress with your cluster's normal queue command, usually:

```bash
squeue -u $USER
```

7. When the job finishes, look in `outputs/`.

The included `submit.slurm` is written with Unix line endings. If you edit it on Windows and Slurm later reports DOS line breaks, run `dos2unix submit.slurm` on the cluster before submitting.

## Run Interactively Without Slurm

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python run_flexddm_cluster.py
```

## Expected Outputs

For fit jobs, expect `fit_parameters.csv`, `condition_summary.csv`, `fit_result.csv`, `result.json`, and posterior predictive PNG files when available.

For validation jobs, expect `validation_table.csv`, `result.json`, and a validation diagnostic PNG when available. Synthetic parameter/model recovery exports do not need a data CSV.
'''


def _run_fit_request(payload: dict, progress=None, cancel_check=None) -> dict[str, object]:
    def report(message: str, value: float | None = None, detail: dict[str, object] | None = None) -> None:
        if progress is not None:
            progress(message, value, detail)

    report("Loading data...", 0.02)
    all_data = _load_payload_data(payload)
    config = _fit_config(payload)
    solver = payload.get("solver", "simulation_de")
    fit_scope = str(payload.get("fitScope", "all")).strip().lower()
    if fit_scope == "selected" and not str(payload.get("participantId", "")).strip():
        raise ValueError("Selected-ID fitting needs at least one participant ID or range, such as 1 or 1-10, 20.")
    participant_ids = _participant_ids_from_payload(all_data, payload, default_all=(fit_scope == "all"))
    fp_config = _fp_config(payload)
    model_name = str(payload.get("model", "StandardDDM"))

    if participant_ids is not None and len(participant_ids) > 1:
        run_name = f"{_safe_model_name(model_name)}_{'fp' if solver == 'fokker_planck' else 'sim'}_batch"
        run_dir = RUNS / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        fit_rows: list[dict[str, object]] = []
        observed_frames: list[pd.DataFrame] = []
        simulated_frames: list[pd.DataFrame] = []
        model: ModelSpec | FokkerPlanckSpec | None = None
        participant_config = replace(config, workers=1)
        worker_count = min(max(int(config.workers), 1), len(participant_ids))
        worker_slots: dict[int, int] = {}
        worker_slots_lock = threading.Lock()
        participant_progress = {index: 0.0 for index in range(1, len(participant_ids) + 1)}
        participant_progress_lock = threading.Lock()

        def aggregate_progress(index: int, value: float) -> float:
            with participant_progress_lock:
                participant_progress[index] = max(0.0, min(1.0, float(value)))
                return 0.98 * (sum(participant_progress.values()) / len(participant_progress))

        def current_worker_slot() -> int:
            thread_id = threading.get_ident()
            with worker_slots_lock:
                if thread_id not in worker_slots:
                    worker_slots[thread_id] = len(worker_slots) + 1
                return worker_slots[thread_id]

        def fit_participant(index: int, participant_id: str):
            worker_slot = current_worker_slot()
            report(
                f"Fitting participant {index} of {len(participant_ids)} ({participant_id})...",
                aggregate_progress(index, 0.0),
                {
                    "workerSlot": worker_slot,
                    "participantIndex": index,
                    "participantId": participant_id,
                    "participantProgress": 0.0,
                    "status": "running",
                },
            )
            participant_data = all_data[all_data["id"].astype(str) == participant_id].reset_index(drop=True)
            participant_data.attrs.update(all_data.attrs)
            participant_data = _refresh_data_constants(participant_data)

            def fit_progress(detail: dict[str, object], participant_index=index, participant=participant_id) -> None:
                evaluations = detail.get("evaluations")
                expected = float(detail.get("expectedEvaluations") or 1)
                eval_progress = min(float(evaluations or 0) / expected, 0.95)
                progress_detail = dict(detail)
                progress_detail.update(
                    {
                        "workerSlot": worker_slot,
                        "participantIndex": participant_index,
                        "participantId": participant,
                        "participantProgress": eval_progress,
                        "status": "running",
                    }
                )
                report(
                    f"Fitting participant {participant_index} of {len(participant_ids)} ({participant})",
                    aggregate_progress(participant_index, eval_progress),
                    progress_detail,
                )

            model, fitter_result, simulated = _fit_once(
                model_name,
                participant_data,
                participant_config,
                solver,
                fp_config,
                payload.get("fpObjective", "g2"),
                participant_id,
                fit_progress,
                cancel_check,
            )
            fit_row = {
                "participant": participant_id,
                "model": fitter_result.model_name,
                "g2": fitter_result.g2,
                "bic": fitter_result.bic,
                "objective": fitter_result.objective_name,
                "objective_score": fitter_result.objective_score,
                "objective_bic": fitter_result.objective_bic,
                **fitter_result.parameters,
            }
            report(
                f"Finished participant {index} of {len(participant_ids)} ({participant_id})",
                aggregate_progress(index, 1.0),
                {
                    "workerSlot": worker_slot,
                    "participantIndex": index,
                    "participantId": participant_id,
                    "participantProgress": 1.0,
                    "status": "finished",
                },
            )
            return index, fit_row, participant_data, simulated, model

        participant_results = []
        if worker_count > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(fit_participant, index, participant_id) for index, participant_id in enumerate(participant_ids, 1)]
                for future in as_completed(futures):
                    if cancel_check is not None and cancel_check():
                        raise InterruptedError("Fit cancelled.")
                    participant_results.append(future.result())
        else:
            for index, participant_id in enumerate(participant_ids, 1):
                participant_results.append(fit_participant(index, participant_id))

        for _, fit_row, participant_data, simulated, fitted_model in sorted(participant_results, key=lambda row: row[0]):
            fit_rows.append(fit_row)
            observed_frames.append(participant_data)
            simulated_frames.append(simulated)
            model = fitted_model

        observed = pd.concat(observed_frames, ignore_index=True)
        simulated = pd.concat(simulated_frames, ignore_index=True)
        observed.attrs.update(all_data.attrs)
        simulated.attrs.update(all_data.attrs)
        fit_frame = pd.DataFrame(fit_rows)
        participant_covariates = _participant_covariate_frame(observed)
        if not participant_covariates.empty:
            fit_frame = fit_frame.merge(participant_covariates, left_on="participant", right_on="id", how="left").drop(columns=["id"], errors="ignore")
            fit_rows = fit_frame.to_dict(orient="records")
        covariate_names = [column for column in participant_covariates.columns if column != "id"]
        covariate_summary = _fit_covariate_summary(fit_frame, covariate_names)
        report("Writing posterior predictive diagnostics...", 0.98)
        ppc_path, ppc_options = _write_ppc_diagnostics(
            observed,
            simulated,
            run_dir,
            "",
            bool(payload.get("posteriorPredictive", True)),
        )
        fit_frame.to_csv(run_dir / "fit_result.csv", index=False)
        return {
            "model": fit_rows[0]["model"] if fit_rows else model_name,
            "parameters": fit_rows[0] if fit_rows else {},
            "parameterRows": fit_rows,
            "scoreName": "G2",
            "objectiveName": str(fit_rows[0]["objective"]) if fit_rows else "G2",
            "objectiveScore": float(sum(float(row["objective_score"]) for row in fit_rows if row["objective_score"] is not None)),
            "objectiveBic": float(sum(float(row["objective_bic"]) for row in fit_rows if row["objective_bic"] is not None)),
            "g2": float(sum(float(row["g2"]) for row in fit_rows)),
            "bic": float(sum(float(row["bic"]) for row in fit_rows)),
            "rowsFit": len(observed),
            "participantsFit": len(participant_ids),
            "outputCsv": str(run_dir / "fit_result.csv"),
            "ppcPath": str(ppc_path) if ppc_path else None,
            "ppcUrl": f"/runs/{run_dir.name}/posterior_predictive.png" if ppc_path else None,
            "ppcOptions": ppc_options,
            "conditionSummary": _condition_summary(observed, simulated),
            "fitCovariateSummary": covariate_summary,
        }

    data = _select_data(all_data, payload)
    report(f"Fitting {len(data)} rows...", 0.05)

    def fit_progress(detail: dict[str, object]) -> None:
        expected = float(detail.get("expectedEvaluations") or 1)
        evaluations = float(detail.get("evaluations") or 0)
        value = 0.05 + 0.8 * min(evaluations / expected, 0.95)
        report("Fitting", value, detail)

    model, fitter_result, simulated = _fit_once(
        model_name,
        data,
        config,
        solver,
        fp_config,
        payload.get("fpObjective", "g2"),
        progress_callback=fit_progress,
        cancel_check=cancel_check,
    )
    run_name = f"{_safe_model_name(model.name)}_fp" if solver == "fokker_planck" else _safe_model_name(model.name)
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    report("Writing posterior predictive diagnostics...", 0.9)
    ppc_path, ppc_options = _write_ppc_diagnostics(
        data,
        simulated,
        run_dir,
        "",
        bool(payload.get("posteriorPredictive", True)),
    )
    output_row = {"model": fitter_result.model_name, "g2": fitter_result.g2, "bic": fitter_result.bic, **fitter_result.parameters}
    if fitter_result.objective_name != "G2":
        output_row.update(
            {
                "objective": fitter_result.objective_name,
                "objective_score": fitter_result.objective_score,
                "objective_bic": fitter_result.objective_bic,
            }
        )
    pd.DataFrame([output_row]).to_csv(run_dir / "fit_result.csv", index=False)
    return _fit_payload(model.name, data, simulated, fitter_result, run_dir, ppc_path, ppc_options)


def _validation_parameter_count(model_name: str, solver: str) -> int:
    try:
        model = _load_fp_model(model_name) if solver == "fokker_planck" else _load_model(model_name)
        return max(1, len(getattr(model, "parameter_names", []) or []))
    except Exception:
        return 5


def _estimated_validation_eta_seconds(
    mode: str,
    model_names: list[str],
    simulations: int,
    trials: int,
    config: FitConfig,
    solver: str,
    participant_count: int | None = None,
) -> int:
    workers = max(int(config.workers or 1), 1)
    parameter_counts = [_validation_parameter_count(name, solver) for name in model_names] or [5]
    mean_parameters = max(1.0, float(sum(parameter_counts)) / len(parameter_counts))
    evaluations_per_fit = max(1, int((config.maxiter + 1) * config.popsize * mean_parameters))
    seconds_per_evaluation = 0.09 if solver == "fokker_planck" else 0.13
    if mode == "model":
        fit_count = max(1, simulations) * max(1, len(model_names)) * max(1, len(model_names))
    elif mode == "internal":
        fit_count = max(1, int(participant_count or 1)) * 2
    elif mode == "curvature":
        fit_count = 1 + int(mean_parameters * mean_parameters)
    elif mode == "profile":
        fit_count = int(mean_parameters * 31)
    else:
        fit_count = max(1, simulations)
    generation_units = 1 if mode in {"curvature", "profile"} else max(1, simulations)
    generation_overhead = generation_units * max(1, trials) * (0.0015 if solver == "fokker_planck" else 0.003)
    refine_multiplier = 1.15 if config.local_refine else 1.0
    estimate = ((fit_count * evaluations_per_fit * seconds_per_evaluation * refine_multiplier) + generation_overhead) / workers
    return int(max(1, round(estimate)))


def _run_validation_request(payload: dict, progress=None, cancel_check=None) -> dict[str, object]:
    def report(message: str, value: float | None = None, detail: dict[str, object] | None = None) -> None:
        if progress is not None:
            progress(message, value, detail)

    mode = payload.get("mode", "parameter")
    config = _fit_config(payload)
    simulations = int(payload.get("simulations", 100))
    trials = int(payload.get("trialsPerSimulation", 100))
    model_names = [str(name) for name in payload.get("models", ["StandardDDM"]) if str(name).strip()]
    solver = payload.get("solver", "simulation_de")
    if mode != "internal":
        initial_eta = _estimated_validation_eta_seconds(mode, model_names, simulations, trials, config, solver)
        report(f"Preparing {mode} validation...", 0.001, {"etaSeconds": initial_eta})
    else:
        report(f"Preparing {mode} validation...", 0.0)
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Validation cancelled.")
    if mode == "internal":
        if len(model_names) != 1:
            raise ValueError("Internal consistency reliability uses exactly one model.")
        data = _select_data(_load_payload_data(payload), payload)
        participant_ids = _participant_ids_from_payload(data, payload, default_all=True) or []
        initial_eta = _estimated_validation_eta_seconds(mode, model_names, simulations, trials, config, solver, len(participant_ids))
        report("Running internal consistency reliability...", 0.001, {"etaSeconds": initial_eta})
        result = _internal_consistency_reliability(
            model_names[0],
            data,
            config,
            payload.get("solver", "simulation_de"),
            _fp_config(payload),
            payload.get("fpObjective", "g2"),
            report,
            cancel_check,
        )
        report("Internal consistency diagnostics ready.", 0.95)
        return result
    if mode in {"curvature", "profile"} and len(model_names) != 1:
        raise ValueError("This validation diagnostic uses exactly one model.")
    if mode == "curvature":
        data = _select_data(_load_payload_data(payload), payload)
        report("Running parameter-curvature diagnostics...", 0.02)
        return _curvature_diagnostics(
            model_names[0],
            data,
            config,
            solver,
            _fp_config(payload),
            payload.get("fpObjective", "g2"),
            _validation_diagnostic_options(payload),
            report,
            cancel_check,
        )
    if mode == "profile":
        report("Running likelihood profile diagnostics...", 0.02)
        return _profile_diagnostics(
            model_names[0],
            trials,
            config,
            solver,
            _fp_config(payload),
            payload.get("fpObjective", "g2"),
            _validation_diagnostic_options(payload),
            report,
            cancel_check,
        )
    if mode == "parameter" and len(model_names) != 1:
        raise ValueError("Parameter recovery uses exactly one model. Choose one model or switch to model recovery.")
    if mode == "model" and len(model_names) < 2:
        raise ValueError("Model recovery needs two or more models.")
    if solver == "fokker_planck":
        selected = [_load_fp_model(name) for name in model_names]
        report("Running Fokker-Planck validation...", 0.0)
        if mode == "model":
            return _model_recovery_diagnostics(selected, simulations, trials, config, _fp_config(payload), True, report, cancel_check)
        else:
            return _parameter_recovery_diagnostics(selected[0], simulations, trials, config, _fp_config(payload), True, report, cancel_check)
    else:
        selected = [_load_model(name) for name in model_names]
        report("Running simulation validation...", 0.0)
        if mode == "model":
            return _model_recovery_diagnostics(selected, simulations, trials, config, _fp_config(payload), False, report, cancel_check)
        else:
            return _parameter_recovery_diagnostics(selected[0], simulations, trials, config, _fp_config(payload), False, report, cancel_check)


def _parameter_recovery_diagnostics(
    model,
    simulations: int,
    trials_per_simulation: int,
    fit_config: FitConfig,
    fp_config: FokkerPlanckConfig,
    use_fp: bool,
    report,
    cancel_check=None,
    parameter_bounds=None,
) -> dict[str, object]:
    if use_fp and not hasattr(model, "sample_dataframe"):
        raise ValueError(f"{model.name} does not yet support synthetic FP parameter recovery because it needs trial-wise design data.")
    rng = np.random.default_rng(13 if use_fp else 1)
    fallback = _validation_fallback_data(model)
    bounds = model.bounds_for_data(fallback)
    generated_values_by_simulation = [
        np.array([rng.uniform(lower, upper) for lower, upper in bounds])
        for _ in range(simulations)
    ]
    worker_count = min(max(int(fit_config.workers), 1), max(simulations, 1))
    worker_fit_config = replace(fit_config, workers=1)
    worker_slots: dict[int, int] = {}
    worker_slots_lock = threading.Lock()
    progress_by_simulation = {index: 0.0 for index in range(simulations)}
    progress_lock = threading.Lock()

    def current_worker_slot() -> int:
        thread_id = threading.get_ident()
        with worker_slots_lock:
            if thread_id not in worker_slots:
                worker_slots[thread_id] = len(worker_slots) + 1
            return worker_slots[thread_id]

    def aggregate_progress(index: int, value: float) -> float:
        with progress_lock:
            progress_by_simulation[index] = max(0.0, min(1.0, float(value)))
            return 0.98 * (sum(progress_by_simulation.values()) / max(len(progress_by_simulation), 1))

    def run_synthetic_participant(simulation_index: int) -> list[dict[str, object]]:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Validation cancelled.")
        slot = current_worker_slot()
        label = f"Worker {slot} | Synthetic participant {simulation_index + 1}"
        report(
            f"Parameter recovery synthetic participant {simulation_index + 1} of {simulations}",
            aggregate_progress(simulation_index, 0.0),
            {"workerSlot": slot, "label": label, "participantProgress": 0.0, "status": "running"},
        )
        generated_values = generated_values_by_simulation[simulation_index]

        def fit_progress(detail: dict[str, object]) -> None:
            evaluations = float(detail.get("evaluations") or 0)
            expected = float(detail.get("expectedEvaluations") or 1)
            local_progress = min(evaluations / expected, 0.95)
            progress_detail = dict(detail)
            progress_detail.update({"workerSlot": slot, "label": label, "participantProgress": local_progress, "status": "running"})
            report(f"Parameter recovery synthetic participant {simulation_index + 1} of {simulations}", aggregate_progress(simulation_index, local_progress), progress_detail)

        if use_fp:
            synthetic = model.sample_dataframe(generated_values, n_trials=trials_per_simulation, participant_id=simulation_index, config=fp_config)
            result = fit_fokker_planck_model(model, synthetic, worker_fit_config, fp_config, progress_callback=fit_progress, cancel_callback=cancel_check)
        else:
            synthetic = model.simulate_dataframe(generated_values, n_trials=trials_per_simulation, participant_id=simulation_index)
            result = fit_model(model, synthetic, worker_fit_config, progress_callback=fit_progress, cancel_callback=cancel_check)
        generated = dict(zip(model.parameter_names, map(float, generated_values)))
        rows = []
        for parameter in model.parameter_names:
            rows.append(
                {
                    "simulation": simulation_index + 1,
                    "parameter": parameter,
                    "generated": generated[parameter],
                    "recovered": result.parameters[parameter],
                }
            )
        report(
            f"Finished synthetic participant {simulation_index + 1} of {simulations}",
            aggregate_progress(simulation_index, 1.0),
            {"workerSlot": slot, "label": label, "participantProgress": 1.0, "status": "finished"},
        )
        return rows

    estimate_rows: list[dict[str, object]] = []
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_synthetic_participant, simulation_index) for simulation_index in range(simulations)]
            for future in as_completed(futures):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("Validation cancelled.")
                estimate_rows.extend(future.result())
    else:
        for simulation_index in range(simulations):
            estimate_rows.extend(run_synthetic_participant(simulation_index))

    estimate_frame = pd.DataFrame(estimate_rows)
    run_dir = RUNS / f"parameter_recovery_{_safe_model_name(model.name)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    estimate_frame.to_csv(run_dir / "parameter_recovery_estimates.csv", index=False)
    rows = []
    for parameter, group in estimate_frame.groupby("parameter", sort=True):
        correlation = None
        p_value = None
        if len(group) >= 3 and group["generated"].nunique() > 1 and group["recovered"].nunique() > 1:
            from scipy.stats import pearsonr

            correlation, p_value = pearsonr(group["generated"], group["recovered"])
        rows.append(
            {
                "parameter": parameter,
                "simulations": int(group["simulation"].nunique()),
                "pearson_r": None if correlation is None else round(float(correlation), 4),
                "p_value": None if p_value is None else round(float(p_value), 4),
            }
        )
    plot_path = run_dir / "parameter_recovery.png"
    plot_parameter_recovery(estimate_frame, plot_path, title=f"{model.name} parameter recovery")
    report("Parameter recovery diagnostics ready.", 1.0)
    return {
        "mode": "parameter",
        "rows": rows,
        "plotUrl": f"/runs/{run_dir.name}/{plot_path.name}",
        "estimateCsv": f"/runs/{run_dir.name}/parameter_recovery_estimates.csv",
    }


def _curvature_diagnostics(
    model_name: str,
    data: pd.DataFrame,
    fit_config: FitConfig,
    solver: str,
    fp_config: FokkerPlanckConfig,
    fp_objective: str,
    diagnostic_options: dict[str, float],
    report,
    cancel_check=None,
) -> dict[str, object]:
    if data.empty:
        raise ValueError("Parameter-curvature validation needs mapped data.")
    model, result, _ = _fit_once(
        model_name,
        data,
        fit_config,
        solver,
        fp_config,
        fp_objective,
        progress_callback=lambda detail: report("Fitting model for curvature diagnostics...", min(0.45, _detail_progress(detail) * 0.45), detail),
        cancel_check=cancel_check,
    )
    names = list(model.parameter_names)
    values = np.array([float(result.parameters[name]) for name in names], dtype=float)
    bounds = model.bounds_for_data(data)
    report("Estimating Hessian by finite differences...", 0.50)
    hessian = _finite_difference_hessian(
        lambda vector: _validation_objective_score(model, data, vector, fit_config, solver, fp_config, fp_objective),
        values,
        bounds,
        step_fraction=diagnostic_options["hessian_step_fraction"],
        min_step=diagnostic_options["hessian_min_step"],
        cancel_check=cancel_check,
    )
    covariance, correlation = _covariance_and_correlation_from_hessian(hessian, diagnostic_options["hessian_ridge"])
    run_dir = RUNS / f"curvature_{_safe_model_name(model.name)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            rows.append(
                {
                    "parameter_1": left,
                    "parameter_2": right,
                    "hessian": float(hessian[i, j]),
                    "covariance": float(covariance[i, j]),
                    "correlation": float(correlation[i, j]),
                    "high_collinearity": bool(i != j and abs(float(correlation[i, j])) > 0.85),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "parameter_curvature_matrix.csv", index=False)
    plot_path = run_dir / "parameter_curvature.png"
    _plot_curvature_heatmap(correlation, names, plot_path, solver, result.objective_name)
    report("Parameter-curvature diagnostics ready.", 1.0)
    max_offdiag = 0.0
    if len(names) > 1:
        mask = ~np.eye(len(names), dtype=bool)
        max_offdiag = float(np.nanmax(np.abs(correlation[mask])))
    return {
        "mode": "curvature",
        "rows": rows,
        "plotUrl": f"/runs/{run_dir.name}/{plot_path.name}",
        "estimateCsv": f"/runs/{run_dir.name}/parameter_curvature_matrix.csv",
    }


def _profile_diagnostics(
    model_name: str,
    trials_per_simulation: int,
    fit_config: FitConfig,
    solver: str,
    fp_config: FokkerPlanckConfig,
    fp_objective: str,
    diagnostic_options: dict[str, float],
    report,
    cancel_check=None,
) -> dict[str, object]:
    model = _load_fp_model(model_name) if solver == "fokker_planck" else _load_model(model_name)
    fallback = _validation_fallback_data(model)
    bounds = model.bounds_for_data(fallback)
    true_values = np.array([(lower + upper) / 2 for lower, upper in bounds], dtype=float)
    if solver == "fokker_planck":
        if not hasattr(model, "sample_dataframe"):
            raise ValueError(f"{model.name} cannot generate synthetic data for profile validation.")
        synthetic = model.sample_dataframe(true_values, n_trials=trials_per_simulation, participant_id="profile", config=fp_config)
    else:
        synthetic = model.simulate_dataframe(true_values, n_trials=trials_per_simulation, participant_id="profile")
    names = list(model.parameter_names)
    grid_points = max(5, int(diagnostic_options["profile_grid_points"]))
    bound_trim = float(np.clip(diagnostic_options["profile_bound_trim"], 0.0, 0.45))
    rows = []
    total = max(1, len(names) * grid_points)
    done = 0
    for parameter_index, parameter in enumerate(names):
        lower, upper = bounds[parameter_index]
        span = max(float(upper) - float(lower), 1e-9)
        grid_lower = float(lower) + bound_trim * span
        grid_upper = float(upper) - bound_trim * span
        if grid_upper <= grid_lower:
            grid_lower, grid_upper = float(lower), float(upper)
        for value in np.linspace(grid_lower, grid_upper, grid_points):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Validation cancelled.")
            candidate = true_values.copy()
            candidate[parameter_index] = float(value)
            score = _validation_objective_score(model, synthetic, candidate, fit_config, solver, fp_config, fp_objective)
            rows.append(
                {
                    "parameter": parameter,
                    "value": float(value),
                    "true_value": float(true_values[parameter_index]),
                    "objective": float(score),
                    "log_likelihood_equivalent": float(-0.5 * score),
                }
            )
            done += 1
            if done % 5 == 0:
                report("Sweeping likelihood/profile slices...", 0.05 + 0.9 * (done / total))
    frame = pd.DataFrame(rows)
    summaries = []
    for parameter, group in frame.groupby("parameter", sort=False):
        best = group.loc[group["objective"].idxmin()]
        sorted_group = group.sort_values("value")
        objective_range = float(group["objective"].max() - group["objective"].min())
        summaries.append(
            {
                "parameter": parameter,
                "true_value": float(best["true_value"]),
                "best_slice_value": float(best["value"]),
                "objective_range": objective_range,
                "flat_profile_flag": bool(objective_range < 1.0),
                "near_true_peak": bool(abs(float(best["value"]) - float(best["true_value"])) <= (float(sorted_group["value"].max() - sorted_group["value"].min()) / max(grid_points - 1, 1))),
            }
        )
    run_dir = RUNS / f"profiles_{_safe_model_name(model.name)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(run_dir / "likelihood_profiles.csv", index=False)
    plot_path = run_dir / "likelihood_profiles.png"
    _plot_likelihood_profiles(frame, plot_path, solver, fp_objective)
    report("Likelihood profile diagnostics ready.", 1.0)
    return {
        "mode": "profile",
        "rows": summaries,
        "plotUrl": f"/runs/{run_dir.name}/{plot_path.name}",
        "estimateCsv": f"/runs/{run_dir.name}/likelihood_profiles.csv",
    }


def _validation_objective_score(
    model,
    data: pd.DataFrame,
    values: np.ndarray,
    fit_config: FitConfig,
    solver: str,
    fp_config: FokkerPlanckConfig,
    fp_objective: str,
) -> float:
    values = np.asarray(values, dtype=float)
    try:
        if solver == "fokker_planck":
            if str(fp_objective).strip().lower() == "likelihood":
                return float(model.neg2_log_likelihood(values, data, fp_config))
            targets = empirical_targets(data, list(fit_config.cdf_quantiles), list(fit_config.caf_quantiles))
            try:
                props = model.predict_props(values, targets, fp_config, data=data)
            except TypeError:
                props = model.predict_props(values, targets, fp_config)
            return float(g2_statistic(targets, props))
        targets = empirical_targets(data, list(fit_config.cdf_quantiles), list(fit_config.caf_quantiles))
        simulated = model.simulate_dataframe(
            values,
            n_trials=fit_config.simulated_trials or max(len(data), model.default_trials),
            dt=fit_config.simulation_dt,
            noise_sd=fit_config.simulation_noise_sd,
            seed=fit_config.seed,
            data_like=data,
        )
        return float(g2_statistic(targets, simulated_props(simulated, targets)))
    except Exception:
        return float(np.iinfo(np.int64).max)


def _validation_diagnostic_options(payload: dict) -> dict[str, float]:
    def number(name: str, default: float, lower: float, upper: float) -> float:
        try:
            value = float(payload.get(name, default))
        except (TypeError, ValueError):
            value = default
        return float(np.clip(value, lower, upper))

    return {
        "hessian_step_fraction": number("hessianStepFraction", 0.01, 1e-5, 0.25),
        "hessian_min_step": number("hessianMinStep", 0.0001, 1e-8, 0.25),
        "hessian_ridge": number("hessianRidge", 1e-6, 0.0, 0.1),
        "profile_grid_points": number("profileGridPoints", 31, 5, 201),
        "profile_bound_trim": number("profileBoundTrim", 0.01, 0.0, 0.45),
    }


def _finite_difference_hessian(
    objective,
    values: np.ndarray,
    bounds: list[tuple[float, float]],
    step_fraction: float = 0.01,
    min_step: float = 1e-4,
    cancel_check=None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    hessian = np.zeros((n, n), dtype=float)
    widths = np.array([max(upper - lower, 1e-6) for lower, upper in bounds], dtype=float)
    steps = np.maximum(widths * step_fraction, np.maximum(np.abs(values) * step_fraction * 0.1, min_step))

    def clipped(candidate: np.ndarray) -> np.ndarray:
        out = candidate.copy()
        for idx, (lower, upper) in enumerate(bounds):
            out[idx] = min(float(upper), max(float(lower), float(out[idx])))
        return out

    f0 = float(objective(values))
    for i in range(n):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Validation cancelled.")
        hi = steps[i]
        ei = np.zeros(n, dtype=float)
        ei[i] = 1.0
        plus = clipped(values + ei * hi)
        minus = clipped(values - ei * hi)
        f_plus = float(objective(plus))
        f_minus = float(objective(minus))
        denom = max((plus[i] - values[i]) * (values[i] - minus[i]), 1e-12)
        hessian[i, i] = (f_plus - 2 * f0 + f_minus) / denom
        for j in range(i + 1, n):
            hj = steps[j]
            ej = np.zeros(n, dtype=float)
            ej[j] = 1.0
            fpp = float(objective(clipped(values + ei * hi + ej * hj)))
            fpm = float(objective(clipped(values + ei * hi - ej * hj)))
            fmp = float(objective(clipped(values - ei * hi + ej * hj)))
            fmm = float(objective(clipped(values - ei * hi - ej * hj)))
            denom = max(4 * hi * hj, 1e-12)
            hessian[i, j] = hessian[j, i] = (fpp - fpm - fmp + fmm) / denom
    return hessian


def _covariance_and_correlation_from_hessian(hessian: np.ndarray, ridge_fraction: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(hessian, dtype=float)
    ridge = max(float(np.nanmax(np.abs(np.diag(matrix)))) * max(ridge_fraction, 0.0), 1e-12)
    covariance = np.linalg.pinv(matrix + np.eye(matrix.shape[0]) * ridge)
    diagonal = np.sqrt(np.maximum(np.abs(np.diag(covariance)), 1e-12))
    correlation = covariance / np.outer(diagonal, diagonal)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return covariance, correlation


def _plot_curvature_heatmap(correlation: np.ndarray, names: list[str], output_path: Path, solver: str, objective_name: str) -> None:
    import matplotlib.pyplot as plt

    size = max(5, min(12, 1.1 * len(names) + 3))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)), labels=names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), labels=names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{correlation[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#18212f")
    ax.set_title(f"Parameter curvature correlations\n{solver}, {objective_name}")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Correlation from inverse Hessian")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_likelihood_profiles(frame: pd.DataFrame, output_path: Path, solver: str, fp_objective: str) -> None:
    import matplotlib.pyplot as plt

    parameters = list(frame["parameter"].drop_duplicates())
    cols = min(3, max(1, len(parameters)))
    rows = int(np.ceil(len(parameters) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, parameter in zip(axes.ravel(), parameters):
        group = frame.loc[frame["parameter"] == parameter].sort_values("value")
        ax.plot(group["value"], group["log_likelihood_equivalent"], color="#2a9d8f", linewidth=2)
        true_value = float(group["true_value"].iloc[0])
        ax.axvline(true_value, color="#e76f51", linestyle="--", linewidth=1.4, label="ground truth")
        ax.set_title(parameter)
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Log-likelihood equivalent")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes.ravel()[len(parameters) :]:
        ax.axis("off")
    fig.suptitle(f"Likelihood / objective profiles ({solver}, {fp_objective})", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _detail_progress(detail: dict[str, object]) -> float:
    evaluations = float(detail.get("evaluations") or 0)
    expected = float(detail.get("expectedEvaluations") or 1)
    return max(0.0, min(1.0, evaluations / max(expected, 1.0)))


def _apply_parameter_overrides(model, rows):
    if not rows:
        return model
    normalized = _normalize_parameter_bounds(rows)
    if not normalized:
        return model
    parameters = []
    changed = False
    for parameter in model.parameters:
        if parameter.name in normalized:
            lower, upper = normalized[parameter.name]
            parameters.append(Parameter(parameter.name, lower, upper))
            changed = True
        else:
            parameters.append(parameter)
    if not changed:
        return model
    parameter_tuple = tuple(parameters)
    if isinstance(model, (ModelSpec, FokkerPlanckSpec, ProcessModelSpec)):
        updated_model = replace(model, parameters=parameter_tuple)
        for attr in ("racing_drifts", "racing_bound", "racing_nondecision"):
            if hasattr(model, attr):
                try:
                    object.__setattr__(updated_model, attr, getattr(model, attr))
                except Exception:
                    pass
        return updated_model
    clone = copy.copy(model)
    clone.parameters = parameter_tuple
    return clone


def _model_schematic_preview(payload: dict) -> dict[str, object]:
    model_name = _safe_model_name(payload.get("modelName", "PreviewModel")) or "PreviewModel"
    code = str(payload.get("code", "") or "")
    if not code.strip():
        raise ValueError("Add or load model code before generating a schematic.")
    model = _load_preview_model_from_code(model_name, code, payload.get("parameterBounds") or [])
    run_dir = RUNS / "model_schematics"
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_path = run_dir / f"{model_name}_{uuid.uuid4().hex[:8]}.png"
    trial_row = payload.get("trialRow")
    if not isinstance(trial_row, dict):
        trial_row = None
    parameter_values = _parameter_value_payload(payload)
    plot_model_schematic(model, plot_path, title=f"{getattr(model, 'name', model_name)} schematic", trial_row=trial_row, parameter_values=parameter_values)
    data = model_schematic_data(model, trial_row=trial_row, parameter_values=parameter_values)
    return {
        "plotUrl": f"/runs/{run_dir.name}/{plot_path.name}",
        "model": getattr(model, "name", model_name),
        "mode": "process" if hasattr(model, "drift") and hasattr(model, "bound") else "simulation",
        "schematic": data,
    }


def _model_trajectory_preview(payload: dict) -> dict[str, object]:
    model_name = _safe_model_name(payload.get("modelName", "PreviewModel")) or "PreviewModel"
    code = str(payload.get("code", "") or "")
    if not code.strip():
        raise ValueError("Add or load model code before simulating a trajectory.")
    model = _load_preview_model_from_code(model_name, code, payload.get("parameterBounds") or [])
    trial_row = payload.get("trialRow")
    if not isinstance(trial_row, dict):
        trial_row = None
    return {
        "trajectory": model_trajectory_data(
            model,
            trial_row=trial_row,
            parameter_values=_parameter_value_payload(payload),
            condition=payload.get("condition"),
            seed=int(payload.get("seed", 1)),
        )
    }


def _model_preview_error_payload(exc: Exception) -> dict[str, object]:
    line = getattr(exc, "lineno", None)
    offset = getattr(exc, "offset", None)
    source_text = getattr(exc, "text", None)
    if line is None:
        traceback_object = exc.__traceback__
        preview_line = None
        while traceback_object is not None:
            filename = str(traceback_object.tb_frame.f_code.co_filename)
            if "model_preview_modules" in filename:
                preview_line = traceback_object.tb_lineno
            traceback_object = traceback_object.tb_next
        line = preview_line
    message = str(exc).strip() or exc.__class__.__name__
    if isinstance(exc, SyntaxError) and getattr(exc, "msg", None):
        message = str(exc.msg)
    return {
        "error": message,
        "errorKind": exc.__class__.__name__,
        "errorLine": line,
        "errorOffset": offset,
        "errorText": str(source_text).strip() if source_text else "",
    }


def _parameter_value_payload(payload: dict) -> dict[str, float]:
    raw = payload.get("parameterValues")
    if not isinstance(raw, dict):
        return {}
    values: dict[str, float] = {}
    for name, value in raw.items():
        safe_name = _safe_parameter_name(str(name))
        if not safe_name:
            continue
        try:
            values[safe_name] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _load_preview_model_from_code(model_name: str, code: str, parameter_bounds: list[dict[str, object]]):
    preview_dir = RUNS / "model_preview_modules"
    preview_dir.mkdir(parents=True, exist_ok=True)
    path = preview_dir / f"{model_name}_{uuid.uuid4().hex}.py"
    path.write_text(code, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location(f"preview_{model_name}_{uuid.uuid4().hex}", path)
        if spec is None or spec.loader is None:
            raise ValueError("Could not prepare the model preview module.")
        module = importlib.util.module_from_spec(spec)
        bounds = _normalize_parameter_bounds(parameter_bounds)
        if bounds:
            module.PARAMETER_BOUNDS = bounds
        spec.loader.exec_module(module)
        process_model = getattr(module, "process_model", None)
        fp_model = getattr(module, "fp_model", None)
        model = getattr(module, "model", None)
        selected = process_model if isinstance(process_model, ProcessModelSpec) else model if hasattr(model, "racing_drifts") else fp_model if _is_fp_model(fp_model) else model
        if selected is None:
            raise ValueError("The editor code must expose process_model, fp_model, or model.")
        return _apply_parameter_overrides(selected, parameter_bounds)
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def _model_recovery_diagnostics(
    models: list,
    simulations: int,
    trials_per_simulation: int,
    fit_config: FitConfig,
    fp_config: FokkerPlanckConfig,
    use_fp: bool,
    report,
    cancel_check=None,
) -> dict[str, object]:
    for model in models:
        if use_fp and not hasattr(model, "sample_dataframe"):
            raise ValueError(f"{model.name} does not yet support synthetic FP model recovery because it needs trial-wise design data.")
    rng = np.random.default_rng(17 if use_fp else 2)
    total_steps = max(len(models) * simulations, 1)
    worker_count = min(max(int(fit_config.workers), 1), total_steps)
    worker_fit_config = replace(fit_config, workers=1)
    worker_slots: dict[int, int] = {}
    worker_slots_lock = threading.Lock()
    progress_by_task = {index: 0.0 for index in range(total_steps)}
    progress_lock = threading.Lock()
    tasks: list[tuple[int, object, int, np.ndarray]] = []
    task_index = 0
    for generating_model in models:
        fallback = _validation_fallback_data(generating_model)
        bounds = generating_model.bounds_for_data(fallback)
        for simulation_index in range(simulations):
            values = np.array([rng.uniform(lower, upper) for lower, upper in bounds])
            tasks.append((task_index, generating_model, simulation_index, values))
            task_index += 1

    def current_worker_slot() -> int:
        thread_id = threading.get_ident()
        with worker_slots_lock:
            if thread_id not in worker_slots:
                worker_slots[thread_id] = len(worker_slots) + 1
            return worker_slots[thread_id]

    def aggregate_progress(task_id: int, value: float) -> float:
        with progress_lock:
            progress_by_task[task_id] = max(0.0, min(1.0, float(value)))
            return 0.98 * (sum(progress_by_task.values()) / max(len(progress_by_task), 1))

    def run_model_recovery_task(task_id: int, generating_model, simulation_index: int, values: np.ndarray) -> tuple[str, str]:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Validation cancelled.")
        slot = current_worker_slot()
        generated_label = _clean_model_label(generating_model.name)
        label = f"Worker {slot} | {generated_label} synthetic {simulation_index + 1}"
        report(
            f"Model recovery {generating_model.name}: synthetic participant {simulation_index + 1} of {simulations}",
            aggregate_progress(task_id, 0.0),
            {"workerSlot": slot, "label": label, "participantProgress": 0.0, "status": "running"},
        )
        if use_fp:
            synthetic = generating_model.sample_dataframe(values, n_trials=trials_per_simulation, participant_id=simulation_index, config=fp_config)
        else:
            synthetic = generating_model.simulate_dataframe(values, n_trials=trials_per_simulation, participant_id=simulation_index)
        fits = []
        for candidate_index, candidate in enumerate(models):
            local_base = candidate_index / max(len(models), 1)

            def fit_progress(detail: dict[str, object], base=local_base) -> None:
                evaluations = float(detail.get("evaluations") or 0)
                expected = float(detail.get("expectedEvaluations") or 1)
                candidate_progress = min(evaluations / expected, 0.95)
                local_progress = min(base + candidate_progress / max(len(models), 1), 0.98)
                progress_detail = dict(detail)
                progress_detail.update({"workerSlot": slot, "label": label, "participantProgress": local_progress, "status": "running"})
                report(f"Model recovery {generated_label}: synthetic participant {simulation_index + 1} of {simulations}", aggregate_progress(task_id, local_progress), progress_detail)

            if use_fp:
                fits.append(fit_fokker_planck_model(candidate, synthetic, worker_fit_config, fp_config, progress_callback=fit_progress, cancel_callback=cancel_check))
            else:
                fits.append(fit_model(candidate, synthetic, worker_fit_config, progress_callback=fit_progress, cancel_callback=cancel_check))
        winner = _clean_model_label(min(fits, key=lambda item: item.bic).model_name)
        report(
            f"Finished model recovery synthetic participant {simulation_index + 1} of {simulations}",
            aggregate_progress(task_id, 1.0),
            {"workerSlot": slot, "label": label, "participantProgress": 1.0, "status": "finished"},
        )
        return generated_label, winner

    winner_rows: list[tuple[str, str]] = []
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_model_recovery_task, *task) for task in tasks]
            for future in as_completed(futures):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("Validation cancelled.")
                winner_rows.append(future.result())
    else:
        for task in tasks:
            winner_rows.append(run_model_recovery_task(*task))

    rows: list[dict[str, object]] = []
    for generating_model in models:
        generated_label = _clean_model_label(generating_model.name)
        winners = {_clean_model_label(candidate.name): 0 for candidate in models}
        for generated_by, winner in winner_rows:
            if generated_by == generated_label:
                winners[winner] = winners.get(winner, 0) + 1
        for candidate_name, wins in winners.items():
            rows.append(
                {
                    "generatedBy": generated_label,
                    "fitModel": candidate_name,
                    "count": wins,
                    "proportion": wins / simulations if simulations else 0,
                }
            )
    row_frame = pd.DataFrame(rows)
    run_dir = RUNS / "model_recovery"
    run_dir.mkdir(parents=True, exist_ok=True)
    row_frame.to_csv(run_dir / "model_recovery_matrix.csv", index=False)
    plot_path = run_dir / "model_recovery.png"
    plot_model_recovery(row_frame, plot_path)
    report("Model recovery diagnostics ready.", 1.0)
    return {"mode": "model", "rows": rows, "plotUrl": f"/runs/{run_dir.name}/{plot_path.name}", "estimateCsv": f"/runs/{run_dir.name}/model_recovery_matrix.csv"}


def _validation_fallback_data(model) -> pd.DataFrame:
    conditions = getattr(model, "conditions", ("0",))
    first = str(conditions[0] if conditions else "0")
    fallback = pd.DataFrame({"id": [0], "condition": [first], "rt": [0.45], "accuracy": [1]})
    fallback.attrs["bound_constants"] = {"min_rt": 0.45}
    return fallback


def _clean_model_label(name: str) -> str:
    return str(name).replace(" (Fokker-Planck)", "")


def _raw_rows_for(data: pd.DataFrame, *, unfiltered: bool = False) -> pd.DataFrame:
    key = "raw_unfiltered_rows" if unfiltered else "raw_rows"
    raw_rows = data.attrs.get(key)
    if isinstance(raw_rows, pd.DataFrame):
        return raw_rows
    return data


def _preview_records(data: pd.DataFrame, limit: int = 8) -> list[dict[str, object]]:
    raw_rows = _raw_rows_for(data).head(limit).reset_index(drop=True).copy()
    flex_ids = data["id"].astype(str).head(limit).reset_index(drop=True).tolist() if "id" in data else []
    if flex_ids:
        raw_rows.insert(0, "FlexDDM_ID", flex_ids)
    return raw_rows.to_dict(orient="records")


def _column_filter_metadata(data: pd.DataFrame) -> list[dict[str, object]]:
    discrete_numeric_unique_limit = 12
    raw_rows = _raw_rows_for(data)
    metadata: list[dict[str, object]] = []
    for column in raw_rows.columns:
        series = raw_rows[column]
        nonmissing = series.dropna()
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_count = int(numeric.notna().sum())
        is_numeric = bool(len(nonmissing) and numeric_count / max(len(nonmissing), 1) >= 0.9)
        unique_values = list(dict.fromkeys(nonmissing.astype(str).tolist()))
        use_discrete_checkboxes = is_numeric and 0 < len(unique_values) <= discrete_numeric_unique_limit
        row: dict[str, object] = {"name": str(column), "type": "categorical" if use_discrete_checkboxes or not is_numeric else "numeric"}
        if is_numeric and not use_discrete_checkboxes:
            clean_numeric = numeric.dropna()
            row["min"] = float(clean_numeric.min()) if not clean_numeric.empty else None
            row["max"] = float(clean_numeric.max()) if not clean_numeric.empty else None
        else:
            values = unique_values
            row["values"] = values[:250]
            row["truncated"] = len(values) > 250
        metadata.append(row)
    return metadata


def _subset_data_with_attrs(data: pd.DataFrame, mask: pd.Series | np.ndarray | list[bool]) -> pd.DataFrame:
    mask_series = pd.Series(mask, index=data.index).fillna(False).astype(bool)
    attrs = dict(data.attrs)
    raw_rows = attrs.get("raw_rows")
    subset = data.loc[mask_series].copy()
    if isinstance(raw_rows, pd.DataFrame):
        if raw_rows.index.equals(data.index):
            raw_subset = raw_rows.loc[mask_series].copy()
        elif len(raw_rows) == len(data):
            raw_subset = raw_rows.iloc[np.flatnonzero(mask_series.to_numpy())].copy()
        else:
            raw_subset = raw_rows.head(0).copy()
        attrs["raw_rows"] = raw_subset.reset_index(drop=True)
    subset = subset.reset_index(drop=True)
    subset.attrs.update(attrs)
    return subset


def _data_preview(data: pd.DataFrame, all_data: pd.DataFrame | None = None, preview_limit: int = 8) -> dict:
    all_data = all_data if all_data is not None else data
    response_kind = str(data.attrs.get("response_kind", "accuracy"))
    response_label = "Accuracy" if response_kind == "accuracy" else "Choice / upper-bound response"
    participant_summary = (
        data.groupby("id", dropna=False)
        .agg(n=("rt", "size"), mean_rt=("rt", "mean"), response_mean=("accuracy", "mean"))
        .reset_index()
    )
    participant_summary["id"] = participant_summary["id"].astype(str)
    participant_summary["mean_rt"] = participant_summary["mean_rt"].round(4)
    participant_summary["response_mean"] = participant_summary["response_mean"].round(4)
    condition_summary = (
        data.groupby("condition", dropna=False)
        .agg(n=("rt", "size"), mean_rt=("rt", "mean"), response_mean=("accuracy", "mean"))
        .reset_index()
    )
    condition_summary["condition"] = condition_summary["condition"].astype(str)
    condition_summary["mean_rt"] = condition_summary["mean_rt"].round(4)
    condition_summary["response_mean"] = condition_summary["response_mean"].round(4)
    response_summary = (
        data.groupby("accuracy", dropna=False)
        .agg(n=("rt", "size"), mean_rt=("rt", "mean"))
        .reset_index()
    )
    response_summary["accuracy"] = response_summary["accuracy"].astype(str)
    response_summary["mean_rt"] = response_summary["mean_rt"].round(4)
    condition_response_summary = (
        data.groupby(["condition", "accuracy"], dropna=False)
        .agg(n=("rt", "size"), mean_rt=("rt", "mean"))
        .reset_index()
    )
    condition_response_summary["condition"] = condition_response_summary["condition"].astype(str)
    condition_response_summary["accuracy"] = condition_response_summary["accuracy"].astype(str)
    condition_response_summary["mean_rt"] = condition_response_summary["mean_rt"].round(4)
    hist_counts, hist_edges = pd.cut(data["rt"], bins=12, retbins=True, include_lowest=True)
    rt_histogram = [
        {
            "label": f"{hist_edges[index]:.3f}-{hist_edges[index + 1]:.3f}",
            "count": int((hist_counts == hist_counts.cat.categories[index]).sum()),
        }
        for index in range(len(hist_edges) - 1)
    ]
    supplied_constants = dict(data.attrs.get("bound_constants", {}))
    constants = dict(supplied_constants)
    constants.setdefault("min_rt", float(data["rt"].min()))
    constant_sources = dict(data.attrs.get("bound_constant_sources", {}))
    constant_sources.setdefault("min_rt", "min(rt)")
    data_plot_url = _write_data_overview_plot(data, response_label)
    covariate_rows = _covariate_summary(data)
    return {
        "rows": len(data),
        "participants": int(data["id"].nunique()),
        "responseKind": response_kind,
        "responseLabel": response_label,
        "responseColumn": str(data.attrs.get("response_column", "accuracy")),
        "participantIds": _sort_id_values(data["id"].unique()),
        "allParticipantIds": _sort_id_values(all_data["id"].unique()),
        "conditions": sorted(map(str, data["condition"].unique())),
        "filterNotes": list(data.attrs.get("filter_notes", [])),
        "rtMin": float(data["rt"].min()),
        "rtMax": float(data["rt"].max()),
        "dataConstants": [
            {
                "name": name,
                "source": constant_sources.get(name, "user expression"),
                "value": value,
                "locked": name == "min_rt",
            }
            for name, value in sorted(constants.items(), key=lambda item: (item[0] != "min_rt", item[0]))
        ],
        "dataPlotUrl": data_plot_url,
        "covariates": covariate_rows,
        "participantSummary": participant_summary.head(200).to_dict(orient="records"),
        "conditionSummary": condition_summary.head(200).to_dict(orient="records"),
        "responseSummary": response_summary.head(200).to_dict(orient="records"),
        "conditionResponseSummary": condition_response_summary.head(400).to_dict(orient="records"),
        "rtHistogram": rt_histogram,
        "columns": list(_raw_rows_for(data).columns),
        "internalColumns": list(data.columns),
        "conditionColumns": list(data.attrs.get("condition_columns", [])),
        "conditionDisplay": str(data.attrs.get("condition_display", "condition")),
        "filterColumns": _column_filter_metadata(all_data),
        "preview": _preview_records(data, limit=preview_limit),
    }


def _write_data_overview_plot(data: pd.DataFrame, response_label: str) -> str | None:
    if data.empty:
        return None
    run_dir = RUNS / "data_preview"
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_path = run_dir / f"data_overview_{uuid.uuid4().hex[:8]}.png"
    plot_data = _sample_for_preview_plot(data)
    plot_data_overview(plot_data, plot_path, response_label=response_label, title="")
    return f"/runs/{run_dir.name}/{plot_path.name}"


def _sample_for_preview_plot(data: pd.DataFrame) -> pd.DataFrame:
    if len(data) <= DATA_PREVIEW_PLOT_MAX_ROWS:
        return data
    sampled = data.sample(n=DATA_PREVIEW_PLOT_MAX_ROWS, random_state=17).sort_index().reset_index(drop=True)
    sampled.attrs.update(data.attrs)
    return sampled


def _covariate_columns(data: pd.DataFrame) -> list[str]:
    excluded = {"id", "original_id", "condition", "rt", "accuracy", "trial"}
    excluded.update(column for column in data.columns if str(column).startswith("condition_"))
    return [str(column) for column in data.columns if str(column) not in excluded]


def _covariate_summary(data: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped = data.groupby("id", dropna=False, sort=False)
    for column in _covariate_columns(data):
        series = data[column]
        nonmissing = series.dropna()
        if nonmissing.empty:
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric = numeric.notna().any()
        per_participant_unique = grouped[column].nunique(dropna=True)
        nonmissing_per_participant = grouped[column].count()
        if bool((per_participant_unique <= 1).all()):
            covariate_type = "participant-level"
        elif bool((nonmissing_per_participant <= 1).all()):
            covariate_type = "participant-level sparse"
        else:
            covariate_type = "trial-level"
        row = {
            "name": column,
            "type": covariate_type,
            "nonmissing": int(nonmissing.shape[0]),
            "unique": int(nonmissing.nunique()),
        }
        if is_numeric:
            clean = numeric.dropna()
            row.update(
                {
                    "mean": round(float(clean.mean()), 4),
                    "sd": round(float(clean.std()), 4) if clean.shape[0] > 1 else 0.0,
                    "min": round(float(clean.min()), 4),
                    "max": round(float(clean.max()), 4),
                }
            )
        else:
            common = nonmissing.astype(str).value_counts().head(3)
            row["top_values"] = ", ".join(f"{name} ({count})" for name, count in common.items())
        rows.append(row)
    return rows


def _internal_consistency_reliability(
    model_name: str,
    data: pd.DataFrame,
    config: FitConfig,
    solver: str,
    fp_config: FokkerPlanckConfig,
    fp_objective: str,
    report,
    cancel_check=None,
) -> dict[str, object]:
    participant_ids = _sort_id_values(data["id"].unique())
    if len(participant_ids) < 2:
        raise ValueError("Internal consistency reliability needs at least two participants to compute correlations.")
    run_dir = RUNS / f"internal_consistency_{_safe_model_name(model_name)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    worker_count = min(max(int(config.workers), 1), len(participant_ids))
    worker_config = replace(config, workers=1)
    worker_slots: dict[int, int] = {}
    worker_slots_lock = threading.Lock()
    progress_by_participant = {index: 0.0 for index in range(1, len(participant_ids) + 1)}
    progress_lock = threading.Lock()

    def current_worker_slot() -> int:
        thread_id = threading.get_ident()
        with worker_slots_lock:
            if thread_id not in worker_slots:
                worker_slots[thread_id] = len(worker_slots) + 1
            return worker_slots[thread_id]

    def aggregate_progress(index: int, value: float) -> float:
        with progress_lock:
            progress_by_participant[index] = max(0.0, min(1.0, float(value)))
            return 0.98 * (sum(progress_by_participant.values()) / max(len(progress_by_participant), 1))

    def fit_participant_halves(index: int, participant_id: str) -> list[dict[str, object]]:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Validation cancelled.")
        slot = current_worker_slot()
        label = f"Worker {slot} | Participant {index} ({participant_id})"
        participant_data = data[data["id"].astype(str) == participant_id].reset_index(drop=True)
        participant_data.attrs.update(data.attrs)
        odd, even = _split_odd_even(participant_data)
        if len(odd) < 10 or len(even) < 10:
            report(
                f"Skipped participant {index} of {len(participant_ids)}",
                aggregate_progress(index, 1.0),
                {"workerSlot": slot, "label": label, "participantProgress": 1.0, "status": "skipped"},
            )
            return []
        report(
            f"Fitting split halves for participant {index} of {len(participant_ids)}",
            aggregate_progress(index, 0.0),
            {"workerSlot": slot, "label": label, "participantProgress": 0.0, "status": "running"},
        )

        def half_progress(offset: float, detail: dict[str, object]) -> None:
            evaluations = float(detail.get("evaluations") or 0)
            expected = float(detail.get("expectedEvaluations") or 1)
            local = offset + 0.48 * min(evaluations / expected, 0.95)
            progress_detail = dict(detail)
            progress_detail.update({"workerSlot": slot, "label": label, "participantProgress": local, "status": "running"})
            report(f"Fitting split halves for participant {index} of {len(participant_ids)}", aggregate_progress(index, local), progress_detail)

        if solver == "fokker_planck":
            model = _load_fp_model(model_name, odd)
            _validate_condition_compatibility(model, odd)
            _validate_condition_compatibility(model, even)
            odd_result = fit_fokker_planck_model(model, odd, worker_config, fp_config, objective_name=fp_objective, progress_callback=lambda detail: half_progress(0.0, detail), cancel_callback=cancel_check)
            even_result = fit_fokker_planck_model(model, even, worker_config, fp_config, objective_name=fp_objective, progress_callback=lambda detail: half_progress(0.5, detail), cancel_callback=cancel_check)
        else:
            model = _load_model(model_name, odd)
            _validate_condition_compatibility(model, odd)
            _validate_condition_compatibility(model, even)
            odd_result = fit_model(model, odd, worker_config, progress_callback=lambda detail: half_progress(0.0, detail), cancel_callback=cancel_check)
            even_result = fit_model(model, even, worker_config, progress_callback=lambda detail: half_progress(0.5, detail), cancel_callback=cancel_check)
        rows = []
        for parameter in model.parameter_names:
            rows.append(
                {
                    "participant": participant_id,
                    "parameter": parameter,
                    "odd": odd_result.parameters[parameter],
                    "even": even_result.parameters[parameter],
                }
            )
        report(
            f"Finished participant {index} of {len(participant_ids)}",
            aggregate_progress(index, 1.0),
            {"workerSlot": slot, "label": label, "participantProgress": 1.0, "status": "finished"},
        )
        return rows

    estimates: list[dict[str, object]] = []
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(fit_participant_halves, index, participant_id) for index, participant_id in enumerate(participant_ids, 1)]
            for future in as_completed(futures):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("Validation cancelled.")
                estimates.extend(future.result())
    else:
        for index, participant_id in enumerate(participant_ids, 1):
            estimates.extend(fit_participant_halves(index, participant_id))

    estimate_frame = pd.DataFrame(estimates)
    if estimate_frame.empty:
        raise ValueError("No participants had enough odd/even trials for internal consistency reliability.")
    estimate_frame.to_csv(run_dir / "internal_consistency_estimates.csv", index=False)
    rows: list[dict[str, object]] = []
    for parameter, group in estimate_frame.groupby("parameter", sort=True):
        correlation = None
        reliability = None
        if len(group) >= 2 and group["odd"].nunique() > 1 and group["even"].nunique() > 1:
            correlation = float(group["odd"].corr(group["even"]))
            reliability = spearman_brown(correlation)
        rows.append(
            {
                "parameter": parameter,
                "participants": int(group["participant"].nunique()),
                "split_half_r": None if correlation is None else round(correlation, 4),
                "spearman_brown_r": None if reliability is None or not np.isfinite(reliability) else round(float(reliability), 4),
                "odd_mean": round(float(group["odd"].mean()), 4),
                "even_mean": round(float(group["even"].mean()), 4),
            }
        )
    plot_path = run_dir / "internal_consistency.png"
    plot_internal_consistency(estimate_frame, plot_path, title=f"{model_name} odd-even parameter reliability")
    return {
        "mode": "internal consistency reliability",
        "rows": rows,
        "plotUrl": f"/runs/{run_dir.name}/{plot_path.name}",
        "estimateCsv": f"/runs/{run_dir.name}/internal_consistency_estimates.csv",
    }


def _split_odd_even(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_numbers = pd.to_numeric(data["trial"], errors="coerce") if "trial" in data else pd.Series(np.arange(1, len(data) + 1))
    if trial_numbers.isna().any():
        trial_numbers = pd.Series(np.arange(1, len(data) + 1))
    odd = data[(trial_numbers.astype(int) % 2) == 1].reset_index(drop=True)
    even = data[(trial_numbers.astype(int) % 2) == 0].reset_index(drop=True)
    odd.attrs.update(data.attrs)
    even.attrs.update(data.attrs)
    return odd, even


def _sort_id_values(values) -> list[str]:
    return sorted((str(value) for value in values), key=lambda value: (not _is_integerish(value), int(float(value)) if _is_integerish(value) else value))


def _fit_once(
    model_name: str,
    data: pd.DataFrame,
    config: FitConfig,
    solver: str,
    fp_config: FokkerPlanckConfig,
    fp_objective: str = "g2",
    simulated_participant_id: object = "simulated",
    progress_callback=None,
    cancel_check=None,
) -> tuple[ModelSpec | FokkerPlanckSpec, FitResult, pd.DataFrame]:
    if solver == "fokker_planck":
        model = _load_fp_model(model_name, data)
        _validate_condition_compatibility(model, data)
        fitter_result = fit_fokker_planck_model(
            model,
            data,
            config,
            fp_config,
            ppc_output_path=None,
            objective_name=fp_objective,
            progress_callback=progress_callback,
            cancel_callback=cancel_check,
        )
        simulated = sample_like_observed(model, fitter_result.parameters.values(), data, simulated_participant_id, fp_config)
        return model, fitter_result, simulated

    model = _load_model(model_name, data)
    _validate_condition_compatibility(model, data)
    fitter_result = fit_model(model, data, config, ppc_output_path=None, progress_callback=progress_callback, cancel_callback=cancel_check)
    simulated = sample_like_observed(model, fitter_result.parameters.values(), data, simulated_participant_id, fp_config)
    return model, fitter_result, simulated


def _fit_payload(
    model_name: str,
    data: pd.DataFrame,
    simulated: pd.DataFrame,
    result: FitResult,
    run_dir: Path,
    ppc_path: Path | None,
    ppc_options: list[dict[str, str]] | None = None,
) -> dict:
    return {
        "model": result.model_name,
        "parameters": result.parameters,
        "scoreName": "G2",
        "objectiveName": result.objective_name,
        "objectiveScore": result.objective_score,
        "objectiveBic": result.objective_bic,
        "g2": result.g2,
        "bic": result.bic,
        "rowsFit": len(data),
        "outputCsv": str(run_dir / "fit_result.csv"),
        "ppcPath": str(ppc_path) if ppc_path else None,
        "ppcUrl": f"/runs/{run_dir.name}/posterior_predictive.png" if ppc_path else None,
        "ppcOptions": ppc_options or [],
        "conditionSummary": _condition_summary(data, simulated),
        "fitCovariateSummary": [],
    }


def _participant_covariate_frame(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    covariates = _covariate_summary(data)
    participant_level = {row["name"] for row in covariates if str(row.get("type", "")).startswith("participant-level")}
    if not participant_level:
        return pd.DataFrame()
    for participant_id, group in data.groupby("id", dropna=False):
        row: dict[str, object] = {"id": str(participant_id)}
        for column in sorted(participant_level):
            values = group[column].dropna() if column in group else pd.Series(dtype=object)
            if values.empty:
                row[column] = np.nan
            else:
                row[column] = values.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_covariate_summary(fit_frame: pd.DataFrame, covariate_names: list[str]) -> list[dict[str, object]]:
    if fit_frame.empty:
        return []
    reserved = {"participant", "model", "g2", "bic", "objective", "objective_score", "objective_bic"}
    covariate_set = set(covariate_names)
    parameter_names = [column for column in fit_frame.columns if column not in reserved and column not in covariate_set]
    rows: list[dict[str, object]] = []
    for covariate in sorted(covariate_set):
        covariate_values = pd.to_numeric(fit_frame.get(covariate), errors="coerce")
        if covariate_values.notna().sum() < 3 or covariate_values.nunique(dropna=True) < 2:
            continue
        for parameter in parameter_names:
            parameter_values = pd.to_numeric(fit_frame.get(parameter), errors="coerce")
            valid = covariate_values.notna() & parameter_values.notna()
            if valid.sum() < 3 or parameter_values[valid].nunique() < 2:
                continue
            rows.append(
                {
                    "covariate": covariate,
                    "parameter": parameter,
                    "n": int(valid.sum()),
                    "r": round(float(covariate_values[valid].corr(parameter_values[valid])), 4),
                }
            )
    return rows


def _write_ppc_diagnostics(
    observed: pd.DataFrame,
    simulated: pd.DataFrame,
    run_dir: Path,
    title: str,
    enabled: bool,
) -> tuple[Path | None, list[dict[str, str]]]:
    if not enabled:
        return None, []

    from .plotting import plot_posterior_predictive

    ppc_path = run_dir / "posterior_predictive.png"
    plot_posterior_predictive(observed, simulated, ppc_path, title=title)
    options = [{"label": "All participants", "url": f"/runs/{run_dir.name}/posterior_predictive.png"}]

    observed_ids = [str(item) for item in sorted(observed["id"].astype(str).unique())]
    simulated_ids = set(simulated["id"].astype(str).unique())
    if len(observed_ids) > 1:
        for participant_id in observed_ids:
            if participant_id not in simulated_ids:
                continue
            participant_path = run_dir / f"posterior_predictive_id_{_safe_model_name(participant_id)}.png"
            participant_observed = observed[observed["id"].astype(str) == participant_id]
            participant_simulated = simulated[simulated["id"].astype(str) == participant_id]
            participant_observed.attrs.update(observed.attrs)
            participant_simulated.attrs.update(simulated.attrs)
            plot_posterior_predictive(
                participant_observed,
                participant_simulated,
                participant_path,
                title="",
            )
            options.append({"label": f"Participant {participant_id}", "url": f"/runs/{run_dir.name}/{participant_path.name}"})
    return ppc_path, options


def _condition_summary(observed: pd.DataFrame, simulated: pd.DataFrame) -> list[dict[str, object]]:
    response_kind = str(observed.attrs.get("response_kind", "accuracy"))
    response_column = str(observed.attrs.get("response_column", "response"))
    response_mean_label = "mean_accuracy" if response_kind == "accuracy" else "mean_choice_1"
    observed = observed.copy()
    simulated = simulated.copy()
    observed.attrs.clear()
    simulated.attrs.clear()
    observed["source"] = "observed"
    simulated["source"] = "simulated"
    combined = pd.concat([observed[["source", "condition", "rt", "accuracy"]], simulated[["source", "condition", "rt", "accuracy"]]])
    rows: list[dict[str, object]] = []
    for (source, condition), group in combined.groupby(["source", "condition"], dropna=False):
        rows.append(
            {
                "source": source,
                "condition": str(condition),
                "n": int(len(group)),
                "mean_rt": round(float(group["rt"].mean()), 4),
                "response_column": response_column,
                response_mean_label: round(float(group["accuracy"].mean()), 4),
            }
        )
    return rows


def _validate_condition_compatibility(model: ModelSpec | FokkerPlanckSpec, data: pd.DataFrame) -> None:
    model_conditions = getattr(model, "conditions", None)
    if not model_conditions:
        return
    expected = {str(condition) for condition in model_conditions}
    observed = {str(condition) for condition in data["condition"].astype(str).unique()}
    missing = sorted(observed - expected)
    if not missing:
        return
    expected_preview = ", ".join(sorted(expected)[:12])
    if len(expected) > 12:
        expected_preview += f", ... ({len(expected)} total)"
    observed_preview = ", ".join(missing[:12])
    if len(missing) > 12:
        observed_preview += f", ... ({len(missing)} incompatible labels)"
    raise ValueError(
        f"{model.name} is not compatible with the loaded data conditions. "
        f"Observed condition label(s) not produced by this model: {observed_preview}. "
        f"Expected labels include: {expected_preview}."
    )


def _save_model_code(model_name: str, code: str, parameter_bounds: list[dict[str, object]] | None = None) -> Path:
    GENERATED_MODELS.mkdir(parents=True, exist_ok=True)
    destination = GENERATED_MODELS / f"{model_name}.py"
    bounds_destination = _bounds_path(model_name)
    old_text = destination.read_text(encoding="utf-8") if destination.exists() else None
    old_bounds = bounds_destination.read_text(encoding="utf-8") if bounds_destination.exists() else None
    if parameter_bounds:
        _save_parameter_bounds(model_name, parameter_bounds)
    code = _strip_parameter_bounds_block(code)
    destination.write_text(code, encoding="utf-8")
    try:
        model = _load_model(model_name)
        _validate_model_runtime(model)
    except Exception:
        if old_text is None:
            destination.unlink(missing_ok=True)
        else:
            destination.write_text(old_text, encoding="utf-8")
        if old_bounds is None:
            bounds_destination.unlink(missing_ok=True)
        else:
            bounds_destination.write_text(old_bounds, encoding="utf-8")
        raise
    return destination


def _delete_saved_model(model_name: str) -> dict[str, object]:
    name = _safe_model_name(model_name)
    if not name:
        raise ValueError("Choose a custom model to delete.")
    root = GENERATED_MODELS.resolve()
    model_path = (GENERATED_MODELS / f"{name}.py").resolve()
    if not model_path.is_file() or root not in model_path.parents:
        raise ValueError("Saved custom model not found.")
    removed: list[str] = []
    for path in (model_path, _bounds_path(name).resolve(), _metadata_path(name).resolve()):
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return {"deleted": name, "removed": removed}


def _bounds_path(model_name: str) -> Path:
    return GENERATED_MODELS / f"{_safe_model_name(model_name)}.bounds.json"


def _strip_parameter_bounds_block(code: str) -> str:
    return re.sub(r"\n?PARAMETER_BOUNDS\s*=\s*\{.*?\}\s*\n\n?", "\n\n", code, count=1, flags=re.S).lstrip()


def _save_parameter_bounds(model_name: str, parameter_bounds: list[dict[str, object]]) -> None:
    entries = _normalize_parameter_bounds(parameter_bounds)
    _bounds_path(model_name).write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _metadata_path(model_name: str) -> Path:
    return GENERATED_MODELS / f"{_safe_model_name(model_name)}.meta.json"


def _load_model_metadata(model_name: str) -> dict[str, object]:
    path = _metadata_path(model_name)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_model_metadata(model_name: str, description: str) -> None:
    _metadata_path(model_name).write_text(json.dumps({"description": description}, indent=2), encoding="utf-8")


def _model_descriptions(generated: list[str]) -> dict[str, str]:
    descriptions = dict(TEMPLATE_DESCRIPTIONS)
    for name in generated:
        meta = _load_model_metadata(name)
        description = str(meta.get("description", "")).strip()
        if description:
            descriptions[name] = description
    return descriptions


def _normalize_parameter_bounds(parameter_bounds: list[dict[str, object]]) -> dict[str, tuple[float, float | str]]:
    entries: dict[str, tuple[float, float | str]] = {}
    for row in parameter_bounds:
        name = _safe_parameter_name(str(row.get("name", "")))
        if not name:
            continue
        lower = float(row.get("lower", 0.0))
        upper_value = row.get("upper", 1.0)
        upper: float | str = str(upper_value).strip() if isinstance(upper_value, str) else float(upper_value)
        if isinstance(upper, str) and not _safe_parameter_name(upper):
            try:
                upper = float(upper)
            except ValueError:
                raise ValueError(f"Upper bound for {name!r} must be numeric or a named data constant such as min_rt.")
        entries[name] = (lower, upper)
    return entries


def _load_parameter_bounds(model_name: str, code: str | None = None) -> dict[str, tuple[float, float | str]]:
    path = _bounds_path(model_name)
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        parsed = {}
        for name, bounds in raw.items():
            upper = bounds[1]
            if isinstance(upper, str):
                try:
                    upper = float(upper)
                except ValueError:
                    if not _safe_parameter_name(upper):
                        raise ValueError(f"Saved upper bound for {name!r} must be numeric or a named data constant.")
            parsed[str(name)] = (float(bounds[0]), upper)
        return parsed
    if code is not None:
        legacy = _legacy_parameter_bounds_from_code(code)
        if legacy:
            return legacy
    return {}


def _legacy_parameter_bounds_from_code(code: str) -> dict[str, tuple[float, float | str]]:
    entries: dict[str, tuple[float, float | str]] = {}
    for match in re.finditer(r"""["']([A-Za-z_][A-Za-z0-9_]*)["']\s*:\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)""", code):
        entries[match.group(1)] = (float(match.group(2)), _parse_upper_bound(match.group(3)))
    for match in re.finditer(r"""(?:param|Parameter)\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)""", code):
        entries[match.group(1)] = (float(match.group(2)), _parse_upper_bound(match.group(3)))
    return entries


def _parse_upper_bound(value: str) -> float | str:
    cleaned = value.strip().strip("\"'")
    if _safe_parameter_name(cleaned):
        return cleaned
    return float(cleaned)


def _bounds_rows(bounds: dict[str, tuple[float, float | str]]) -> list[dict[str, object]]:
    return [{"name": name, "lower": lower, "upper": upper} for name, (lower, upper) in bounds.items()]


def _safe_parameter_name(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        return value
    return ""


def _discover_generated_models() -> tuple[list[str], list[str], list[str]]:
    if not GENERATED_MODELS.exists():
        return [], [], []
    valid: list[str] = []
    invalid: list[str] = []
    saved = sorted(path.stem for path in GENERATED_MODELS.glob("*.py"))
    for name in saved:
        try:
            model = _load_model(name)
            _validate_model_runtime(model)
            valid.append(name)
        except Exception:
            invalid.append(name)
    return sorted(valid), saved, sorted(invalid)


def _discover_generated_fp_models() -> list[str]:
    if not GENERATED_MODELS.exists():
        return []
    valid: list[str] = []
    for path in GENERATED_MODELS.glob("*.py"):
        try:
            _load_fp_model(path.stem)
            valid.append(path.stem)
        except Exception:
            continue
    return sorted(valid)


def _validate_model_runtime(model: ModelSpec) -> None:
    fallback = pd.DataFrame({"id": [0], "condition": ["0"], "rt": [0.45], "accuracy": [1]})
    bounds = model.bounds_for_data(fallback)
    values = [(lower + upper) / 2 for lower, upper in bounds]
    simulated = model.simulate_dataframe(values, n_trials=8)
    if simulated.empty:
        raise ValueError(f"{model.name} produced no simulated rows.")
    if not pd.to_numeric(simulated["rt"], errors="coerce").notna().all():
        raise ValueError(f"{model.name} produced non-finite RT values in a smoke simulation.")
    if not pd.to_numeric(simulated["accuracy"], errors="coerce").notna().all():
        raise ValueError(f"{model.name} produced non-finite accuracy values in a smoke simulation.")


def _load_payload_data(payload: dict) -> pd.DataFrame:
    condition_columns = [str(item).strip() for item in (payload.get("conditionColumns") or []) if str(item).strip()]
    primary_condition = condition_columns[0] if condition_columns else str(payload.get("condition", "")).strip()
    data = load_rt_data(
        payload["path"],
        ColumnMap(
            participant=payload.get("participant", "PPT"),
            condition=primary_condition,
            rt=payload.get("rt", "RT"),
            accuracy=payload.get("accuracy", "Correct"),
            response_kind=payload.get("responseKind", "accuracy"),
        ),
    )
    data = _assign_condition_columns(data, condition_columns)
    data = _apply_data_filters(data, payload)
    data.attrs["custom_constants_text"] = str(payload.get("customConstants", "")).strip()
    constants = _parse_data_constants(str(payload.get("customConstants", "")).strip(), data)
    if constants:
        data.attrs["bound_constants"] = constants
        data.attrs["bound_constant_sources"] = _parse_data_constant_sources(str(payload.get("customConstants", "")).strip())
    return data


def _assign_condition_columns(data: pd.DataFrame, condition_columns: list[str]) -> pd.DataFrame:
    assigned = data.copy()
    attrs = dict(data.attrs)
    raw_rows = attrs.get("raw_rows")
    if not condition_columns:
        assigned["condition"] = "all"
        attrs["condition_columns"] = []
        attrs["condition_display"] = "all"
        assigned.attrs.update(attrs)
        return assigned
    if not isinstance(raw_rows, pd.DataFrame):
        raw_rows = assigned
    missing = [column for column in condition_columns if column not in raw_rows.columns]
    if missing:
        available = ", ".join(map(str, raw_rows.columns))
        raise ValueError(f"Assigned condition column(s) not found: {missing}. Available columns: {available}")
    primary = condition_columns[0]
    labels = raw_rows[primary].reset_index(drop=True).fillna("all").astype(str).str.strip()
    assigned["condition"] = labels.replace("", "all").to_numpy()
    attrs["condition_columns"] = list(condition_columns)
    attrs["condition_display"] = primary
    assigned.attrs.update(attrs)
    return assigned


def _data_columns(payload: dict) -> dict[str, object]:
    path = str(payload.get("path", "")).strip()
    if not path:
        raise ValueError("Choose a CSV file first.")
    sample = pd.read_csv(path, encoding="utf-8-sig", nrows=200)
    columns = [str(column) for column in sample.columns]
    lower_map = {column.lower(): column for column in columns}

    def pick(candidates: list[str]) -> str:
        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]
        for column in columns:
            cleaned = re.sub(r"[^a-z0-9]+", "", column.lower())
            if any(re.sub(r"[^a-z0-9]+", "", candidate.lower()) == cleaned for candidate in candidates):
                return column
        return ""

    suggestions = {
        "participant": pick(["PPT", "participant", "participant_id", "subject", "subject_id", "id"]),
        "condition": pick(["Condition", "condition", "trial_type", "block", "block_name"]),
        "rt": pick(["RT", "rt", "response_time", "reaction_time"]),
        "accuracy": pick(["Correct", "correct", "accuracy", "response", "choice", "ChoiceGood"]),
    }
    return {"columns": columns, "suggestions": suggestions, "rowsSampled": int(len(sample))}


def _apply_data_filters(data: pd.DataFrame, payload: dict) -> pd.DataFrame:
    filtered = data.copy()
    filtered.attrs.update(data.attrs)
    filter_notes = []
    for index, row_filter in enumerate(payload.get("columnFilters") or [], start=1):
        column = str(row_filter.get("column", "")).strip()
        if not column:
            continue
        raw_rows = _raw_rows_for(filtered)
        if column not in raw_rows.columns:
            raise ValueError(f"Data filter {index} refers to unknown column {column!r}.")
        filter_type = str(row_filter.get("type", "")).strip() or "categorical"
        series = raw_rows[column]
        if filter_type == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            mask = pd.Series(True, index=filtered.index)
            lower = str(row_filter.get("min", "")).strip()
            upper = str(row_filter.get("max", "")).strip()
            if lower:
                lower_value = float(lower)
                mask &= numeric >= lower_value
                filter_notes.append(f"{column} >= {lower_value:g}")
            if upper:
                upper_value = float(upper)
                mask &= numeric <= upper_value
                filter_notes.append(f"{column} <= {upper_value:g}")
            if not lower and not upper:
                continue
            filtered = _subset_data_with_attrs(filtered, mask.fillna(False))
        else:
            included = {str(value) for value in row_filter.get("include", []) if str(value) != ""}
            if not included and row_filter.get("exclude"):
                excluded = {str(value) for value in row_filter.get("exclude", []) if str(value) != ""}
                mask = ~series.astype(str).isin(excluded)
                filtered = _subset_data_with_attrs(filtered, mask)
                preview = ", ".join(list(excluded)[:6])
                suffix = "..." if len(excluded) > 6 else ""
                filter_notes.append(f"{column} excludes {preview}{suffix}")
                continue
            if not included:
                continue
            mask = series.astype(str).isin(included)
            filtered = _subset_data_with_attrs(filtered, mask)
            preview = ", ".join(list(included)[:6])
            suffix = "..." if len(included) > 6 else ""
            filter_notes.append(f"{column} includes {preview}{suffix}")
    keep_where = str(payload.get("filterExpression", "")).strip()
    if keep_where:
        mask = _evaluate_filter_expression(keep_where, filtered)
        filtered = _subset_data_with_attrs(filtered, mask)
        filter_notes.append(keep_where)
    if filtered.empty:
        raise ValueError("Data cleaning removed every row. Relax the column filters.")
    filtered.attrs["filter_notes"] = filter_notes
    return _renumber_analysis_ids(filtered)


def _renumber_analysis_ids(data: pd.DataFrame) -> pd.DataFrame:
    renumbered = data.copy()
    attrs = dict(data.attrs)
    original_ids = renumbered["id"].astype(str).tolist()
    cleaned_ids = _renumber_ids(renumbered["id"])
    if "original_id" in renumbered.columns:
        renumbered = renumbered.drop(columns=["original_id"])
    if cleaned_ids != original_ids:
        renumbered.insert(1, "original_id", original_ids)
    renumbered["id"] = cleaned_ids
    renumbered.attrs.update(attrs)
    return renumbered


def _renumber_ids(values: pd.Series) -> list[str]:
    unique = list(dict.fromkeys(values.astype(str).tolist()))
    unique = sorted(unique, key=lambda value: (not _is_integerish(value), int(float(value)) if _is_integerish(value) else value))
    mapping = {value: str(index + 1) for index, value in enumerate(unique)}
    return [mapping[str(value)] for value in values.astype(str)]


def _evaluate_filter_expression(expression: str, data: pd.DataFrame) -> pd.Series:
    safe_locals: dict[str, object] = {}
    for column in data.columns:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(column)):
            safe_locals[str(column)] = data[column]
    try:
        result = eval(expression, {"__builtins__": {}}, safe_locals)
    except Exception as exc:
        raise ValueError(f"Could not evaluate keep-expression {expression!r}: {exc}") from exc
    if not isinstance(result, pd.Series):
        raise ValueError("Keep-expression must produce a row-wise boolean result, such as trial_type == 'attribute'.")
    mask = result.fillna(False).astype(bool)
    if len(mask) != len(data):
        raise ValueError("Keep-expression produced the wrong number of rows.")
    return mask


def _parse_data_constants(text: str, data: pd.DataFrame | None = None) -> dict[str, float]:
    constants: dict[str, float] = {}
    for name, expression in _parse_data_constant_entries(text):
        constants[name] = _evaluate_data_constant(expression, data)
    return constants


def _parse_data_constant_sources(text: str) -> dict[str, str]:
    return {name: expression for name, expression in _parse_data_constant_entries(text)}


def _parse_data_constant_entries(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    if not text:
        return entries
    for token in re.split(r"[\n;]+", text):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError("Data constants must use name=value entries separated by semicolons or new lines.")
        name, value = token.split("=", 1)
        name = _safe_parameter_name(name)
        if not name:
            raise ValueError("Data constant names must be valid Python identifiers.")
        expression = value.strip()
        if not expression:
            raise ValueError(f"Data constant {name!r} needs a source expression.")
        entries.append((name, expression))
    return entries


def _evaluate_data_constant(expression: str, data: pd.DataFrame | None = None) -> float:
    try:
        return float(expression)
    except ValueError:
        pass
    if data is None:
        raise ValueError(f"Data constant {expression!r} is not a number and no data are available for expressions.")

    def clean(values):
        return pd.to_numeric(values, errors="coerce").dropna()

    safe_locals: dict[str, object] = {
        "min": lambda values: float(clean(values).min()),
        "max": lambda values: float(clean(values).max()),
        "mean": lambda values: float(clean(values).mean()),
        "median": lambda values: float(clean(values).median()),
        "quantile": lambda values, q: float(clean(values).quantile(float(q))),
        "std": lambda values: float(clean(values).std()),
        "n": lambda values: int(clean(values).shape[0]),
    }
    for column in data.columns:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(column)):
            safe_locals[str(column)] = data[column]
    try:
        value = eval(expression, {"__builtins__": {}}, safe_locals)
    except Exception as exc:
        raise ValueError(f"Could not evaluate data constant expression {expression!r}: {exc}") from exc
    return float(value)


def _select_data(data: pd.DataFrame, payload: dict) -> pd.DataFrame:
    participant_ids = _participant_ids_from_payload(data, payload)
    if participant_ids is not None:
        selected = _subset_data_with_attrs(data, data["id"].astype(str).isin(participant_ids))
        if selected.empty:
            raise ValueError(f"No rows found for participant specification {payload.get('participantId')!r}.")
        return _refresh_data_constants(selected)
    selected = data.reset_index(drop=True)
    selected.attrs.update(data.attrs)
    raw_rows = selected.attrs.get("raw_rows")
    if isinstance(raw_rows, pd.DataFrame):
        selected.attrs["raw_rows"] = raw_rows.reset_index(drop=True)
    return _refresh_data_constants(selected)


def _select_preview_data(data: pd.DataFrame, payload: dict) -> pd.DataFrame:
    participant_id = str(payload.get("previewParticipantId", "")).strip()
    if participant_id and participant_id.lower() != "all":
        selected = _subset_data_with_attrs(data, data["id"].astype(str) == participant_id)
        if selected.empty:
            raise ValueError(f"No preview rows found for participant {participant_id!r}.")
        return _refresh_data_constants(selected)
    selected = data.reset_index(drop=True)
    selected.attrs.update(data.attrs)
    raw_rows = selected.attrs.get("raw_rows")
    if isinstance(raw_rows, pd.DataFrame):
        selected.attrs["raw_rows"] = raw_rows.reset_index(drop=True)
    return _refresh_data_constants(selected)


def _refresh_data_constants(data: pd.DataFrame) -> pd.DataFrame:
    text = str(data.attrs.get("custom_constants_text", "")).strip()
    data.attrs.pop("bound_constants", None)
    data.attrs.pop("bound_constant_sources", None)
    constants = _parse_data_constants(text, data)
    if constants:
        data.attrs["bound_constants"] = constants
        data.attrs["bound_constant_sources"] = _parse_data_constant_sources(text)
    return data


def _participant_ids_from_payload(data: pd.DataFrame, payload: dict, default_all: bool = False) -> list[str] | None:
    spec = str(payload.get("participantId", "")).strip()
    available = list(dict.fromkeys(data["id"].astype(str).tolist()))
    if not spec:
        return available if default_all else None
    available_set = set(available)
    requested = _parse_participant_spec(spec, available)
    missing = [participant_id for participant_id in requested if participant_id not in available_set]
    if missing:
        raise ValueError(f"Participant id(s) not found: {', '.join(missing)}")
    return requested


def _parse_participant_spec(spec: str, available_ids: list[str]) -> list[str]:
    requested: list[str] = []
    seen: set[str] = set()
    numeric_order = {int(float(value)): value for value in available_ids if _is_integerish(value)}
    for token in re.split(r"[,;]+", spec):
        token = token.strip()
        if not token:
            continue
        range_match = re.fullmatch(r"([+-]?\d+(?:\.0+)?)\s*(?:-|:|to|through)\s*([+-]?\d+(?:\.0+)?)", token, flags=re.I)
        if range_match:
            start = int(float(range_match.group(1)))
            end = int(float(range_match.group(2)))
            step = 1 if end >= start else -1
            expanded = [numeric_order.get(value, str(value)) for value in range(start, end + step, step)]
        else:
            expanded = [token]
        for participant_id in expanded:
            if participant_id not in seen:
                requested.append(participant_id)
                seen.add(participant_id)
    return requested


def _is_integerish(value: str) -> bool:
    return re.fullmatch(r"[+-]?\d+(?:\.0+)?", value.strip()) is not None


def _fit_config(payload: dict) -> FitConfig:
    return FitConfig(
        maxiter=int(payload.get("maxiter", 12)),
        popsize=int(payload.get("popsize", 5)),
        seed=int(payload.get("seed", 10)),
        polish=bool(payload.get("polish", False)),
        workers=int(payload.get("workers", 1)),
        local_refine=bool(payload.get("localRefine", False)),
        simulated_trials=int(payload.get("simulatedTrials", 80)),
        progress_interval=1,
        simulation_dt=float(payload.get("simulationDt", 0.01)),
        simulation_noise_sd=float(payload.get("simulationNoiseSd", 0.01)),
        cdf_quantiles=tuple(_parse_quantiles(payload.get("cdfQuantiles", "0.10,0.30,0.50,0.70,0.90"), "CDF")),
        caf_quantiles=tuple(_parse_quantiles(payload.get("cafQuantiles", "0.25,0.50,0.75"), "CAF")),
    )


def _parse_quantiles(value: object, label: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        quantiles = [float(item) for item in value]
    else:
        quantiles = [float(item.strip()) for item in re.split(r"[,;\s]+", str(value)) if item.strip()]
    if not quantiles:
        raise ValueError(f"{label} quantiles cannot be empty.")
    if any(item <= 0 or item >= 1 for item in quantiles):
        raise ValueError(f"{label} quantiles must be between 0 and 1.")
    if quantiles != sorted(quantiles):
        raise ValueError(f"{label} quantiles must be in ascending order.")
    return quantiles


def _fp_config(payload: dict) -> FokkerPlanckConfig:
    return FokkerPlanckConfig(
        dt=float(payload.get("fpDt", 0.02)),
        dx=float(payload.get("fpDx", 0.01)),
        t_max=float(payload.get("fpTmax", 3.0)),
        seed=int(payload.get("seed", 10)),
    )


def _load_model(name: str, data: pd.DataFrame | None = None) -> ModelSpec:
    if name in BUILTIN_MODELS:
        if name in BUILTIN_FP_MODELS and isinstance(BUILTIN_FP_MODELS[name], FokkerPlanckSpec) and name != "DSTP":
            fp_model = _adapt_builtin_two_condition_fp_model(BUILTIN_FP_MODELS[name], data)
            return ProcessModelSpec(
                name=fp_model.name,
                parameters=fp_model.parameters,
                conditions=fp_model.conditions,
                drift=fp_model.drift,
                bound=fp_model.bound,
                noise=fp_model.noise,
                nondecision=fp_model.nondecision,
                starting_position=fp_model.starting_position,
                default_trials=fp_model.default_trials,
            ).to_simulation_model()
        return BUILTIN_MODELS[name]
    if name in MODEL_TEMPLATES:
        module = _load_template_module(name)
        model = getattr(module, "model", None)
        process_model = getattr(module, "process_model", None)
        if model is None and isinstance(process_model, ProcessModelSpec):
            model = process_model.to_simulation_model()
        if not isinstance(model, ModelSpec):
            raise ValueError(f"Template {name!r} must expose model = ModelSpec(...) or process_model = ProcessModelSpec(...).")
        return model
    path = (GENERATED_MODELS / f"{_safe_model_name(name)}.py").resolve()
    if not path.is_file() or not str(path).startswith(str(GENERATED_MODELS.resolve())):
        raise ValueError(f"Unknown model {name!r}.")
    spec = importlib.util.spec_from_file_location(f"generated_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load generated model {path}.")
    module = importlib.util.module_from_spec(spec)
    bounds = _load_parameter_bounds(name, path.read_text(encoding="utf-8"))
    if bounds:
        module.PARAMETER_BOUNDS = bounds
    spec.loader.exec_module(module)
    model = getattr(module, "model", None)
    process_model = getattr(module, "process_model", None)
    if model is None and isinstance(process_model, ProcessModelSpec):
        model = process_model.to_simulation_model()
    if not isinstance(model, ModelSpec):
        raise ValueError(f"Generated file {path} must expose model = ModelSpec(...) or process_model = ProcessModelSpec(...).")
    return model


def _load_fp_model(name: str, data: pd.DataFrame | None = None):
    if name in BUILTIN_FP_MODELS:
        model = BUILTIN_FP_MODELS[name]
        if isinstance(model, FokkerPlanckSpec):
            return _adapt_builtin_two_condition_fp_model(model, data)
        return model
    if name in MODEL_TEMPLATES:
        module = _load_template_module(name)
        model = getattr(module, "fp_model", None)
        process_model = getattr(module, "process_model", None)
        if model is None and isinstance(process_model, ProcessModelSpec):
            model = process_model.to_fokker_planck_model()
        if not _is_fp_model(model):
            raise ValueError(f"Template {name!r} does not expose FP-readable process functions.")
        return model
    path = (GENERATED_MODELS / f"{_safe_model_name(name)}.py").resolve()
    if not path.is_file() or not str(path).startswith(str(GENERATED_MODELS.resolve())):
        raise ValueError(f"Unknown Fokker-Planck model {name!r}.")
    spec = importlib.util.spec_from_file_location(f"generated_fp_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load generated model {path}.")
    module = importlib.util.module_from_spec(spec)
    bounds = _load_parameter_bounds(name, path.read_text(encoding="utf-8"))
    if bounds:
        module.PARAMETER_BOUNDS = bounds
    spec.loader.exec_module(module)
    model = getattr(module, "fp_model", None)
    process_model = getattr(module, "process_model", None)
    if model is None and isinstance(process_model, ProcessModelSpec):
        model = process_model.to_fokker_planck_model()
    if not _is_fp_model(model):
        raise ValueError(
            f"{name!r} does not expose fp_model = FokkerPlanckSpec(...) or process_model = ProcessModelSpec(...). "
            "Fokker-Planck requires process functions, not only an opaque simulator."
        )
    return model


def _adapt_builtin_two_condition_fp_model(model: FokkerPlanckSpec, data: pd.DataFrame | None) -> FokkerPlanckSpec:
    if data is None or data.empty or "condition" not in data:
        return model
    expected = tuple(str(condition) for condition in (model.conditions or ()))
    observed = tuple(dict.fromkeys(data["condition"].astype(str).tolist()))
    if set(observed).issubset(set(expected)):
        return model
    if expected != ("0", "1") or len(observed) != 2:
        return model

    first_label, second_label = observed

    def map_condition(condition: object) -> str:
        condition_text = str(condition)
        if condition_text == first_label:
            return "0"
        if condition_text == second_label:
            return "1"
        return condition_text

    if _component_ignores_x(model.drift) or _component_has_unused_x(model.drift):
        def drift(t, condition, *params):
            return model.drift(0.0, t, map_condition(condition), *params)
    else:
        def drift(x, t, condition, *params):
            return model.drift(x, t, map_condition(condition), *params)

    if _component_ignores_x(model.noise) or _component_has_unused_x(model.noise):
        def noise(t, condition, *params):
            return model.noise(0.0, t, map_condition(condition), *params)
    else:
        def noise(x, t, condition, *params):
            return model.noise(x, t, map_condition(condition), *params)

    def bound(t, condition, *params):
        return model.bound(t, map_condition(condition), *params)

    return replace(
        model,
        conditions=observed,
        drift=drift,
        noise=noise,
        bound=bound,
        starting_position=_wrap_time_condition_component(model.starting_position, map_condition),
        nondecision=_wrap_time_condition_component(model.nondecision, map_condition),
    )


def _wrap_time_condition_component(component, map_condition):
    if not callable(component):
        return component

    def wrapped(t, condition, *params, **kwargs):
        return component(t, map_condition(condition), *params)

    return wrapped


def _is_fp_model(model: object) -> bool:
    required = ("parameter_names", "bounds_for_data", "neg2_log_likelihood", "predict_props")
    return isinstance(model, FokkerPlanckSpec) or all(hasattr(model, name) for name in required)


def _load_template_module(name: str):
    module = types.ModuleType(f"template_{_safe_model_name(name)}")
    module.__file__ = f"<FlexDDM template {name}>"
    module.PARAMETER_BOUNDS = MODEL_TEMPLATE_BOUNDS.get(name, {})
    exec(compile(MODEL_TEMPLATES[name], module.__file__, "exec"), module.__dict__)
    return module


def _fp_parameter_recovery(
    model: FokkerPlanckSpec,
    simulations: int,
    trials_per_simulation: int,
    fit_config: FitConfig,
    fp_config: FokkerPlanckConfig,
) -> list[dict[str, object]]:
    import numpy as np

    fallback = pd.DataFrame({"id": [0], "condition": ["0"], "rt": [0.45], "accuracy": [1]})
    rng = np.random.default_rng(13)
    generated_rows = []
    recovered_rows = []
    for index in range(simulations):
        bounds = model.bounds_for_data(fallback)
        values = np.array([rng.uniform(lower, upper) for lower, upper in bounds])
        synthetic = model.sample_dataframe(values, n_trials=trials_per_simulation, participant_id=index, config=fp_config)
        result = fit_fokker_planck_model(model, synthetic, fit_config, fp_config)
        generated_rows.append(dict(zip(model.parameter_names, map(float, values))))
        recovered_rows.append(result.parameters)
    generated = pd.DataFrame(generated_rows)
    recovered = pd.DataFrame(recovered_rows)
    rows = []
    for parameter in model.parameter_names:
        correlation = None
        if simulations >= 2 and generated[parameter].nunique() > 1 and recovered[parameter].nunique() > 1:
            correlation = float(generated[parameter].corr(recovered[parameter]))
        rows.append({"parameter": parameter, "correlation": correlation})
    return rows


def _fp_model_recovery(
    models: list[FokkerPlanckSpec],
    simulations: int,
    trials_per_simulation: int,
    fit_config: FitConfig,
    fp_config: FokkerPlanckConfig,
) -> list[dict[str, object]]:
    import numpy as np

    fallback = pd.DataFrame({"id": [0], "condition": ["0"], "rt": [0.45], "accuracy": [1]})
    rng = np.random.default_rng(17)
    rows = []
    for generating_model in models:
        winners = {candidate.name: 0 for candidate in models}
        for index in range(simulations):
            bounds = generating_model.bounds_for_data(fallback)
            values = np.array([rng.uniform(lower, upper) for lower, upper in bounds])
            synthetic = generating_model.sample_dataframe(values, n_trials=trials_per_simulation, participant_id=index, config=fp_config)
            fits = [fit_fokker_planck_model(candidate, synthetic, fit_config, fp_config) for candidate in models]
            winner = min(fits, key=lambda item: item.bic).model_name.replace(" (Fokker-Planck)", "")
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


def _safe_model_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", value.strip().replace(" ", "_"))
    if not cleaned:
        return "GeneratedModel"
    if cleaned[0].isdigit():
        cleaned = f"Model_{cleaned}"
    return cleaned


if __name__ == "__main__":
    run()
