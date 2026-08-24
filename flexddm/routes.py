from __future__ import annotations


def _app():
    from . import app

    return app


def handle_get(handler, route: str, query: dict[str, list[str]]) -> None:
    app = _app()

    if route == "/":
        handler._send_file(app.FRONTEND_DIR / "index.html", app.FRONTEND_DIR)
        return
    if route.startswith("/static/"):
        handler._send_file(app.FRONTEND_DIR / route.removeprefix("/"), app.FRONTEND_DIR)
        return
    if route == "/api/about":
        handler._send_json(
            {
                "name": "FlexDDM",
                "description": app.APP_DESCRIPTION,
                "version": app.APP_VERSION,
                "buildDate": app.APP_BUILD_DATE,
                "runtime": f"Python {app.sys.version.split()[0]}",
                "contactName": app.APP_CONTACT_NAME,
                "contactInstitution": app.APP_CONTACT_INSTITUTION,
                "contactEmail": app.APP_CONTACT_EMAIL,
                "citation": app.APP_CITATION,
                "copyright": app.APP_COPYRIGHT_NOTICE,
            }
        )
        return
    if route == "/api/models":
        generated, saved, invalid = app._discover_generated_models()
        fp_models = sorted(set(app.BUILTIN_FP_MODELS).union(app._discover_generated_fp_models()))
        seeded_custom = app._available_seeded_custom_models()
        handler._send_json(
            {
                "builtins": app.describe_models(),
                "generated": generated,
                "saved": saved,
                "invalidGenerated": invalid,
                "fokkerPlanckModels": fp_models,
                "templates": sorted(app.MODEL_STARTER_TEMPLATES),
                "fokkerPlanckTemplates": list(app.FOKKER_PLANCK_TEMPLATES),
                "simulationTemplates": list(app.SIMULATION_TEMPLATES),
                "lockedTemplates": ["StandardDDM", "DMC", "DSTP", "SSP"],
                "seededCustom": sorted(seeded_custom),
                "descriptions": app._model_descriptions(generated),
            }
        )
        return
    if route == "/api/model-parameters":
        name = query.get("name", ["StandardDDM"])[0]
        solver = query.get("solver", ["simulation_de"])[0]
        model = app._load_fp_model(name) if solver == "fokker_planck" else app._load_model(name)
        handler._send_json({"name": name, "parameterBounds": [{"name": p.name, "lower": p.lower, "upper": p.upper} for p in model.parameters]})
        return
    if route == "/api/model-template":
        template = query.get("name", ["Blank"])[0]
        handler._send_json(
            {
                "name": template,
                "code": app.MODEL_TEMPLATES.get(template, app.MODEL_TEMPLATES["Blank"]),
                "parameterBounds": app._bounds_rows(app.MODEL_TEMPLATE_BOUNDS.get(template, app.MODEL_TEMPLATE_BOUNDS["Blank"])),
                "description": app.TEMPLATE_DESCRIPTIONS.get(template.replace(" - Fokker-Planck", "").replace(" - Simulation", ""), ""),
            }
        )
        return
    if route == "/api/model-code":
        _send_model_code(handler, query)
        return
    if route == "/api/default-data-path":
        name = query.get("name", ["ulrich2015.csv"])[0]
        path = app._default_data_path(name)
        handler._send_json({"path": str(path), "exists": path.is_file()})
        return
    if route.startswith("/runs/"):
        handler._send_file(app.RUNS / route.removeprefix("/runs/"), app.RUNS)
        return
    if route == "/api/job-status":
        job_id = query.get("job", [""])[0]
        with app.JOBS_LOCK:
            if job_id in app.JOBS:
                app.JOBS[job_id]["elapsedSeconds"] = int(round(app.time.time() - float(app.JOBS[job_id].get("startedAt", app.time.time()))))
            job = dict(app.JOBS.get(job_id, {"status": "missing", "message": "Job not found"}))
        handler._send_json(job, status=200 if job.get("status") != "missing" else 404)
        return

    handler._send_json({"error": "Not found"}, status=404)


def handle_post(handler, route: str, payload: dict) -> None:
    app = _app()

    if route == "/api/infer":
        inference = app.infer_parameters(payload.get("pseudocode", ""))
        handler._send_json(
            {
                "parameters": [
                    {"name": parameter.name, "lower": parameter.lower, "upper": parameter.upper}
                    for parameter in inference.parameters
                ],
                "unresolved": list(inference.unresolved_names),
                "compilerStatus": "Parameter inference only. Use the Python editor for runnable custom models.",
            }
        )
        return
    if route == "/api/data-preview":
        all_data = app._load_payload_data(payload)
        selected = app._select_preview_data(all_data, payload)
        handler._send_json(app._data_preview(selected, all_data))
        return
    if route == "/api/data-columns":
        handler._send_json(app._data_columns(payload))
        return
    if route == "/api/data-settings-load":
        handler._send_json(app._load_data_settings(payload))
        return
    if route == "/api/data-settings-save":
        handler._send_json(app._save_data_settings(payload))
        return
    if route == "/api/generate-model":
        model_name = app._safe_model_name(payload.get("modelName", "GeneratedModel"))
        skeleton = app.model_skeleton_from_pseudocode(model_name, payload.get("pseudocode", ""))
        destination = app._save_model_code(model_name, skeleton)
        handler._send_json({"path": str(destination)})
        return
    if route == "/api/save-model":
        _save_model(handler, payload)
        return
    if route == "/api/delete-model":
        handler._send_json(app._delete_saved_model(str(payload.get("modelName", ""))))
        return
    if route == "/api/model-schematic":
        _send_model_preview(handler, payload, app._model_schematic_preview)
        return
    if route == "/api/model-trajectory":
        _send_model_preview(handler, payload, app._model_trajectory_preview)
        return
    if route == "/api/fit":
        handler._send_json(app._run_fit_request(payload))
        return
    if route == "/api/fit-job":
        handler._send_json(app._start_job("fit", payload, app._run_fit_request))
        return
    if route == "/api/cancel-job":
        handler._send_json(app._cancel_job(str(payload.get("jobId", ""))))
        return
    if route == "/api/browse-file":
        handler._send_json(app._browse_csv_file())
        return
    if route == "/api/save-artifact":
        handler._send_json(app._save_artifact(payload))
        return
    if route == "/api/read-clipboard":
        handler._send_json({"text": app._read_system_clipboard()})
        return
    if route == "/api/export-cluster-job":
        handler._send_json(app._export_cluster_job(payload))
        return
    if route == "/api/simulate":
        handler._send_json(app._run_simulation_request(payload))
        return
    if route == "/api/simulated-preview":
        handler._send_json(app._simulated_preview_request(payload))
        return
    if route == "/api/validate":
        handler._send_json(app._run_validation_request(payload))
        return
    if route == "/api/validate-job":
        handler._send_json(app._start_job("validation", payload, app._run_validation_request))
        return

    handler._send_json({"error": "Not found"}, status=404)


def _send_model_code(handler, query: dict[str, list[str]]) -> None:
    app = _app()

    raw_name = query.get("name", [""])[0]
    seeded_custom = app._available_seeded_custom_models()
    if raw_name in seeded_custom:
        handler._send_json(
            {
                "name": raw_name,
                "code": seeded_custom[raw_name],
                "parameterBounds": app._bounds_rows(app.MODEL_TEMPLATE_BOUNDS.get(raw_name, {})),
                "description": app.TEMPLATE_DESCRIPTIONS.get(raw_name, ""),
            }
        )
        return

    name = app._safe_model_name(raw_name)
    path = (app.GENERATED_MODELS / f"{name}.py").resolve()
    if not path.is_file() or not str(path).startswith(str(app.GENERATED_MODELS.resolve())):
        if name in seeded_custom:
            handler._send_json(
                {
                    "name": name,
                    "code": seeded_custom[name],
                    "parameterBounds": app._bounds_rows(app.MODEL_TEMPLATE_BOUNDS.get(name, {})),
                    "description": app.TEMPLATE_DESCRIPTIONS.get(name, ""),
                }
            )
            return
        handler._send_json({"error": "Model file not found"}, status=404)
        return

    raw_code = path.read_text(encoding="utf-8")
    handler._send_json(
        {
            "name": name,
            "code": app._strip_parameter_bounds_block(raw_code),
            "parameterBounds": app._bounds_rows(app._load_parameter_bounds(name, raw_code)),
            "description": app._load_model_metadata(name).get("description", ""),
        }
    )


def _save_model(handler, payload: dict) -> None:
    app = _app()

    model_name = app._safe_model_name(payload.get("modelName", "CustomModel"))
    destination = app._save_model_code(model_name, payload.get("code", ""), payload.get("parameterBounds"))
    app._save_model_metadata(model_name, str(payload.get("description", "")).strip())
    model = app._load_model(model_name)
    app._validate_model_runtime(model)
    code = destination.read_text(encoding="utf-8")
    handler._send_json(
        {
            "path": str(destination),
            "code": code,
            "parameterBounds": app._bounds_rows(app._load_parameter_bounds(model_name, code)),
            "model": model.name,
            "parameters": [{"name": p.name, "lower": p.lower, "upper": p.upper} for p in model.parameters],
        }
    )


def _send_model_preview(handler, payload: dict, preview_function) -> None:
    app = _app()

    try:
        handler._send_json(preview_function(payload))
    except Exception as exc:
        handler._send_json(app._model_preview_error_payload(exc), status=400)
