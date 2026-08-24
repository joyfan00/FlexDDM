const $ = (id) => document.getElementById(id);
    let modelInventory = {builtins: [], generated: [], templates: [], fokkerPlanckTemplates: [], simulationTemplates: [], seededCustom: [], lockedTemplates: []};
    let currentModelName = "MyFlexModel";
    let currentModelDescription = "";
    let currentModelKind = "process";
    let currentModelConditions = ["0", "1"];
    let currentHiddenImports = "from __future__ import annotations\n\nfrom flexddm import define_model\n";
    let currentModelSource = "";
    let currentTemplateName = "";
    let currentTemplateDirty = false;
    let currentTemplateConditionKey = "";
    let pendingDeleteModelName = "";
    let parameterBounds = {};
    let parameterRenderTimer = null;
    let activeFitJob = null;
    let activeValidationJob = null;
    let lastDataPreview = null;
    let lastFilterPreview = null;
    let dataLoaded = false;
    let csvColumns = [];
    let mappingAutoCollapsed = false;
    let fitHasResults = false;
    let validationHasResults = false;
    let dataConstantRows = [];
    let columnFilterRows = [];
    let conditionColumnRows = [];
    let dataPreviewRequestId = 0;
    let filterPreviewRequestId = 0;
    let schematicTimer = null;
    let schematicRequestId = 0;
    let schematicData = null;
    let schematicParameterValues = {};
    let schematicTrajectories = [];
    let schematicAnimation = null;
    let schematicSeed = 1;
    let schematicError = null;
    let schematicLabelBoxes = [];
    let schematicLabelOverrides = {};
    let schematicLabelDrag = null;
    let modelCodeErrorLine = null;
    let simulateParameterRows = [];
    let simulateCovariateRows = [];
    let lastSimulatedRows = [];
    let lastSimulatedPreview = null;
    let pendingClusterExport = null;
    let tutorialRunning = false;
    let tutorialStepIndex = 0;
    let tutorialPreviousPane = "dataPane";
    let tutorialCheckTimer = null;
    const DATA_SETTINGS_VERSION = "2026-08-23-condition-columns-1";
    const CLUSTER_EXPORT_ETA_SECONDS = 300;
    resetStaleDataSettings();

    function showPane(id) {
      if (!id) return;
      document.querySelectorAll(".pane").forEach(p => p.classList.toggle("active", p.id === id));
      document.querySelectorAll(".tab[data-pane]").forEach(t => t.classList.toggle("active", t.dataset.pane === id));
      if (id === "modelPane") refreshCurrentTemplateForConditions().catch(error => setModelStatus(error.message, true));
    }
    document.querySelectorAll(".tab[data-pane]").forEach(tab => tab.onclick = () => {
      if (tutorialRunning) endTutorial(false);
      showPane(tab.dataset.pane);
    });

    async function post(path, payload) {
      const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
      const data = await response.json();
      if (!response.ok) {
        const error = new Error(data.error || "Request failed");
        error.payload = data;
        throw error;
      }
      return data;
    }
    async function browseCsvPath() {
      $("dataStatus").textContent = "Opening file browser...";
      try {
        const data = await post("/api/browse-file", {});
        if (data.path) {
          $("csvPath").value = data.path;
          columnFilterRows = [];
          await loadCsvColumns(data.path);
        } else {
          $("dataStatus").textContent = "File selection cancelled.";
        }
      } catch (error) {
        $("dataStatus").innerHTML = `<span class="warn">${error.message}</span>`;
      }
    }
    async function getJson(path) {
      const response = await fetch(path);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Request failed");
      return data;
    }
    function activePaneId() {
      return document.querySelector(".pane.active")?.id || "dataPane";
    }
    function openTutorialIntro() {
      tutorialPreviousPane = activePaneId();
      $("tutorialTab").classList.add("tutorial-active");
      $("tutorialIntroModal").classList.remove("hidden");
    }
    function closeTutorialIntro(restore = true) {
      $("tutorialIntroModal").classList.add("hidden");
      if (!tutorialRunning) $("tutorialTab").classList.remove("tutorial-active");
      if (restore && tutorialPreviousPane) showPane(tutorialPreviousPane);
    }
    let tutorialMappingResetDone = false;
    let tutorialFilterResetDone = false;
    let tutorialCsvBaseline = "";
    let tutorialModelBrowserOpenedForTemplate = false;
    let tutorialModelBrowserOpenedForCustom = false;
    let tutorialDmcTemplateChosen = false;
    let tutorialCustomSourceChosen = false;
    let tutorialCustomModelChosen = false;
    let tutorialBoundaryModalOpened = false;
    let tutorialBoundaryParamsTouched = false;
    let tutorialBoundaryAppliedThisStep = false;
    let tutorialSaveModalOpened = false;
    let tutorialModelSavedThisStep = false;
    function tutorialCsvIsUlrich() {
      return /ulrich2015\.csv$/i.test(String($("csvPath").value || "").replace(/\\/g, "/")) && csvColumns.length > 0;
    }
    function resetTutorialMappingOnce() {
      if (tutorialMappingResetDone || !csvColumns.length) return;
      ["participantCol", "rtCol", "accuracyCol"].forEach(id => { $(id).value = ""; });
      $("responseKind").value = "";
      conditionColumnRows = normalizeConditionColumnRows([]);
      renderConditionColumns();
      mappingAutoCollapsed = false;
      tutorialMappingResetDone = true;
    }
    function resetTutorialFiltersOnce() {
      if (tutorialFilterResetDone) return;
      columnFilterRows = normalizeColumnFilterRows([]);
      if ($("columnFilters")) renderColumnFilters((lastFilterPreview || lastDataPreview || {}).filterColumns || []);
      if ($("filterStatus")) $("filterStatus").textContent = "";
      if ($("filterDataTable")) $("filterDataTable").innerHTML = "";
      tutorialFilterResetDone = true;
    }
    function tutorialCoreMappingComplete() {
      return $("participantCol").value === "PPT"
        && $("rtCol").value === "RT"
        && $("accuracyCol").value === "Correct"
        && $("responseKind").value === "accuracy";
    }
    function tutorialConditionReady() {
      return conditionColumnsPayload()[0] === "Condition";
    }
    function tutorialRtFilterReady() {
      return columnFiltersPayload().some(row => row.column === "RT" && row.type === "numeric" && Number(row.min) === 0.15 && Number(row.max) === 2);
    }
    function tutorialLoadedDmc() {
      return tutorialDmcTemplateChosen
        && /DMC/i.test(String($("loadedModelLabel").textContent || currentModelName || ""))
        && !/simulation/i.test(String($("loadedModelLabel").textContent || ""));
    }
    function tutorialSchematicValuesReady() {
      return Math.abs(Number(schematicParameterValues.mu_c) - 0.325) < 0.002
        && Math.abs(Number(schematicParameterValues.characteristic_time) - 0.225) < 0.002;
    }
    function tutorialBothConditionTrajectories() {
      const conditions = new Set(schematicTrajectories.map(row => String(row.condition)));
      return conditions.size >= 2;
    }
    function tutorialBoundaryAdded() {
      return /def\s+bound\s*\(/.test($("modelCode").value || "");
    }
    function isTutorialModelName(value) {
      return /Tutorial[\s_-]*Model/i.test(String(value || ""));
    }
    function tutorialModelSaved() {
      return tutorialModelSavedThisStep && isTutorialModelName($("loadedModelLabel").textContent || currentModelName || "");
    }
    function tutorialClusterPromptVisible() {
      return !$("clusterExportModal").classList.contains("hidden");
    }
    function tutorialCurrentStepTitle() {
      return tutorialSteps[tutorialStepIndex]?.title || "";
    }
    function tutorialBoundaryParamsReady() {
      return tutorialCollapseRateReady() && tutorialCollapseShapeReady();
    }
    function tutorialCollapseRateReady() {
      const rate = parameterBounds.collapse_rate || {};
      return parameterExists("collapse_rate")
        && Number(rate.lower) === 0
        && Number(rate.upper) === 4;
    }
    function tutorialCollapseShapeReady() {
      const shape = parameterBounds.collapse_shape || {};
      return parameterExists("collapse_shape")
        && Math.abs(Number(shape.lower) - 0.2) < 1e-9
        && Number(shape.upper) === 4;
    }
    function tutorialBoundaryCodeReady() {
      const code = $("functionModalCode") ? $("functionModalCode").value : "";
      return /def\s+bound\s*\([^)]*collapse_rate[^)]*collapse_shape[^)]*\)\s*:/.test(code)
        && /return\s+alpha\s*\/\s*2\s*\*\s*math\.exp\s*\(\s*-collapse_rate\s*\*\s*\(\s*t\s*\*\*\s*collapse_shape\s*\)\s*\)/.test(code);
    }
    function tutorialCollapseSlidersReady() {
      return Math.abs(Number(schematicParameterValues.collapse_rate) - 1.25) < 0.03
        && Math.abs(Number(schematicParameterValues.collapse_shape) - 1.50) < 0.03;
    }
    function tutorialSaveNameReady() {
      return String($("saveModelName")?.value || "").trim() === "Tutorial Model";
    }
    function tutorialCustomModelSourceReady() {
      const custom = document.querySelector("input[name='modelLoadSource'][value='custom']");
      return tutorialCustomSourceChosen && !$("modelLoadModal").classList.contains("hidden") && custom && custom.checked;
    }
    function tutorialDmcFitReady() {
      return String($("fitModelSelect").value || "") === "DMC";
    }
    function tutorialParameterRecoverySelected() {
      return String($("validationMode").value || "") === "parameter";
    }
    function tutorialDmcSimulationReady() {
      return String($("simulateModelSelect").value || "") === "DMC";
    }
    function tutorialSimulationParameter() {
      return simulateParameterRows.find(row => row.name === "mu_c") || simulateParameterRows[0] || null;
    }
    function tutorialSimulationMeanReady() {
      const row = tutorialSimulationParameter();
      return Boolean(row) && Math.abs(Number(row.mean) - 0.325) < 0.004;
    }
    function tutorialSimulationSdReady() {
      const row = tutorialSimulationParameter();
      return Boolean(row) && Math.abs(Number(row.sd) - 0.075) < 0.004;
    }
    function tutorialSimulationSkewReady() {
      const row = tutorialSimulationParameter();
      return Boolean(row) && Math.abs(Number(row.skew) - 1.25) < 0.04;
    }
    function tutorialSimulationCovariateReady() {
      return simulateCovariateRows.some(row =>
        String(row.name || "").trim().toLowerCase() === "age"
        && String(row.level || "") === "participant"
        && Math.abs(Number((row.targets || {}).mu_c ?? row.correlation) - 0.6) < 0.02
      );
    }
    function tutorialSyntheticDataReady() {
      return lastSimulatedRows.length > 0 && !$("simulateResultsArea").classList.contains("hidden");
    }
    function selectBoundFunctionInEditor() {
      const textarea = $("modelCode");
      const code = textarea.value || "";
      const match = code.match(/def\s+bound\s*\([^]*?(?=\n\s*def\s+|\n#|\n\s*process_model|\n\s*model\s*=|$)/);
      if (!match) return;
      textarea.focus();
      textarea.setSelectionRange(match.index, match.index + match[0].length);
    }
    function selectDmcTemplateView() {
      const template = document.querySelector("input[name='modelLoadSource'][value='template']");
      if (template) template.checked = true;
      if ($("showSimulationTemplates")) $("showSimulationTemplates").checked = false;
      renderModelLoadCards();
    }
    function resetLoadModelTutorialStep() {
      tutorialModelBrowserOpenedForTemplate = false;
      tutorialDmcTemplateChosen = false;
      closeModelLoader();
    }
    function prepareChooseDmcTutorialStep() {
      tutorialDmcTemplateChosen = false;
      openModelLoader({tutorialSilent: true});
      selectDmcTemplateView();
    }
    function resetAddBoundaryTutorialStep() {
      tutorialBoundaryModalOpened = false;
      closeFunctionModal(true);
    }
    function prepareBoundaryModalTutorialStep() {
      if ($("functionModal").classList.contains("hidden")) openFunctionModal("bound", {tutorialSilent: true});
    }
    function prepareBoundaryParamsTutorialStep() {
      prepareBoundaryModalTutorialStep();
    }
    function prepareBoundaryCodeTutorialStep() {
      tutorialBoundaryAppliedThisStep = false;
      prepareBoundaryModalTutorialStep();
    }
    function resetSaveModelTutorialStep() {
      tutorialSaveModalOpened = false;
      tutorialModelSavedThisStep = false;
      closeSaveModelModal(true);
    }
    function prepareSaveTutorialModelStep() {
      tutorialModelSavedThisStep = false;
      if ($("saveModelModal").classList.contains("hidden")) openSaveModelModal({tutorialSilent: true});
    }
    function resetFindCustomTutorialStep() {
      tutorialModelBrowserOpenedForCustom = false;
      tutorialCustomSourceChosen = false;
      tutorialCustomModelChosen = false;
      closeModelLoader();
    }
    function prepareCustomModelsTutorialStep() {
      tutorialCustomSourceChosen = false;
      openModelLoader({tutorialSilent: true});
      const template = document.querySelector("input[name='modelLoadSource'][value='template']");
      if (template) template.checked = true;
      renderModelLoadCards();
    }
    function prepareTutorialModelCardStep() {
      tutorialCustomModelChosen = false;
      openModelLoader({tutorialSilent: true});
      const custom = document.querySelector("input[name='modelLoadSource'][value='custom']");
      if (custom) custom.checked = true;
      renderModelLoadCards();
    }
    function syncTutorialRestrictions() {
      const title = tutorialRunning ? tutorialCurrentStepTitle() : "";
      document.querySelectorAll("#modelHelperBar button").forEach(button => {
        const allowBoundary = title === "Add Model Pieces" && button.dataset.functionComponent === "bound";
        const restricted = title === "Add Model Pieces" && !allowBoundary;
        button.classList.toggle("tutorial-disabled", restricted);
        button.disabled = restricted;
      });
      if ($("applyFunctionAdd")) {
        const parameterStep = title === "Add Collapse Rate" || title === "Add Collapse Shape";
        const boundaryStep = title === "Type Boundary Code" || title === "Apply Boundary Function" || parameterStep;
        const ready = ["Type Boundary Code", "Apply Boundary Function"].includes(title) ? tutorialBoundaryCodeReady() : !boundaryStep;
        $("applyFunctionAdd").disabled = boundaryStep && !ready;
      }
      if ($("applySaveModel")) {
        $("applySaveModel").disabled = title === "Save Tutorial Model" && !tutorialSaveNameReady();
      }
      if ($("cancelFunctionAdd")) {
        $("cancelFunctionAdd").disabled = ["Add Collapse Rate", "Add Collapse Shape", "Type Boundary Code", "Apply Boundary Function"].includes(title);
      }
      if ($("cancelSaveModel")) {
        $("cancelSaveModel").disabled = title === "Save Tutorial Model";
      }
    }
    const tutorialSteps = [
      {
        pane: "dataPane",
        target: "#csvPath",
        title: "Choose Data",
        body: "FlexDDM starts from trial-level data. Each row should represent one decision with a response time, a response or accuracy value, a participant identifier, and any trial information the model may use.",
      },
      {
        pane: "dataPane",
        target: "#browseCsv",
        title: "Browse For A CSV",
        body: "Use Browse to select the built-in Ulrich dataset: data/ulrich2015.csv. Even if another file is already loaded, choose this one for the tutorial so every later step matches.",
        waitText: "Choose data/ulrich2015.csv to continue.",
        waitFor: () => tutorialCsvIsUlrich(),
        autoAdvance: true,
      },
      {
        pane: "dataPane",
        target: "#dataMappingDetails",
        title: "Map Core Columns",
        body: "Choose PPT for participant, RT for response time, Correct for response, and Accuracy for response meaning. These tell FlexDDM which original columns carry the core decision information.",
        waitText: "Choose PPT, RT, Correct, and Accuracy to continue.",
        ensure: () => {
          if (!$("dataMappingDetails").classList.contains("hidden")) $("dataMappingDetails").open = true;
          resetTutorialMappingOnce();
        },
        waitFor: () => tutorialCoreMappingComplete(),
        autoAdvance: true,
      },
      {
        pane: "dataPane",
        target: "#dataConditionsDetails",
        fallback: "#dataMappingDetails",
        title: "Assign Conditions",
        body: "Choose Condition as the first condition column. Conditions define trial types for this tutorial model, but FlexDDM also supports condition-free custom models when every row can be treated the same way.",
        ensure: () => {
          if (!$("dataConditionsDetails").classList.contains("hidden")) $("dataConditionsDetails").open = true;
        },
        waitText: "Choose Condition as the first condition column.",
        waitFor: () => tutorialConditionReady(),
        autoAdvance: true,
      },
      {
        pane: "dataPane",
        target: "#filterControlsDetails",
        fallback: "#filterContent",
        title: "Filter Analysis Rows",
        body: "Open Filters and add an RT numeric filter that keeps trials from 0.15 to 2 seconds. This removes implausibly fast responses and very slow responses without changing the original CSV.",
        ensure: () => {
          if (mappingComplete()) $("filterControlsDetails").open = true;
          resetTutorialFiltersOnce();
        },
        waitText: "Create an RT filter with minimum 0.15 and maximum 2.",
        waitFor: () => tutorialRtFilterReady(),
        autoAdvance: true,
      },
      {
        pane: "dataPane",
        target: "#dataConstantsDetails",
        fallback: "#filterContent",
        title: "Define Data Constants",
        body: "Data constants are reusable values derived from the selected data, such as min_RT. They can become model bounds or fixed quantities without being hard-coded into a model file.",
        ensure: () => { if (mappingComplete()) $("dataConstantsDetails").open = true; },
      },
      {
        pane: "dataPane",
        target: "#dataDiagnosticsExportFrame",
        fallback: "#filterStatus",
        title: "Inspect Data Diagnostics",
        body: "Diagnostics let you check whether the loaded and filtered data look sensible before modeling. In diffusion modeling, RT distributions and response proportions are often the first clues about whether a model is aimed at the right structure.",
      },
      {
        pane: "dataPane",
        target: "#filterDataTable",
        fallback: "#filterStatus",
        title: "Preview Rows",
        body: "Preview rows show the actual filtered data that will flow downstream. The cleaned participant ID is added for analysis convenience, but original column names are preserved so the table still matches your source file.",
      },
      {
        pane: "modelPane",
        target: ".loaded-model-row",
        title: "Load A Model",
        body: "Click Browse to open the model browser. The tutorial will use the regular DMC template, not the simulation version, so you can see a time-varying drift model that still works with Fokker-Planck.",
        waitText: "Click Browse to open the model browser.",
        ensure: resetLoadModelTutorialStep,
        waitFor: () => tutorialModelBrowserOpenedForTemplate && !$("modelLoadModal").classList.contains("hidden"),
        autoAdvance: true,
      },
      {
        pane: "modelPane",
        target: '[data-load-template*="DMC"]',
        fallback: "#modelLoadCards",
        title: "Choose DMC",
        body: "Choose the DMC card. DMC models conflict by combining an automatic pulse with a controlled drift process, so the predicted evidence path can curve over time.",
        ensure: prepareChooseDmcTutorialStep,
        waitText: "Click the DMC template card.",
        waitFor: () => tutorialLoadedDmc(),
        autoAdvance: true,
      },
      {
        pane: "modelPane",
        target: "#modelCode",
        title: "Edit Minimal Python",
        body: "This is the minimal Python users edit: mathematical pieces such as drift, boundary, noise, start, and non-decision time. Behind the scenes FlexDDM converts solver-critical model pieces to compiled C++ or machine-code paths for faster fitting.",
      },
      {
        pane: "modelPane",
        target: ".model-workspace",
        fallback: ".schematic-panel",
        title: "Read And Adjust The Schematic",
        body: "Move the sliders so mu_c is 0.325 and characteristic_time is 0.225. This increases the controlled drift and shifts the timing of the automatic conflict pulse, so the curve changes before any fitting occurs.",
        waitText: "Set mu_c to 0.325 and characteristic_time to 0.225.",
        waitFor: () => tutorialSchematicValuesReady(),
      },
      {
        pane: "modelPane",
        target: "#schematicLegendControls",
        fallback: "#modelSchematicExportFrame",
        title: "Simulate Schematic Paths",
        body: "Use the condition buttons above the schematic to simulate one noisy trajectory from each condition. The smooth lines are the model's average path; the simulated paths show trial-to-trial Brownian variability.",
        waitText: "Simulate trajectories for two conditions.",
        waitFor: () => tutorialBothConditionTrajectories(),
      },
      {
        pane: "modelPane",
        target: "#modelHelperBar",
        title: "Add Model Pieces",
        body: "Now click Add Boundary. During this tutorial, the other model-piece buttons are temporarily unavailable so we can focus on one change at a time.",
        waitText: "Click Add Boundary.",
        ensure: resetAddBoundaryTutorialStep,
        waitFor: () => tutorialBoundaryModalOpened && !$("functionModal").classList.contains("hidden") && /bound/.test($("functionModalTitle").textContent || ""),
        autoAdvance: true,
      },
      {
        pane: "modelPane",
        target: "#functionParameterRow",
        fallback: "#functionModal",
        title: "Add Collapse Rate",
        body: "Add collapse_rate with bounds 0 to 4. This free parameter controls how quickly the boundary collapses.",
        ensure: prepareBoundaryParamsTutorialStep,
        waitText: "Add collapse_rate [0, 4].",
        waitFor: () => tutorialBoundaryParamsTouched && tutorialCollapseRateReady(),
        autoAdvance: true,
      },
      {
        pane: "modelPane",
        target: "#functionParameterRow",
        fallback: "#functionModal",
        title: "Add Collapse Shape",
        body: "Now add collapse_shape with bounds 0.2 to 4. This controls the curvature of the collapse over time.",
        ensure: prepareBoundaryParamsTutorialStep,
        waitText: "Add collapse_shape [0.2, 4].",
        waitFor: () => tutorialBoundaryParamsTouched && tutorialCollapseShapeReady(),
        autoAdvance: true,
      },
      {
        pane: "modelPane",
        target: "#functionModalCode",
        fallback: "#functionModal",
        title: "Type Boundary Code",
        body: "Replace the return line with exactly: return alpha / 2 * math.exp(-collapse_rate * (t ** collapse_shape)). FlexDDM will only enable Apply Function once the boundary function includes both new parameters and this expression.",
        copyText: "return alpha / 2 * math.exp(-collapse_rate * (t ** collapse_shape))",
        ensure: prepareBoundaryCodeTutorialStep,
        waitText: "Enter the collapsing-bound return line exactly.",
        waitFor: () => tutorialBoundaryCodeReady(),
        autoAdvance: true,
      },
      {
        pane: "modelPane",
        target: "#applyFunctionAdd",
        fallback: "#functionModal",
        title: "Apply Boundary Function",
        body: "That boundary function is written correctly. Click Apply Function to place it into the model code.",
        waitText: "Click Apply Function.",
        waitFor: () => tutorialBoundaryAppliedThisStep && tutorialBoundaryAdded(),
        autoAdvance: true,
      },
      {
        pane: "modelPane",
        target: "#modelCode",
        title: "Boundary Added",
        body: "The new bound(...) function is now part of the editable model code. Because it defines a time-varying boundary, the schematic can show how the response threshold collapses across time.",
        ensure: selectBoundFunctionInEditor,
      },
      {
        pane: "modelPane",
        target: ".schematic-panel",
        title: "Adjust Collapse Shape",
        body: "Now use the new sliders to set collapse_rate to 1.25 and collapse_shape to 1.50. The boundary should pull inward over time, showing how less evidence is needed later in the trial.",
        waitText: "Set collapse_rate to 1.25 and collapse_shape to 1.50.",
        waitFor: () => tutorialCollapseSlidersReady(),
      },
      {
        pane: "modelPane",
        target: "#parameterBounds",
        title: "Set Free Parameters",
        body: "The free-parameter table defines the search bounds used during fitting. Bounds should be broad enough to let the model find good values, but narrow enough to keep impossible or meaningless values out.",
      },
      {
        pane: "modelPane",
        target: "#saveModel",
        title: "Save Custom Models",
        body: "Click Save Model to open the save dialog. Saving records the editable code and the free-parameter bounds together.",
        waitText: "Click Save Model.",
        ensure: resetSaveModelTutorialStep,
        waitFor: () => tutorialSaveModalOpened && !$("saveModelModal").classList.contains("hidden"),
        autoAdvance: true,
      },
      {
        pane: "modelPane",
        target: "#saveModelModal",
        title: "Save Tutorial Model",
        body: "Name this model exactly Tutorial Model, then click Save. The Save button stays disabled until the tutorial name is entered correctly.",
        ensure: prepareSaveTutorialModelStep,
        waitText: "Name it Tutorial Model and save it.",
        waitFor: () => tutorialModelSaved(),
        autoAdvance: true,
      },
      {
        pane: "modelPane",
        target: ".loaded-model-row",
        title: "Find Your Custom Model",
        body: "Click Browse again. This time we will look at where saved custom models live, instead of loading another built-in template.",
        waitText: "Click Browse to open the model browser.",
        ensure: resetFindCustomTutorialStep,
        waitFor: () => tutorialModelBrowserOpenedForCustom && !$("modelLoadModal").classList.contains("hidden"),
        autoAdvance: true,
      },
      {
        pane: "modelPane",
        target: ".segmented",
        fallback: "#modelLoadModal",
        title: "Custom Models",
        body: "Click Custom to see saved models, including Tutorial Model. Built-in templates are locked examples; custom models are the versions you save and reuse.",
        ensure: prepareCustomModelsTutorialStep,
        waitText: "Choose Custom in the model browser.",
        waitFor: () => tutorialCustomModelSourceReady(),
        autoAdvance: true,
      },
      {
        pane: "modelPane",
        target: '[data-load-custom*="Tutorial"]',
        fallback: "#modelLoadCards",
        title: "Tutorial Model Card",
        body: "This is where Tutorial Model now lives. You can load it later from Custom, or select it from Fit and Validate after saving.",
        ensure: prepareTutorialModelCardStep,
        waitText: "Click the Tutorial Model card.",
        waitFor: () => tutorialCustomModelChosen && $("modelLoadModal").classList.contains("hidden") && isTutorialModelName($("loadedModelLabel").textContent || currentModelName || ""),
        autoAdvance: true,
      },
      {
        pane: "simulatePane",
        target: "#simulateModelSelect",
        title: "Choose A Simulation Model",
        body: "The Simulate pane generates synthetic trial-level datasets from a model. Choose DMC here too, so the synthetic data come from the same kind of time-varying conflict model you just edited.",
        ensure: () => { $("simulateSetupDetails").open = true; },
        waitText: "Choose DMC as the simulation model.",
        waitFor: () => tutorialDmcSimulationReady(),
        autoAdvance: true,
      },
      {
        pane: "simulatePane",
        target: '[data-sim-param-name="mu_c"]',
        fallback: "#simulateParameterControls",
        title: "Tweak A Parameter Mean",
        body: "Each free parameter is drawn from a bounded distribution. For mu_c, set the mean to 0.325. This makes the controlled drift a little stronger on average across synthetic participants.",
        ensure: () => { $("simulateSetupDetails").open = true; },
        waitText: "Set the mu_c mean to 0.325.",
        waitFor: () => tutorialSimulationMeanReady(),
        autoAdvance: true,
      },
      {
        pane: "simulatePane",
        target: '[data-sim-param-name="mu_c"]',
        fallback: "#simulateParameterControls",
        title: "Tweak A Parameter SD",
        body: "Now set the mu_c SD to 0.075. The SD controls how different synthetic participants are from one another while staying inside the model's allowed bounds.",
        ensure: () => { $("simulateSetupDetails").open = true; },
        waitText: "Set the mu_c SD to 0.075.",
        waitFor: () => tutorialSimulationSdReady(),
        autoAdvance: true,
      },
      {
        pane: "simulatePane",
        target: '[data-sim-param-name="mu_c"]',
        fallback: "#simulateParameterControls",
        title: "Tweak A Parameter Skew",
        body: "Set the mu_c skew to 1.25. Skew changes the shape of the participant-level parameter distribution, which is useful when you expect many people near one value but a longer tail in one direction.",
        ensure: () => { $("simulateSetupDetails").open = true; },
        waitText: "Set the mu_c skew to 1.25.",
        waitFor: () => tutorialSimulationSkewReady(),
        autoAdvance: true,
      },
      {
        pane: "simulatePane",
        target: "#simulateCovariateDetails",
        title: "Add A Synthetic Trait",
        body: "Open Trial Covariates / Participant Traits and click Add Covariate / Trait. The default age trait is participant-level and is correlated with a parameter, showing how synthetic individual differences can be built into a dataset.",
        ensure: () => {
          $("simulateSetupDetails").open = true;
          $("simulateCovariateDetails").open = true;
        },
        waitText: "Add the default age participant trait.",
        waitFor: () => tutorialSimulationCovariateReady(),
        autoAdvance: true,
      },
      {
        pane: "simulatePane",
        target: "#runSimulate",
        title: "Create Synthetic Data",
        body: "Click Create Synthetic Data. FlexDDM will draw participant parameters, generate trial rows, attach the age trait, and then summarize the synthetic dataset just like the Data pane summarizes empirical data.",
        ensure: () => { $("simulateSetupDetails").open = true; },
        waitText: "Create synthetic data and wait for the results.",
        waitFor: () => tutorialSyntheticDataReady(),
        autoAdvance: true,
      },
      {
        pane: "simulatePane",
        target: "#simulateTableExportFrame",
        fallback: "#simulateResultsArea",
        title: "Synthetic Data Rows",
        body: "The table contains the full generated dataset, not just a tiny preview. You can scroll through the simulated participant IDs, conditions, responses, RTs, free-parameter values, and any covariates or traits you created.",
      },
      {
        pane: "simulatePane",
        target: "#simulateDiagnosticsExportFrame",
        fallback: "#simulateResultsArea",
        title: "Synthetic Diagnostics",
        body: "These diagnostics show the distributions and response patterns produced by the simulation. This is a quick way to ask whether a model produces plausible data before fitting it to real participants.",
      },
      {
        pane: "simulatePane",
        target: "#saveSimulatedTable",
        fallback: "#simulateTableExportFrame",
        title: "Save Or Reuse Synthetic Data",
        body: "The small save icon downloads the synthetic data table. If you want to analyze it like an empirical dataset, you can return to Data and load that saved CSV.",
      },
      {
        pane: "validationPane",
        target: "#validationMode",
        title: "Choose Validation",
        body: "Validation asks whether a model can recover what it should. Parameter recovery checks whether known parameters can be estimated again; model recovery checks whether the correct model wins; internal consistency checks odd/even stability. Parameter curvature/Hessian checks whether fitted parameters are collinear, and likelihood profiles sweep one parameter at a time to reveal flat, jagged, or multi-peaked objective regions. Choose Parameter recovery for this tutorial.",
        ensure: () => { closeModelLoader(); $("validationSetupDetails").open = true; },
        waitText: "Choose Parameter recovery.",
        waitFor: () => tutorialParameterRecoverySelected(),
      },
      {
        pane: "validationPane",
        target: "#validationSingleModelRow",
        title: "Select Validation Models",
        body: "Parameter recovery and internal consistency use one model. Model recovery uses multiple models and summarizes whether simulated datasets are classified back to the model that generated them.",
        ensure: () => { $("validationSetupDetails").open = true; },
      },
      {
        pane: "validationPane",
        target: "#validationSimsRow",
        fallback: "#validationSetupDetails",
        title: "Synthetic Participants",
        body: "For recovery analyses, synthetic participants are artificial datasets generated from known values. More synthetic participants give a more stable validation picture, but they take longer.",
        ensure: () => { $("validationSetupDetails").open = true; },
      },
      {
        pane: "validationPane",
        target: "#runValidation",
        title: "Run Validation",
        body: "Set synthetic participants to 200, then click Run Validation. This starts a parameter-recovery check so you can see how FlexDDM handles longer validation jobs.",
        ensure: () => {
          $("validationSetupDetails").open = true;
          $("validationMode").value = "parameter";
          $("validationSims").value = "200";
          updateValidationModeUI();
        },
        waitText: "Run validation and wait for the long-job popup.",
        waitFor: () => tutorialClusterPromptVisible(),
        autoAdvance: true,
      },
      {
        pane: "validationPane",
        target: "#clusterExportModal",
        fallback: "#clusterExportModal",
        title: "Cluster Package Option",
        body: "This popup appears when the ETA is over the 5-minute threshold. Export Cluster Package builds a zip containing the selected data, filters, model, options, and runnable Slurm files. For the tutorial, click Abort so we do not wait for the full validation.",
        waitText: "Click Abort in the cluster popup.",
        waitFor: () => $("clusterExportModal").classList.contains("hidden"),
        autoAdvance: true,
      },
      {
        pane: "fitPane",
        target: "#fitModelSelect",
        title: "Choose A Fit Model",
        body: "Choose DMC for the fit model. Fit uses this selected model independently from whatever you are editing in the model pane.",
        ensure: () => { $("fitSetupDetails").open = true; },
        waitText: "Choose DMC as the fit model.",
        waitFor: () => tutorialDmcFitReady(),
      },
      {
        pane: "fitPane",
        target: "#fitBackend",
        title: "Choose A Fitting Backend",
        body: "Fokker-Planck fits use numerical density solvers for compatible process models. Simulation fits are slower but more flexible because they compare observed data to forward simulated behavior.",
        ensure: () => { $("fitSetupDetails").open = true; },
      },
      {
        pane: "fitPane",
        target: "#participantIdRow",
        fallback: "#fitScopeGroup",
        title: "Choose Fit Scope",
        body: "Choose Selected IDs and enter participant 1. This fits one participant so the tutorial finishes quickly while still showing the full fitting workflow.",
        ensure: () => {
          $("fitSetupDetails").open = true;
          const selected = document.querySelector("input[name='fitScope'][value='selected']");
          if (selected) selected.checked = true;
          syncFitScopeControls();
        },
        waitText: "Choose Selected IDs and enter 1.",
        waitFor: () => fitScopeValue() === "selected" && String($("participantId").value).trim() === "1",
      },
      {
        pane: "fitPane",
        target: "#runFit",
        dynamicTarget: () => activeFitJob ? "#fitStatus" : "#runFit",
        title: "Fit The Model",
        body: "Run the fit. FlexDDM will optimize the selected model for participant 1, then produce parameter estimates, summaries, and posterior predictive checks.",
        ensure: () => { $("fitSetupDetails").open = true; },
        waitText: "Run the fit and wait for it to complete.",
        waitFor: () => fitHasResults,
        autoAdvance: true,
      },
      {
        pane: "fitPane",
        target: "#fitTableExportFrame",
        fallback: "#fitStatus",
        title: "Parameter CSV",
        body: "The first CSV output stores the fitted parameter values. These are the estimates you would report, compare across people, or use in downstream analyses.",
      },
      {
        pane: "fitPane",
        target: "#conditionTableExportFrame",
        fallback: "#fitStatus",
        title: "Condition Summary CSV",
        body: "The condition summary CSV compares observed and model-generated behavior by condition. It helps you see whether the fit captures condition-level patterns.",
      },
      {
        pane: "fitPane",
        target: "#ppcExportFrame",
        fallback: "#fitStatus",
        title: "Posterior Predictive Visualization",
        body: "The posterior predictive figure overlays observed and simulated response-time densities. Good overlap means the fitted model is reproducing the shape of the data, not just a single score.",
      },
      {
        pane: "optionsPane",
        target: "#seed",
        title: "Set General Options",
        body: "The seed makes simulations and searches reproducible. The other options are advanced controls for users who want to tune solver speed, precision, parallelization, and objective calculations.",
      },
      {
        pane: "aboutPane",
        target: "#aboutContact",
        title: "Questions And Citation",
        body: "Please reach out if you have questions, ideas, or run into trouble. And if FlexDDM helps your research, please cite our work so others can find and evaluate the tool too.",
      },
    ];
    function startTutorial() {
      closeTutorialIntro(false);
      tutorialRunning = true;
      tutorialStepIndex = 0;
      tutorialCsvBaseline = $("csvPath").value || "";
      tutorialMappingResetDone = false;
      tutorialFilterResetDone = false;
      tutorialModelBrowserOpenedForTemplate = false;
      tutorialModelBrowserOpenedForCustom = false;
      tutorialDmcTemplateChosen = false;
      tutorialCustomSourceChosen = false;
      tutorialCustomModelChosen = false;
      tutorialBoundaryModalOpened = false;
      tutorialBoundaryParamsTouched = false;
      tutorialBoundaryAppliedThisStep = false;
      tutorialSaveModalOpened = false;
      tutorialModelSavedThisStep = false;
      ["modelLoadModal", "functionModal", "saveModelModal"].forEach(id => {
        if ($(id)) $(id).classList.add("hidden");
      });
      $("tutorialTab").classList.add("tutorial-active");
      $("tutorialLayer").classList.remove("hidden");
      renderTutorialStep();
      clearInterval(tutorialCheckTimer);
      tutorialCheckTimer = setInterval(updateTutorialGate, 350);
    }
    function endTutorial(restorePrevious = false) {
      tutorialRunning = false;
      clearInterval(tutorialCheckTimer);
      tutorialCheckTimer = null;
      $("tutorialLayer").classList.add("hidden");
      $("tutorialIntroModal").classList.add("hidden");
      $("tutorialTab").classList.remove("tutorial-active");
      syncTutorialRestrictions();
      if (restorePrevious && tutorialPreviousPane) showPane(tutorialPreviousPane);
    }
    function tutorialTarget(step) {
      if (typeof step.dynamicTarget === "function") {
        const dynamic = step.dynamicTarget();
        if (dynamic) {
          const element = document.querySelector(dynamic);
          if (element) return element;
        }
      }
      const selectors = [step.target, step.fallback, `#${step.pane}`].filter(Boolean);
      for (const selector of selectors) {
        const element = document.querySelector(selector);
        if (!element) continue;
        const rect = element.getBoundingClientRect();
        const visible = rect.width > 0 && rect.height > 0 && getComputedStyle(element).visibility !== "hidden";
        if (visible) return element;
      }
      return document.querySelector(`#${step.pane}`) || document.body;
    }
    function renderTutorialStep() {
      if (!tutorialRunning) return;
      const step = tutorialSteps[tutorialStepIndex];
      const paneChanged = activePaneId() !== step.pane;
      showPane(step.pane);
      if (paneChanged) shakeTutorialPane();
      if (typeof step.ensure === "function") step.ensure();
      const target = tutorialTarget(step);
      target.scrollIntoView({block: "center", inline: "nearest", behavior: "smooth"});
      $("tutorialTitle").textContent = step.title;
      $("tutorialBody").textContent = step.body;
      const copyCode = $("tutorialCopyCode");
      copyCode.classList.toggle("hidden", !step.copyText);
      copyCode.innerHTML = step.copyText ? `<code>${escapeHtml(step.copyText)}</code><button class="secondary" type="button" id="tutorialCopyCodeButton">Copy</button>` : "";
      if (step.copyText && $("tutorialCopyCodeButton")) {
        $("tutorialCopyCodeButton").onclick = async () => {
          await navigator.clipboard.writeText(step.copyText);
        };
      }
      $("tutorialProgress").textContent = `Step ${tutorialStepIndex + 1} of ${tutorialSteps.length}`;
      $("tutorialBack").disabled = tutorialStepIndex === 0;
      $("tutorialNext").textContent = tutorialStepIndex === tutorialSteps.length - 1 ? "Finish" : "Okay";
      syncTutorialRestrictions();
      setTimeout(positionTutorial, 220);
      updateTutorialGate();
    }
    function shakeTutorialPane() {
      const main = document.querySelector("main");
      if (!main) return;
      main.classList.remove("tutorial-pane-shift");
      void main.offsetWidth;
      main.classList.add("tutorial-pane-shift");
      window.setTimeout(() => main.classList.remove("tutorial-pane-shift"), 540);
    }
    function safeTutorialWait(step) {
      if (!step.waitFor) return true;
      try { return Boolean(step.waitFor()); } catch (error) { return false; }
    }
    function updateTutorialGate() {
      if (!tutorialRunning) return;
      const step = tutorialSteps[tutorialStepIndex];
      const ready = safeTutorialWait(step);
      $("tutorialNext").disabled = !ready;
      $("tutorialWaiting").classList.toggle("hidden", ready || !step.waitText);
      $("tutorialWaiting").textContent = ready ? "" : step.waitText;
      syncTutorialRestrictions();
      positionTutorial();
      if (ready && step.autoAdvance && !step._autoAdvanced) {
        step._autoAdvanced = true;
        setTimeout(() => {
          if (tutorialRunning && tutorialSteps[tutorialStepIndex] === step) nextTutorialStep();
          step._autoAdvanced = false;
        }, 450);
      }
    }
    function maybeAdvanceTutorial(delay = 80) {
      if (!tutorialRunning) return;
      const step = tutorialSteps[tutorialStepIndex];
      if (!step || !safeTutorialWait(step)) {
        updateTutorialGate();
        return;
      }
      if (step.autoAdvance || step.advanceOnReady) {
        window.setTimeout(() => {
          if (tutorialRunning && tutorialSteps[tutorialStepIndex] === step && safeTutorialWait(step)) {
            nextTutorialStep();
          }
        }, delay);
      } else {
        updateTutorialGate();
      }
    }
    function positionTutorial() {
      if (!tutorialRunning) return;
      const step = tutorialSteps[tutorialStepIndex];
      const target = tutorialTarget(step);
      const rect = target.getBoundingClientRect();
      const pad = 8;
      const spotlight = $("tutorialSpotlight");
      spotlight.style.left = `${Math.max(8, rect.left - pad)}px`;
      spotlight.style.top = `${Math.max(8, rect.top - pad)}px`;
      spotlight.style.width = `${Math.min(window.innerWidth - 16, rect.width + pad * 2)}px`;
      spotlight.style.height = `${Math.min(window.innerHeight - 16, rect.height + pad * 2)}px`;
      const bubble = $("tutorialBubble");
      const bubbleRect = bubble.getBoundingClientRect();
      const gap = 14;
      const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
      const candidates = [
        {left: rect.right + gap, top: rect.top, preference: 0},
        {left: rect.left - bubbleRect.width - gap, top: rect.top, preference: 1},
        {left: rect.left, top: rect.bottom + gap, preference: 2},
        {left: rect.left, top: rect.top - bubbleRect.height - gap, preference: 3},
        {left: (window.innerWidth - bubbleRect.width) / 2, top: window.innerHeight - bubbleRect.height - 18, preference: 4}
      ].map(candidate => ({
        left: clamp(candidate.left, 16, Math.max(16, window.innerWidth - bubbleRect.width - 16)),
        top: clamp(candidate.top, 16, Math.max(16, window.innerHeight - bubbleRect.height - 16)),
        preference: candidate.preference
      }));
      const targetBox = {left: rect.left - pad, right: rect.right + pad, top: rect.top - pad, bottom: rect.bottom + pad};
      const scored = candidates.map(candidate => {
        const box = {left: candidate.left, right: candidate.left + bubbleRect.width, top: candidate.top, bottom: candidate.top + bubbleRect.height};
        const overlapX = Math.max(0, Math.min(box.right, targetBox.right) - Math.max(box.left, targetBox.left));
        const overlapY = Math.max(0, Math.min(box.bottom, targetBox.bottom) - Math.max(box.top, targetBox.top));
        return {...candidate, overlap: overlapX * overlapY};
      }).sort((a, b) => (a.overlap - b.overlap) || (a.preference - b.preference));
      bubble.style.left = `${scored[0].left}px`;
      bubble.style.top = `${scored[0].top}px`;
    }
    function nextTutorialStep() {
      if (!tutorialRunning) return;
      if (tutorialStepIndex >= tutorialSteps.length - 1) {
        endTutorial(false);
        return;
      }
      tutorialStepIndex += 1;
      renderTutorialStep();
    }
    function previousTutorialStep() {
      if (!tutorialRunning || tutorialStepIndex === 0) return;
      tutorialStepIndex -= 1;
      renderTutorialStep();
    }
    async function loadAboutPane() {
      const about = await getJson("/api/about");
      $("aboutName").textContent = about.name || "FlexDDM";
      $("aboutDescription").textContent = about.description || "";
      $("aboutVersion").textContent = about.version ? `Version ${about.version}` : "";
      $("aboutBuild").textContent = [about.buildDate, about.runtime].filter(Boolean).join(" | ");
      $("aboutContact").textContent = [about.contactName, about.contactInstitution, about.contactEmail].filter(Boolean).join(" | ");
      $("aboutCitation").textContent = about.citation || "";
      $("aboutCopyright").textContent = about.copyright || "";
    }
    function table(rows) {
      if (!rows || rows.length === 0) return "";
      const cols = Object.keys(rows[0]);
      const head = "<tr>" + cols.map(c => `<th>${escapeHtml(c)}</th>`).join("") + "</tr>";
      const body = rows.map(r => "<tr>" + cols.map(c => `<td>${escapeHtml(r[c] === null || r[c] === undefined ? "" : r[c])}</td>`).join("") + "</tr>").join("");
      return `<table>${head}${body}</table>`;
    }
    function editableTarget(element) {
      return element && (element.matches?.("textarea, input[type='text'], input:not([type]), input[type='number']") || element.isContentEditable);
    }
    function installPasteContextMenu() {
      document.addEventListener("click", () => closePastePopover());
      document.addEventListener("contextmenu", (event) => {
        if (!editableTarget(event.target)) return;
        event.preventDefault();
        closePastePopover();
        const popover = document.createElement("div");
        popover.className = "paste-popover";
        popover.style.left = `${Math.min(event.clientX, window.innerWidth - 94)}px`;
        popover.style.top = `${Math.min(event.clientY, window.innerHeight - 42)}px`;
        popover.innerHTML = `<button type="button">Paste</button>`;
        document.body.appendChild(popover);
        popover.querySelector("button").onclick = async (clickEvent) => {
          clickEvent.stopPropagation();
          try {
            const text = await readClipboardText();
            insertTextIntoEditable(event.target, text);
          } finally {
            closePastePopover();
          }
        };
      });
    }
    function closePastePopover() {
      document.querySelectorAll(".paste-popover").forEach(node => node.remove());
    }
    async function readClipboardText() {
      if (navigator.clipboard && window.isSecureContext) {
        try {
          const text = await navigator.clipboard.readText();
          if (text) return text;
        } catch (error) {}
      }
      const data = await post("/api/read-clipboard", {});
      return data.text || "";
    }
    function insertTextIntoEditable(target, text) {
      if (!text) return;
      if (target.isContentEditable) {
        document.execCommand("insertText", false, text);
        return;
      }
      const start = target.selectionStart ?? String(target.value || "").length;
      const end = target.selectionEnd ?? start;
      target.value = String(target.value || "").slice(0, start) + text + String(target.value || "").slice(end);
      target.selectionStart = target.selectionEnd = start + text.length;
      target.dispatchEvent(new Event("input", {bubbles: true}));
      target.dispatchEvent(new Event("change", {bubbles: true}));
      target.focus();
    }
    document.addEventListener("wheel", (event) => {
      const preview = event.target.closest && event.target.closest(".preview-table");
      if (!preview || preview.scrollWidth <= preview.clientWidth) return;
      if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
        preview.scrollLeft += event.deltaX;
        event.preventDefault();
        event.stopPropagation();
      } else if (event.shiftKey && Math.abs(event.deltaY) > 0) {
        preview.scrollLeft += event.deltaY;
        event.preventDefault();
        event.stopPropagation();
      }
    }, {passive: false});
    function dataSettingsKey(path = $("csvPath").value) {
      return path ? `flexddm:data-settings:${path}` : "";
    }
    function resetStaleDataSettings() {
      try {
        const versionKey = "flexddm:data-settings-version";
        if (localStorage.getItem(versionKey) === DATA_SETTINGS_VERSION) return;
        Object.keys(localStorage)
          .filter(key => key.startsWith("flexddm:data-settings:") || key.startsWith("flexddmDataSettings"))
          .forEach(key => localStorage.removeItem(key));
        localStorage.setItem(versionKey, DATA_SETTINGS_VERSION);
      } catch (error) {}
    }
    function saveDataSettings() {
      const key = dataSettingsKey();
      if (!key) return;
      const settings = {
        participant: $("participantCol").value || "",
        condition: firstAssignedConditionColumn(),
        conditionColumns: conditionColumnsPayload(),
        rt: $("rtCol").value || "",
        accuracy: $("accuracyCol").value || "",
        responseKind: $("responseKind").value || "",
        columnFilterRows: normalizeColumnFilterRows(columnFilterRows),
        dataConstantRows: normalizeDataConstantRows(dataConstantRows).filter(row => !row.locked)
      };
      try {
        localStorage.setItem(key, JSON.stringify(settings));
      } catch (error) {}
      post("/api/data-settings-save", {path: $("csvPath").value, version: DATA_SETTINGS_VERSION, settings}).catch(() => {});
    }
    async function loadDataSettings(path = $("csvPath").value) {
      const key = dataSettingsKey(path);
      if (!key) return null;
      try {
        const local = JSON.parse(localStorage.getItem(key) || "null");
        if (local) return local;
      } catch (error) {
      }
      try {
        const remote = await post("/api/data-settings-load", {path, version: DATA_SETTINGS_VERSION});
        return remote.settings || null;
      } catch (error) {
        return null;
      }
    }
    function setSelectOptions(id, columns, selected = "") {
      const select = $(id);
      if (!select) return;
      const values = Array.isArray(columns) ? columns : [];
      select.innerHTML = [`<option value="">Choose column</option>`]
        .concat(values.map(column => `<option value="${escapeHtml(column)}">${escapeHtml(column)}</option>`))
        .join("");
      select.value = values.includes(selected) ? selected : "";
    }
    function coreMappingComplete() {
      return Boolean($("csvPath").value && $("participantCol").value && $("rtCol").value && $("accuracyCol").value && $("responseKind").value);
    }
    function mappingComplete() {
      return coreMappingComplete();
    }
    function normalizeConditionColumnRows(rows) {
      const actual = (rows || []).map(row => ({column: row.column || ""}));
      const hasBlank = actual.some(row => !String(row.column || "").trim());
      if (!hasBlank) actual.push({column: ""});
      return actual;
    }
    function conditionColumnsPayload() {
      return normalizeConditionColumnRows(conditionColumnRows).map(row => row.column).filter(Boolean);
    }
    function firstAssignedConditionColumn() {
      return conditionColumnsPayload()[0] || "";
    }
    function currentConditionVariableName() {
      const first = firstAssignedConditionColumn();
      return first ? sanitizePythonName(first) : "condition";
    }
    function renderConditionColumns() {
      conditionColumnRows = normalizeConditionColumnRows(conditionColumnRows);
      const options = [`<option value="">No condition</option>`]
        .concat(csvColumns.map(column => `<option value="${escapeHtml(column)}">${escapeHtml(column)}</option>`))
        .join("");
      $("conditionColumns").innerHTML = conditionColumnRows.map((row, index) => `
        <div class="condition-column-row">
          <div><label>${index === 0 ? "Primary condition column" : "Additional condition column"}</label><select data-condition-index="${index}">${options}</select></div>
          <button class="secondary" type="button" data-remove-condition-column="${index}" ${String(row.column || "").trim() ? "" : "disabled"}>Remove</button>
        </div>
      `).join("");
      $("conditionColumns").querySelectorAll("select[data-condition-index]").forEach(select => {
        const index = Number(select.dataset.conditionIndex);
        select.value = conditionColumnRows[index].column || "";
        select.onchange = async () => {
          conditionColumnRows[index].column = select.value;
          conditionColumnRows = normalizeConditionColumnRows(conditionColumnRows);
          saveDataSettings();
          renderConditionColumns();
          if (mappingComplete()) {
            await previewData("");
            await previewFilterData();
          }
          maybeAdvanceTutorial();
        };
      });
      $("conditionColumns").querySelectorAll("button[data-remove-condition-column]").forEach(button => {
        button.onclick = async () => {
          conditionColumnRows.splice(Number(button.dataset.removeConditionColumn), 1);
          conditionColumnRows = normalizeConditionColumnRows(conditionColumnRows);
          saveDataSettings();
          renderConditionColumns();
          if (mappingComplete()) {
            await previewData("");
            await previewFilterData();
          }
        };
      });
    }
    function setDataLoadedState(loaded) {
      dataLoaded = Boolean(loaded);
      $("dataMappingDetails").classList.toggle("hidden", csvColumns.length === 0);
      $("dataConditionsDetails").classList.toggle("hidden", !coreMappingComplete());
      $("filterContent").classList.toggle("hidden", !dataLoaded);
      if (!dataLoaded) {
        $("participantViewRow").classList.add("hidden");
        $("filterOverviewImage").style.display = "none";
        $("filterDataTable").innerHTML = "";
        setExportReady("dataDiagnosticsExportFrame", false);
      }
    }
    let filterPreviewTimer = null;
    function scheduleFilterPreview(delay = 250) {
      clearTimeout(filterPreviewTimer);
      filterPreviewTimer = setTimeout(() => {
        previewFilterData().catch(error => {
          $("filterStatus").innerHTML = `<span class="warn">${escapeHtml(error.message)}</span>`;
        });
      }, delay);
    }
    async function loadCsvColumns(path = $("csvPath").value) {
      if (!path) return;
      const previousPath = $("csvPath").value;
      const data = await post("/api/data-columns", {path});
      if (path !== $("csvPath").value && previousPath !== path) return;
      csvColumns = data.columns || [];
      const suggestions = data.suggestions || {};
      const stored = await loadDataSettings(path) || {};
      setSelectOptions("participantCol", csvColumns, stored.participant || suggestions.participant || "");
      setSelectOptions("rtCol", csvColumns, stored.rt || suggestions.rt || "");
      setSelectOptions("accuracyCol", csvColumns, stored.accuracy || suggestions.accuracy || "");
      $("responseKind").value = stored.responseKind || "";
      const storedConditions = Array.isArray(stored.conditionColumns) ? stored.conditionColumns : (stored.condition ? [stored.condition] : []);
      const suggestedConditions = storedConditions.length ? storedConditions : (suggestions.condition ? [suggestions.condition] : []);
      conditionColumnRows = normalizeConditionColumnRows(suggestedConditions.map(column => ({column})));
      renderConditionColumns();
      columnFilterRows = normalizeColumnFilterRows(stored.columnFilterRows || []);
      dataConstantRows = normalizeDataConstantRows(stored.dataConstantRows || []);
      mappingAutoCollapsed = false;
      setDataLoadedState(false);
      if (mappingComplete()) {
        await previewData("");
        await previewFilterData();
      } else {
        $("dataStatus").textContent = `${csvColumns.length} columns found. Complete column mapping to preview the data.`;
      }
      maybeAdvanceTutorial();
    }
    function boolValue(id) { return $(id).value === "true"; }
    function payloadBase(includeFilters = true) {
      const payload = {
        path: $("csvPath").value,
        participant: $("participantCol").value,
        condition: firstAssignedConditionColumn(),
        conditionColumns: conditionColumnsPayload(),
        rt: $("rtCol").value,
        accuracy: $("accuracyCol").value,
        responseKind: $("responseKind").value
      };
      if (includeFilters) {
        payload.columnFilters = columnFiltersPayload();
        payload.customConstants = dataConstantsPayload();
      } else {
        payload.columnFilters = [];
        payload.customConstants = "";
      }
      return payload;
    }
    function optionPayload() {
      return {
        seed: Number($("seed").value),
        simulatedTrials: Number($("simulatedTrials").value),
        maxiter: Number($("maxiter").value),
        popsize: Number($("popsize").value),
        workers: Number($("workers").value),
        simulationDt: Number($("simulationDt").value),
        simulationNoiseSd: Number($("simulationNoiseSd").value),
        polish: boolValue("polish"),
        localRefine: boolValue("localRefine"),
        fpDt: Number($("dtFp").value),
        fpDx: Number($("dxFp").value),
        fpTmax: Number($("tmaxFp").value),
        fpObjective: $("fpObjective").value,
        hessianStepFraction: Number($("hessianStepFraction").value),
        hessianMinStep: Number($("hessianMinStep").value),
        hessianRidge: Number($("hessianRidge").value),
        profileGridPoints: Number($("profileGridPoints").value),
        profileBoundTrim: Number($("profileBoundTrim").value)
      };
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[ch]));
    }
    function setModelStatus(message = "", isError = false) {
      const status = $("modelStatus");
      if (!status) return;
      status.classList.toggle("hidden", !message);
      status.innerHTML = isError ? `<span class="warn">${escapeHtml(message)}</span>` : escapeHtml(message);
    }
    const saveIconSvg = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3h12l2 2v16H5z"></path><path d="M8 3v6h8V3"></path><path d="M8 21v-7h8v7"></path></svg>`;
    const trashIconSvg = `<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"></path><path d="M8 6V4h8v2"></path><path d="M19 6l-1 15H6L5 6"></path><path d="M10 11v6"></path><path d="M14 11v6"></path></svg>`;
    function installSaveIcons() {
      document.querySelectorAll(".save-icon").forEach(button => button.innerHTML = saveIconSvg);
      $("saveDataDiagnostics").onclick = () => saveImageUrl("filterOverviewImage", "flexddm_data_diagnostics.png", "dataStatus");
      $("saveModelSchematic").onclick = () => saveCanvas("modelSchematicCanvas", "flexddm_model_schematic.png", "modelSchematicStatus");
      $("saveFitTable").onclick = () => saveTable("fitTable", "flexddm_fit_parameters.csv", "fitStatus");
      $("saveConditionTable").onclick = () => saveTable("conditionTable", "flexddm_condition_summary.csv", "fitStatus");
      $("saveFitCovariateTable").onclick = () => saveTable("fitCovariateTable", "flexddm_participant_covariate_associations.csv", "fitStatus");
      $("savePpcImage").onclick = () => saveImageUrl("ppcImage", "flexddm_posterior_predictive.png", "fitStatus");
      $("saveValidationTable").onclick = () => saveTable("validationTable", "flexddm_validation_table.csv", "validationStatus");
      $("saveValidationImage").onclick = () => saveImageUrl("validationImage", "flexddm_validation_diagnostics.png", "validationStatus");
      $("saveSimulatedTable").onclick = () => saveTable("simulateDataTable", "flexddm_synthetic_data.csv", "simulateStatus");
      $("saveSimulatedDiagnostics").onclick = () => saveImageUrl("simulateOverviewImage", "flexddm_synthetic_data_diagnostics.png", "simulateStatus");
    }
    function setExportReady(frameId, ready) {
      $(frameId).classList.toggle("ready", Boolean(ready));
    }
    function tableCsv(tableWrapId) {
      const tableElement = $(tableWrapId).querySelector("table");
      if (!tableElement) throw new Error("There is no table to save yet.");
      const rows = Array.from(tableElement.querySelectorAll("tr")).map(row =>
        Array.from(row.querySelectorAll("th,td")).map(cell => {
          const text = cell.textContent || "";
          return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
        }).join(",")
      );
      return rows.join("\n") + "\n";
    }
    async function saveTable(tableWrapId, filename, statusId) {
      try {
        const data = await post("/api/save-artifact", {kind: "csv", filename, content: tableCsv(tableWrapId)});
        if (data.saved) $(statusId).textContent = `Saved ${data.path}`;
      } catch (error) {
        $(statusId).innerHTML = `<span class="warn">${error.message}</span>`;
      }
    }
    async function saveImageUrl(imageId, filename, statusId) {
      try {
        const image = $(imageId);
        if (!image.src || image.style.display === "none") throw new Error("There is no image to save yet.");
        const data = await post("/api/save-artifact", {kind: "image-url", filename, sourceUrl: image.getAttribute("src") || image.src});
        if (data.saved) $(statusId).textContent = `Saved ${data.path}`;
      } catch (error) {
        $(statusId).innerHTML = `<span class="warn">${error.message}</span>`;
      }
    }
    async function saveCanvas(canvasId, filename, statusId) {
      try {
        const canvas = $(canvasId);
        const data = await post("/api/save-artifact", {kind: "canvas-png", filename, dataUrl: canvas.toDataURL("image/png")});
        if (data.saved) {
          $(statusId).classList.remove("hidden");
          $(statusId).textContent = `Saved ${data.path}`;
        }
      } catch (error) {
        $(statusId).classList.remove("hidden");
        $(statusId).innerHTML = `<span class="warn">${error.message}</span>`;
      }
    }
    function progressLine(job) {
      const pct = `${Math.round((job.progress === undefined ? 0 : job.progress) * 100)}%`;
      const eta = job.etaSeconds === undefined ? "" : ` | ETA ${formatSeconds(job.etaSeconds)}`;
      const elapsed = job.elapsedSeconds === undefined ? "" : `Elapsed ${formatSeconds(job.elapsedSeconds)}`;
      if (Array.isArray(job.statusRows) && job.statusRows.length) {
        const rows = job.statusRows.map((row, index) => {
          const rowPct = row.progress === undefined || row.progress === null ? "" : ` | ${Math.round(Number(row.progress) * 100)}%`;
          const status = row.status ? ` | ${row.status}` : "";
          const elapsedCell = index === 0 ? elapsed : "";
          return `<div class="status-line"><span>${escapeHtml(`${row.label || "Worker"}${rowPct}${status}`)}</span><span class="elapsed">${escapeHtml(elapsedCell)}</span></div>`;
        });
        rows.push(`<div class="status-line"><span>${escapeHtml(`Overall | ${pct}${eta}`)}</span><span class="elapsed"></span></div>`);
        return rows.join("");
      }
      if (job.workerProgress && Object.keys(job.workerProgress).length) {
        const rows = Object.values(job.workerProgress)
          .sort((a, b) => Number(a.slot || 0) - Number(b.slot || 0))
          .map((worker, index) => {
            const workerPct = worker.participantProgress === undefined || worker.participantProgress === null
              ? ""
              : ` | ${Math.round(Number(worker.participantProgress) * 100)}%`;
            const participant = worker.participantIndex
              ? `Participant ${worker.participantIndex}${worker.participantId ? ` (${worker.participantId})` : ""}`
              : (worker.message || "Waiting");
            const status = worker.status === "finished" ? "done" : "running";
            const elapsedCell = index === 0 ? elapsed : "";
            return `<div class="status-line"><span>${escapeHtml(`Worker ${worker.slot} | ${participant}${workerPct} | ${status}`)}</span><span class="elapsed">${escapeHtml(elapsedCell)}</span></div>`;
          });
        rows.push(`<div class="status-line"><span>${escapeHtml(`Overall | ${pct}${eta}`)}</span><span class="elapsed"></span></div>`);
        return rows.join("");
      }
      const message = job.message || "";
      const participant = message.match(/participant\s+(\d+)\s+of\s+(\d+)/i);
      const label = participant ? `Participant ${participant[1]} of ${participant[2]}` : (job.kind === "validation" ? "Validation" : "Fitting");
      return `<div class="status-line"><span>${escapeHtml(`${label} | ${pct}${eta}`)}</span><span class="elapsed">${escapeHtml(elapsed)}</span></div>`;
    }
    function formatSeconds(seconds) {
      seconds = Math.max(0, Number(seconds) || 0);
      const mins = Math.floor(seconds / 60);
      const secs = seconds % 60;
      return mins ? `${mins}m ${secs}s` : `${secs}s`;
    }
    function setFitButtonRunning(isRunning) {
      $("runFit").textContent = isRunning ? "Cancel Fit" : "Fit Model";
      $("runFit").classList.toggle("cancel", isRunning);
      $("fitTab").classList.toggle("running", isRunning);
    }
    function setValidationButtonRunning(isRunning) {
      $("runValidation").textContent = isRunning ? "Cancel Validation" : "Run Validation";
      $("runValidation").classList.toggle("cancel", isRunning);
      $("validateTab").classList.toggle("running", isRunning);
    }
    function fitScopeValue() {
      return (document.querySelector("input[name='fitScope']:checked") || {}).value || "all";
    }
    function syncFitScopeControls() {
      const scope = fitScopeValue();
      $("participantIdRow").classList.toggle("hidden", scope !== "selected");
      if (scope !== "selected") $("participantId").value = "";
    }
    function openClusterExportModal(kind, payload, etaSeconds, statusElement, jobId) {
      pendingClusterExport = {kind, payload, statusElement, jobId};
      $("clusterExportCpus").value = String((payload || {}).workers || $("workers").value || 4);
      if ($("clusterExportPythonModule").value.trim() === "python/3.10") $("clusterExportPythonModule").value = "";
      $("clusterExportModalTitle").textContent = "Long Job Detected";
      $("clusterExportMessage").textContent = `This ${kind} job is estimated to take about ${formatSeconds(etaSeconds)}. You can continue locally, export a cluster-ready package, or abort the local run.`;
      $("clusterExportStatus").textContent = "";
      $("clusterExportStatus").classList.add("hidden");
      $("clusterExportMessage").classList.remove("hidden");
      $("clusterExportSettings").classList.remove("hidden");
      $("clusterExportGuide").classList.add("hidden");
      $("clusterExportGuide").innerHTML = "";
      $("clusterExportAbort").classList.remove("hidden");
      $("clusterExportDismiss").classList.remove("hidden");
      $("clusterExportDismiss").textContent = "Continue Anyway";
      $("clusterExportApply").classList.remove("hidden");
      $("clusterExportApply").disabled = false;
      $("clusterExportApply").textContent = "Export Cluster Package";
      delete $("clusterExportApply").dataset.downloadUrl;
      $("clusterExportModal").classList.remove("hidden");
      maybeAdvanceTutorial();
    }
    function closeClusterExportModal() {
      $("clusterExportModal").classList.add("hidden");
      maybeAdvanceTutorial();
    }
    function clusterSettingsPayload() {
      return {
        time: $("clusterExportTime").value,
        cpus: $("clusterExportCpus").value,
        mem: $("clusterExportMem").value,
        partition: $("clusterExportPartition").value,
        account: $("clusterExportAccount").value,
        pythonModule: $("clusterExportPythonModule").value,
      };
    }
    async function downloadClusterPackage(url) {
      const exportData = pendingClusterExport?.exportData || {};
      const filename = exportData.archiveName || "flexddm_cluster_package.zip";
      return await post("/api/save-artifact", {kind: "file-url", filename, sourceUrl: url});
    }
    async function copyTextToClipboard(text) {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return;
      }
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      textarea.setAttribute("readonly", "");
      document.body.appendChild(textarea);
      textarea.select();
      const ok = document.execCommand("copy");
      textarea.remove();
      if (!ok) throw new Error("Clipboard copy was blocked.");
    }
    function showClusterInstructions() {
      if (!pendingClusterExport) return;
      const settings = clusterSettingsPayload();
      const kind = pendingClusterExport.kind || "analysis";
      const isValidation = kind === "validation";
      const exportData = pendingClusterExport.exportData || {};
      const archiveName = exportData.archiveName || `flexddm_${kind}_cluster_package.zip`;
      const folderName = exportData.folderName || `flexddm_${kind}_cluster_package`;
      const clusterCommand = `unzip ${archiveName}\ncd ${folderName}\nsbatch submit.slurm\nsqueue -u $USER`;
      $("clusterExportModalTitle").textContent = "Run The Cluster Package";
      $("clusterExportMessage").classList.add("hidden");
      $("clusterExportSettings").classList.add("hidden");
      $("clusterExportStatus").classList.add("hidden");
      $("clusterExportAbort").classList.add("hidden");
      $("clusterExportApply").classList.add("hidden");
      $("clusterExportDismiss").classList.remove("hidden");
      $("clusterExportDismiss").textContent = "Done";
      const runLabel = isValidation ? "validation" : "fit";
      $("clusterExportGuide").innerHTML = `
        <div class="cluster-flow">
          <div class="cluster-step-card"><span class="cluster-step-num">1</span><strong>Move the zip</strong><span>Upload the downloaded package to the cluster with your usual transfer tool.</span></div>
          <div class="cluster-step-card"><span class="cluster-step-num">2</span><strong>Unzip it</strong><span>Open a cluster terminal, unpack the folder, and move inside it.</span></div>
          <div class="cluster-step-card"><span class="cluster-step-num">3</span><strong>Submit Slurm</strong><span>The included script already requests ${escapeHtml(settings.cpus || "4")} cores, ${escapeHtml(settings.mem || "16G")} memory, and ${escapeHtml(settings.time || "12:00:00")}.</span></div>
          <div class="cluster-step-card"><span class="cluster-step-num">4</span><strong>Collect outputs</strong><span>When the job finishes, download the outputs folder with tables, figures, logs, and the run manifest.</span></div>
        </div>
        <div class="cluster-command-wrap">
          <button class="cluster-copy-button" id="copyClusterCommand" type="button" title="Copy cluster commands">Copy</button>
          <div class="cluster-command" id="clusterPrimaryCommand"></div>
        </div>
        <div class="cluster-output-grid">
          <div class="cluster-output-card"><strong>Before submitting</strong>Open <code>submit.slurm</code> if your cluster uses a different partition, account, Python module, or environment setup. If Python module is blank, the script tries current Python first, then searches <code>module avail</code> for Python 3.9+.</div>
          <div class="cluster-output-card"><strong>What runs</strong><code>run_flexddm_cluster.py</code> rebuilds the selected data, model, filters, options, and ${escapeHtml(runLabel)} request exactly from this app state.</div>
          <div class="cluster-output-card"><strong>What you get</strong>The job writes CSV tables, PNG figures, a JSON manifest, and stdout/stderr logs into <code>outputs/</code>.</div>
          <div class="cluster-output-card"><strong>How to bring it back</strong>Download <code>outputs/</code> from the cluster. The files match the same result types shown in the FlexDDM pane.</div>
        </div>
        <div class="cluster-command"># If submit.slurm was edited on Windows and Slurm reports DOS line breaks:
dos2unix submit.slurm</div>
        <div class="notice">The zip also includes <code>README_CLUSTER.md</code> with the same commands and a checklist you can keep next to the job files.</div>
      `;
      $("clusterPrimaryCommand").textContent = clusterCommand;
      $("copyClusterCommand").onclick = async () => {
        try {
          await copyTextToClipboard(clusterCommand);
          $("copyClusterCommand").textContent = "Copied";
          setTimeout(() => {
            if ($("copyClusterCommand")) $("copyClusterCommand").textContent = "Copy";
          }, 1400);
        } catch (error) {
          $("copyClusterCommand").textContent = "Blocked";
          setTimeout(() => {
            if ($("copyClusterCommand")) $("copyClusterCommand").textContent = "Copy";
          }, 1400);
        }
      };
      $("clusterExportGuide").classList.remove("hidden");
    }
    async function exportPendingClusterJob() {
      if (!pendingClusterExport) return;
      const downloadUrl = $("clusterExportApply").dataset.downloadUrl;
      if (downloadUrl) {
        try {
          const saved = await downloadClusterPackage(downloadUrl);
          if (saved.saved) showClusterInstructions();
        } catch (error) {
          $("clusterExportStatus").classList.remove("hidden");
          $("clusterExportStatus").innerHTML = `<span class="warn">${escapeHtml(error.message)}</span>`;
        }
        return;
      }
      const exportPayload = pendingClusterExport.payload || {};
      const needsData = pendingClusterExport.kind === "fit" || (pendingClusterExport.kind === "validation" && exportPayload.mode === "internal");
      if (needsData && !String(exportPayload.path || "").trim()) {
        $("clusterExportStatus").classList.remove("hidden");
        $("clusterExportStatus").innerHTML = `<span class="warn">Cluster export needs mapped data. Load a CSV and complete the data mapping before exporting.</span>`;
        return;
      }
      $("clusterExportStatus").textContent = "";
      $("clusterExportStatus").classList.add("hidden");
      $("clusterExportApply").disabled = true;
      $("clusterExportApply").textContent = "Building...";
      pendingClusterExport.exporting = true;
      try {
        if (pendingClusterExport.jobId) {
          await cancelJob(pendingClusterExport.jobId);
        }
        const data = await post("/api/export-cluster-job", {
          kind: pendingClusterExport.kind,
          payload: pendingClusterExport.payload,
          clusterSettings: clusterSettingsPayload(),
        });
        pendingClusterExport.exportData = data;
        $("clusterExportApply").dataset.downloadUrl = data.url;
        $("clusterExportApply").disabled = false;
        $("clusterExportApply").textContent = "Download Cluster Package";
      } catch (error) {
        pendingClusterExport.exporting = false;
        $("clusterExportApply").disabled = false;
        $("clusterExportApply").textContent = "Export Cluster Package";
        $("clusterExportStatus").classList.remove("hidden");
        $("clusterExportStatus").innerHTML = `<span class="warn">${escapeHtml(error.message)}</span>`;
      }
    }
    async function abortPendingClusterJob() {
      if (!pendingClusterExport || !pendingClusterExport.jobId) {
        closeClusterExportModal();
        return;
      }
      $("clusterExportStatus").textContent = "Aborting local job...";
      try {
        await cancelJob(pendingClusterExport.jobId);
        if (pendingClusterExport.statusElement) {
          pendingClusterExport.statusElement.textContent = `Cancelling ${pendingClusterExport.kind}...`;
        }
        closeClusterExportModal();
      } catch (error) {
        $("clusterExportStatus").innerHTML = `<span class="warn">${escapeHtml(error.message)}</span>`;
      }
    }
    async function runJob(path, payload, statusElement, onComplete, onStarted, clusterKind = "") {
      const started = await post(path, payload);
      if (onStarted) onStarted(started.jobId);
      let clusterPrompted = false;
      return new Promise((resolve, reject) => {
        const timer = setInterval(async () => {
          try {
            const job = await getJson(`/api/job-status?job=${encodeURIComponent(started.jobId)}`);
            statusElement.innerHTML = progressLine(job);
            if (
              clusterKind &&
              !clusterPrompted &&
              job.status === "running" &&
              (Number(job.etaSeconds || 0) >= CLUSTER_EXPORT_ETA_SECONDS || (tutorialRunning && clusterKind === "validation" && Number(job.progress || 0) > 0.02))
            ) {
              clusterPrompted = true;
              openClusterExportModal(clusterKind, payload, Number(job.etaSeconds || 0), statusElement, started.jobId);
            }
            if (job.status === "complete") {
              clearInterval(timer);
              if (job.result && job.elapsedSeconds !== undefined) job.result.elapsedSeconds = job.elapsedSeconds;
              onComplete(job.result);
              resolve(job.result);
            } else if (job.status === "cancelled") {
              clearInterval(timer);
              if (pendingClusterExport && pendingClusterExport.jobId === started.jobId && pendingClusterExport.exporting) {
                statusElement.innerHTML = "";
                const exported = new Error("Cluster export selected.");
                exported.clusterPrompt = true;
                reject(exported);
                return;
              }
              const cancelledMessage = job.message || `${clusterKind === "validation" ? "Validation" : "Fit"} cancelled.`;
              statusElement.textContent = cancelledMessage;
              reject(new Error(cancelledMessage));
            } else if (job.status === "error") {
              clearInterval(timer);
              statusElement.innerHTML = `<span class="warn">${job.error || job.message}</span>`;
              reject(new Error(job.error || job.message));
            }
          } catch (error) {
            clearInterval(timer);
            statusElement.innerHTML = `<span class="warn">${error.message}</span>`;
            reject(error);
          }
        }, 1000);
      });
    }
    async function cancelJob(jobId) {
      if (!jobId) return;
      await post("/api/cancel-job", {jobId});
    }
    async function waitForJobToStop(jobId, statusElement) {
      if (!jobId) return;
      for (let attempt = 0; attempt < 120; attempt++) {
        const job = await getJson(`/api/job-status?job=${encodeURIComponent(jobId)}`);
        statusElement.textContent = job.status === "running" ? "Cancelling job..." : (job.message || job.status);
        if (["cancelled", "complete", "error", "missing"].includes(job.status)) return;
        await new Promise(resolve => setTimeout(resolve, 500));
      }
      throw new Error("Previous fit did not stop in time.");
    }
    function renderFitResult(data) {
      fitHasResults = true;
      $("fitResultsArea").classList.remove("hidden");
      $("fitSetupDetails").open = false;
      const participants = data.participantsFit ? ` | participants ${data.participantsFit}` : "";
      const scoreName = data.scoreName || "G2";
      const optimized = data.objectiveName && data.objectiveName !== scoreName
        ? ` | optimized ${data.objectiveName} ${data.objectiveScore.toFixed(3)} | ${data.objectiveName} BIC ${data.objectiveBic.toFixed(3)}`
        : "";
      const elapsed = data.elapsedSeconds === undefined ? "" : `Elapsed ${formatSeconds(data.elapsedSeconds)}`;
      const summary = `${data.model}${participants} | rows ${data.rowsFit} | ${scoreName} ${data.g2.toFixed(3)} | ${scoreName} BIC ${data.bic.toFixed(3)}${optimized}`;
      $("fitStatus").innerHTML = `<div class="status-line"><span>${escapeHtml(summary)}</span><span class="elapsed">${escapeHtml(elapsed)}</span></div>`;
      $("fitTable").innerHTML = table(data.parameterRows || [data.parameters]);
      $("conditionTable").innerHTML = table(data.conditionSummary);
      setExportReady("fitTableExportFrame", Boolean((data.parameterRows || [data.parameters]).length));
      setExportReady("conditionTableExportFrame", Boolean((data.conditionSummary || []).length));
      const covariateRows = data.fitCovariateSummary || [];
      $("fitCovariatePanel").classList.toggle("hidden", covariateRows.length === 0);
      $("fitCovariateTable").innerHTML = covariateRows.length ? table(covariateRows) : "";
      setExportReady("fitCovariateTableExportFrame", covariateRows.length > 0);
      const ppcOptions = data.ppcOptions && data.ppcOptions.length ? data.ppcOptions : (data.ppcUrl ? [{label: "All participants", url: data.ppcUrl}] : []);
      $("ppcSelect").innerHTML = ppcOptions.map((option) => `<option value="${option.url}">${option.label}</option>`).join("");
      $("ppcSelectRow").classList.toggle("hidden", ppcOptions.length <= 1);
      if (ppcOptions.length) {
        $("ppcImage").src = `${ppcOptions[0].url}?t=${Date.now()}`;
        $("ppcImage").style.display = "block";
        setExportReady("ppcExportFrame", true);
      } else {
        $("ppcImage").style.display = "none";
        setExportReady("ppcExportFrame", false);
      }
    }
    function renderParticipantViewSelect(data) {
      const ids = (data.allParticipantIds || data.participantIds || []).map(String);
      const current = $("participantViewSelect").value || "all";
      if (ids.length <= 1) {
        $("participantViewRow").classList.add("hidden");
        $("participantViewSelect").innerHTML = "";
        return;
      }
      $("participantViewRow").classList.remove("hidden");
      const selected = current !== "all" && ids.includes(current) ? current : "all";
      $("participantViewSelect").innerHTML = [`<option value="all">All participants</option>`]
        .concat(ids.map(id => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`))
        .join("");
      $("participantViewSelect").value = selected;
      $("participantViewSelect").onchange = handleParticipantViewChange;
    }
    function handleParticipantViewChange() {
      previewFilterData().catch(error => {
        $("filterStatus").innerHTML = `<span class="warn">${escapeHtml(error.message)}</span>`;
      });
    }
    async function refreshModels() {
      modelInventory = await getJson("/api/models");
      const locked = modelInventory.lockedTemplates || ["StandardDDM", "DMC", "DSTP", "SSP"];
      const custom = Array.from(new Set([...(modelInventory.seededCustom || []), ...(modelInventory.generated || [])]));
      const groupedOptions = `<optgroup label="Built-in models">${locked.map(name => `<option value="${name}">${name}</option>`).join("")}</optgroup><optgroup label="Custom models">${custom.map(name => `<option value="${name}">${name}</option>`).join("")}</optgroup>`;
      $("fitModelSelect").innerHTML = groupedOptions;
      $("simulateModelSelect").innerHTML = groupedOptions;
      $("validationModelSelect").innerHTML = groupedOptions;
      renderValidationModelChecklist(locked, custom);
      updateValidationModeUI();
      await refreshSimulateParameters();
      if ((modelInventory.invalidGenerated || []).length) {
        setModelStatus(`Saved files needing simulator fixes: ${modelInventory.invalidGenerated.join(", ")}`, true);
      }
      renderModelLoadCards();
    }
    function numericBound(value, fallback) {
      const number = Number(value);
      return Number.isFinite(number) ? number : fallback;
    }
    async function refreshSimulateParameters() {
      const select = $("simulateModelSelect");
      if (!select || !select.value) return;
      try {
        const data = await getJson(`/api/model-parameters?name=${encodeURIComponent(select.value)}&solver=simulation_de`);
        simulateParameterRows = (data.parameterBounds || []).map(row => {
          const lower = numericBound(row.lower, 0);
          const upper = numericBound(row.upper, lower + 1);
          const width = Math.max(upper - lower, 1e-6);
          return {
            name: row.name,
            lower,
            upper,
            mean: Math.min(upper, Math.max(lower, lower + width / 2)),
            sd: Math.max(width / 6, 0.001),
            skew: 0
          };
        });
        simulateCovariateRows = simulateCovariateRows.map(row => normalizeSimCovariateRow(row));
        renderSimulateParameterControls();
        renderSimulateCovariates();
      } catch (error) {
        $("simulateParameterControls").innerHTML = `<div class="notice"><span class="warn">${escapeHtml(error.message)}</span></div>`;
      }
    }
    function renderSimulateParameterControls() {
      const wrap = $("simulateParameterControls");
      if (!simulateParameterRows.length) {
        wrap.innerHTML = `<div class="notice">No free parameters were detected for this model.</div>`;
        return;
      }
      wrap.innerHTML = simulateParameterRows.map((row, index) => {
        const step = Math.max((row.upper - row.lower) / 500, 0.001);
        const sdMax = Math.max(row.upper - row.lower, step);
        return `<div class="simulate-param-card" data-sim-param-index="${index}" data-sim-param-name="${escapeHtml(row.name)}">
          <strong>${escapeHtml(row.name)}</strong>
          <canvas data-sim-param-canvas="${index}" width="320" height="90"></canvas>
          <div class="simulate-param-controls">
            <div class="simulate-param-control">
              <label>Mean</label>
              <input type="range" data-sim-param-field="mean" data-sim-param-kind="slider" data-sim-param-index="${index}" min="${row.lower}" max="${row.upper}" step="${step}" value="${row.mean}">
              <input type="number" data-sim-param-field="mean" data-sim-param-kind="number" data-sim-param-index="${index}" min="${row.lower}" max="${row.upper}" step="${step}" value="${formatNumber(row.mean)}">
            </div>
            <div class="simulate-param-control">
              <label>SD</label>
              <input type="range" data-sim-param-field="sd" data-sim-param-kind="slider" data-sim-param-index="${index}" min="0" max="${sdMax}" step="${step}" value="${row.sd}">
              <input type="number" data-sim-param-field="sd" data-sim-param-kind="number" data-sim-param-index="${index}" min="0" max="${sdMax}" step="${step}" value="${formatNumber(row.sd)}">
            </div>
            <div class="simulate-param-control">
              <label>Skew</label>
              <input type="range" data-sim-param-field="skew" data-sim-param-kind="slider" data-sim-param-index="${index}" min="-3" max="3" step="0.05" value="${row.skew}">
              <input type="number" data-sim-param-field="skew" data-sim-param-kind="number" data-sim-param-index="${index}" min="-3" max="3" step="0.05" value="${formatNumber(row.skew)}">
            </div>
          </div>
        </div>`;
      }).join("");
      wrap.querySelectorAll("input[data-sim-param-field]").forEach(input => {
        const index = Number(input.dataset.simParamIndex);
        const field = input.dataset.simParamField;
        const commit = (updateCurrent = true) => {
          const row = simulateParameterRows[index];
          if (!row) return false;
          const raw = String(input.value || "").trim();
          if (!raw || raw === "-" || raw === "." || raw === "-.") {
            if (updateCurrent) input.value = formatNumber(row[field]);
            return false;
          }
          let value = Number(raw);
          if (!Number.isFinite(value)) {
            if (updateCurrent) input.value = formatNumber(row[field]);
            return false;
          }
          if (field === "mean") value = Math.min(row.upper, Math.max(row.lower, value));
          if (field === "sd") value = Math.max(0, value);
          if (field === "skew") value = Math.min(3, Math.max(-3, value));
          row[field] = value;
          syncSimParamInputs(index, field, updateCurrent ? null : input);
          drawSimParamKdes();
          maybeAdvanceTutorial();
          return true;
        };
        if (input.dataset.simParamKind === "slider") {
          input.oninput = () => commit(true);
        } else {
          input.oninput = () => commit(false);
          input.onblur = () => commit(true);
          input.onkeydown = (event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commit(true);
              input.blur();
            }
          };
        }
      });
      drawSimParamKdes();
    }
    function syncSimParamInputs(index, field, skipInput = null) {
      const row = simulateParameterRows[index];
      if (!row) return;
      $("simulateParameterControls").querySelectorAll(`input[data-sim-param-index="${index}"][data-sim-param-field="${field}"]`).forEach(input => {
        if (skipInput && input === skipInput) return;
        const value = row[field];
        input.value = input.type === "number" ? formatNumber(value) : String(value);
      });
    }
    function drawSimParamKdes() {
      $("simulateParameterControls").querySelectorAll("canvas[data-sim-param-canvas]").forEach(canvas => {
        const row = simulateParameterRows[Number(canvas.dataset.simParamCanvas)];
        if (row) drawSimParamKde(canvas, row);
      });
    }
    function drawSimParamKde(canvas, row) {
      const ctx = canvas.getContext("2d");
      const width = canvas.width, height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#fbfcfd";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#dde4ec";
      ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
      const lower = row.lower, upper = row.upper;
      const sd = Math.max(Number(row.sd) || 0.001, 0.001);
      const alpha = (Number(row.skew) || 0) * 2.0;
      const delta = alpha / Math.sqrt(1 + alpha * alpha);
      const varianceFactor = Math.max(1 - (2 * delta * delta / Math.PI), 0.08);
      const scale = sd / Math.sqrt(varianceFactor);
      const location = Number(row.mean) - scale * delta * Math.sqrt(2 / Math.PI);
      const values = [];
      let maxY = 1e-9;
      for (let i = 0; i < 100; i += 1) {
        const x = lower + (upper - lower) * i / 99;
        const z = (x - location) / Math.max(scale, 1e-9);
        const phi = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
        const cdfApprox = 1 / (1 + Math.exp(-1.702 * alpha * z));
        const y = (2 / Math.max(scale, 1e-9)) * phi * cdfApprox;
        values.push(y);
        maxY = Math.max(maxY, y);
      }
      ctx.beginPath();
      values.forEach((y, i) => {
        const xPx = 8 + (width - 16) * i / 99;
        const yPx = height - 9 - (height - 22) * y / maxY;
        if (i === 0) ctx.moveTo(xPx, yPx);
        else ctx.lineTo(xPx, yPx);
      });
      ctx.strokeStyle = "#2a9d8f";
      ctx.lineWidth = 2.2;
      ctx.stroke();
      const meanX = 8 + (width - 16) * (row.mean - lower) / Math.max(upper - lower, 1e-9);
      ctx.strokeStyle = "#e76f51";
      ctx.beginPath(); ctx.moveTo(meanX, 10); ctx.lineTo(meanX, height - 8); ctx.stroke();
    }
    function addSimCovariateRow() {
      const next = simulateCovariateRows.length + 1;
      simulateCovariateRows.push({
        name: next === 1 ? "age" : `covariate_${next}`,
        level: next === 1 ? "participant" : "trial",
        mean: next === 1 ? 40 : 0,
        sd: next === 1 ? 12 : 1,
        skew: 0,
        target: simulateParameterRows[0]?.name || "",
        correlation: next === 1 ? 0.6 : 0
      });
      renderSimulateCovariates();
      maybeAdvanceTutorial();
    }
    function normalizeSimCovariateRow(row) {
      const normalized = {...row};
      const targets = {};
      if (normalized.targets && typeof normalized.targets === "object") {
        Object.entries(normalized.targets).forEach(([name, value]) => {
          if (simulateParameterRows.some(param => param.name === name)) targets[name] = Number(value) || 0;
        });
      }
      if (!Object.keys(targets).length && normalized.target) {
        targets[normalized.target] = Number(normalized.correlation) || 0;
      }
      normalized.targets = targets;
      normalized.target = Object.keys(targets)[0] || simulateParameterRows[0]?.name || "";
      normalized.correlation = normalized.target ? Number(targets[normalized.target] || 0) : 0;
      return normalized;
    }
    function renderSimulateCovariates() {
      const wrap = $("simulateCovariates");
      wrap.innerHTML = simulateCovariateRows.length
        ? simulateCovariateRows.map((rawRow, index) => {
          const row = normalizeSimCovariateRow(rawRow);
          simulateCovariateRows[index] = row;
          const targetRows = simulateParameterRows.map(param => {
            const value = Number(row.targets?.[param.name] || 0);
            return `<div class="simulate-correlation-row">
              <label>${escapeHtml(param.name)}</label>
              <input type="range" min="-0.99" max="0.99" step="0.01" value="${value}" data-sim-cov-target="${escapeHtml(param.name)}" data-sim-cov-index="${index}">
              <input type="number" min="-0.99" max="0.99" step="0.01" value="${formatNumber(value)}" data-sim-cov-target="${escapeHtml(param.name)}" data-sim-cov-index="${index}">
            </div>`;
          }).join("");
          return `<div class="simulate-covariate-card" data-sim-cov-index="${index}">
            <strong>Covariate / trait ${index + 1}</strong>
            <div class="simulate-covariate-controls">
              <div><label>Name</label><input data-sim-cov-field="name" data-sim-cov-index="${index}" value="${escapeHtml(row.name)}"></div>
              <div><label>Level</label><select data-sim-cov-field="level" data-sim-cov-index="${index}"><option value="participant" ${row.level === "participant" ? "selected" : ""}>Participant trait</option><option value="trial" ${row.level === "trial" ? "selected" : ""}>Trial covariate</option></select></div>
              <div><label>Mean</label><input type="number" data-sim-cov-field="mean" data-sim-cov-index="${index}" value="${escapeHtml(row.mean)}"></div>
              <div><label>SD</label><input type="number" data-sim-cov-field="sd" data-sim-cov-index="${index}" value="${escapeHtml(row.sd)}"></div>
              <div><label>Skew</label><input type="number" data-sim-cov-field="skew" data-sim-cov-index="${index}" value="${escapeHtml(row.skew)}"></div>
              <div><button class="secondary" type="button" data-remove-sim-cov="${index}">Remove</button></div>
            </div>
            <div class="simulate-correlation-grid">
              <label>Parameter correlations</label>
              ${targetRows || `<div class="notice">Load a model with free parameters to set correlations.</div>`}
            </div>
          </div>`;
        }).join("")
        : `<div class="notice">No synthetic covariates or traits yet.</div>`;
      wrap.querySelectorAll("[data-sim-cov-field]").forEach(input => {
        input.oninput = input.onchange = () => {
          const row = simulateCovariateRows[Number(input.dataset.simCovIndex)];
          if (!row) return;
          const field = input.dataset.simCovField;
          row[field] = ["mean", "sd", "skew", "correlation"].includes(field) ? Number(input.value) : input.value;
          maybeAdvanceTutorial();
        };
      });
      wrap.querySelectorAll("[data-sim-cov-target]").forEach(input => {
        input.oninput = input.onchange = () => {
          const row = simulateCovariateRows[Number(input.dataset.simCovIndex)];
          if (!row) return;
          const target = input.dataset.simCovTarget;
          row.targets = row.targets || {};
          const value = Math.min(0.99, Math.max(-0.99, Number(input.value) || 0));
          row.targets[target] = value;
          if (Math.abs(value) < 1e-12) delete row.targets[target];
          row.target = Object.keys(row.targets)[0] || "";
          row.correlation = row.target ? Number(row.targets[row.target] || 0) : 0;
          wrap.querySelectorAll(`[data-sim-cov-index="${input.dataset.simCovIndex}"][data-sim-cov-target="${CSS.escape(target)}"]`).forEach(peer => {
            if (peer !== input) peer.value = peer.type === "number" ? formatNumber(value) : String(value);
          });
          maybeAdvanceTutorial();
        };
      });
      wrap.querySelectorAll("[data-remove-sim-cov]").forEach(button => {
        button.onclick = () => {
          simulateCovariateRows.splice(Number(button.dataset.removeSimCov), 1);
          renderSimulateCovariates();
        };
      });
    }
    async function runSimulate() {
      $("simulateStatus").textContent = "Simulating synthetic data...";
      $("runSimulate").disabled = true;
      try {
        const data = await post("/api/simulate", {
          model: $("simulateModelSelect").value,
          participants: $("simulateParticipants").value,
          trials: $("simulateTrials").value,
          seed: $("seed").value,
          dt: $("dt") ? $("dt").value : 0.01,
          noiseSd: $("noiseSd") ? $("noiseSd").value : 0.01,
          parameters: simulateParameterRows,
          covariates: simulateCovariateRows
        });
        lastSimulatedRows = data.data || [];
        renderSimulatedPreview(data.preview || {});
        $("simulateResultsArea").classList.remove("hidden");
        $("simulateSetupDetails").open = false;
        $("simulateStatus").textContent = `${data.rows} synthetic rows created from ${data.model}.`;
        maybeAdvanceTutorial();
      } catch (error) {
        $("simulateStatus").innerHTML = `<span class="warn">${escapeHtml(error.message)}</span>`;
      } finally {
        $("runSimulate").disabled = false;
      }
    }
    async function refreshSimulatedParticipantView() {
      if (!lastSimulatedRows.length) return;
      const participantId = $("simulateParticipantViewSelect").value || "all";
      const data = await post("/api/simulated-preview", {
        rows: lastSimulatedRows,
        previewParticipantId: participantId === "all" ? "" : participantId
      });
      renderSimulatedPreview(data);
    }
    function renderSimulatedPreview(data) {
      lastSimulatedPreview = data || {};
      $("simulateDataTable").innerHTML = table(lastSimulatedPreview.preview || []);
      setExportReady("simulateTableExportFrame", Boolean((lastSimulatedPreview.preview || []).length));
      const ids = (lastSimulatedPreview.allParticipantIds || []).map(String);
      $("simulateParticipantViewRow").classList.toggle("hidden", ids.length <= 1);
      const current = $("simulateParticipantViewSelect").value || "all";
      $("simulateParticipantViewSelect").innerHTML = [`<option value="all">All participants</option>`].concat(ids.map(id => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`)).join("");
      $("simulateParticipantViewSelect").value = ids.includes(current) ? current : "all";
      if (lastSimulatedPreview.dataPlotUrl) {
        $("simulateOverviewImage").src = `${lastSimulatedPreview.dataPlotUrl}?t=${Date.now()}`;
        $("simulateOverviewImage").style.display = "block";
        setExportReady("simulateDiagnosticsExportFrame", true);
      } else {
        $("simulateOverviewImage").style.display = "none";
        setExportReady("simulateDiagnosticsExportFrame", false);
      }
    }
    function modelDescription(name) {
      const descriptions = modelInventory.descriptions || {};
      const cleaned = String(name || "").replace(" - Fokker-Planck", "").replace(" - Simulation", "");
      return descriptions[name] || descriptions[cleaned] || "";
    }
    function templateDisplayName(name) {
      return String(name || "").replace(" - Fokker-Planck", "").replace(" - Simulation", "");
    }
    function openModelLoader(options = {}) {
      if (tutorialRunning && !options.tutorialSilent) {
        const title = tutorialCurrentStepTitle();
        if (title === "Load A Model") tutorialModelBrowserOpenedForTemplate = true;
        if (title === "Find Your Custom Model") tutorialModelBrowserOpenedForCustom = true;
      }
      renderModelLoadCards();
      $("modelLoadModal").classList.remove("hidden");
      maybeAdvanceTutorial();
    }
    function closeModelLoader() {
      $("modelLoadModal").classList.add("hidden");
    }
    function renderModelLoadCards() {
      const source = document.querySelector("input[name='modelLoadSource']:checked")?.value || "template";
      const showSimulation = Boolean($("showSimulationTemplates") && $("showSimulationTemplates").checked);
      $("simulationTemplateToggleLabel").classList.toggle("hidden", source !== "template");
      const cards = source === "custom" ? customModelCards() : templateModelCards(showSimulation);
      $("modelLoadCards").innerHTML = cards.length
        ? cards.map(card => modelLoadCardHtml(card)).join("")
        : `<div class="notice">${source === "custom" ? "You have no custom models." : "No models are available in this group yet."}</div>`;
      $("modelLoadCards").querySelectorAll("[data-load-template]").forEach(button => {
        button.onclick = async () => {
          if (tutorialRunning && tutorialCurrentStepTitle() === "Choose DMC" && templateDisplayName(button.dataset.loadTemplate) === "DMC" && !String(button.dataset.loadTemplate || "").includes("Simulation")) {
            tutorialDmcTemplateChosen = true;
          }
          await loadTemplate(button.dataset.loadTemplate);
          closeModelLoader();
          maybeAdvanceTutorial();
        };
      });
      $("modelLoadCards").querySelectorAll("[data-load-custom]").forEach(button => {
        button.onclick = async () => {
          const completingTutorialModelCard = tutorialRunning
            && tutorialCurrentStepTitle() === "Tutorial Model Card"
            && isTutorialModelName(button.dataset.loadCustom || "");
          if (completingTutorialModelCard) {
            tutorialCustomModelChosen = true;
          }
          await loadCustomModel(button.dataset.loadCustom);
          closeModelLoader();
          if (completingTutorialModelCard && tutorialRunning && tutorialCurrentStepTitle() === "Tutorial Model Card") {
            window.setTimeout(() => {
              if (tutorialRunning && tutorialCurrentStepTitle() === "Tutorial Model Card") nextTutorialStep();
            }, 80);
            return;
          }
          maybeAdvanceTutorial();
        };
      });
      $("modelLoadCards").querySelectorAll("[data-delete-custom]").forEach(button => {
        button.onclick = (event) => {
          event.preventDefault();
          event.stopPropagation();
          openDeleteModelModal(button.dataset.deleteCustom || "");
        };
      });
    }
    function templateModelCards(showSimulation) {
      const names = showSimulation
        ? (modelInventory.simulationTemplates || [])
        : (modelInventory.fokkerPlanckTemplates || modelInventory.lockedTemplates || []);
      return names.map(name => ({
        name,
        title: templateDisplayName(name),
        description: modelDescription(name),
        warning: !showSimulation && templateDisplayName(name) === "DSTP"
          ? "This template uses a response-process approximation for the Fokker-Planck solver. Use the simulation template for the full stage-switching DSTP."
          : "",
        attr: "data-load-template",
        deletable: false
      }));
    }
    function customModelCards() {
      const seeded = new Set(modelInventory.seededCustom || []);
      const deletable = new Set([...(modelInventory.generated || []), ...(modelInventory.saved || [])]);
      const names = Array.from(new Set([...seeded, ...deletable]));
      return names.map(name => ({
        name,
        title: name,
        description: modelDescription(name) || "Saved custom model.",
        warning: "",
        attr: "data-load-custom",
        deletable: deletable.has(name)
      }));
    }
    function modelLoadCardHtml(card) {
      const warning = card.warning
        ? `<span class="model-card-warning" title="${escapeHtml(card.warning)}">!</span>`
        : "";
      const cardButton = `<button class="model-card" type="button" ${card.attr}="${escapeHtml(card.name)}">
        ${warning}
        <span class="model-card-title">${escapeHtml(card.title)}</span>
        <span class="model-card-body">${escapeHtml(card.description || "")}</span>
      </button>`;
      if (!card.deletable) return cardButton;
      return `<div class="model-card-wrap has-delete">
        ${cardButton}
        <button class="model-delete-button" type="button" data-delete-custom="${escapeHtml(card.name)}" title="Delete ${escapeHtml(card.title)}" aria-label="Delete ${escapeHtml(card.title)}">${trashIconSvg}</button>
      </div>`;
    }
    function openDeleteModelModal(name) {
      pendingDeleteModelName = String(name || "");
      $("deleteModelMessage").textContent = `Delete custom model "${pendingDeleteModelName}"? This action cannot be undone.`;
      $("deleteModelError").classList.add("hidden");
      $("deleteModelError").textContent = "";
      $("deleteModelModal").classList.remove("hidden");
    }
    function closeDeleteModelModal() {
      pendingDeleteModelName = "";
      $("deleteModelModal").classList.add("hidden");
    }
    async function confirmDeleteModel() {
      const name = pendingDeleteModelName;
      if (!name) return;
      try {
        await post("/api/delete-model", {modelName: name});
        const deletedCurrent = currentModelSource === "custom" && currentModelName === name;
        closeDeleteModelModal();
        await refreshModels();
        if (deletedCurrent) {
          const fallback = (modelInventory.fokkerPlanckTemplates || [])[0] || (modelInventory.lockedTemplates || [])[0] || "";
          if (fallback) await loadTemplate(fallback);
        } else {
          renderModelLoadCards();
        }
        setModelStatus(`Deleted ${name}.`);
      } catch (error) {
        $("deleteModelError").classList.remove("hidden");
        $("deleteModelError").innerHTML = `<span class="warn">${escapeHtml(error.message)}</span>`;
      }
    }
    function refreshCovariateInsertOptions() {
      return;
    }
    function renderValidationModelChecklist(locked, custom) {
      const rows = [
        `<div class="muted">Built-in models</div>`,
        ...locked.map((name, index) => `<label><input type="checkbox" value="${name}" ${index < 2 ? "checked" : ""}>${name}</label>`),
        custom.length ? `<div class="muted">Custom models</div>` : "",
        ...custom.map(name => `<label><input type="checkbox" value="${name}">${name}</label>`)
      ];
      $("validationModelChecklist").innerHTML = rows.join("");
      $("validationModelChecklist").querySelectorAll("input[type=checkbox]").forEach(input => {
        input.onchange = () => {};
      });
    }
    function selectedValidationModels() {
      const mode = $("validationMode").value;
      if (mode !== "model") {
        const single = $("validationModelSelect").value || "StandardDDM";
        return single ? [single] : [];
      }
      const checked = Array.from($("validationModelChecklist").querySelectorAll("input[type=checkbox]:checked"))
        .map(input => input.value)
        .filter(Boolean);
      return checked.length ? checked : [($("validationModelSelect").value || "StandardDDM")];
    }
    async function loadTemplate(name) {
      if (!name) return;
      const data = await getJson(`/api/model-template?name=${encodeURIComponent(name)}`);
      const adapted = adaptTemplateConditions(data.code);
      const description = modelDescription(name) || data.description || "";
      setEditorFromFullCode(adapted.code, {name: templateDisplayName(name), description});
      parameterBounds = boundsRowsToMap(data.parameterBounds) || readParameterBoundsFromCode(data.code);
      renderParameterBounds();
      scheduleModelSchematic();
      currentModelName = `${templateDisplayName(name).replace(/[^A-Za-z0-9_]/g, "_")}_Custom`;
      currentModelDescription = description;
      currentModelSource = "template";
      currentTemplateName = name;
      currentTemplateDirty = false;
      currentTemplateConditionKey = templateConditionKey();
      $("loadedModelLabel").textContent = `${templateDisplayName(name)}${name.includes("Simulation") ? " (simulation template)" : " (template)"}`;
      setModelStatus("");
      updateModelHelperVisibility();
      maybeAdvanceTutorial();
    }
    async function refreshCurrentTemplateForConditions() {
      if (currentModelSource !== "template" || !currentTemplateName || currentTemplateDirty) return;
      const nextKey = templateConditionKey();
      if (nextKey === currentTemplateConditionKey) return;
      const templateName = currentTemplateName;
      const data = await getJson(`/api/model-template?name=${encodeURIComponent(templateName)}`);
      if (currentModelSource !== "template" || currentTemplateName !== templateName || currentTemplateDirty) return;
      const adapted = adaptTemplateConditions(data.code);
      const description = modelDescription(templateName) || data.description || currentModelDescription || "";
      setEditorFromFullCode(adapted.code, {name: templateDisplayName(templateName), description});
      parameterBounds = boundsRowsToMap(data.parameterBounds) || readParameterBoundsFromCode(data.code);
      renderParameterBounds();
      scheduleModelSchematic();
      currentModelName = `${templateDisplayName(templateName).replace(/[^A-Za-z0-9_]/g, "_")}_Custom`;
      currentModelDescription = description;
      currentTemplateConditionKey = nextKey;
      $("loadedModelLabel").textContent = `${templateDisplayName(templateName)}${templateName.includes("Simulation") ? " (simulation template)" : " (template)"}`;
      updateModelHelperVisibility();
    }
    function templateConditionKey() {
      const columns = conditionColumnsPayload();
      if (!columns.length) return "template-default-conditions";
      return JSON.stringify({
        columns,
        labels: currentConditionLabels(),
      });
    }
    function markCurrentModelEdited() {
      if (currentModelSource === "template") currentTemplateDirty = true;
    }
    async function loadCustomModel(name) {
      if (!name) return;
      const data = await getJson(`/api/model-code?name=${encodeURIComponent(name)}`);
      setEditorFromFullCode(data.code, {name, description: data.description || modelDescription(name)});
      parameterBounds = boundsRowsToMap(data.parameterBounds) || readParameterBoundsFromCode(data.code);
      renderParameterBounds();
      scheduleModelSchematic();
      currentModelName = name;
      currentModelDescription = data.description || modelDescription(name);
      currentModelSource = "custom";
      currentTemplateName = "";
      currentTemplateDirty = false;
      currentTemplateConditionKey = "";
      $("loadedModelLabel").textContent = `${name} (custom)`;
      setModelStatus("");
      updateModelHelperVisibility();
      maybeAdvanceTutorial();
    }
    function currentConditionLabels() {
      const preview = lastFilterPreview || lastDataPreview || {};
      return Array.isArray(preview.conditions) ? preview.conditions.map(String) : [];
    }
    function pythonString(value) {
      return JSON.stringify(String(value));
    }
    function adaptTemplateConditions(code) {
      const hasAssignedCondition = conditionColumnsPayload().length > 0;
      const labels = hasAssignedCondition ? currentConditionLabels() : [];
      const conditionArg = hasAssignedCondition ? currentConditionVariableName() : "condition";
      let adapted = code;
      if (conditionArg !== "condition") {
        adapted = renameConditionArgument(adapted, conditionArg);
      }
      if (!labels.length || !adapted.match(/conditions\s*=\s*\(\s*["']0["']\s*,\s*["']1["']\s*,?\s*\)/)) {
        return {code: adapted, changed: adapted !== code};
      }
      const tupleText = `conditions=(${labels.map(pythonString).join(", ")}${labels.length === 1 ? "," : ""})`;
      adapted = adapted.replace(/conditions\s*=\s*\(\s*["']0["']\s*,\s*["']1["']\s*,?\s*\)/g, tupleText);
      const argRe = new RegExp(`\\b${conditionArg}\\s*==\\s*["']0["']`, "g");
      adapted = adapted.replace(argRe, `${conditionArg} == ${pythonString(labels[0])}`);
      return {code: adapted, changed: adapted !== code};
    }
    function renameConditionArgument(code, nextName) {
      const safe = sanitizePythonName(nextName || "condition");
      let updated = String(code || "");
      modelFunctionNames().concat(["gamma_pulse", "normal_pdf", "update_values"]).forEach(name => {
        const block = findFunctionBlock(updated, name);
        if (!block || !block.args.includes("condition")) return;
        const functionText = updated.slice(block.start, block.end);
        const rewritten = functionText.replace(/\bcondition\b/g, safe);
        updated = updated.slice(0, block.start) + rewritten + updated.slice(block.end);
      });
      return updated;
    }
    function setEditorFromFullCode(fullCode, meta = {}) {
      const split = splitModelCode(fullCode);
      currentHiddenImports = split.imports || defaultImportsForKind(split.kind);
      currentModelKind = split.kind;
      currentModelConditions = split.conditions.length ? split.conditions : currentConditionLabels().length ? currentConditionLabels() : ["0", "1"];
      currentModelDescription = meta.description || "";
      $("modelCode").value = split.visible;
      clearModelCodeErrorLine();
      updateModelHelperVisibility();
    }
    function splitModelCode(fullCode) {
      let code = String(fullCode || "").replace(/\r\n/g, "\n");
      const split = splitTopLevelImports(code);
      const imports = split.imports;
      code = split.code;
      const conditions = extractConditions(fullCode);
      const hasSimulate = /def\s+simulate\s*\(/.test(code);
      const hasProcess = /process_model\s*=\s*define_model\s*\(/.test(code);
      code = removeAssignmentCallBlock(code, "process_model", "define_model");
      code = removeAssignmentCallBlock(code, "model", "ModelSpec");
      code = code.replace(/^\s*model\s*=\s*process_model\.to_simulation_model\(\)\s*$/gm, "");
      code = code.replace(/^\s*fp_model\s*=\s*[A-Za-z_][A-Za-z0-9_]*\([^\n]*\)\s*$/gm, "");
      return {imports, visible: code.trim() + "\n", kind: hasSimulate && !hasProcess ? "simulation" : "process", conditions};
    }
    function splitTopLevelImports(code) {
      const lines = String(code || "").split("\n");
      const imports = [];
      let index = 0;
      while (index < lines.length) {
        const trimmed = lines[index].trim();
        if (
          !trimmed ||
          /^from\s+__future__\s+import\s+annotations\b/.test(trimmed) ||
          /^import\s+[A-Za-z_][A-Za-z0-9_.]*(?:\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?$/.test(trimmed) ||
          /^from\s+[A-Za-z_][A-Za-z0-9_.]*\s+import\s+/.test(trimmed)
        ) {
          imports.push(lines[index]);
          index += 1;
          continue;
        }
        break;
      }
      const importText = imports.join("\n").trimEnd();
      return {imports: importText ? `${importText}\n` : "", code: lines.slice(index).join("\n")};
    }
    function removeAssignmentCallBlock(code, variableName, functionName) {
      const startRe = new RegExp(`(^|\\n)\\s*${variableName}\\s*=\\s*${functionName}\\s*\\(`);
      const match = code.match(startRe);
      if (!match) return code;
      const start = match.index + (match[1] ? 1 : 0);
      const open = code.indexOf("(", start);
      const close = findMatchingParenInText(code, open);
      if (close < 0) return code;
      let end = close + 1;
      while (end < code.length && code[end] !== "\n") end += 1;
      return (code.slice(0, start) + code.slice(end)).trim() + "\n";
    }
    function findMatchingParenInText(text, openIndex) {
      let depth = 0;
      let quote = null;
      for (let i = openIndex; i < text.length; i++) {
        const ch = text[i];
        const prev = i > 0 ? text[i - 1] : "";
        if (quote) {
          if (ch === quote && prev !== "\\") quote = null;
          continue;
        }
        if (ch === "\"" || ch === "'") quote = ch;
        else if (ch === "(" || ch === "[" || ch === "{") depth += 1;
        else if (ch === ")" || ch === "]" || ch === "}") {
          depth -= 1;
          if (depth === 0) return i;
        }
      }
      return -1;
    }
    function extractConditions(code) {
      const match = String(code || "").match(/conditions\s*=\s*\(([\s\S]*?)\)/);
      if (!match) return [];
      const found = [];
      const re = /["']([^"']+)["']/g;
      let item;
      while ((item = re.exec(match[1])) !== null) found.push(item[1]);
      return found;
    }
    function defaultImportsForKind(kind) {
      if (kind === "simulation") return "from __future__ import annotations\n\nimport math\nimport numpy as np\n\nfrom flexddm import ModelSpec, SimulationResult, parameters_from_bounds\n";
      return "from __future__ import annotations\n\nimport math\nimport numpy as np\n\nfrom flexddm import define_model\n";
    }
    function composeModelCode() {
      const visible = $("modelCode").value.trim();
      const imports = normalizeHiddenImports(currentHiddenImports || defaultImportsForKind(currentModelKind), visible);
      if (/process_model\s*=\s*define_model\s*\(|model\s*=\s*ModelSpec\s*\(/.test(visible)) return `${imports}\n\n${visible}\n`;
      const driftNames = visibleDriftFunctionNames(visible);
      if (driftNames.length > 1) {
        return `${racingImports(imports)}\n\n${visible}\n\n${racingSimulatorScaffold(driftNames)}\n`;
      }
      if (currentModelKind === "simulation" || /def\s+simulate\s*\(/.test(visible)) {
        return `${imports}\n\n${visible}\n\nmodel = ModelSpec(\n    name=${pythonString(currentModelName || "CustomModel")},\n    parameters=parameters_from_bounds(),\n    simulate=simulate,\n    conditions=${pythonTuple(currentModelConditions)},\n)\n`;
      }
      const components = ["drift", "bound", "noise", "starting_position", "nondecision"].filter(name => new RegExp(`def\\s+${name}\\s*\\(`).test(visible));
      const lines = [
        "process_model = define_model(",
        `    name=${pythonString(currentModelName || "CustomModel")},`,
        `    conditions=${pythonTuple(currentModelConditions)},`,
      ];
      if (components.includes("drift")) lines.push("    drift=drift,");
      if (components.includes("bound")) lines.push("    bound=bound,");
      if (components.includes("noise")) lines.push("    noise=noise,");
      if (components.includes("starting_position")) lines.push("    starting_position=starting_position,");
      if (components.includes("nondecision")) lines.push("    nondecision=nondecision,");
      else if (inferParameterNamesFromCode(visible).includes("tau")) lines.push('    nondecision="tau",');
      lines.push(")");
      return `${imports}\n\n${visible}\n\n${lines.join("\n")}\n\nmodel = process_model.to_simulation_model()\n`;
    }
    function visibleDriftFunctionNames(code) {
      const names = [];
      const re = /^def\s+(drift(?:_[A-Za-z0-9_]+)?)\s*\(/gm;
      let match;
      while ((match = re.exec(code)) !== null) {
        if (!names.includes(match[1])) names.push(match[1]);
      }
      return names.sort((a, b) => a === "drift" ? -1 : b === "drift" ? 1 : a.localeCompare(b, undefined, {numeric: true}));
    }
    function racingImports(imports) {
      let text = String(imports || "").trim();
      if (!/^import\s+inspect\b/m.test(text)) text += "\nimport inspect";
      if (!/^import\s+numpy\s+as\s+np\b/m.test(text)) text += "\nimport numpy as np";
      if (!/from\s+flexddm\s+import\s+/.test(text)) {
        text += "\nfrom flexddm import ModelSpec, SimulationResult, parameters_from_bounds";
      } else if (!/ModelSpec/.test(text) || !/SimulationResult/.test(text) || !/parameters_from_bounds/.test(text)) {
        text += "\nfrom flexddm import ModelSpec, SimulationResult, parameters_from_bounds";
      }
      if (!/RacingFokkerPlanckSpec/.test(text)) text += "\nfrom flexddm.racing import RacingFokkerPlanckSpec";
      return text.trim();
    }
    function racingSimulatorScaffold(driftNames) {
      const tupleNames = `(${driftNames.join(", ")}${driftNames.length === 1 ? "," : ""})`;
      return `RACING_DRIFTS = ${tupleNames}\n\n` +
`def _call_racing_component(component, x, t, condition, params, trial_row=None):\n` +
`    signature = inspect.signature(component).parameters\n` +
`    kwargs = {name: params[name] for name in signature if name in params}\n` +
`    if "trial_row" in signature:\n` +
`        kwargs["trial_row"] = trial_row or {}\n` +
`    if "covariates" in signature:\n` +
`        kwargs["covariates"] = trial_row or {}\n` +
`    names = list(signature)\n` +
`    if names and names[0] == "x":\n` +
`        return float(component(x, t, condition, **kwargs))\n` +
`    return float(component(t, condition, **kwargs))\n\n` +
`def _racing_simulate(*values, runtime=None):\n` +
`    params = dict(zip(PARAMETER_BOUNDS.keys(), values))\n` +
`    trials = runtime.trials if runtime is not None else 800\n` +
`    dt = runtime.dt if runtime is not None else 0.01\n` +
`    seed = runtime.seed if runtime is not None else 50\n` +
`    design = runtime.design if runtime is not None else None\n` +
`    rng = np.random.default_rng(seed)\n` +
`    trial = np.arange(1, trials + 1)\n` +
`    response = np.zeros(trials, dtype=int)\n` +
`    rt = np.zeros(trials, dtype=float)\n` +
`    condition_out = np.empty(trials, dtype=object)\n` +
`    for index in range(trials):\n` +
`        trial_row = design.iloc[index % len(design)].to_dict() if design is not None and len(design) else {}\n` +
`        condition = str(trial_row.get("condition", ${pythonTuple(currentModelConditions)}[index % len(${pythonTuple(currentModelConditions)})]))\n` +
`        condition_out[index] = condition\n` +
`        evidence = np.zeros(len(RACING_DRIFTS), dtype=float)\n` +
`        time = 0.0\n` +
`        nondecision_time = float(nondecision(0.0, condition, **{name: params[name] for name in inspect.signature(nondecision).parameters if name in params})) if "nondecision" in globals() else 0.0\n` +
`        while time < 10.0:\n` +
`            boundary_value = abs(float(bound(time, condition, **{name: params[name] for name in inspect.signature(bound).parameters if name in params}))) if "bound" in globals() else abs(float(params.get("alpha", 0.3))) / 2\n` +
`            for drift_index, drift_function in enumerate(RACING_DRIFTS):\n` +
`                evidence[drift_index] += _call_racing_component(drift_function, evidence[drift_index], time, condition, params, trial_row=trial_row) * dt\n` +
`                evidence[drift_index] += float(params.get("noise_sd", 0.1)) * np.sqrt(dt) * rng.normal()\n` +
`            winner = np.flatnonzero(evidence >= boundary_value)\n` +
`            if len(winner):\n` +
`                response[index] = 1 if int(winner[0]) == 0 else 0\n` +
`                rt[index] = time + max(nondecision_time, 0.0)\n` +
`                break\n` +
`            time += dt\n` +
`        if rt[index] <= 0:\n` +
`            response[index] = 1 if int(np.argmax(evidence)) == 0 else 0\n` +
`            rt[index] = time + max(nondecision_time, 0.0)\n` +
`    return SimulationResult(trial=trial, accuracy=response, rt=rt, condition=condition_out)\n\n` +
`model = ModelSpec(\n` +
`    name=${pythonString(currentModelName || "CustomRacingModel")},\n` +
`    parameters=parameters_from_bounds(),\n` +
`    simulate=_racing_simulate,\n` +
`    conditions=${pythonTuple(currentModelConditions)},\n` +
`)\n` +
`object.__setattr__(model, "racing_drifts", RACING_DRIFTS)\n` +
`object.__setattr__(model, "racing_bound", bound if "bound" in globals() else None)\n` +
`object.__setattr__(model, "racing_nondecision", nondecision if "nondecision" in globals() else None)\n\n` +
`fp_model = RacingFokkerPlanckSpec(\n` +
`    name=${pythonString(currentModelName || "CustomRacingModel")},\n` +
`    parameters=parameters_from_bounds(),\n` +
`    conditions=${pythonTuple(currentModelConditions)},\n` +
`    drifts=RACING_DRIFTS,\n` +
`    bound=bound if "bound" in globals() else (lambda t, condition, alpha: alpha / 2),\n` +
`    noise=noise if "noise" in globals() else 0.1,\n` +
`    nondecision=nondecision if "nondecision" in globals() else 0.0,\n` +
`)\n`;
    }
    function normalizeHiddenImports(imports, visible) {
      let text = String(imports || defaultImportsForKind(currentModelKind)).trim();
      if (visible.includes("math.") && !/^import\s+math\b/m.test(text)) text += "\nimport math";
      if ((visible.includes("np.") || visible.includes("numpy.")) && !/^import\s+numpy\s+as\s+np\b/m.test(text)) text += "\nimport numpy as np";
      if (currentModelKind === "simulation" && !/ModelSpec/.test(text)) text += "\nfrom flexddm import ModelSpec, SimulationResult, parameters_from_bounds";
      if (currentModelKind !== "simulation" && !/define_model/.test(text)) text += "\nfrom flexddm import define_model";
      return text;
    }
    function pythonTuple(values) {
      const items = (values && values.length ? values : ["0", "1"]).map(pythonString).join(", ");
      return `(${items}${values && values.length === 1 ? "," : ""})`;
    }
    let pendingFunctionComponent = "";
    function availableCovariates() {
      const preview = lastFilterPreview || lastDataPreview || {};
      return Array.isArray(preview.covariates) ? preview.covariates : [];
    }
    function covariateOptionsHtml() {
      return availableCovariates().map(row => `<option value="${escapeHtml(row.name)}">${escapeHtml(row.name)}</option>`).join("")
        + `<option value="__custom__">Custom covariate</option>`;
    }
    function selectedCovariateName(selectId, customId) {
      const selected = $(selectId).value;
      return selected === "__custom__" ? ($(customId).value || "custom_covariate") : selected;
    }
    function selectedCovariateIsNumeric(selectId, typeId) {
      if ($(selectId).value === "__custom__") return $(typeId).value !== "label";
      const covariate = $(selectId).value;
      const row = availableCovariates().find(item => item.name === covariate) || {};
      return Object.prototype.hasOwnProperty.call(row, "mean");
    }
    function syncCovariateCustomControls(prefix) {
      const isCustom = $(`${prefix}Variable`).value === "__custom__";
      $(`${prefix}CustomWrap`).classList.toggle("hidden", !isCustom);
      $(`${prefix}Type`).disabled = !isCustom;
      const covariate = selectedCovariateName(`${prefix}Variable`, `${prefix}Custom`);
      $(`${prefix}Coefficient`).value = `${sanitizePythonName(covariate)}_weight`;
    }
    function updateModelHelperVisibility() {
      const visible = $("modelCode") ? $("modelCode").value : "";
      const simulationOnly = currentModelKind === "simulation" || /def\s+simulate\s*\(/.test(visible);
      if ($("modelHelperBar")) {
        $("modelHelperBar").classList.toggle("process-hidden", simulationOnly);
      }
    }
    function modelFunctionNames() {
      return ["simulate", "drift", "bound", "noise", "starting_position", "nondecision"];
    }
    function visibleFreeParameters() {
      const names = [];
      const add = (name) => {
        if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(name) && !names.includes(name)) names.push(name);
      };
      inferParameterNamesFromCode($("modelCode").value).forEach(add);
      Object.keys(parameterBounds || {}).forEach(add);
      return names;
    }
    function functionArgsForComponent(component, functionName = component) {
      const params = visibleFreeParameters();
      const args = ["t", currentConditionVariableName()];
      params.forEach(name => {
        if (!args.includes(name)) args.push(name);
      });
      return args;
    }
    function componentSnippet(component, functionName = component) {
      const args = functionArgsForComponent(component, functionName);
      const suffix = args.join(", ");
      if (component === "drift") {
        if (functionName !== "drift") return racingDriftSnippet(functionName);
        const rateName = args.includes("drift_rate") ? "drift_rate" : "0.3";
        return `def ${functionName}(${suffix}):\n    return ${rateName}\n`;
      }
      if (component === "bound") return `def bound(${suffix}):\n    return alpha / 2\n`;
      if (component === "noise") return `def noise(${suffix}):\n    return 0.1\n`;
      if (component === "starting_position") return `def starting_position(${suffix}):\n    return (beta - 0.5) * alpha\n`;
      if (component === "nondecision") return `def nondecision(${suffix}):\n    return ${args.includes("tau") ? "tau" : "0.0"}\n`;
      return `def ${component}(${suffix}):\n    return 0.0\n`;
    }
    function racingDriftSnippet(functionName) {
      const indexText = functionName.replace(/^drift_?/, "") || "2";
      const responseName = indexText === "2" ? "incorrect" : `response_${indexText}`;
      const params = visibleFreeParameters();
      const labels = currentModelConditions && currentModelConditions.length ? currentModelConditions : ["0", "1"];
      const firstCondition = pythonString(labels[0] || "0");
      let firstParam;
      let otherParam;
      if (params.includes("delta_c") || params.includes("delta_i")) {
        firstParam = `${responseName}_drift_c`;
        otherParam = `${responseName}_drift_i`;
      } else {
        firstParam = `${responseName}_drift_${sanitizePythonName(labels[0] || "a").toLowerCase()}`;
        otherParam = `${responseName}_drift_${sanitizePythonName(labels[1] || "b").toLowerCase()}`;
      }
      parameterBounds[firstParam] = parameterBounds[firstParam] || defaultBoundsFor(firstParam);
      parameterBounds[otherParam] = parameterBounds[otherParam] || defaultBoundsFor(otherParam);
      const args = ["t", "condition"];
      params.concat([firstParam, otherParam]).forEach(name => {
        if (!args.includes(name)) args.push(name);
      });
      const conditionArg = currentConditionVariableName();
      return `def ${functionName}(${args.join(", ")}):\n    return ${firstParam} if ${conditionArg} == ${firstCondition} else ${otherParam}\n`;
    }
    function insertAtCursor(textarea, text) {
      const start = textarea.selectionStart || 0;
      const end = textarea.selectionEnd || 0;
      textarea.value = textarea.value.slice(0, start) + text + textarea.value.slice(end);
      textarea.selectionStart = textarea.selectionEnd = start + text.length;
      textarea.focus();
    }
    function findDefineModelClose(code) {
      const start = code.indexOf("define_model(");
      if (start < 0) return -1;
      let depth = 0;
      let quote = null;
      for (let i = start; i < code.length; i++) {
        const ch = code[i];
        const prev = i > 0 ? code[i - 1] : "";
        if (quote) {
          if (ch === quote && prev !== "\\") quote = null;
          continue;
        }
        if (ch === "\"" || ch === "'") {
          quote = ch;
        } else if (ch === "(" || ch === "[" || ch === "{") {
          depth += 1;
        } else if (ch === ")" || ch === "]" || ch === "}") {
          depth -= 1;
          if (depth === 0) return i;
        }
      }
      return -1;
    }
    function ensureDefineModelArgument(argName) {
      if (!/define_model\s*\(/.test($("modelCode").value)) return currentModelKind !== "simulation";
      const textarea = $("modelCode");
      const code = textarea.value;
      if (!code.includes("define_model(") || code.includes(`${argName}=`)) return false;
      const close = findDefineModelClose(code);
      if (close < 0) return false;
      const before = code.slice(0, close);
      const insertion = before.trimEnd().endsWith(",") ? `    ${argName}=${argName},\\n` : `,\\n    ${argName}=${argName}\\n`;
      textarea.value = before + insertion + code.slice(close);
      return true;
    }
    function openParameterModal() {
      $("parameterModalName").value = "";
      $("parameterModalLower").value = "-1.0";
      $("parameterModalUpper").value = "1.0";
      showParameterModalError("");
      $("parameterModal").classList.remove("hidden");
      $("parameterModalName").focus();
    }
    function closeParameterModal() {
      $("parameterModal").classList.add("hidden");
    }
    function showParameterModalError(message) {
      $("parameterModalError").classList.toggle("hidden", !message);
      $("parameterModalError").innerHTML = message ? `<span class="warn">${escapeHtml(message)}</span>` : "";
    }
    function parameterExists(parameter) {
      return visibleFreeParameters().includes(parameter) || Object.prototype.hasOwnProperty.call(parameterBounds || {}, parameter);
    }
    function applyParameterAdd(name = null, lower = null, upper = null, options = {}) {
      const raw = name || $("parameterModalName").value;
      if (!String(raw || "").trim()) return null;
      const parameter = sanitizePythonName(raw);
      if (!parameter) return null;
      if (parameterExists(parameter)) {
        const message = "This parameter already exists.";
        if (options.source === "function") showFunctionModalError(message);
        else showParameterModalError(message);
        return null;
      }
      const defaultBounds = defaultBoundsFor(parameter);
      parameterBounds[parameter] = {lower: lower || $("parameterModalLower").value || defaultBounds.lower, upper: upper || $("parameterModalUpper").value || defaultBounds.upper};
      $("modelCode").value = addParameterToModelFunctions($("modelCode").value, parameter);
      markCurrentModelEdited();
      renderParameterBounds();
      scheduleModelSchematic();
      updateModelHelperVisibility();
      if (!name) closeParameterModal();
      if (!options.quiet) setModelStatus(`Added ${parameter} to available model functions.`);
      maybeAdvanceTutorial();
      return parameter;
    }
    function addParameterToModelFunctions(code, parameter) {
      let updated = String(code || "");
      modelFunctionNames().forEach(name => {
        updated = addParameterToNamedFunction(updated, name, parameter);
      });
      return updated;
    }
    function addParameterToNamedFunction(code, name, parameter) {
      const re = new RegExp(`^def\\s+${name}\\s*\\(([^)]*)\\)\\s*:`, "m");
      const match = code.match(re);
      if (!match) return code;
      const args = match[1].split(",").map(arg => arg.trim()).filter(Boolean);
      const argNames = args.map(arg => arg.split("=")[0].trim());
      if (argNames.includes(parameter)) return code;
      const specialIndex = args.findIndex(arg => ["runtime", "trial_row"].includes(arg.split("=")[0].trim()));
      const insertAt = specialIndex >= 0 ? specialIndex : args.length;
      args.splice(insertAt, 0, parameter);
      const header = `def ${name}(${args.join(", ")}):`;
      return code.slice(0, match.index) + header + code.slice(match.index + match[0].length);
    }
    function openFunctionModal(component, options = {}) {
      if (tutorialRunning && !options.tutorialSilent && tutorialCurrentStepTitle() === "Add Model Pieces" && component === "bound") {
        tutorialBoundaryModalOpened = true;
      }
      pendingFunctionComponent = component;
      const functionName = component;
      let existing = findFunctionBlock($("modelCode").value, component);
      $("functionModalTitle").textContent = existing ? `Edit ${component}(...)` : `Add ${functionName}(...)`;
      $("functionModalCode").value = existing ? $("modelCode").value.slice(existing.start, existing.end).trim() + "\n" : componentSnippet(component, functionName);
      const notice = $("functionModalNotice");
      notice.classList.add("hidden");
      notice.textContent = "";
      prepareFunctionModalTools();
      $("functionModal").classList.remove("hidden");
      $("functionModalCode").focus();
      syncTutorialRestrictions();
      maybeAdvanceTutorial();
    }
    function closeFunctionModal(force = false) {
      if (!force && tutorialRunning && ["Add Collapse Rate", "Add Collapse Shape", "Type Boundary Code", "Apply Boundary Function"].includes(tutorialCurrentStepTitle())) {
        showFunctionModalError("Finish this tutorial step or Exit the tutorial before closing this popup.");
        return;
      }
      $("functionModal").classList.add("hidden");
      pendingFunctionComponent = "";
    }
    function applyFunctionModal() {
      const code = $("functionModalCode").value.trim() + "\n";
      if (tutorialRunning && ["Add Collapse Rate", "Add Collapse Shape", "Type Boundary Code", "Apply Boundary Function"].includes(tutorialCurrentStepTitle()) && !tutorialBoundaryCodeReady()) {
        showFunctionModalError("For the tutorial, add collapse_rate and collapse_shape, then use: return alpha / 2 * math.exp(-collapse_rate * (t ** collapse_shape))");
        syncTutorialRestrictions();
        return;
      }
      const match = code.match(/^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/m);
      if (!match) {
        showFunctionModalError("Function code must start with a Python def line.");
        return;
      }
      const fnName = match[1];
      $("modelCode").value = replaceOrAppendFunction($("modelCode").value, fnName, code);
      markCurrentModelEdited();
      renderParameterBounds();
      scheduleModelSchematic();
      updateModelHelperVisibility();
      if (tutorialRunning && tutorialCurrentStepTitle() === "Apply Boundary Function" && fnName === "bound") {
        tutorialBoundaryAppliedThisStep = true;
      }
      closeFunctionModal(true);
      setModelStatus("");
      maybeAdvanceTutorial();
    }
    function replaceOrAppendFunction(code, name, functionCode) {
      const existing = findFunctionBlock(code, name);
      if (existing) return code.slice(0, existing.start).trimEnd() + "\n\n" + functionCode.trimEnd() + "\n" + code.slice(existing.end).trimStart();
      return code.trimEnd() + "\n\n" + functionCode.trimEnd() + "\n";
    }
    function showFunctionModalError(message, warn = true) {
      $("functionModalError").classList.toggle("hidden", !message);
      $("functionModalError").innerHTML = message ? (warn ? `<span class="warn">${escapeHtml(message)}</span>` : escapeHtml(message)) : "";
    }
    function addParameterInsideFunctionModal() {
      const raw = $("functionParameterName").value;
      if (!String(raw || "").trim()) {
        showFunctionModalError("Enter a parameter name first.");
        return;
      }
      const requested = sanitizePythonName(raw);
      if (
        tutorialRunning
        && ["Add Collapse Rate", "Add Collapse Shape", "Type Boundary Code", "Apply Boundary Function"].includes(tutorialCurrentStepTitle())
        && ["collapse_rate", "collapse_shape"].includes(requested)
        && parameterExists(requested)
      ) {
        const defaults = defaultBoundsFor(requested);
        parameterBounds[requested] = {
          lower: $("functionParameterLower").value || defaults.lower,
          upper: $("functionParameterUpper").value || defaults.upper
        };
        $("modelCode").value = addParameterToModelFunctions($("modelCode").value, requested);
        markCurrentModelEdited();
        let existingFunctionCode = $("functionModalCode").value;
        const existingMatch = existingFunctionCode.match(/^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/m);
        if (existingMatch) existingFunctionCode = addParameterToNamedFunction(existingFunctionCode, existingMatch[1], requested);
        $("functionModalCode").value = existingFunctionCode;
        renderParameterBounds();
        scheduleModelSchematic();
        $("functionParameterName").value = "";
        showFunctionModalError(`Updated ${requested}.`, false);
        tutorialBoundaryParamsTouched = true;
        syncTutorialRestrictions();
        maybeAdvanceTutorial();
        return;
      }
      const parameter = applyParameterAdd(raw, $("functionParameterLower").value, $("functionParameterUpper").value, {source: "function", quiet: true});
      if (!parameter) return;
      let functionCode = $("functionModalCode").value;
      const match = functionCode.match(/^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/m);
      if (match) functionCode = addParameterToNamedFunction(functionCode, match[1], parameter);
      $("functionModalCode").value = functionCode;
      renderParameterBounds();
      $("functionParameterName").value = "";
      showFunctionModalError(`Added ${parameter}.`, false);
      if (tutorialRunning && ["Add Collapse Rate", "Add Collapse Shape", "Type Boundary Code", "Apply Boundary Function"].includes(tutorialCurrentStepTitle())) {
        tutorialBoundaryParamsTouched = true;
      }
      syncTutorialRestrictions();
      maybeAdvanceTutorial();
    }
    function prepareFunctionModalTools() {
      showFunctionModalError("");
      $("functionParameterName").value = "";
      $("functionParameterLower").value = "-1.0";
      $("functionParameterUpper").value = "1.0";
      const covariates = availableCovariates();
      $("functionCovariateRow").classList.remove("hidden");
      $("functionCovariateVariable").innerHTML = covariateOptionsHtml();
      $("functionCovariateVariable").value = covariates[0]?.name || "__custom__";
      $("functionCovariateCustom").value = "custom_covariate";
      syncCovariateCustomControls("functionCovariate");
      renderFunctionCovariateEffectOptions();
    }
    function renderFunctionCovariateEffectOptions() {
      const covariate = selectedCovariateName("functionCovariateVariable", "functionCovariateCustom");
      const isNumeric = selectedCovariateIsNumeric("functionCovariateVariable", "functionCovariateType");
      if (!covariate) {
        $("functionCovariateEffect").innerHTML = `<option value="">Enter a covariate name</option>`;
        return;
      }
      if (!isNumeric) {
        $("functionCovariateEffect").innerHTML = `<option value="label">Label lookup for branching</option>`;
        return;
      }
      const fnName = currentFunctionModalName();
      const positiveRecommended = ["bound", "noise"].includes(fnName);
      const options = [
        {value: "additive", label: "Additive shift"},
        {value: "positive_multiplier", label: "Positive multiplier"},
        {value: "helper_only", label: "Lookup only"}
      ];
      $("functionCovariateEffect").innerHTML = options.map(option => `<option value="${option.value}">${option.label}</option>`).join("");
      $("functionCovariateEffect").value = positiveRecommended ? "positive_multiplier" : "additive";
    }
    function currentFunctionModalName() {
      const match = $("functionModalCode").value.match(/^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/m);
      return match ? match[1] : pendingFunctionComponent;
    }
    function addCovariateInsideFunctionModal() {
      const covariate = selectedCovariateName("functionCovariateVariable", "functionCovariateCustom");
      if (!covariate) {
        showFunctionModalError("Enter a covariate name first.");
        return;
      }
      const isNumeric = selectedCovariateIsNumeric("functionCovariateVariable", "functionCovariateType");
      const coefficient = sanitizePythonName($("functionCovariateCoefficient").value || `${covariate}_weight`);
      const effect = $("functionCovariateEffect").value || (isNumeric ? "additive" : "label");
      if (effect === "positive_multiplier") ensureMathImport($("modelCode").value);
      let mainCode = ensureCovariateHelper($("modelCode").value, covariate, `${sanitizePythonName(covariate)}_${isNumeric ? "value" : "label"}`, isNumeric);
      if (isNumeric && effect !== "label" && !parameterBounds[coefficient]) {
        parameterBounds[coefficient] = defaultBoundsFor(coefficient);
        mainCode = addParameterToModelFunctions(mainCode, coefficient);
      }
      const result = insertCovariateIntoStandaloneFunction($("functionModalCode").value, {covariate, coefficient, effect, isNumeric});
      if (!result.ok) {
        showFunctionModalError(result.message);
        return;
      }
      $("modelCode").value = mainCode;
      markCurrentModelEdited();
      $("functionModalCode").value = result.code;
      renderParameterBounds();
      scheduleModelSchematic();
      showFunctionModalError(result.message, false);
    }
    function insertCovariateIntoStandaloneFunction(functionCode, spec) {
      const target = currentFunctionModalName();
      const helperName = `${sanitizePythonName(spec.covariate)}_${spec.isNumeric ? "value" : "label"}`;
      const fn = findFunctionBlock(functionCode, target);
      if (!fn) return {ok: false, message: `Could not find ${target}(...).`};
      let args = fn.args.slice();
      if (!args.map(arg => arg.split("=")[0].trim()).includes("trial_row")) args.push("trial_row=None");
      if (spec.isNumeric && spec.effect !== "label" && !args.map(arg => arg.split("=")[0].trim()).includes(spec.coefficient)) {
        const trialIndex = args.findIndex(arg => arg.split("=")[0].trim() === "trial_row");
        args.splice(Math.max(trialIndex, 0), 0, spec.coefficient);
      }
      const body = functionCode.slice(fn.headerEnd, fn.end);
      const returnMatch = body.match(/^(\s*)return\s+(.+)$/m);
      if (!returnMatch && !["label", "helper_only"].includes(spec.effect)) {
        return {ok: false, message: `${target}(...) has no simple return line FlexDDM can edit automatically.`};
      }
      let newBody = body;
      if (spec.effect === "helper_only") {
        newBody = insertAfterFunctionHeader(body, `    ${sanitizePythonName(spec.covariate)}_current = ${helperName}(trial_row)\n`);
      } else if (spec.effect === "label") {
        newBody = insertAfterFunctionHeader(body, `    current_${sanitizePythonName(spec.covariate)} = ${helperName}(trial_row)\n`);
      } else {
        const indent = returnMatch[1];
        const expression = returnMatch[2].trim();
        const replacement = spec.effect === "positive_multiplier"
          ? `${indent}base_value = ${expression}\n${indent}return max(1e-9, base_value * math.exp(${spec.coefficient} * ${helperName}(trial_row)))`
          : `${indent}base_value = ${expression}\n${indent}return base_value + ${spec.coefficient} * ${helperName}(trial_row)`;
        newBody = body.slice(0, returnMatch.index) + replacement + body.slice(returnMatch.index + returnMatch[0].length);
      }
      const header = `def ${target}(${args.join(", ")}):`;
      const updated = functionCode.slice(0, fn.start) + header + newBody + functionCode.slice(fn.end);
      return {ok: true, code: updated, message: `Added ${spec.covariate} to ${target}(...).`};
    }
    function convertDriftToEvidenceDependent() {
      const textarea = $("modelCode");
      const code = textarea.value;
      const match = code.match(/def\s+drift\s*\(([^)]*)\)\s*:/);
      if (!match) return false;
      const args = match[1].split(",").map(arg => arg.trim()).filter(Boolean);
      if (args[0] === "x") return true;
      if (args[0] !== "t") return false;
      const replacement = `def drift(x, ${args.join(", ")}):`;
      textarea.value = code.slice(0, match.index) + replacement + code.slice(match.index + match[0].length);
      return true;
    }
    function insertCovariateSnippet() {
      openCovariateModal();
    }
    function openConditionModal() {
      const functions = processFunctionsInCode($("modelCode").value).filter(fn => fn.name !== "simulate");
      if (!functions.length) {
        setModelStatus("Add or load a process function before adding condition logic.", true);
        return;
      }
      const conditionColumns = conditionColumnsPayload();
      $("conditionModalTarget").innerHTML = functions.map(fn => `<option value="${escapeHtml(fn.name)}">${escapeHtml(fn.name)}(${escapeHtml(fn.args.join(", "))})</option>`).join("");
      $("conditionModalColumn").innerHTML = conditionColumns.map(column => `<option value="${escapeHtml(column)}">${escapeHtml(column)}</option>`).join("")
        + `<option value="__custom__">${conditionColumns.length ? "Custom condition variable" : "Custom condition variable"}</option>`;
      $("conditionModalColumn").value = conditionColumns[0] || "__custom__";
      $("conditionModalCustomWrap").classList.toggle("hidden", $("conditionModalColumn").value !== "__custom__");
      $("conditionModalTarget").value = (functions.find(fn => fn.name === "drift") || functions[0]).name;
      renderConditionModalCode();
      $("conditionModal").classList.remove("hidden");
    }
    function closeConditionModal() {
      $("conditionModal").classList.add("hidden");
      $("conditionModalError").classList.add("hidden");
    }
    function renderConditionModalCode() {
      const target = $("conditionModalTarget").value;
      const column = selectedConditionModalVariable();
      const fn = findFunctionBlock($("modelCode").value, target);
      if (!fn) return;
      const conditionArg = sanitizePythonName(column || currentConditionVariableName());
      let code = $("modelCode").value.slice(fn.start, fn.end).trimEnd();
      code = code.replace(/\bcondition\b/g, conditionArg);
      const currentFn = findFunctionBlock(code, target);
      const args = currentFn ? currentFn.args.map(arg => arg.split("=")[0].trim()) : [];
      if (!fn.args.includes(conditionArg)) {
        if (!args.includes(conditionArg)) code = addParameterToNamedFunction(code, target, conditionArg);
      }
      const labels = $("conditionModalColumn").value === "__custom__" ? ["level_1", "level_2"] : currentConditionLabels();
      const block = labels.length
        ? labels.map((label, index) => `${index === 0 ? "if" : "elif"} ${conditionArg} == ${pythonString(label)}:\n        return base_value`).join("\n    ") + "\n    return base_value"
        : `if ${conditionArg} == "all":\n        return base_value\n    return base_value`;
      const body = code.slice(code.indexOf(":") + 1);
      const returnMatch = body.match(/^(\s*)return\s+(.+)$/m);
      if (returnMatch && !/\bbase_value\b/.test(body)) {
        const expression = returnMatch[2].trim();
        const replacement = `${returnMatch[1]}base_value = ${expression}\n    ${block}`;
        code = code.slice(0, code.indexOf(":") + 1) + body.slice(0, returnMatch.index) + replacement + body.slice(returnMatch.index + returnMatch[0].length);
      }
      $("conditionModalCode").value = code.trimEnd() + "\n";
    }
    function selectedConditionModalVariable() {
      const selected = $("conditionModalColumn").value;
      if (selected === "__custom__") return $("conditionModalCustom").value || "condition";
      return selected || firstAssignedConditionColumn() || "condition";
    }
    function applyConditionModal() {
      const code = $("conditionModalCode").value.trim() + "\n";
      const match = code.match(/^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/m);
      if (!match) {
        $("conditionModalError").classList.remove("hidden");
        $("conditionModalError").innerHTML = `<span class="warn">Condition code must start with a Python def line.</span>`;
        return;
      }
      $("modelCode").value = replaceOrAppendFunction($("modelCode").value, match[1], code);
      markCurrentModelEdited();
      renderParameterBounds();
      scheduleModelSchematic();
      closeConditionModal();
      setModelStatus("");
    }
    function openCovariateModal() {
      const covariates = availableCovariates();
      const functions = processFunctionsInCode($("modelCode").value);
      if (!functions.length) {
        setModelStatus("Add or load a model function such as drift(...), bound(...), noise(...), or simulate(...) before inserting a covariate.", true);
        return;
      }
      $("covariateModalVariable").innerHTML = covariateOptionsHtml();
      $("covariateModalVariable").value = covariates[0]?.name || "__custom__";
      $("covariateModalCustom").value = "custom_covariate";
      syncCovariateCustomControls("covariateModal");
      $("covariateModalTarget").innerHTML = functions.map(fn => `<option value="${escapeHtml(fn.name)}">${escapeHtml(fn.name)}(${escapeHtml(fn.args.join(", "))})</option>`).join("");
      const preferred = functions.find(fn => fn.name === "drift") || functions[0];
      $("covariateModalTarget").value = preferred.name;
      renderCovariateEffectOptions();
      $("covariateModal").classList.remove("hidden");
      updateCovariateModalPreview();
    }
    function closeCovariateModal() {
      $("covariateModal").classList.add("hidden");
    }
    function renderCovariateEffectOptions() {
      const covariate = selectedCovariateName("covariateModalVariable", "covariateModalCustom");
      const isNumeric = selectedCovariateIsNumeric("covariateModalVariable", "covariateModalType");
      const target = $("covariateModalTarget").value || "";
      if (target === "simulate") {
        $("covariateModalEffect").innerHTML = `<option value="helper_only">Make covariate available inside simulate(...)</option>`;
        return;
      }
      if (!isNumeric) {
        $("covariateModalEffect").innerHTML = `<option value="label">Insert label lookup for branching</option>`;
        return;
      }
      const positiveRecommended = ["bound", "noise"].includes(target);
      const options = [
        {value: "additive", label: "Additive shift: base + b * covariate"},
        {value: "positive_multiplier", label: "Positive multiplier: base * exp(b * covariate)"},
        {value: "helper_only", label: "Only insert helper and coefficient"}
      ];
      $("covariateModalEffect").innerHTML = options.map(option => `<option value="${option.value}">${option.label}</option>`).join("");
      $("covariateModalEffect").value = positiveRecommended ? "positive_multiplier" : "additive";
    }
    function updateCovariateModalPreview() {
      const covariate = selectedCovariateName("covariateModalVariable", "covariateModalCustom") || "covariate";
      const coefficient = sanitizePythonName($("covariateModalCoefficient").value || `${covariate}_weight`);
      const helper = `${sanitizePythonName(covariate)}_value`;
      const target = $("covariateModalTarget").value || "drift";
      const effect = $("covariateModalEffect").value || "additive";
      if (target === "simulate") {
        $("covariateModalPreview").textContent = `FlexDDM will add ${coefficient} to simulate(...) and create ${helper}(trial_row). Inside the trial loop, ${sanitizePythonName(covariate)}_current will hold this row's covariate value.`;
        return;
      }
      let line = `${covariate}_effect = ${coefficient} * ${helper}(trial_row)`;
      if (effect === "positive_multiplier") line = `return max(1e-9, base_value * math.exp(${coefficient} * ${helper}(trial_row)))`;
      else if (effect === "additive") line = `return base_value + ${coefficient} * ${helper}(trial_row)`;
      else if (effect === "label") line = `current_${sanitizePythonName(covariate)} = ${sanitizePythonName(covariate)}_label(trial_row)`;
      $("covariateModalPreview").textContent = `FlexDDM will update ${target}(...) and add:\n${line}`;
    }
    function applyCovariateInsertion() {
      const covariate = selectedCovariateName("covariateModalVariable", "covariateModalCustom");
      if (!covariate) return;
      const isNumeric = selectedCovariateIsNumeric("covariateModalVariable", "covariateModalType");
      const coefficient = sanitizePythonName($("covariateModalCoefficient").value || `${covariate}_weight`);
      const target = $("covariateModalTarget").value;
      const effect = $("covariateModalEffect").value || (isNumeric ? "additive" : "label");
      const result = insertCovariateIntoFunction($("modelCode").value, {covariate, coefficient, target, effect, isNumeric});
      if (!result.ok) {
        setModelStatus(result.message, true);
        return;
      }
      $("modelCode").value = result.code;
      markCurrentModelEdited();
      if (isNumeric && effect !== "label" && !parameterBounds[coefficient]) {
        parameterBounds[coefficient] = defaultBoundsFor(coefficient);
        renderParameterBounds();
      }
      closeCovariateModal();
      setModelStatus(result.message);
      renderParameterBounds();
      scheduleModelSchematic();
    }
    function processFunctionsInCode(code) {
      const functions = [];
      const re = /^def\s+(simulate|drift(?:_[A-Za-z0-9_]+)?|bound|noise|starting_position|nondecision)\s*\(([^)]*)\)\s*:/gm;
      let match;
      while ((match = re.exec(code)) !== null) {
        functions.push({name: match[1], args: match[2].split(",").map(arg => arg.trim()).filter(Boolean), start: match.index, headerEnd: re.lastIndex});
      }
      return functions;
    }
    function sanitizePythonName(value) {
      let cleaned = String(value || "").replace(/[^A-Za-z0-9_]/g, "_");
      if (!cleaned || /^[0-9]/.test(cleaned)) cleaned = `covariate_${cleaned}`;
      return cleaned;
    }
    function insertCovariateIntoFunction(code, spec) {
      const helperName = `${sanitizePythonName(spec.covariate)}_${spec.isNumeric ? "value" : "label"}`;
      let updated = ensureCovariateHelper(code, spec.covariate, helperName, spec.isNumeric);
      if (spec.effect === "positive_multiplier") updated = ensureMathImport(updated);
      const fn = findFunctionBlock(updated, spec.target);
      if (!fn) return {ok: false, message: `Could not find ${spec.target}(...).`};
      let args = fn.args.slice();
      if (spec.target === "simulate") {
        if (spec.isNumeric && !args.map(arg => arg.split("=")[0].trim()).includes(spec.coefficient)) {
          const runtimeIndex = args.findIndex(arg => arg.split("=")[0].trim() === "runtime");
          args.splice(runtimeIndex >= 0 ? runtimeIndex : args.length, 0, spec.coefficient);
        }
        if (!args.map(arg => arg.split("=")[0].trim()).includes("runtime")) args.push("runtime=None");
        const body = updated.slice(fn.headerEnd, fn.end);
        let newBody = body;
        const covName = sanitizePythonName(spec.covariate);
        if (!body.includes(`${covName}_current`)) {
          const loopRe = /^(\s*)for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+range\(([^)]*)\)\s*:/m;
          const loopMatch = body.match(loopRe);
          if (loopMatch) {
            const indent = loopMatch[1] + "    ";
            const indexName = loopMatch[2];
            const insertion = `\n${indent}trial_row = runtime.design.iloc[${indexName}].to_dict() if runtime is not None and getattr(runtime, "design", None) is not None and hasattr(runtime.design, "iloc") and ${indexName} < len(runtime.design) else {}\n${indent}${covName}_current = ${helperName}(trial_row)\n`;
            newBody = body.slice(0, loopMatch.index + loopMatch[0].length) + insertion + body.slice(loopMatch.index + loopMatch[0].length);
          } else {
            newBody = insertAfterFunctionHeader(body, `    trial_row = {}\n    ${covName}_current = ${helperName}(trial_row)\n`);
          }
        }
        const header = `def ${spec.target}(${args.join(", ")}):`;
        updated = updated.slice(0, fn.start) + header + newBody + updated.slice(fn.end);
        return {ok: true, code: updated, message: `Made ${spec.covariate} available inside simulate(...). ${spec.isNumeric ? `${spec.coefficient} is now a free parameter.` : "The lookup is ready for branching."}`};
      }
      if (!args.includes("trial_row")) args.push("trial_row=None");
      if (spec.isNumeric && spec.effect !== "label" && !args.includes(spec.coefficient)) {
        const trialIndex = args.findIndex(arg => arg.split("=")[0].trim() === "trial_row");
        args.splice(Math.max(trialIndex, 0), 0, spec.coefficient);
      }
      const body = updated.slice(fn.headerEnd, fn.end);
      const returnMatch = body.match(/^(\s*)return\s+(.+)$/m);
      if (!returnMatch && spec.effect !== "label") {
        return {ok: false, message: `${spec.target}(...) has no simple return line FlexDDM can edit automatically.`};
      }
      let newBody = body;
      if (spec.effect === "helper_only") {
        newBody = insertAfterFunctionHeader(body, `    ${sanitizePythonName(spec.covariate)}_effect = ${spec.coefficient} * ${helperName}(trial_row)\n`);
      } else if (spec.effect === "label") {
        newBody = insertAfterFunctionHeader(body, `    current_${sanitizePythonName(spec.covariate)} = ${helperName}(trial_row)\n`);
      } else {
        const indent = returnMatch[1];
        const expression = returnMatch[2].trim();
        let replacement;
        if (spec.effect === "positive_multiplier") {
          replacement = `${indent}base_value = ${expression}\n${indent}return max(1e-9, base_value * math.exp(${spec.coefficient} * ${helperName}(trial_row)))`;
        } else {
          replacement = `${indent}base_value = ${expression}\n${indent}return base_value + ${spec.coefficient} * ${helperName}(trial_row)`;
        }
        newBody = body.slice(0, returnMatch.index) + replacement + body.slice(returnMatch.index + returnMatch[0].length);
      }
      const header = `def ${spec.target}(${args.join(", ")}):`;
      updated = updated.slice(0, fn.start) + header + newBody + updated.slice(fn.end);
      return {ok: true, code: updated, message: `Added ${spec.covariate} to ${spec.target}(...). ${spec.isNumeric && spec.effect !== "label" ? `${spec.coefficient} is now a free parameter.` : "The lookup is ready for branching."}`};
    }
    function findFunctionBlock(code, name) {
      const re = new RegExp(`^def\\s+${name}\\s*\\(([^)]*)\\)\\s*:`, "m");
      const match = code.match(re);
      if (!match) return null;
      const start = match.index;
      const headerEnd = start + match[0].length;
      const after = code.slice(headerEnd);
      const next = after.search(/\n(?=def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(|process_model\s*=|model\s*=)/);
      const end = next < 0 ? code.length : headerEnd + next;
      return {start, headerEnd, end, args: match[1].split(",").map(arg => arg.trim()).filter(Boolean)};
    }
    function insertAfterFunctionHeader(body, line) {
      const newline = body.startsWith("\n") ? "\n" : "";
      const rest = body.startsWith("\n") ? body.slice(1) : body;
      return newline + line + rest;
    }
    function ensureCovariateHelper(code, covariate, helperName, isNumeric) {
      if (code.includes(`def ${helperName}(`)) return code;
      const helper = isNumeric
        ? `\n\ndef ${helperName}(trial_row=None):\n    try:\n        return float((trial_row or {}).get(${pythonString(covariate)}, 0.0) or 0.0)\n    except (TypeError, ValueError):\n        return 0.0\n`
        : `\n\ndef ${helperName}(trial_row=None):\n    return str((trial_row or {}).get(${pythonString(covariate)}, \"\"))\n`;
      const firstComponent = code.search(/^def\s+(simulate|drift|bound|noise|starting_position|nondecision)\s*\(/m);
      const insertAt = firstComponent >= 0 ? firstComponent : Math.max(code.indexOf("process_model ="), 0);
      return code.slice(0, insertAt).trimEnd() + helper + "\n" + code.slice(insertAt).trimStart();
    }
    function ensureMathImport(code) {
      if (!/^import\s+math\b/m.test(currentHiddenImports || "")) currentHiddenImports = `${currentHiddenImports || ""}\nimport math\n`;
      return code;
    }
    function nextDriftName(code) {
      const used = new Set(["drift"]);
      const re = /def\s+(drift(?:_\d+)?)\s*\(/g;
      let match;
      while ((match = re.exec(code)) !== null) used.add(match[1]);
      let index = 2;
      while (used.has(`drift_${index}`)) index += 1;
      return `drift_${index}`;
    }
    function readParameterBoundsFromCode(code) {
      const bounds = {};
      const boundsRe = /["']([A-Za-z_][A-Za-z0-9_]*)["']\s*:\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)/g;
      let match;
      while ((match = boundsRe.exec(code)) !== null) {
        bounds[match[1]] = {lower: cleanBoundValue(match[2]), upper: cleanBoundValue(match[3])};
      }
      const paramRe = /param\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)/g;
      while ((match = paramRe.exec(code)) !== null) {
        bounds[match[1]] = {lower: cleanBoundValue(match[2]), upper: cleanBoundValue(match[3])};
      }
      const parameterRe = /Parameter\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)/g;
      while ((match = parameterRe.exec(code)) !== null) {
        bounds[match[1]] = {lower: cleanBoundValue(match[2]), upper: cleanBoundValue(match[3])};
      }
      return bounds;
    }
    function rowsFromBoundsMap(bounds) {
      return Object.keys(bounds || {}).map(name => ({name, lower: bounds[name].lower, upper: bounds[name].upper}));
    }
    function boundsRowsToMap(rows) {
      if (!rows || !rows.length) return null;
      const bounds = {};
      rows.forEach(row => {
        if (row.name) bounds[row.name] = {lower: row.lower, upper: row.upper};
      });
      return bounds;
    }
    function cleanBoundValue(value) {
      return String(value).trim().replace(/^["']|["']$/g, "");
    }
    function inferParameterNamesFromCode(code) {
      const ignored = new Set(["self", "x", "t", "time", "condition", currentConditionVariableName(), "conditions", "runtime", "trial_row", "covariates", "trial_index", "nTrials", "n_trials", "dt", "var", "noiseseed", "rng", "q_left", "q_right", "choice", "reward"]);
      const names = [];
      const add = (name) => {
        name = name.replace(/^\*+/, "").trim();
        if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(name) && !ignored.has(name) && !names.includes(name)) names.push(name);
      };
      const defRe = /def\s+(simulate|drift(?:_[A-Za-z0-9_]+)?|bound|noise|starting_position|nondecision|update_values)\s*\(([^)]*)\)\s*:/g;
      let match;
      while ((match = defRe.exec(code)) !== null) {
        match[2].split(",").map(part => part.split("=")[0].trim()).forEach(add);
      }
      Object.keys(readParameterBoundsFromCode(code)).forEach(add);
      return names;
    }
    function defaultBoundsFor(name) {
      if (name === "alpha" || name.endsWith("_alpha")) return {lower: "0.07", upper: "0.38"};
      if (name === "beta" || name.endsWith("_beta")) return {lower: "0.0", upper: "1.0"};
      if (name === "tau" || name.includes("nondecision")) return {lower: "0.15", upper: "min_rt"};
      if (name.includes("shape")) return {lower: "0.5", upper: "5.0"};
      if (name.includes("time") || name.includes("rate")) return {lower: "0.01", upper: "2.0"};
      if (name.includes("drift")) return {lower: "-1.0", upper: "1.0"};
      if (name.includes("noise") || name.includes("sd")) return {lower: "0.01", upper: "2.0"};
      return {lower: "-1.0", upper: "1.0"};
    }
    function renderParameterBounds() {
      const names = inferParameterNamesFromCode($("modelCode").value);
      const parsed = readParameterBoundsFromCode($("modelCode").value);
      names.forEach(name => {
        if (!parameterBounds[name]) parameterBounds[name] = parsed[name] || defaultBoundsFor(name);
      });
      if (!names.length) {
        $("parameterBounds").innerHTML = "<table><tr><td>No free parameters detected yet.</td></tr></table>";
        return;
      }
      const rows = names.map(name => {
        const bounds = parameterBounds[name] || defaultBoundsFor(name);
        return `<tr><td>${name}</td><td><input data-param="${name}" data-bound="lower" value="${bounds.lower}"></td><td><input data-param="${name}" data-bound="upper" value="${bounds.upper}"></td></tr>`;
      }).join("");
      $("parameterBounds").innerHTML = `<table class="param-table"><tr><th>Parameter</th><th>Lower</th><th>Upper</th></tr>${rows}</table>`;
      $("parameterBounds").querySelectorAll("input[data-param]").forEach(input => {
        input.oninput = () => {
          const name = input.dataset.param;
          parameterBounds[name] = parameterBounds[name] || defaultBoundsFor(name);
          parameterBounds[name][input.dataset.bound] = input.value;
          scheduleModelSchematic();
        };
      });
    }
    async function refreshValidationParameterBounds() {
      const firstModel = ($("validationModels").value.split(",")[0] || "StandardDDM").trim();
      if (!firstModel) return;
      try {
        const data = await getJson(`/api/model-parameters?name=${encodeURIComponent(firstModel)}&solver=${encodeURIComponent($("validationBackend").value)}`);
        validationParameterBounds = boundsRowsToMap(data.parameterBounds) || {};
        renderValidationParameterBounds();
      } catch (error) {
        $("validationParameterBounds").innerHTML = `<table><tr><td class="warn">${escapeHtml(error.message)}</td></tr></table>`;
      }
    }
    function renderValidationParameterBounds() {
      const names = Object.keys(validationParameterBounds);
      if (!names.length) {
        $("validationParameterBounds").innerHTML = "<table><tr><td>No parameters loaded yet.</td></tr></table>";
        return;
      }
      const rows = names.map(name => {
        const bounds = validationParameterBounds[name];
        return `<tr><td>${name}</td><td><input data-vparam="${name}" data-bound="lower" value="${bounds.lower}"></td><td><input data-vparam="${name}" data-bound="upper" value="${bounds.upper}"></td></tr>`;
      }).join("");
      $("validationParameterBounds").innerHTML = `<table class="param-table"><tr><th>Parameter</th><th>Lower</th><th>Upper</th></tr>${rows}</table>`;
      $("validationParameterBounds").querySelectorAll("input[data-vparam]").forEach(input => {
        input.oninput = () => {
          const name = input.dataset.vparam;
          validationParameterBounds[name] = validationParameterBounds[name] || {lower: "0", upper: "1"};
          validationParameterBounds[name][input.dataset.bound] = input.value;
        };
      });
    }
    function parameterBoundsPayload() {
      return inferParameterNamesFromCode($("modelCode").value).map(name => {
        const bounds = parameterBounds[name] || defaultBoundsFor(name);
        return {name, lower: bounds.lower, upper: bounds.upper};
      });
    }
    function scheduleParameterRender() {
      clearTimeout(parameterRenderTimer);
      parameterRenderTimer = setTimeout(() => {
        renderParameterBounds();
        scheduleModelSchematic();
      }, 250);
    }
    function scheduleModelSchematic() {
      clearTimeout(schematicTimer);
      schematicTimer = setTimeout(renderModelSchematic, 700);
    }
    function requestModelSchematicNow() {
      clearTimeout(schematicTimer);
      renderModelSchematic();
    }
    function schematicParameterValuePayload() {
      return {...schematicParameterValues};
    }
    async function renderModelSchematic() {
      const code = composeModelCode();
      if (!code.trim()) {
        schematicData = null;
        schematicTrajectories = [];
        schematicError = null;
        clearModelCodeErrorLine();
        drawSchematicCanvas();
        renderSchematicLegendControls([]);
        $("schematicSliders").innerHTML = "";
        $("modelSchematicStatus").classList.add("hidden");
        return;
      }
      const requestId = ++schematicRequestId;
      const preview = lastFilterPreview || lastDataPreview || {};
      const trialRow = Array.isArray(preview.preview) && preview.preview.length ? preview.preview[0] : null;
      $("schematicCanvasWrap").classList.add("loading");
      $("modelSchematicStatus").classList.add("hidden");
      try {
        const data = await post("/api/model-schematic", {
          modelName: currentModelName || "PreviewModel",
          code,
          parameterBounds: parameterBoundsPayload(),
          parameterValues: schematicParameterValuePayload(),
          trialRow
        });
        if (requestId !== schematicRequestId) return;
        schematicData = data.schematic || null;
        schematicError = null;
        clearModelCodeErrorLine();
        schematicTrajectories = [];
        renderSchematicSliders(schematicData ? schematicData.parameters : []);
        renderSchematicLegendControls(schematicData ? schematicData.paths : []);
        drawSchematicCanvas();
        $("schematicCanvasWrap").classList.remove("loading");
        $("modelSchematicStatus").classList.add("hidden");
      } catch (error) {
        schematicData = null;
        schematicError = normalizeSchematicError(error);
        setModelCodeErrorLine(schematicError.line);
        schematicTrajectories = [];
        drawSchematicCanvas();
        renderSchematicLegendControls([]);
        $("schematicCanvasWrap").classList.remove("loading");
        $("modelSchematicStatus").classList.add("hidden");
      }
    }
    function normalizeSchematicError(error) {
      const payload = error && error.payload ? error.payload : {};
      return {
        message: payload.error || (error && error.message) || "Model preview error",
        kind: payload.errorKind || "Error",
        line: payload.errorLine || null,
        offset: payload.errorOffset || null,
        text: payload.errorText || ""
      };
    }
    function setModelCodeErrorLine(line) {
      modelCodeErrorLine = visibleEditorLineFromPreviewLine(Number(line) || null);
      updateModelCodeErrorHighlight();
    }
    function visibleEditorLineFromPreviewLine(line) {
      if (!line) return null;
      const hidden = String(currentHiddenImports || "").trim();
      const hiddenLines = hidden ? hidden.split("\n").length + 1 : 0;
      const visibleLine = line - hiddenLines;
      const visibleCount = String($("modelCode").value || "").split("\n").length;
      if (visibleLine >= 1 && visibleLine <= visibleCount) return visibleLine;
      if (line >= 1 && line <= visibleCount) return line;
      return Math.max(1, Math.min(visibleCount, visibleLine));
    }
    function clearModelCodeErrorLine() {
      modelCodeErrorLine = null;
      const textarea = $("modelCode");
      textarea.classList.remove("code-error");
      textarea.style.backgroundImage = "";
      textarea.title = "";
    }
    function updateModelCodeErrorHighlight() {
      const textarea = $("modelCode");
      if (!modelCodeErrorLine) {
        clearModelCodeErrorLine();
        return;
      }
      const style = getComputedStyle(textarea);
      const fontSize = parseFloat(style.fontSize) || 13;
      let lineHeight = parseFloat(style.lineHeight);
      if (!Number.isFinite(lineHeight)) lineHeight = fontSize * 1.45;
      const top = (modelCodeErrorLine - 1) * lineHeight - textarea.scrollTop + 8;
      const visible = top > -lineHeight && top < textarea.clientHeight;
      textarea.classList.add("code-error");
      textarea.title = `Preview error near line ${modelCodeErrorLine}`;
      textarea.style.backgroundImage = visible
        ? `linear-gradient(to bottom, transparent 0, transparent ${Math.max(top, 0)}px, rgba(220,38,38,0.18) ${Math.max(top, 0)}px, rgba(220,38,38,0.18) ${Math.max(top + lineHeight, 0)}px, transparent ${Math.max(top + lineHeight, 0)}px)`
        : "";
    }
    function schematicColors() {
      return ["#0072B2", "#CC79A7", "#0f766e", "#9a5b05"];
    }
    function renderSchematicLegendControls(paths) {
      const controls = $("schematicLegendControls");
      if (!paths || !paths.length || !schematicData || schematicData.mode !== "process") {
        controls.innerHTML = "";
        return;
      }
      const colors = schematicColors();
      controls.innerHTML = paths.map((path, index) => {
        const condition = String(path.condition);
        const label = condition.length > 32 ? condition.slice(0, 31) + "..." : condition;
        return `<span class="schematic-legend-item">
          <span class="schematic-swatch" style="background:${colors[index % colors.length]}"></span>
          <span class="schematic-legend-label" title="${escapeHtml(condition)}">${escapeHtml(label)}</span>
          <button class="secondary" data-sim-condition="${escapeHtml(condition)}">Simulate</button>
        </span>`;
      }).join("");
      controls.querySelectorAll("button[data-sim-condition]").forEach(button => {
        button.onclick = () => simulateSchematicTrajectory(button.dataset.simCondition);
      });
    }
    function renderSchematicSliders(parameters) {
      if (!parameters || !parameters.length) {
        $("schematicSliders").innerHTML = "";
        schematicParameterValues = {};
        return;
      }
      parameters.forEach(parameter => {
        if (!Object.prototype.hasOwnProperty.call(schematicParameterValues, parameter.name)) {
          schematicParameterValues[parameter.name] = Number(parameter.value);
        } else {
          schematicParameterValues[parameter.name] = Math.min(Number(parameter.upper), Math.max(Number(parameter.lower), Number(schematicParameterValues[parameter.name])));
        }
      });
      const activeNames = new Set(parameters.map(parameter => parameter.name));
      Object.keys(schematicParameterValues).forEach(name => {
        if (!activeNames.has(name)) delete schematicParameterValues[name];
      });
      $("schematicSliders").innerHTML = parameters.map(parameter => {
        const lower = Number(parameter.lower);
        const upper = Number(parameter.upper);
        const span = Math.max(Math.abs(upper - lower), 1e-6);
        const step = span / 200;
        const value = Number(schematicParameterValues[parameter.name]);
        return `<div class="schematic-slider-row">
          <span>${escapeHtml(parameter.name)}</span>
          <input type="range" data-schematic-param="${escapeHtml(parameter.name)}" min="${lower}" max="${upper}" step="${step}" value="${value}">
          <input type="number" data-schematic-number="${escapeHtml(parameter.name)}" min="${lower}" max="${upper}" step="${step}" value="${formatNumber(value)}">
        </div>`;
      }).join("");
      $("schematicSliders").querySelectorAll("input[data-schematic-param]").forEach(input => {
        const commit = () => {
          schematicParameterValues[input.dataset.schematicParam] = Number(input.value);
          requestModelSchematicNow();
        };
        input.oninput = () => {
          schematicParameterValues[input.dataset.schematicParam] = Number(input.value);
          $("schematicSliders").querySelectorAll("input[data-schematic-number]").forEach(numberInput => {
            if (numberInput.dataset.schematicNumber === input.dataset.schematicParam) numberInput.value = formatNumber(Number(input.value));
          });
        };
        input.onchange = commit;
        input.onpointerup = commit;
        input.onkeyup = (event) => {
          if (event.key === "Enter") commit();
        };
      });
      $("schematicSliders").querySelectorAll("input[data-schematic-number]").forEach(input => {
        const commit = () => {
          const parameter = parameters.find(item => item.name === input.dataset.schematicNumber);
          if (!parameter) return;
          const lower = Number(parameter.lower);
          const upper = Number(parameter.upper);
          const value = Math.min(upper, Math.max(lower, Number(input.value)));
          if (!Number.isFinite(value)) return;
          schematicParameterValues[input.dataset.schematicNumber] = value;
          input.value = formatNumber(value);
          $("schematicSliders").querySelectorAll("input[data-schematic-param]").forEach(slider => {
            if (slider.dataset.schematicParam === input.dataset.schematicNumber) slider.value = String(value);
          });
          requestModelSchematicNow();
        };
        input.onchange = commit;
        input.onblur = commit;
        input.onkeydown = (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            commit();
          }
        };
      });
    }
    function formatNumber(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) return "";
      return Math.abs(number) >= 10 ? number.toFixed(1) : number.toFixed(3);
    }
    function drawSchematicCanvas() {
      const canvas = $("modelSchematicCanvas");
      const ctx = canvas.getContext("2d");
      const ratio = window.devicePixelRatio || 1;
      const cssWidth = Math.max(360, Math.round(canvas.getBoundingClientRect().width || 620));
      const cssHeight = Math.round(cssWidth * 0.82);
      canvas.width = Math.round(cssWidth * ratio);
      canvas.height = Math.round(cssHeight * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, cssWidth, cssHeight);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, cssWidth, cssHeight);
      if (schematicError) {
        drawSchematicError(ctx, schematicError, cssWidth, cssHeight);
        return;
      }
      if (!schematicData || schematicData.mode !== "process") {
        ctx.fillStyle = "#5c6674";
        ctx.font = "13px Segoe UI, Arial, sans-serif";
        ctx.fillText("Load a process-readable model to preview its schematic.", 18, 32);
        return;
      }
      const margin = {left: 76, right: 104, top: 36, bottom: 68};
      const plot = {x: margin.left, y: margin.top, width: cssWidth - margin.left - margin.right, height: cssHeight - margin.top - margin.bottom};
      const xMin = Number(schematicData.xMin), xMax = Number(schematicData.xMax), yMin = Number(schematicData.yMin), yMax = Number(schematicData.yMax);
      const xScale = (x) => plot.x + ((Number(x) - xMin) / Math.max(xMax - xMin, 1e-9)) * plot.width;
      const yScale = (y) => plot.y + plot.height - ((Number(y) - yMin) / Math.max(yMax - yMin, 1e-9)) * plot.height;
      const colors = schematicColors();
      drawGrid(ctx, plot);
      drawTicks(ctx, plot, xMin, xMax, yMin, yMax);
      drawAxis(ctx, plot);
      drawLine(ctx, schematicData.times, schematicData.upper, xScale, yScale, "#18212f", 2.2);
      drawLine(ctx, schematicData.times, schematicData.lower, xScale, yScale, "#18212f", 2.2);
      drawDashedLine(ctx, [xMin, xMax], [0, 0], xScale, yScale, "#5c6674");
      (schematicData.paths || []).forEach((path, index) => drawLine(ctx, path.x, path.y, xScale, yScale, colors[index % colors.length], 2.3));
      schematicTrajectories.forEach((trajectory, index) => {
        const n = Math.max(2, Math.floor((trajectory.x.length || 0) * trajectory.progress));
        const pathIndex = (schematicData.paths || []).findIndex(path => String(path.condition) === String(trajectory.condition));
        const color = colors[(pathIndex >= 0 ? pathIndex : index) % colors.length];
        drawLine(ctx, trajectory.x.slice(0, n), trajectory.y.slice(0, n), xScale, yScale, color, 1.5, 0.65);
      });
      drawLabels(ctx, schematicData, plot, xScale, yScale);
      drawTitles(ctx, plot, cssWidth, cssHeight);
    }
    function drawSchematicError(ctx, error, width, height) {
      ctx.save();
      ctx.fillStyle = "#fff7ed";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#fed7aa";
      ctx.lineWidth = 2;
      ctx.strokeRect(14, 14, width - 28, height - 28);
      ctx.fillStyle = "#9a3412";
      ctx.font = "700 16px Segoe UI, Arial, sans-serif";
      ctx.fillText("Model preview error", 34, 50);
      ctx.font = "13px Segoe UI, Arial, sans-serif";
      ctx.fillStyle = "#18212f";
      const location = error.line ? `Line ${error.line}${error.offset ? `, column ${error.offset}` : ""}` : "Line unavailable";
      const details = [`${error.kind || "Error"}: ${error.message || "The model could not be previewed."}`, location];
      if (error.text) details.push(error.text);
      let y = 82;
      details.forEach((text, index) => {
        wrapCanvasText(ctx, text, 34, y, width - 68, 18);
        y += 18 * (Math.ceil(ctx.measureText(text).width / Math.max(width - 68, 1)) + 1);
        if (index === 1) y += 5;
      });
      ctx.fillStyle = "#5c6674";
      ctx.fillText("Fix the highlighted code line, then the schematic will refresh.", 34, Math.min(height - 36, y + 14));
      ctx.restore();
    }
    function wrapCanvasText(ctx, text, x, y, maxWidth, lineHeight) {
      const words = String(text || "").split(/\s+/);
      let line = "";
      words.forEach(word => {
        const test = line ? `${line} ${word}` : word;
        if (ctx.measureText(test).width > maxWidth && line) {
          ctx.fillText(line, x, y);
          line = word;
          y += lineHeight;
        } else {
          line = test;
        }
      });
      if (line) ctx.fillText(line, x, y);
    }
    function drawGrid(ctx, plot) {
      ctx.save();
      ctx.strokeStyle = "#eef1f5";
      ctx.lineWidth = 1;
      for (let i = 1; i < 5; i += 1) {
        const x = plot.x + (plot.width * i / 5);
        ctx.beginPath(); ctx.moveTo(x, plot.y); ctx.lineTo(x, plot.y + plot.height); ctx.stroke();
        const y = plot.y + (plot.height * i / 5);
        ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.width, y); ctx.stroke();
      }
      ctx.restore();
    }
    function drawTicks(ctx, plot, xMin, xMax, yMin, yMax) {
      ctx.save();
      ctx.strokeStyle = "#18212f";
      ctx.fillStyle = "#5c6674";
      ctx.lineWidth = 1;
      ctx.font = "11px Segoe UI, Arial, sans-serif";
      for (let i = 0; i <= 5; i += 1) {
        const value = xMin + (xMax - xMin) * i / 5;
        const x = plot.x + plot.width * i / 5;
        ctx.beginPath(); ctx.moveTo(x, plot.y + plot.height); ctx.lineTo(x, plot.y + plot.height + 5); ctx.stroke();
        ctx.textAlign = "center";
        ctx.fillText(value.toFixed(1), x, plot.y + plot.height + 20);
      }
      for (let i = 0; i <= 4; i += 1) {
        const value = yMin + (yMax - yMin) * i / 4;
        const y = plot.y + plot.height - plot.height * i / 4;
        ctx.beginPath(); ctx.moveTo(plot.x - 5, y); ctx.lineTo(plot.x, y); ctx.stroke();
        ctx.textAlign = "right";
        ctx.fillText(value.toFixed(2), plot.x - 9, y + 4);
      }
      ctx.restore();
    }
    function drawAxis(ctx, plot) {
      ctx.save();
      ctx.strokeStyle = "#18212f";
      ctx.lineWidth = 1.6;
      ctx.strokeRect(plot.x, plot.y, plot.width, plot.height);
      ctx.restore();
    }
    function drawLine(ctx, xs, ys, xScale, yScale, color, width, alpha = 1) {
      if (!xs || !ys || xs.length < 2 || ys.length < 2) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = width;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(xScale(xs[0]), yScale(ys[0]));
      for (let i = 1; i < Math.min(xs.length, ys.length); i += 1) ctx.lineTo(xScale(xs[i]), yScale(ys[i]));
      ctx.stroke();
      ctx.restore();
    }
    function drawDashedLine(ctx, xs, ys, xScale, yScale, color) {
      ctx.save();
      ctx.setLineDash([7, 5]);
      drawLine(ctx, xs, ys, xScale, yScale, color, 1.4, 1);
      ctx.restore();
    }
    function drawLabels(ctx, data, plot, xScale, yScale) {
      schematicLabelBoxes = [];
      ctx.save();
      ctx.font = "12px Segoe UI, Arial, sans-serif";
      drawEditableLabel(ctx, "upper_boundary", "upper boundary", plot.x + plot.width + 9, yScale(data.upper[data.upper.length - 1]) + 4, "#18212f");
      drawEditableLabel(ctx, "lower_boundary", "lower boundary", plot.x + plot.width + 9, yScale(data.lower[data.lower.length - 1]) + 4, "#18212f");
      drawEditableLabel(ctx, "z_start", "z start", plot.x + 10, yScale(data.start) - 6, "#5c6674");
      const colors = schematicColors();
      (data.paths || []).forEach((path, index) => {
        const n = Math.min(path.x?.length || 0, path.y?.length || 0);
        if (n < 2) return;
        const key = `drift_${String(path.condition).replace(/[^A-Za-z0-9_]/g, "_") || index}`;
        const state = schematicLabelOverrides[key] || {};
        const defaultPointIndex = Math.max(1, Math.min(n - 1, Math.floor(n * 0.58)));
        const pointIndex = Math.max(1, Math.min(n - 1, Number.isFinite(state.anchorIndex) ? Math.round(state.anchorIndex) : defaultPointIndex));
        const pointX = xScale(path.x[pointIndex]);
        const pointY = yScale(path.y[pointIndex]);
        const text = `drift(t): ${path.condition}`;
        const defaultX = Math.min(plot.x + plot.width - 98, Math.max(plot.x + 10, pointX + 14));
        const defaultY = Math.min(plot.y + plot.height - 10, Math.max(plot.y + 18, pointY - 14 - index * 4));
        const label = drawEditableLabel(ctx, key, text, defaultX, defaultY, colors[index % colors.length]);
        ctx.strokeStyle = colors[index % colors.length];
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        ctx.moveTo(label.x + label.width + 4, label.y - 4);
        ctx.lineTo(pointX, pointY);
        ctx.stroke();
        ctx.fillStyle = colors[index % colors.length];
        ctx.beginPath();
        ctx.arc(pointX, pointY, 4, 0, Math.PI * 2);
        ctx.fill();
        const screenPoints = [];
        for (let i = 0; i < n; i += 1) screenPoints.push({index: i, x: xScale(path.x[i]), y: yScale(path.y[i])});
        schematicLabelBoxes.push({
          key,
          type: "anchor",
          x: pointX,
          y: pointY,
          left: pointX - 8,
          right: pointX + 8,
          top: pointY - 8,
          bottom: pointY + 8,
          screenPoints
        });
        ctx.globalAlpha = 1;
      });
      ctx.restore();
    }
    function drawEditableLabel(ctx, key, defaultText, defaultX, defaultY, color) {
      const state = schematicLabelOverrides[key] || {};
      const text = state.text || defaultText;
      const x = Number.isFinite(state.x) ? state.x : defaultX;
      const y = Number.isFinite(state.y) ? state.y : defaultY;
      ctx.save();
      ctx.font = "12px Segoe UI, Arial, sans-serif";
      const width = Math.ceil(ctx.measureText(text).width);
      const height = 16;
      ctx.fillStyle = color || "#18212f";
      ctx.fillText(text, x, y);
      ctx.restore();
      const box = {key, type: "label", text, x, y, width, height, left: x - 3, top: y - height + 3, right: x + width + 3, bottom: y + 4};
      schematicLabelBoxes.push(box);
      return {x, y, width, height, text};
    }
    function schematicLabelAt(event) {
      const canvas = $("modelSchematicCanvas");
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      for (let i = schematicLabelBoxes.length - 1; i >= 0; i -= 1) {
        const box = schematicLabelBoxes[i];
        if (x >= box.left && x <= box.right && y >= box.top && y <= box.bottom) return {box, x, y};
      }
      return null;
    }
    function nearestEditableSchematicLabel(event, maxDistance = 90) {
      const canvas = $("modelSchematicCanvas");
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      let best = null;
      let bestDistance = Infinity;
      schematicLabelBoxes.forEach(box => {
        if ((box.type || "label") !== "label") return;
        const cx = (box.left + box.right) / 2;
        const cy = (box.top + box.bottom) / 2;
        const distance = Math.hypot(cx - x, cy - y);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = {box, x, y};
        }
      });
      return best && bestDistance <= maxDistance ? best : null;
    }
    function beginSchematicLabelDrag(event) {
      const hit = schematicLabelAt(event);
      if (!hit) return;
      schematicLabelDrag = {
        key: hit.box.key,
        type: hit.box.type || "label",
        offsetX: hit.x - hit.box.x,
        offsetY: hit.y - hit.box.y,
        screenPoints: hit.box.screenPoints || null
      };
      event.preventDefault();
    }
    function moveSchematicLabelDrag(event) {
      const canvas = $("modelSchematicCanvas");
      const hit = schematicLabelAt(event);
      canvas.style.cursor = schematicLabelDrag || hit ? "move" : "default";
      if (!schematicLabelDrag) return;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left - schematicLabelDrag.offsetX;
      const y = event.clientY - rect.top - schematicLabelDrag.offsetY;
      if (schematicLabelDrag.type === "anchor" && schematicLabelDrag.screenPoints) {
        const rawX = event.clientX - rect.left;
        const rawY = event.clientY - rect.top;
        let nearest = schematicLabelDrag.screenPoints[0];
        let best = Infinity;
        schematicLabelDrag.screenPoints.forEach(point => {
          const dist = (point.x - rawX) ** 2 + (point.y - rawY) ** 2;
          if (dist < best) {
            best = dist;
            nearest = point;
          }
        });
        schematicLabelOverrides[schematicLabelDrag.key] = {...(schematicLabelOverrides[schematicLabelDrag.key] || {}), anchorIndex: nearest.index};
      } else {
        schematicLabelOverrides[schematicLabelDrag.key] = {...(schematicLabelOverrides[schematicLabelDrag.key] || {}), x, y};
      }
      drawSchematicCanvas();
      event.preventDefault();
    }
    function endSchematicLabelDrag() {
      schematicLabelDrag = null;
    }
    function editSchematicLabel(event) {
      const hit = schematicLabelAt(event) || nearestEditableSchematicLabel(event);
      if (!hit) return;
      if (hit.box.type === "anchor") return;
      event.preventDefault();
      event.stopPropagation();
      const wrap = $("schematicCanvasWrap");
      wrap.querySelectorAll(".schematic-label-editor").forEach(node => node.remove());
      const input = document.createElement("input");
      input.className = "schematic-label-editor";
      input.value = schematicLabelOverrides[hit.box.key]?.text || hit.box.text || "";
      input.style.left = `${Math.max(4, hit.box.left)}px`;
      input.style.top = `${Math.max(4, hit.box.top)}px`;
      input.style.width = `${Math.min(260, Math.max(140, hit.box.width + 34))}px`;
      wrap.appendChild(input);
      const commit = () => {
        if (!input.isConnected) return;
        const nextText = String(input.value || "").trim();
        schematicLabelOverrides[hit.box.key] = {
          ...(schematicLabelOverrides[hit.box.key] || {}),
          text: nextText || hit.box.text
        };
        input.remove();
        drawSchematicCanvas();
      };
      input.addEventListener("keydown", (keyEvent) => {
        if (keyEvent.key === "Enter") {
          keyEvent.preventDefault();
          commit();
        }
        if (keyEvent.key === "Escape") {
          keyEvent.preventDefault();
          input.remove();
          drawSchematicCanvas();
        }
      });
      input.addEventListener("blur", commit);
      input.focus();
      input.select();
    }
    function drawTitles(ctx, plot, width, height) {
      ctx.save();
      ctx.fillStyle = "#18212f";
      ctx.font = "13px Segoe UI, Arial, sans-serif";
      ctx.translate(22, plot.y + plot.height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText("Evidence", 0, 0);
      ctx.restore();
      ctx.save();
      ctx.fillStyle = "#18212f";
      ctx.font = "13px Segoe UI, Arial, sans-serif";
      ctx.fillText("Time", plot.x + plot.width / 2 - 14, height - 17);
      ctx.restore();
    }
    async function simulateSchematicTrajectory(condition) {
      if (!schematicData || schematicData.mode !== "process") return;
      const preview = lastFilterPreview || lastDataPreview || {};
      const trialRow = Array.isArray(preview.preview) && preview.preview.length ? preview.preview[0] : null;
      $("schematicLegendControls").querySelectorAll("button").forEach(button => { button.disabled = true; });
      try {
        const data = await post("/api/model-trajectory", {
          modelName: currentModelName || "PreviewModel",
          code: composeModelCode(),
          parameterBounds: parameterBoundsPayload(),
          parameterValues: schematicParameterValuePayload(),
          trialRow,
          condition,
          seed: schematicSeed++
        });
        const trajectory = {...data.trajectory, progress: 0};
        schematicTrajectories.push(trajectory);
        if (schematicTrajectories.length > 3) schematicTrajectories.shift();
        animateSchematicTrajectory(trajectory);
      } catch (error) {
        $("modelSchematicStatus").innerHTML = `<span class="warn">${error.message}</span>`;
        $("modelSchematicStatus").classList.remove("hidden");
      } finally {
        $("schematicLegendControls").querySelectorAll("button").forEach(button => { button.disabled = !schematicData || schematicData.mode !== "process"; });
      }
    }
    function animateSchematicTrajectory(trajectory) {
      if (schematicAnimation) cancelAnimationFrame(schematicAnimation);
      const started = performance.now();
      const duration = 900;
      const step = (now) => {
        trajectory.progress = Math.min(1, (now - started) / duration);
        drawSchematicCanvas();
        if (trajectory.progress < 1) schematicAnimation = requestAnimationFrame(step);
      };
      schematicAnimation = requestAnimationFrame(step);
    }
    function renderHistogram(rows) {
      if (!rows || !rows.length) return "";
      const maxCount = Math.max(...rows.map(row => row.count || 0), 1);
      return rows.map(row => {
        const width = Math.max(2, Math.round((row.count || 0) / maxCount * 100));
        return `<div class="hist-row"><span>${row.label}</span><div class="hist-bar"><div class="hist-fill" style="width:${width}%"></div></div><span>${row.count}</span></div>`;
      }).join("");
    }
    function renderCountBars(rows, labelKey, countKey = "n") {
      if (!rows || !rows.length) return "<div class='notice'>Load data to see this visualization.</div>";
      const maxCount = Math.max(...rows.map(row => Number(row[countKey]) || 0), 1);
      return rows.map(row => {
        const width = Math.max(2, Math.round((Number(row[countKey]) || 0) / maxCount * 100));
        return `<div class="bar-row"><span>${escapeHtml(row[labelKey])}</span><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><span>${row[countKey]}</span></div>`;
      }).join("");
    }
    function renderValueBars(rows, labelKey, valueKey = "response_mean") {
      if (!rows || !rows.length) return "<div class='notice'>Load data to see this visualization.</div>";
      return rows.map(row => {
        const value = Math.max(0, Math.min(1, Number(row[valueKey]) || 0));
        const width = Math.max(2, Math.round(value * 100));
        return `<div class="bar-row"><span>${escapeHtml(row[labelKey])}</span><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><span>${value.toFixed(2)}</span></div>`;
      }).join("");
    }
    function dataConstantsPayload() {
      return (dataConstantRows || [])
        .filter(row => !row.locked && String(row.name || "").trim() && String(row.source || "").trim())
        .map(row => `${row.name.trim()}=${row.source.trim()}`)
        .join("; ");
    }
    function normalizeDataConstantRows(rows) {
      const actual = (rows || []).map(row => ({
        name: row.name || "",
        source: row.source || "",
        value: row.value === undefined || row.value === null ? "" : row.value,
        locked: Boolean(row.locked)
      }));
      const hasBlank = actual.some(row => !row.locked && !String(row.name).trim() && !String(row.source).trim());
      if (!hasBlank) actual.push({name: "", source: "", value: "", locked: false});
      return actual;
    }
    function mergeServerConstants(rows) {
      const incoming = normalizeDataConstantRows(rows);
      const incomplete = (dataConstantRows || []).filter(row => !row.locked && (String(row.name || "").trim() || String(row.source || "").trim()) && !(String(row.name || "").trim() && String(row.source || "").trim()));
      const merged = incoming.filter(row => row.locked || String(row.name || "").trim() || String(row.source || "").trim());
      incomplete.forEach(row => merged.push(row));
      return normalizeDataConstantRows(merged);
    }
    function commitDataConstantsIfComplete() {
      dataConstantRows = normalizeDataConstantRows(dataConstantRows);
      const incomplete = dataConstantRows.some(row => !row.locked && ((String(row.name || "").trim() && !String(row.source || "").trim()) || (!String(row.name || "").trim() && String(row.source || "").trim())));
      const complete = dataConstantRows.some(row => !row.locked && String(row.name || "").trim() && String(row.source || "").trim());
      if (!incomplete && complete) scheduleFilterPreview();
    }
    function renderDataConstants(rows) {
      dataConstantRows = mergeServerConstants(rows);
      const body = dataConstantRows.map((row, index) => {
        if (row.locked) {
          return `<tr><td><code>${escapeHtml(row.name)}</code></td><td><code>${escapeHtml(row.source)}</code></td><td>${escapeHtml(row.value)}</td></tr>`;
        }
        return `<tr>
          <td><input data-constant-index="${index}" data-constant-field="name" placeholder="deadline" value="${escapeHtml(row.name)}"></td>
          <td><input data-constant-index="${index}" data-constant-field="source" placeholder="1.5 or quantile(rt, 0.1)" value="${escapeHtml(row.source)}"></td>
          <td>${row.value === "" ? "" : escapeHtml(row.value)}</td>
        </tr>`;
      }).join("");
      $("dataConstants").innerHTML = `<table class="param-table"><tr><th>Name</th><th>Source</th><th>Value</th></tr>${body}</table>`;
      $("dataConstants").querySelectorAll("input[data-constant-index]").forEach(input => {
        input.oninput = () => {
          const index = Number(input.dataset.constantIndex);
          const field = input.dataset.constantField;
          dataConstantRows[index][field] = input.value;
          dataConstantRows = normalizeDataConstantRows(dataConstantRows);
          saveDataSettings();
        };
        input.onkeydown = (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            commitDataConstantsIfComplete();
          }
        };
        input.onblur = commitDataConstantsIfComplete;
      });
    }
    function normalizeColumnFilterRows(rows) {
      const actual = (rows || []).map(row => ({
        column: row.column || "",
        type: row.type || "",
        include: Array.isArray(row.include) ? row.include.map(String) : (Array.isArray(row.exclude) ? [] : []),
        min: row.min || "",
        max: row.max || "",
        valueOrder: Array.isArray(row.valueOrder) ? row.valueOrder.map(String) : [],
        valueScopeKey: row.valueScopeKey || ""
      }));
      const hasBlank = actual.some(row => !String(row.column || "").trim());
      if (!hasBlank) actual.push({column: "", type: "", include: [], min: "", max: "", valueOrder: [], valueScopeKey: ""});
      return actual;
    }
    function filterValueScopeKey() {
      return [
        $("csvPath").value,
        $("participantCol").value,
        conditionColumnsPayload().join(","),
        $("rtCol").value,
        $("accuracyCol").value,
        $("responseKind").value
      ].join("|");
    }
    function metadataForColumn(column, metadata) {
      return (metadata || []).find(row => row.name === column) || null;
    }
    function columnFiltersPayload() {
      const preview = lastFilterPreview || lastDataPreview || {};
      const metadata = preview.filterColumns || [];
      return normalizeColumnFilterRows(columnFilterRows)
        .map(row => {
          const meta = metadataForColumn(row.column, metadata);
          const filterType = meta ? meta.type : row.type;
          if (!row.column || !filterType) return null;
          if (filterType === "numeric") {
            if (!String(row.min || "").trim() && !String(row.max || "").trim()) return null;
            return {column: row.column, type: "numeric", min: row.min, max: row.max};
          }
          const included = (row.include || []).filter(value => String(value).trim() !== "");
          if (!included.length) return null;
          return {column: row.column, type: "categorical", include: included};
        })
        .filter(Boolean);
    }
    function schedulePreviewIfFilterComplete() {
      const usable = columnFiltersPayload().length > 0;
      if (usable || (lastFilterPreview && (lastFilterPreview.filterNotes || []).length)) scheduleFilterPreview();
    }
    function renderColumnFilters(metadata) {
      columnFilterRows = normalizeColumnFilterRows(columnFilterRows);
      const options = [`<option value="">Add a column filter</option>`].concat((metadata || []).map(column => `<option value="${escapeHtml(column.name)}">${escapeHtml(column.name)}</option>`)).join("");
      const rows = columnFilterRows.map((row, index) => {
        const meta = metadataForColumn(row.column, metadata);
        let controls = `<div class="notice">Choose a column to create a filter.</div>`;
        if (meta && meta.type === "numeric") {
          controls = `<div class="numeric-filter">
            <div><label>Minimum kept</label><input data-filter-index="${index}" data-filter-field="min" placeholder="${meta.min === null || meta.min === undefined ? "none" : meta.min}" value="${escapeHtml(row.min)}"></div>
            <div><label>Maximum kept</label><input data-filter-index="${index}" data-filter-field="max" placeholder="${meta.max === null || meta.max === undefined ? "none" : meta.max}" value="${escapeHtml(row.max)}"></div>
          </div>`;
        } else if (meta) {
          const selected = new Set(row.include || []);
          const scopeKey = filterValueScopeKey();
          if (!row.valueOrder || !row.valueOrder.length || row.valueScopeKey !== scopeKey) {
            row.valueOrder = (meta.values || []).map(String);
            row.valueScopeKey = scopeKey;
          }
          const values = Array.from(new Set([...(row.valueOrder || []), ...(row.include || [])]));
          const allChecked = values.length > 0 && values.every(value => selected.has(String(value)));
          controls = `<div><div class="filter-actions"><label><input type="checkbox" data-filter-index="${index}" data-filter-check-all="true" ${allChecked ? "checked" : ""}>Include all values</label></div><div class="muted">Include checked values only</div><div class="filter-values">${
            values.map(value => `<label><input type="checkbox" data-filter-index="${index}" data-filter-value="${escapeHtml(value)}" ${selected.has(String(value)) ? "checked" : ""}>${escapeHtml(value)}</label>`).join("")
          }</div>${meta.truncated ? "<div class='muted'>Showing first 250 unique values.</div>" : ""}</div>`;
        }
        return `<div class="filter-row">
          <div><label>Column</label><select data-filter-index="${index}" data-filter-field="column">${options}</select></div>
          <div>${controls}</div>
        </div>`;
      }).join("");
      $("columnFilters").innerHTML = rows;
      $("columnFilters").querySelectorAll("select[data-filter-field='column']").forEach(select => {
        const index = Number(select.dataset.filterIndex);
        select.value = columnFilterRows[index].column || "";
        select.onchange = () => {
          const hadActiveFilter = columnFiltersPayload().length > 0 || Boolean(lastFilterPreview && (lastFilterPreview.filterNotes || []).length);
          const meta = metadataForColumn(select.value, metadata);
          columnFilterRows[index] = {column: select.value, type: meta ? meta.type : "", include: [], min: "", max: "", valueOrder: meta && meta.values ? meta.values.map(String) : [], valueScopeKey: filterValueScopeKey()};
          columnFilterRows = normalizeColumnFilterRows(columnFilterRows);
          saveDataSettings();
          renderColumnFilters(metadata);
          if (hadActiveFilter && !select.value) scheduleFilterPreview();
          maybeAdvanceTutorial();
        };
      });
      $("columnFilters").querySelectorAll("input[data-filter-field]").forEach(input => {
        input.oninput = () => {
          const index = Number(input.dataset.filterIndex);
          columnFilterRows[index][input.dataset.filterField] = input.value;
          saveDataSettings();
          maybeAdvanceTutorial();
        };
        input.onblur = () => {
          schedulePreviewIfFilterComplete();
          maybeAdvanceTutorial();
        };
        input.onkeydown = (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            schedulePreviewIfFilterComplete();
            maybeAdvanceTutorial();
          }
        };
      });
      $("columnFilters").querySelectorAll("input[data-filter-value]").forEach(input => {
        input.onchange = () => {
          const index = Number(input.dataset.filterIndex);
          const value = input.dataset.filterValue;
          const selected = new Set(columnFilterRows[index].include || []);
          if (input.checked) selected.add(value); else selected.delete(value);
          columnFilterRows[index].include = Array.from(selected);
          saveDataSettings();
          scheduleFilterPreview();
          maybeAdvanceTutorial();
        };
      });
      $("columnFilters").querySelectorAll("input[data-filter-check-all]").forEach(input => {
        input.onchange = () => {
          const index = Number(input.dataset.filterIndex);
          const values = columnFilterRows[index].valueOrder || [];
          columnFilterRows[index].include = input.checked ? values.slice() : [];
          saveDataSettings();
          renderColumnFilters(metadata);
          scheduleFilterPreview();
          maybeAdvanceTutorial();
        };
      });
    }
    function renderDataPreview(data) {
      lastDataPreview = data;
      setDataLoadedState(true);
      $("dataStatus").textContent = `${data.rows} mapped rows | ${data.participants} participants | ${data.responseLabel} from ${data.responseColumn} | RT ${data.rtMin.toFixed(3)}-${data.rtMax.toFixed(3)} | ${data.conditions.length} conditions`;
      refreshCovariateInsertOptions();
      updateModelHelperVisibility();
      if (mappingComplete() && !mappingAutoCollapsed) {
        $("dataMappingDetails").open = false;
        mappingAutoCollapsed = true;
        $("filterContent").classList.remove("hidden");
      }
      saveDataSettings();
      refreshCurrentTemplateForConditions().catch(error => setModelStatus(error.message, true));
      scheduleModelSchematic();
      maybeAdvanceTutorial();
    }
    function renderFilterPreview(data) {
      lastFilterPreview = data;
      setDataLoadedState(true);
      renderParticipantViewSelect(data);
      const filters = data.filterNotes && data.filterNotes.length ? ` | filters: ${data.filterNotes.join("; ")}` : "";
      $("filterStatus").textContent = `${data.rows} analysis rows | ${data.participants} participants | ${data.responseLabel} from ${data.responseColumn} | RT ${data.rtMin.toFixed(3)}-${data.rtMax.toFixed(3)} | ${data.conditions.length} conditions${filters}`;
      $("filterDataTable").innerHTML = table(data.preview);
      if (!$("dataConstants").contains(document.activeElement)) {
        renderDataConstants(data.dataConstants);
      }
      if (!$("columnFilters").contains(document.activeElement)) {
        renderColumnFilters(data.filterColumns || []);
      }
      refreshCovariateInsertOptions();
      updateModelHelperVisibility();
      if (data.dataPlotUrl) {
        $("filterOverviewImage").src = `${data.dataPlotUrl}?t=${Date.now()}`;
        $("filterOverviewImage").style.display = "block";
        setExportReady("dataDiagnosticsExportFrame", true);
      } else {
        $("filterOverviewImage").style.display = "none";
        setExportReady("dataDiagnosticsExportFrame", false);
      }
      saveDataSettings();
      refreshCurrentTemplateForConditions().catch(error => setModelStatus(error.message, true));
      scheduleModelSchematic();
      maybeAdvanceTutorial();
    }
    async function previewData(participantId = "") {
      if (!mappingComplete()) {
        setDataLoadedState(dataLoaded);
        return;
      }
      const requestId = ++dataPreviewRequestId;
      $("dataStatus").textContent = "Loading...";
      const payload = payloadBase(false);
      if (participantId) payload.participantId = participantId;
      const data = await post("/api/data-preview", payload);
      if (requestId !== dataPreviewRequestId || payload.path !== $("csvPath").value) return;
      renderDataPreview(data);
    }
    async function previewFilterData() {
      if (!mappingComplete()) {
        setDataLoadedState(dataLoaded);
        return;
      }
      const requestId = ++filterPreviewRequestId;
      $("filterStatus").textContent = "Loading filtered data...";
      const payload = payloadBase(true);
      const participantView = $("participantViewSelect").value || "all";
      if (participantView !== "all") payload.previewParticipantId = participantView;
      const data = await post("/api/data-preview", payload);
      if (requestId !== filterPreviewRequestId || payload.path !== $("csvPath").value) return;
      renderFilterPreview(data);
    }
    $("browseCsv").onclick = browseCsvPath;
    if ($("previewData")) {
      $("previewData").onclick = async () => {
        try {
          await previewData("");
        } catch (error) { $("dataStatus").innerHTML = `<span class="warn">${error.message}</span>`; }
      };
    }
    $("csvPath").onchange = () => loadCsvColumns().catch(error => {
      $("dataStatus").innerHTML = `<span class="warn">${escapeHtml(error.message)}</span>`;
    });
    ["participantCol", "rtCol", "accuracyCol", "responseKind"].forEach(id => {
      $(id).onchange = async () => {
        saveDataSettings();
        setDataLoadedState(dataLoaded);
        if (!mappingComplete()) return;
        maybeAdvanceTutorial();
        try {
          await previewData("");
          await previewFilterData();
        } catch (error) {
          $("dataStatus").innerHTML = `<span class="warn">${escapeHtml(error.message)}</span>`;
        }
      };
    });
    $("participantViewSelect").onchange = handleParticipantViewChange;
    $("participantViewSelect").addEventListener("change", handleParticipantViewChange);
    $("modelSchematicCanvas").addEventListener("mousedown", beginSchematicLabelDrag);
    $("modelSchematicCanvas").addEventListener("mousemove", moveSchematicLabelDrag);
    $("modelSchematicCanvas").addEventListener("mouseleave", endSchematicLabelDrag);
    window.addEventListener("mouseup", endSchematicLabelDrag);
    $("modelSchematicCanvas").addEventListener("dblclick", editSchematicLabel);
    $("simulateModelSelect").onchange = async () => {
      await refreshSimulateParameters();
      maybeAdvanceTutorial();
    };
    $("addSimCovariate").onclick = addSimCovariateRow;
    $("runSimulate").onclick = runSimulate;
    $("simulateParticipantViewSelect").onchange = () => refreshSimulatedParticipantView().catch(error => {
      $("simulateStatus").innerHTML = `<span class="warn">${escapeHtml(error.message)}</span>`;
    });
    $("tutorialTab").onclick = openTutorialIntro;
    $("tutorialIntroNo").onclick = () => closeTutorialIntro(true);
    $("tutorialIntroYes").onclick = startTutorial;
    $("tutorialExit").onclick = () => endTutorial(true);
    $("tutorialBack").onclick = previousTutorialStep;
    $("tutorialNext").onclick = nextTutorialStep;
    $("tutorialIntroModal").addEventListener("click", (event) => {
      if (event.target === $("tutorialIntroModal")) closeTutorialIntro(true);
    });
    window.addEventListener("resize", positionTutorial);
    document.querySelector("main").addEventListener("scroll", positionTutorial, {passive: true});
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && tutorialRunning) endTutorial(true);
    });
    $("openModelLoader").onclick = openModelLoader;
    $("closeModelLoader").onclick = closeModelLoader;
    $("modelLoadModal").addEventListener("click", (event) => {
      if (event.target === $("modelLoadModal")) closeModelLoader();
    });
    $("cancelDeleteModel").onclick = closeDeleteModelModal;
    $("confirmDeleteModel").onclick = confirmDeleteModel;
    $("deleteModelModal").addEventListener("click", (event) => {
      if (event.target === $("deleteModelModal")) closeDeleteModelModal();
    });
    document.querySelectorAll("input[name='modelLoadSource']").forEach(input => {
      input.onchange = () => {
        if (tutorialRunning && tutorialCurrentStepTitle() === "Custom Models" && input.value === "custom" && input.checked) {
          tutorialCustomSourceChosen = true;
        }
        renderModelLoadCards();
        maybeAdvanceTutorial();
      };
    });
    $("showSimulationTemplates").onchange = renderModelLoadCards;
    $("addParameterButton").onclick = openParameterModal;
    $("pasteModelCode").onclick = pasteModelCodeFromClipboard;
    $("cancelParameterAdd").onclick = closeParameterModal;
    $("applyParameterAdd").onclick = () => applyParameterAdd();
    $("parameterModal").addEventListener("click", (event) => {
      if (event.target === $("parameterModal")) closeParameterModal();
    });
    $("parameterModalName").oninput = () => {
      const parameter = sanitizePythonName($("parameterModalName").value);
      showParameterModalError(parameter && parameterExists(parameter) ? "This parameter already exists." : "");
      if (parameter) {
        const bounds = defaultBoundsFor(parameter);
        $("parameterModalLower").value = bounds.lower;
        $("parameterModalUpper").value = bounds.upper;
      }
    };
    document.querySelectorAll("[data-function-component]").forEach(button => button.onclick = () => openFunctionModal(button.dataset.functionComponent));
    $("functionModalAddParameter").onclick = addParameterInsideFunctionModal;
    $("functionModalAddCovariate").onclick = addCovariateInsideFunctionModal;
    $("addConditionButton").onclick = openConditionModal;
    $("cancelConditionAdd").onclick = closeConditionModal;
    $("applyConditionAdd").onclick = applyConditionModal;
    $("conditionModalTarget").onchange = renderConditionModalCode;
    $("conditionModalColumn").onchange = () => {
      $("conditionModalCustomWrap").classList.toggle("hidden", $("conditionModalColumn").value !== "__custom__");
      renderConditionModalCode();
    };
    $("conditionModalCustom").oninput = renderConditionModalCode;
    $("conditionModal").addEventListener("click", (event) => {
      if (event.target === $("conditionModal")) closeConditionModal();
    });
    $("cancelFunctionAdd").onclick = closeFunctionModal;
    $("applyFunctionAdd").onclick = applyFunctionModal;
    $("functionModal").addEventListener("click", (event) => {
      if (event.target === $("functionModal")) closeFunctionModal();
    });
    $("functionParameterName").oninput = () => {
      const parameter = sanitizePythonName($("functionParameterName").value);
      if (parameter) {
        const bounds = defaultBoundsFor(parameter);
        $("functionParameterLower").value = bounds.lower;
        $("functionParameterUpper").value = bounds.upper;
      }
      syncTutorialRestrictions();
    };
    ["functionParameterLower", "functionParameterUpper", "functionModalCode"].forEach(id => {
      $(id).addEventListener("input", () => {
        syncTutorialRestrictions();
        maybeAdvanceTutorial();
      });
    });
    $("functionCovariateVariable").onchange = () => {
      syncCovariateCustomControls("functionCovariate");
      renderFunctionCovariateEffectOptions();
    };
    $("functionCovariateCustom").oninput = () => {
      syncCovariateCustomControls("functionCovariate");
      renderFunctionCovariateEffectOptions();
    };
    $("functionCovariateType").onchange = renderFunctionCovariateEffectOptions;
    $("functionCovariateEffect").onchange = () => showFunctionModalError("");
    $("insertCovariateSnippet").onclick = insertCovariateSnippet;
    $("cancelCovariateInsert").onclick = closeCovariateModal;
    $("applyCovariateInsert").onclick = applyCovariateInsertion;
    $("covariateModal").addEventListener("click", (event) => {
      if (event.target === $("covariateModal")) closeCovariateModal();
    });
    $("covariateModalVariable").onchange = () => {
      syncCovariateCustomControls("covariateModal");
      renderCovariateEffectOptions();
      updateCovariateModalPreview();
    };
    $("covariateModalCustom").oninput = () => {
      syncCovariateCustomControls("covariateModal");
      renderCovariateEffectOptions();
      updateCovariateModalPreview();
    };
    $("covariateModalType").onchange = () => {
      renderCovariateEffectOptions();
      updateCovariateModalPreview();
    };
    $("covariateModalCoefficient").oninput = updateCovariateModalPreview;
    $("covariateModalTarget").onchange = () => {
      renderCovariateEffectOptions();
      updateCovariateModalPreview();
    };
    $("covariateModalEffect").onchange = updateCovariateModalPreview;
    window.addEventListener("resize", () => drawSchematicCanvas());
    async function pasteModelCodeFromClipboard() {
      try {
        const text = await readClipboardText();
        if (!String(text || "").trim()) {
          setModelStatus("Clipboard is empty.", true);
          return;
        }
        setEditorFromFullCode(text, {name: currentModelName, description: currentModelDescription});
        markCurrentModelEdited();
        parameterBounds = readParameterBoundsFromCode(text);
        renderParameterBounds();
        scheduleModelSchematic();
        updateModelHelperVisibility();
        setModelStatus("Pasted model code from clipboard.");
      } catch (error) {
        setModelStatus(`Clipboard paste failed: ${error.message || error}`, true);
      }
    }
    function openSaveModelModal(options = {}) {
      if (tutorialRunning && !options.tutorialSilent && tutorialCurrentStepTitle() === "Save Custom Models") {
        tutorialSaveModalOpened = true;
      }
      $("saveModelName").value = currentModelName && currentModelName !== "PreviewModel" ? currentModelName : "MyFlexModel";
      $("saveModelDescription").value = currentModelDescription || "";
      $("saveModelModal").classList.remove("hidden");
      $("saveModelName").focus();
      $("saveModelName").select();
      syncTutorialRestrictions();
      maybeAdvanceTutorial();
    }
    function closeSaveModelModal(force = false) {
      if (!force && tutorialRunning && tutorialCurrentStepTitle() === "Save Tutorial Model") {
        setModelStatus("Finish saving Tutorial Model, or Exit the tutorial before closing this popup.", true);
        return;
      }
      $("saveModelModal").classList.add("hidden");
    }
    $("saveModel").onclick = openSaveModelModal;
    $("cancelSaveModel").onclick = closeSaveModelModal;
    $("saveModelModal").addEventListener("click", (event) => {
      if (event.target === $("saveModelModal")) closeSaveModelModal();
    });
    $("saveModelName").addEventListener("input", syncTutorialRestrictions);
    $("applySaveModel").onclick = async () => {
      const requestedName = $("saveModelName").value || "CustomModel";
      if (tutorialRunning && tutorialCurrentStepTitle() === "Save Tutorial Model" && String(requestedName).trim() !== "Tutorial Model") {
        setModelStatus("For the tutorial, name this model exactly Tutorial Model.", true);
        syncTutorialRestrictions();
        return;
      }
      setModelStatus("Saving and checking model...");
      try {
        currentModelName = requestedName.replace(/[^A-Za-z0-9_]/g, "");
        currentModelDescription = $("saveModelDescription").value || "";
        const data = await post("/api/save-model", {modelName: requestedName, description: currentModelDescription, code: composeModelCode(), parameterBounds: parameterBoundsPayload()});
        setEditorFromFullCode(data.code || composeModelCode(), {name: currentModelName, description: currentModelDescription});
        parameterBounds = boundsRowsToMap(data.parameterBounds) || readParameterBoundsFromCode($("modelCode").value);
        renderParameterBounds();
        scheduleModelSchematic();
        closeSaveModelModal(true);
        currentModelSource = "custom";
        currentTemplateName = "";
        currentTemplateDirty = false;
        currentTemplateConditionKey = "";
        $("loadedModelLabel").textContent = `${currentModelName} (custom)`;
        setModelStatus(`Saved ${data.model}\n${data.path}`);
        await refreshModels();
        $("fitModelSelect").value = currentModelName;
        if (tutorialRunning && String(requestedName).trim() === "Tutorial Model") {
          tutorialModelSavedThisStep = true;
        }
        maybeAdvanceTutorial();
      } catch (error) { setModelStatus(error.message, true); }
    };
    $("modelCode").addEventListener("input", () => { markCurrentModelEdited(); updateModelHelperVisibility(); scheduleParameterRender(); });
    $("modelCode").addEventListener("scroll", updateModelCodeErrorHighlight);
    document.querySelectorAll("input[name='fitScope']").forEach(input => {
      input.onchange = syncFitScopeControls;
      input.addEventListener("change", syncFitScopeControls);
    });
    syncFitScopeControls();
    $("clusterExportDismiss").onclick = closeClusterExportModal;
    $("clusterExportApply").onclick = exportPendingClusterJob;
    $("clusterExportAbort").onclick = abortPendingClusterJob;
    $("clusterExportModal").addEventListener("click", (event) => {
      if (event.target === $("clusterExportModal")) closeClusterExportModal();
    });
    $("runFit").onclick = async () => {
      let startedFitJob = null;
      try {
        if (activeFitJob) {
          $("fitStatus").textContent = "Cancelling fit...";
          await cancelJob(activeFitJob);
          await waitForJobToStop(activeFitJob, $("fitStatus"));
          activeFitJob = null;
          setFitButtonRunning(false);
          return;
        }
        $("fitStatus").textContent = "Starting fit...";
        fitHasResults = false;
        $("fitResultsArea").classList.add("hidden");
        $("fitTable").innerHTML = "";
        $("conditionTable").innerHTML = "";
        $("fitCovariatePanel").classList.add("hidden");
        $("fitCovariateTable").innerHTML = "";
        $("ppcSelect").innerHTML = "";
        $("ppcImage").style.display = "none";
        setExportReady("fitTableExportFrame", false);
        setExportReady("conditionTableExportFrame", false);
        setExportReady("fitCovariateTableExportFrame", false);
        setExportReady("ppcExportFrame", false);
        const fitPayload = {...payloadBase(), ...optionPayload(), solver: $("fitBackend").value, model: $("fitModelSelect").value, fitScope: fitScopeValue(), participantId: $("participantId").value, posteriorPredictive: true};
        await runJob(
          "/api/fit-job",
          fitPayload,
          $("fitStatus"),
          (data) => {
            if (activeFitJob === startedFitJob) activeFitJob = null;
            renderFitResult(data);
          },
          (jobId) => {
            startedFitJob = jobId;
            activeFitJob = jobId;
            setFitButtonRunning(true);
          },
          "fit"
        );
        if (activeFitJob === startedFitJob) activeFitJob = null;
        setFitButtonRunning(false);
      } catch (error) {
        if (activeFitJob === startedFitJob) activeFitJob = null;
        setFitButtonRunning(false);
        if (error.clusterPrompt) return;
        $("fitStatus").innerHTML = `<span class="warn">${error.message}</span>`;
      }
    };
    $("ppcSelect").onchange = () => {
      if ($("ppcSelect").value) {
        $("ppcImage").src = `${$("ppcSelect").value}?t=${Date.now()}`;
        $("ppcImage").style.display = "block";
        setExportReady("ppcExportFrame", true);
      }
    };
    function updateValidationModeUI() {
      const mode = $("validationMode").value;
      $("validationParticipantRow").classList.toggle("hidden", !["internal", "curvature"].includes(mode));
      $("validationSingleModelRow").classList.toggle("hidden", mode === "model");
      $("validationModelRecoveryRow").classList.toggle("hidden", mode !== "model");
      $("validationSimsRow").classList.toggle("hidden", ["internal", "curvature", "profile"].includes(mode));
      $("validationTrialsRow").classList.toggle("hidden", ["internal", "curvature"].includes(mode));
    }
    $("validationMode").onchange = () => {
      updateValidationModeUI();
      maybeAdvanceTutorial();
    };
    $("runValidation").onclick = async () => {
      if (activeValidationJob) {
        $("validationStatus").textContent = "Cancelling validation...";
        try {
          await cancelJob(activeValidationJob);
          await waitForJobToStop(activeValidationJob, $("validationStatus"));
        } catch (error) {
          $("validationStatus").innerHTML = `<span class="warn">${error.message}</span>`;
        } finally {
          activeValidationJob = null;
          setValidationButtonRunning(false);
        }
        return;
      }
      $("validationStatus").textContent = "Starting validation...";
      validationHasResults = false;
      $("validationResultsArea").classList.add("hidden");
      $("validationTable").innerHTML = "";
      $("validationImage").style.display = "none";
      setExportReady("validationTableExportFrame", false);
      setExportReady("validationImageExportFrame", false);
      let startedValidationJob = null;
      try {
        const validationPayload = {...payloadBase(), ...optionPayload(), solver: $("validationBackend").value, mode: $("validationMode").value, models: selectedValidationModels(), simulations: Number($("validationSims").value), trialsPerSimulation: Number($("validationTrials").value), simulatedTrials: Number($("validationTrials").value), participantId: $("validationParticipantId").value};
        await runJob(
          "/api/validate-job",
          validationPayload,
          $("validationStatus"),
          (data) => {
            if (activeValidationJob === startedValidationJob) activeValidationJob = null;
            setValidationButtonRunning(false);
            validationHasResults = true;
            $("validationResultsArea").classList.remove("hidden");
            $("validationSetupDetails").open = false;
            $("validationStatus").textContent = `${data.mode} validation complete`;
            $("validationTable").innerHTML = table(data.rows);
            setExportReady("validationTableExportFrame", Boolean((data.rows || []).length));
            if (data.plotUrl) {
              $("validationImage").src = `${data.plotUrl}?t=${Date.now()}`;
              $("validationImage").style.display = "block";
              setExportReady("validationImageExportFrame", true);
            } else {
              setExportReady("validationImageExportFrame", false);
            }
          },
          (jobId) => {
            startedValidationJob = jobId;
            activeValidationJob = jobId;
            setValidationButtonRunning(true);
          },
          "validation"
        );
        if (activeValidationJob === startedValidationJob) activeValidationJob = null;
        setValidationButtonRunning(false);
      } catch (error) {
        if (activeValidationJob === startedValidationJob) activeValidationJob = null;
        setValidationButtonRunning(false);
        if (error.clusterPrompt) return;
        $("validationStatus").innerHTML = `<span class="warn">${error.message}</span>`;
      }
    };
    updateValidationModeUI();
    installSaveIcons();
    installPasteContextMenu();
    loadAboutPane().catch(() => {});
    refreshModels().then(() => {
      const firstTemplate = (modelInventory.fokkerPlanckTemplates || [])[0] || (modelInventory.simulationTemplates || [])[0] || "";
      if (firstTemplate) {
        loadTemplate(firstTemplate);
      }
    });
    setDataLoadedState(false);
