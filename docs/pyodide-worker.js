const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/";
let pyodideReadyPromise = null;
let moduleLoaded = false;

function postStatus(message) {
  self.postMessage({ type: "status", message });
}

function postDebug(label, payload) {
  self.postMessage({ type: "debug", label, payload });
}

function serializeError(error) {
  if (error instanceof Error) {
    return { message: error.message, stack: error.stack };
  }
  return { message: String(error) };
}

async function ensurePyodide() {
  if (!pyodideReadyPromise) {
    postStatus("Pyodideを読み込み中...");
    pyodideReadyPromise = loadPyodide({ indexURL: PYODIDE_URL }).catch((error) => {
      pyodideReadyPromise = null;
      throw error;
    });
  }
  const pyodide = await pyodideReadyPromise;
  if (!moduleLoaded) {
    postStatus("Pythonスクリプトを読み込み中...");
    const response = await fetch("compare_interpolation.py");
    if (!response.ok) {
      throw new Error("compare_interpolation.py を取得できませんでした");
    }
    const code = await response.text();
    pyodide.FS.writeFile("compare_interpolation.py", code);
    await pyodide.runPythonAsync("import compare_interpolation");
    moduleLoaded = true;
    postStatus("Pyodideの初期化が完了しました。");
  }
  return pyodide;
}

async function handleFit(payload) {
  const pyodide = await ensurePyodide();
  const dataset = payload && payload.dataset ? payload.dataset : null;
  const normalize = payload && payload.normalize === true;
  const denormalizeAfterPredict = payload && payload.denormalizeAfterPredict === true;
  const algorithmConfigs = payload && Array.isArray(payload.algorithmConfigs) ? payload.algorithmConfigs : null;
  let pyDataset = null;
  let pyAlgorithmConfigs = null;
  if (dataset !== null) {
    pyDataset = pyodide.toPy(dataset);
  }
  if (algorithmConfigs !== null) {
    pyAlgorithmConfigs = pyodide.toPy(algorithmConfigs);
  }
  try {
    pyodide.globals.set("PY_DATASET", pyDataset);
    pyodide.globals.set("PY_NORMALIZE", normalize);
    pyodide.globals.set("PY_DENORMALIZE_AFTER_PREDICT", denormalizeAfterPredict);
    pyodide.globals.set("PY_ALGORITHM_CONFIGS", pyAlgorithmConfigs);

    // Set up progress callback in Python
    pyodide.globals.set("progress_callback", pyodide.toPy((index, total, name) => {
      postStatus(`学習中... (${index}/${total}) ${name}`);
    }));

    const resultJson = await pyodide.runPythonAsync(`from compare_interpolation import fit_session, normalize_dataset, build_dataset_payload
import json

dataset_candidate = globals().get("PY_DATASET")
dataset_input = None
if dataset_candidate is not None:
    dataset_input = normalize_dataset(dataset_candidate)

progress_cb = globals().get("progress_callback")
normalize_flag = globals().get("PY_NORMALIZE") or False
denormalize_after_predict = globals().get("PY_DENORMALIZE_AFTER_PREDICT") or False
algo_configs_raw = globals().get("PY_ALGORITHM_CONFIGS")
if algo_configs_raw is not None:
    algo_configs = algo_configs_raw.to_py() if hasattr(algo_configs_raw, "to_py") else [c.to_py() if hasattr(c, "to_py") else c for c in algo_configs_raw]
else:
    algo_configs = None
session = fit_session(dataset=dataset_input, progress_callback=progress_cb, normalize=normalize_flag, algorithm_configs=algo_configs)
json.dumps({
    "status": "ok",
    "dataset": build_dataset_payload(
        session.dataset,
        session.dataset_source,
        session.axis_bounds,
        normalize=session.normalize,
        norm_means=session.norm_means,
        norm_stds=session.norm_stds,
        display_normalized=not denormalize_after_predict,
    ),
    "dataset_source": session.dataset_source,
}, ensure_ascii=False)`);
    return JSON.parse(resultJson);
  } finally {
    pyodide.globals.set("PY_DATASET", null);
    pyodide.globals.set("PY_NORMALIZE", null);
    pyodide.globals.set("PY_DENORMALIZE_AFTER_PREDICT", null);
    pyodide.globals.set("PY_ALGORITHM_CONFIGS", null);
    pyodide.globals.set("progress_callback", null);
    if (pyDataset && typeof pyDataset.destroy === "function") {
      pyDataset.destroy();
    }
    if (pyAlgorithmConfigs && typeof pyAlgorithmConfigs.destroy === "function") {
      pyAlgorithmConfigs.destroy();
    }
  }
}

async function handlePredictPlane(payload) {
  const pyodide = await ensurePyodide();
  const axis = payload && payload.sliceAxis ? payload.sliceAxis : "z";
  const value = payload && typeof payload.sliceValue === "number" ? payload.sliceValue : null;
  const denormalizeAfterPredict = payload && payload.denormalizeAfterPredict === true;
  const algorithmConfigs = payload && Array.isArray(payload.algorithmConfigs) ? payload.algorithmConfigs : null;
  let pyAlgorithmConfigs = null;
  if (algorithmConfigs !== null) {
    pyAlgorithmConfigs = pyodide.toPy(algorithmConfigs);
  }
  pyodide.globals.set("PY_SLICE_AXIS", axis);
  pyodide.globals.set("PY_SLICE_VALUE", value);
  pyodide.globals.set("PY_DENORMALIZE_AFTER_PREDICT", denormalizeAfterPredict);
  pyodide.globals.set("PY_ALGORITHM_CONFIGS", pyAlgorithmConfigs);
  pyodide.globals.set("predict_progress_callback", pyodide.toPy((index, total, name) => {
    postStatus(`結果を計算中... (${index + 1}/${total}) ${name}`);
  }));
  postDebug("handlePredictPlane", {
    axis,
    value,
    algorithmConfigs,
  });
  try {
    const resultJson = await pyodide.runPythonAsync(`import json
from compare_interpolation import predict_session_async
slice_axis = globals().get("PY_SLICE_AXIS") or "z"
slice_value = globals().get("PY_SLICE_VALUE")
cb = globals().get("predict_progress_callback")
denormalize_after_predict = globals().get("PY_DENORMALIZE_AFTER_PREDICT") or False
algo_configs_raw = globals().get("PY_ALGORITHM_CONFIGS")
if algo_configs_raw is not None:
    algo_configs = algo_configs_raw.to_py() if hasattr(algo_configs_raw, "to_py") else [c.to_py() if hasattr(c, "to_py") else c for c in algo_configs_raw]
else:
    algo_configs = None
print(f"DEBUG: slice_axis={slice_axis}, slice_value={slice_value}, algo_configs={algo_configs}")
json.dumps(await predict_session_async(slice_axis=slice_axis, slice_value=slice_value, progress_callback=cb, algorithm_configs=algo_configs, denormalize_after_predict=denormalize_after_predict), ensure_ascii=False)`);
    return JSON.parse(resultJson);
  } finally {
    pyodide.globals.set("PY_SLICE_AXIS", null);
    pyodide.globals.set("PY_SLICE_VALUE", null);
    pyodide.globals.set("PY_DENORMALIZE_AFTER_PREDICT", null);
    pyodide.globals.set("PY_ALGORITHM_CONFIGS", null);
    pyodide.globals.set("predict_progress_callback", null);
    if (pyAlgorithmConfigs && typeof pyAlgorithmConfigs.destroy === "function") {
      pyAlgorithmConfigs.destroy();
    }
  }
}

async function handlePredictLine(payload) {
  const pyodide = await ensurePyodide();
  const axis = payload && payload.lineAxis ? payload.lineAxis : "z";
  const fixed = payload && payload.fixedValues ? payload.fixedValues : null;
  const denormalizeAfterPredict = payload && payload.denormalizeAfterPredict === true;
  const algorithmConfigs = payload && Array.isArray(payload.algorithmConfigs) ? payload.algorithmConfigs : null;
  const resolution =
    payload && Number.isFinite(payload.lineResolution)
      ? Number(payload.lineResolution)
      : null;
  let pyFixed = null;
  let pyAlgorithmConfigs = null;
  if (fixed !== null) {
    pyFixed = pyodide.toPy(fixed);
  }
  if (algorithmConfigs !== null) {
    pyAlgorithmConfigs = pyodide.toPy(algorithmConfigs);
  }
  try {
    pyodide.globals.set("PY_LINE_AXIS", axis);
    pyodide.globals.set("PY_FIXED_AXES", pyFixed);
    pyodide.globals.set("PY_LINE_RESOLUTION", resolution);
    pyodide.globals.set("PY_DENORMALIZE_AFTER_PREDICT", denormalizeAfterPredict);
    pyodide.globals.set("PY_ALGORITHM_CONFIGS", pyAlgorithmConfigs);
    pyodide.globals.set("predict_progress_callback", pyodide.toPy((index, total, name) => {
      postStatus(`結果を計算中... (${index + 1}/${total}) ${name}`);
    }));
    const resultJson = await pyodide.runPythonAsync(`import json
from compare_interpolation import predict_line_session_async
line_axis = globals().get("PY_LINE_AXIS") or "z"
fixed_candidate = globals().get("PY_FIXED_AXES")
fixed_values = None
if fixed_candidate is not None:
    fixed_values = fixed_candidate
line_resolution = globals().get("PY_LINE_RESOLUTION")
cb = globals().get("predict_progress_callback")
denormalize_after_predict = globals().get("PY_DENORMALIZE_AFTER_PREDICT") or False
algo_configs_raw = globals().get("PY_ALGORITHM_CONFIGS")
if algo_configs_raw is not None:
    algo_configs = algo_configs_raw.to_py() if hasattr(algo_configs_raw, "to_py") else [c.to_py() if hasattr(c, "to_py") else c for c in algo_configs_raw]
else:
    algo_configs = None
json.dumps(await predict_line_session_async(varying_axis=line_axis, fixed_values=fixed_values, line_resolution=line_resolution, progress_callback=cb, algorithm_configs=algo_configs, denormalize_after_predict=denormalize_after_predict), ensure_ascii=False)`);
    return JSON.parse(resultJson);
  } finally {
    pyodide.globals.set("PY_LINE_AXIS", null);
    pyodide.globals.set("PY_FIXED_AXES", null);
    pyodide.globals.set("PY_LINE_RESOLUTION", null);
    pyodide.globals.set("PY_DENORMALIZE_AFTER_PREDICT", null);
    pyodide.globals.set("PY_ALGORITHM_CONFIGS", null);
    pyodide.globals.set("predict_progress_callback", null);
    if (pyFixed && typeof pyFixed.destroy === "function") {
      pyFixed.destroy();
    }
    if (pyAlgorithmConfigs && typeof pyAlgorithmConfigs.destroy === "function") {
      pyAlgorithmConfigs.destroy();
    }
  }
}

async function handleComputeMetrics() {
  const pyodide = await ensurePyodide();
  pyodide.globals.set("metrics_progress_callback", pyodide.toPy((index, total, name) => {
    postStatus(`滑らかさ計算中... (${index}/${total}) ${name}`);
  }));
  try {
    const resultJson = await pyodide.runPythonAsync(`import json
from compare_interpolation import compute_metrics_session
cb = globals().get("metrics_progress_callback")
json.dumps(compute_metrics_session(progress_callback=cb), ensure_ascii=False)`);
    return JSON.parse(resultJson);
  } finally {
    pyodide.globals.set("metrics_progress_callback", null);
  }
}

async function handleExportSession() {
  const pyodide = await ensurePyodide();
  const resultJson = await pyodide.runPythonAsync(`import json
from compare_interpolation import export_session
json.dumps(export_session(), ensure_ascii=False)`);
  return JSON.parse(resultJson);
}

async function handleImportSession(payload) {
  const pyodide = await ensurePyodide();
  const sessionData = payload && payload.sessionData ? payload.sessionData : null;
  const denormalizeAfterPredict = payload && payload.denormalizeAfterPredict === true;
  if (!sessionData) {
    throw new Error("sessionData is required for import");
  }
  let pySessionData = pyodide.toPy(sessionData);
  try {
    pyodide.globals.set("PY_SESSION_DATA", pySessionData);
    pyodide.globals.set("PY_DENORMALIZE_AFTER_PREDICT", denormalizeAfterPredict);
    const resultJson = await pyodide.runPythonAsync(`import json
from compare_interpolation import import_session, build_dataset_payload
session_data = globals().get("PY_SESSION_DATA")
denormalize_after_predict = globals().get("PY_DENORMALIZE_AFTER_PREDICT") or False
session = import_session(session_data)
json.dumps({
    "status": "ok",
    "dataset": build_dataset_payload(
        session.dataset,
        session.dataset_source,
        session.axis_bounds,
        normalize=session.normalize,
        norm_means=session.norm_means,
        norm_stds=session.norm_stds,
        display_normalized=not denormalize_after_predict,
    ),
    "dataset_source": session.dataset_source,
}, ensure_ascii=False)`);
    return JSON.parse(resultJson);
  } finally {
    pyodide.globals.set("PY_SESSION_DATA", null);
    pyodide.globals.set("PY_DENORMALIZE_AFTER_PREDICT", null);
    if (pySessionData && typeof pySessionData.destroy === "function") {
      pySessionData.destroy();
    }
  }
}

self.onmessage = async (event) => {
  const { id, type, payload } = event.data || {};
  if (!type) {
    return;
  }
  try {
    if (type === "init") {
      await ensurePyodide();
      self.postMessage({ id, result: { status: "ready" } });
      return;
    }
    let result;
    if (type === "fit") {
      result = await handleFit(payload || {});
    } else if (type === "predictPlane") {
      result = await handlePredictPlane(payload || {});
    } else if (type === "predictLine") {
      result = await handlePredictLine(payload || {});
    } else if (type === "computeMetrics") {
      result = await handleComputeMetrics();
    } else if (type === "exportSession") {
      result = await handleExportSession();
    } else if (type === "importSession") {
      result = await handleImportSession(payload || {});
    } else {
      throw new Error(`未知の操作: ${type}`);
    }
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({ id, error: serializeError(error) });
  }
};

importScripts("https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js");
