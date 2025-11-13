const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/";
let pyodideReadyPromise = null;
let moduleLoaded = false;

function postStatus(message) {
  self.postMessage({ type: "status", message });
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
  let pyDataset = null;
  if (dataset !== null) {
    pyDataset = pyodide.toPy(dataset);
  }
  try {
    pyodide.globals.set("PY_DATASET", pyDataset);
    await pyodide.runPythonAsync(`from compare_interpolation import fit_session, normalize_dataset
import json

dataset_candidate = globals().get("PY_DATASET")
dataset_input = None
if dataset_candidate is not None:
    dataset_input = normalize_dataset(dataset_candidate)
fit_session(dataset=dataset_input)`);
    return { status: "ok" };
  } finally {
    pyodide.globals.set("PY_DATASET", null);
    if (pyDataset && typeof pyDataset.destroy === "function") {
      pyDataset.destroy();
    }
  }
}

async function handlePredictPlane(payload) {
  const pyodide = await ensurePyodide();
  const axis = payload && payload.sliceAxis ? payload.sliceAxis : "z";
  const value = payload && typeof payload.sliceValue === "number" ? payload.sliceValue : null;
  pyodide.globals.set("PY_SLICE_AXIS", axis);
  pyodide.globals.set("PY_SLICE_VALUE", value);
  try {
    const resultJson = await pyodide.runPythonAsync(`import json
from compare_interpolation import predict_session
slice_axis = globals().get("PY_SLICE_AXIS") or "z"
slice_value = globals().get("PY_SLICE_VALUE")
json.dumps(predict_session(slice_axis=slice_axis, slice_value=slice_value), ensure_ascii=False)`);
    return JSON.parse(resultJson);
  } finally {
    pyodide.globals.set("PY_SLICE_AXIS", null);
    pyodide.globals.set("PY_SLICE_VALUE", null);
  }
}

async function handlePredictLine(payload) {
  const pyodide = await ensurePyodide();
  const axis = payload && payload.lineAxis ? payload.lineAxis : "z";
  const fixed = payload && payload.fixedValues ? payload.fixedValues : null;
  let pyFixed = null;
  if (fixed !== null) {
    pyFixed = pyodide.toPy(fixed);
  }
  try {
    pyodide.globals.set("PY_LINE_AXIS", axis);
    pyodide.globals.set("PY_FIXED_AXES", pyFixed);
    const resultJson = await pyodide.runPythonAsync(`import json
from compare_interpolation import predict_line_session
line_axis = globals().get("PY_LINE_AXIS") or "z"
fixed_candidate = globals().get("PY_FIXED_AXES")
fixed_values = None
if fixed_candidate is not None:
    fixed_values = fixed_candidate
json.dumps(predict_line_session(varying_axis=line_axis, fixed_values=fixed_values), ensure_ascii=False)`);
    return JSON.parse(resultJson);
  } finally {
    pyodide.globals.set("PY_LINE_AXIS", null);
    pyodide.globals.set("PY_FIXED_AXES", null);
    if (pyFixed && typeof pyFixed.destroy === "function") {
      pyFixed.destroy();
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
    } else {
      throw new Error(`未知の操作: ${type}`);
    }
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({ id, error: serializeError(error) });
  }
};

importScripts("https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js");
