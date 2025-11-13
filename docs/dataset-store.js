(function () {
  const STORAGE_KEY = "sdi.dataset.selection.v1";

  function isValidDataset(dataset) {
    if (!dataset || typeof dataset !== "object") {
      return false;
    }
    if (!Array.isArray(dataset.points) || !Array.isArray(dataset.values)) {
      return false;
    }
    if (dataset.points.length !== dataset.values.length) {
      return false;
    }
    return dataset.points.length > 0;
  }

  function save(selection) {
    try {
      if (!selection || selection.source === "dummy") {
        localStorage.removeItem(STORAGE_KEY);
        return;
      }
      const dataset = selection.dataset;
      if (!isValidDataset(dataset)) {
        localStorage.removeItem(STORAGE_KEY);
        return;
      }
      const payload = {
        version: 1,
        name: selection.name || "インポートデータ",
        dataset,
        timestamp: Date.now(),
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (error) {
      console.warn("データセットの保存に失敗しました", error);
    }
  }

  function load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return null;
      }
      const payload = JSON.parse(raw);
      if (!payload || typeof payload !== "object") {
        return null;
      }
      if (!isValidDataset(payload.dataset)) {
        localStorage.removeItem(STORAGE_KEY);
        return null;
      }
      return {
        name: typeof payload.name === "string" ? payload.name : "インポートデータ",
        dataset: payload.dataset,
      };
    } catch (error) {
      console.warn("保存済みデータセットの読み込みに失敗しました", error);
      return null;
    }
  }

  function clear() {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.warn("データセットのクリアに失敗しました", error);
    }
  }

  window.datasetStore = {
    save,
    load,
    clear,
  };
})();
