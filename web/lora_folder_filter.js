import { app } from "../../../scripts/app.js";

const FILTER_NODE_IDS = new Set([
  "Krea2Merge_LoadLoRA",
  "Krea2Merge_ApplyLoRA",
]);

app.registerExtension({
  name: "Krea2Merge.LoraFolderFilter",

  nodeCreated(node) {
    if (!FILTER_NODE_IDS.has(node.comfyClass)) return;

    const loraNamesWidget = node.widgets?.find((widget) => widget.name === "lora_name");
    const categoryFilterWidget = node.widgets?.find(
      (widget) => widget.name === "category_filter",
    );
    const availableNames = loraNamesWidget?.options?.values;

    if (!loraNamesWidget || !categoryFilterWidget || !Array.isArray(availableNames)) {
      console.warn("[Krea2 Merge] folder filter widgets are unavailable");
      return;
    }

    const fullLoraList = [...availableNames];

    function updateLoraList() {
      const filter = categoryFilterWidget.value;
      let filteredList = fullLoraList;

      if (filter !== "All") {
        const escapedFilter = filter.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const folderPattern = new RegExp(`^${escapedFilter}[\\\\/]`);
        filteredList = fullLoraList.filter((name) => folderPattern.test(name));
      }

      loraNamesWidget.options.values = filteredList;
      if (!filteredList.includes(loraNamesWidget.value)) {
        loraNamesWidget.value = filteredList[0] ?? "";
      }

      if (loraNamesWidget.widget?.inputEl) {
        loraNamesWidget.widget.inputEl.dispatchEvent(new Event("input"));
      }
      node.graph?.setDirtyCanvas(true, true);
    }

    const previousCallback = categoryFilterWidget.callback ?? (() => {});
    categoryFilterWidget.callback = (value, ...args) => {
      previousCallback(value, ...args);
      updateLoraList();
    };

    updateLoraList();
  },
});
