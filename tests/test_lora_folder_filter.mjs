import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

// Run the actual extension against the same lifecycle callbacks ComfyUI uses.
let extension;
const source = readFileSync(new URL("../web/lora_folder_filter.js", import.meta.url), "utf8")
  .replace(/^import .*;\s*/m, "");
vm.runInNewContext(source, {
  app: { registerExtension(value) { extension = value; } },
  console,
  Event,
});

function createNode(type, names, selection = names[0]) {
  const lora = { name: "lora_name", value: selection, options: { values: [...names] } };
  const filter = { name: "category_filter", value: "All" };
  const node = { comfyClass: type, widgets: [filter, lora], graph: { setDirtyCanvas() {} } };
  extension.nodeCreated(node);
  return { node, lora, filter };
}

for (const type of ["Krea2Merge_LoadLoRA", "Krea2Merge_ApplyLoRA"]) {
  test(`${type}: preserves a missing file after restoring a workflow`, () => {
    const { node, lora, filter } = createNode(type, ["chars/a.safetensors", "style/b.sft"]);
    lora.value = "chars/deleted.safetensors";
    filter.value = "chars";
    extension.loadedGraphNode(node);
    assert.equal(lora.value, "chars/deleted.safetensors");
    assert.deepEqual([...lora.options.values], ["chars/a.safetensors"]);
    filter.value = "All";
    filter.callback("All");
    assert.equal(lora.value, "chars/deleted.safetensors");
  });

  test(`${type}: resets an existing selection excluded by a chosen folder`, () => {
    const { lora, filter } = createNode(type, ["chars/a.safetensors", "style/b.sft"]);
    filter.value = "style";
    filter.callback("style");
    assert.equal(lora.value, "style/b.sft");
    assert.deepEqual([...lora.options.values], ["style/b.sft"]);
    filter.value = "All";
    filter.callback("All");
    assert.equal(lora.options.values.length, 2);
    assert.equal(lora.value, "style/b.sft");
  });

  test(`${type}: preserves a missing file with an empty list or missing folder`, () => {
    const { node, lora, filter } = createNode(type, [], "deleted.safetensors");
    filter.value = "removed-folder";
    extension.loadedGraphNode(node);
    assert.equal(lora.value, "deleted.safetensors");
    assert.equal(lora.options.values.length, 0);
  });

  test(`${type}: restores a saved Windows folder containing regex characters`, () => {
    const names = ["chars/a.sft", "styles[1]\\nested\\b.safetensors", "styles1/c.sft"];
    const { node, lora, filter } = createNode(type, names);
    filter.value = "styles[1]\\nested";
    lora.value = names[1];
    extension.loadedGraphNode(node);
    assert.deepEqual([...lora.options.values], [names[1]]);
    assert.equal(lora.value, names[1]);
  });
}
