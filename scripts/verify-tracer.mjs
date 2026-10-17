import assert from "node:assert/strict";
import { writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { traceModel, tracingStatus } from "../server/tracing.mjs";
const status = await tracingStatus();
const model =
  process.argv[2] ||
  status.models.find((model) => model.name === "gemma4:e4B")?.name ||
  status.models.find((model) => model.fitsMemory)?.name;
assert.ok(status.ready, "Build the engine first with npm run setup:tracer.");
assert.ok(model, "Download a compatible GGUF model with Ollama first.");
const trace = await traceModel(
  { model, prompt: "Say hello in one short sentence.", maxTokens: 2 },
  {
    onEvent: (event) => {
      if (event.type === "progress") console.log(event.stage);
    },
  },
);
assert.ok(trace.steps.length > 0);
if (
  trace.templateSource === "Gemma 4 E2B/E4B documented single-turn text format"
) {
  assert.equal(
    trace.contextTokens
      .slice(0, trace.promptTokenCount)
      .map((token) => token.text)
      .join(""),
    "<bos><|turn>user\nSay hello in one short sentence.<turn|>\n<|turn>model\n",
    "The edge Gemma template must preserve its full non-thinking assistant prefix.",
  );
}
for (const step of trace.steps) {
  assert.equal(step.layers.length, trace.layerCount);
  for (const layer of step.layers) {
    assert.equal(layer.activation.values.length, trace.hiddenSize);
    assert.ok(layer.activation.values.every(Number.isFinite));
    for (const row of layer.attention) {
      assert.equal(row.length, step.contextLength);
      assert.ok(
        row.every(
          (value) => Number.isFinite(value) && value >= 0 && value <= 1,
        ),
      );
      assert.ok(
        Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) < 0.001,
        "Attention mass should sum to one for this model.",
      );
    }
  }
  if (step.lens.available)
    assert.ok(
      step.lens.finalReadoutMaxLogitError < 0.05,
      "The final diagnostic readout must match the actual output head.",
    );
}
const root = fileURLToPath(new URL("..", import.meta.url));
await mkdir(resolve(root, ".runtime"), { recursive: true });
await writeFile(
  resolve(root, ".runtime/verified-trace.json"),
  JSON.stringify(trace),
);
console.log(
  JSON.stringify(
    {
      model,
      blocks: trace.layerCount,
      width: trace.hiddenSize,
      steps: trace.steps.length,
      text: trace.text,
      seconds: trace.duration,
      verified: true,
    },
    null,
    2,
  ),
);
