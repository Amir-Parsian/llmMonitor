import { spawn } from "node:child_process";
import { access, open, readFile, readdir, stat } from "node:fs/promises";
import { constants } from "node:fs";
import { homedir, totalmem } from "node:os";
import { resolve, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { ApiError } from "./providers.mjs";
import { StringDecoder } from "node:string_decoder";

const root = fileURLToPath(new URL("..", import.meta.url));
export const TRACER_REVISION = "9d1878a4354e743529fbe09e7350a957f5e39e9a";
const executable = resolve(root, ".runtime/build/bin/tokenscope-tracer");
const modelRoot = () =>
  process.env.OLLAMA_MODELS || resolve(homedir(), ".ollama/models");
let busy = false;

async function isGGUF(path) {
  let file;
  try {
    file = await open(path, "r");
    const bytes = Buffer.alloc(4);
    await file.read(bytes, 0, 4, 0);
    return bytes.toString() === "GGUF";
  } catch {
    return false;
  } finally {
    await file?.close();
  }
}

export async function discoverTraceModels(directory = modelRoot()) {
  const manifests = resolve(directory, "manifests");
  const found = [];
  async function walk(path, depth = 0) {
    if (depth > 8) return;
    let entries;
    try {
      entries = await readdir(path, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const child = resolve(path, entry.name);
      if (entry.isDirectory()) {
        await walk(child, depth + 1);
        continue;
      }
      if (!entry.isFile()) continue;
      const parts = relative(manifests, child).split(sep);
      if (parts.length < 4) continue;
      let manifest;
      try {
        if ((await stat(child)).size > 1_000_000) continue;
        manifest = JSON.parse(await readFile(child, "utf8"));
      } catch {
        continue;
      }
      // Recent Ollama manifests can contain both an Ollama-specific layout and
      // an already-converted llama.cpp runner variant. Prefer that existing variant.
      for (const descriptor of manifest.layers || []) {
        if (
          !descriptor.mediaType?.includes("manifest.v2+json") ||
          !/^sha256:[a-f0-9]{64}$/.test(descriptor.digest) ||
          descriptor.size > 1_000_000
        )
          continue;
        try {
          const variant = JSON.parse(
            await readFile(
              resolve(directory, "blobs", descriptor.digest.replace(":", "-")),
              "utf8",
            ),
          );
          if (variant.runner === "llamacpp") {
            manifest = variant;
            break;
          }
        } catch {}
      }
      const layer = manifest.layers?.find(
        (l) =>
          l.mediaType === "application/vnd.ollama.image.model" &&
          /^sha256:[a-f0-9]{64}$/.test(l.digest),
      );
      if (!layer) continue;
      const modelPath = resolve(
        directory,
        "blobs",
        layer.digest.replace(":", "-"),
      );
      if (!(await isGGUF(modelPath))) continue;
      const [registry, namespace, ...rest] = parts;
      const tag = rest.pop();
      const name = `${registry === "registry.ollama.ai" ? "" : registry + "/"}${namespace === "library" ? "" : namespace + "/"}${rest.join("/")}:${tag}`;
      if (name.startsWith("llamacpp:")) continue;
      found.push({
        name,
        size: layer.size || 0,
        digest: layer.digest,
        modelPath,
        fitsMemory: (layer.size || 0) < totalmem() * 0.7,
      });
    }
  }
  await walk(manifests);
  return found.sort((a, b) => a.name.localeCompare(b.name));
}

export async function tracingStatus() {
  let ready = false;
  try {
    await access(executable, constants.X_OK);
    const build = JSON.parse(
      await readFile(resolve(root, ".runtime/tracer-build.json"), "utf8"),
    );
    ready = build.revision === TRACER_REVISION;
  } catch {}
  const models = (await discoverTraceModels()).map(
    ({ modelPath, ...model }) => model,
  );
  return {
    ready,
    busy,
    models,
    engine: "llama.cpp",
    revision: TRACER_REVISION,
    limits: { maxTokens: 8, maxPromptTokens: 256, maxPromptCharacters: 1200 },
    setupCommand: "npm run setup:tracer",
  };
}

export function validateTraceRequest(input) {
  if (
    !input ||
    typeof input !== "object" ||
    typeof input.prompt !== "string" ||
    !input.prompt.trim() ||
    input.prompt.length > 1200
  )
    throw new ApiError("Enter a question between 1 and 1,200 characters.");
  if (
    typeof input.model !== "string" ||
    !input.model ||
    input.model.length > 256
  )
    throw new ApiError("Choose a downloaded GGUF model.");
  const maxTokens = input.maxTokens ?? 4;
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 8)
    throw new ApiError("Trace between 1 and 8 generation steps.");
  return { model: input.model, prompt: input.prompt, maxTokens };
}

export async function traceModel(input, { signal, onEvent } = {}) {
  const request = validateTraceRequest(input);
  if (busy)
    throw new ApiError(
      "A layer trace is already running. Wait for it to finish or stop that trace.",
      409,
    );
  const status = await tracingStatus();
  if (!status.ready)
    throw new ApiError(
      "The tracing engine is not built yet. Run npm run setup:tracer in the project folder.",
      503,
    );
  const model = (await discoverTraceModels()).find(
    (m) => m.name === request.model,
  );
  if (!model)
    throw new ApiError(
      "This model is not an installed local GGUF model. Refresh the model list.",
    );
  if (!model.fitsMemory)
    throw new ApiError(
      "This model exceeds the tracing memory budget on this machine. Choose a smaller model.",
    );
  if (busy)
    throw new ApiError(
      "Another layer trace has started. Wait for it to finish.",
      409,
    );
  if (signal?.aborted) throw new ApiError("Trace stopped.", 499);
  busy = true;
  try {
    return await new Promise((done, fail) => {
      const child = spawn(executable, [], {
        stdio: ["pipe", "pipe", "pipe"],
        cwd: root,
      });
      const decoder = new StringDecoder("utf8");
      let pending = "",
        stderr = "",
        result = null,
        engineError = null,
        size = 0,
        timedOut = false;
      const stop = () => child.kill("SIGTERM");
      const timer = setTimeout(() => {
        timedOut = true;
        stop();
      }, 300_000);
      signal?.addEventListener("abort", stop, { once: true });
      if (signal?.aborted) stop();
      const clean = () => {
        clearTimeout(timer);
        signal?.removeEventListener("abort", stop);
      };
      child.stdin.on("error", () => {});
      child.stderr.on("data", (chunk) => {
        stderr = (stderr + chunk.toString()).slice(-6000);
      });
      child.stdout.on("data", (chunk) => {
        size += chunk.length;
        if (size > 64 * 1024 * 1024) {
          engineError = "The trace exceeded its 64 MB output limit.";
          stop();
          return;
        }
        pending += decoder.write(chunk);
        let end;
        while ((end = pending.indexOf("\n")) >= 0) {
          const line = pending.slice(0, end);
          pending = pending.slice(end + 1);
          if (!line.trim()) continue;
          try {
            const event = JSON.parse(line);
            if (event.type === "result") result = event.data;
            else if (event.type === "error") engineError = event.error;
            else if (event.type === "progress") onEvent?.(event);
          } catch {
            engineError = "The tracing engine returned an unreadable event.";
            stop();
          }
        }
      });
      child.on("error", (error) => {
        clean();
        fail(
          new ApiError(
            `Could not launch the tracing engine: ${error.message}`,
            503,
          ),
        );
      });
      child.on("close", (code) => {
        clean();
        if (signal?.aborted) return fail(new ApiError("Trace stopped.", 499));
        if (timedOut)
          return fail(
            new ApiError(
              "The trace exceeded five minutes. Try a smaller model or a shorter prompt.",
              504,
            ),
          );
        if (code !== 0 || !result)
          return fail(
            new ApiError(
              engineError ||
                `The tracing engine failed. ${stderr.trim().slice(-800)}`,
              502,
            ),
          );
        const complete = {
          ...result,
          id: crypto.randomUUID(),
          createdAt: new Date().toISOString(),
          model: request.model,
          prompt: request.prompt,
          modelDigest: model.digest,
          requestedSteps: request.maxTokens,
          notice:
            "Measured in a separate instrumented llama.cpp run using your Ollama-downloaded GGUF weights. This is not a trace of an Ollama API response.",
        };
        onEvent?.({ type: "result", data: complete });
        done(complete);
      });
      child.stdin.end(
        JSON.stringify({ ...request, modelPath: model.modelPath }),
      );
    });
  } finally {
    busy = false;
  }
}
