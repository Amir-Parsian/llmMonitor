import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { discoverTraceModels, validateTraceRequest } from "./tracing.mjs";
import { createAppServer } from "./index.mjs";

test("trace requests have bounded prompt length and generation steps", () => {
  assert.equal(
    validateTraceRequest({ model: "gemma4:e4B", prompt: "Hello" }).maxTokens,
    4,
  );
  for (const input of [
    null,
    {},
    { model: "gemma4", prompt: "" },
    { model: "gemma4", prompt: "x".repeat(1201) },
    { model: "gemma4", prompt: "Hello", maxTokens: 0 },
    { model: "gemma4", prompt: "Hello", maxTokens: 9 },
    { model: "gemma4", prompt: "Hello", maxTokens: 1.5 },
  ])
    assert.throws(
      () => validateTraceRequest(input),
      (error) => error.status === 400,
    );
});

test("tracing routes enforce local access, validate requests, and keep model paths private", async () => {
  const server = createAppServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    const response = await fetch(base + "/api/tracing/status");
    assert.equal(response.status, 200);
    const status = await response.json();
    assert.equal(typeof status.ready, "boolean");
    assert.equal(status.limits.maxTokens, 8);
    assert.ok(status.models.every((model) => !("modelPath" in model)));
    assert.equal(
      (
        await fetch(base + "/api/tracing/status", {
          headers: { Origin: "https://remote.example" },
        })
      ).status,
      403,
    );
    for (const body of [
      "{broken",
      "{}",
      JSON.stringify({ model: "any", prompt: "hello", maxTokens: 9 }),
    ]) {
      const response = await fetch(base + "/api/tracing/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      assert.equal(response.status, 400);
      assert.match(response.headers.get("content-type"), /application\/json/);
    }
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("discovery reuses installed llama.cpp variants, rejects invalid blobs, and keeps namespace names", async () => {
  const directory = await mkdtemp(resolve(tmpdir(), "tokenscope-models-"));
  try {
    await mkdir(resolve(directory, "blobs"), { recursive: true });
    const modelDigest = "a".repeat(64),
      variantDigest = "b".repeat(64),
      originalDigest = "c".repeat(64);
    await writeFile(
      resolve(directory, "blobs", "sha256-" + modelDigest),
      "GGUFtest-fixture",
    );
    await writeFile(
      resolve(directory, "blobs", "sha256-" + originalDigest),
      "not-a-GGUF",
    );
    const variant = {
      runner: "llamacpp",
      layers: [
        {
          mediaType: "application/vnd.ollama.image.model",
          digest: "sha256:" + modelDigest,
          size: 16,
        },
      ],
    };
    await writeFile(
      resolve(directory, "blobs", "sha256-" + variantDigest),
      JSON.stringify(variant),
    );
    const manifest = {
      runner: "ggml",
      layers: [
        {
          mediaType: "application/vnd.ollama.image.model",
          digest: "sha256:" + originalDigest,
        },
        {
          mediaType: "application/vnd.docker.distribution.manifest.v2+json",
          digest: "sha256:" + variantDigest,
          size: 200,
        },
      ],
    };
    for (const [path, data] of [
      ["library/gemma4/e4B", manifest],
      ["community/custom/latest", variant],
      [
        "library/bad/latest",
        {
          layers: [
            {
              mediaType: "application/vnd.ollama.image.model",
              digest: "sha256:../../etc/passwd",
            },
          ],
        },
      ],
      [
        "library/not-gguf/latest",
        {
          layers: [
            {
              mediaType: "application/vnd.ollama.image.model",
              digest: "sha256:" + originalDigest,
            },
          ],
        },
      ],
    ]) {
      const target = resolve(directory, "manifests/registry.ollama.ai", path);
      await mkdir(resolve(target, ".."), { recursive: true });
      await writeFile(target, JSON.stringify(data));
    }
    const models = await discoverTraceModels(directory);
    assert.deepEqual(
      models.map((model) => model.name),
      ["community/custom:latest", "gemma4:e4B"],
    );
    assert.ok(
      models.every((model) => model.digest === "sha256:" + modelDigest),
    );
    assert.ok(
      models.every(
        (model) =>
          model.modelPath ===
          resolve(directory, "blobs", "sha256-" + modelDigest),
      ),
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
