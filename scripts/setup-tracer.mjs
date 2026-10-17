import { spawn } from "node:child_process";
import { mkdir, access, writeFile, rename } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { availableParallelism } from "node:os";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const runtime = resolve(root, ".runtime");
const revision = "9d1878a4354e743529fbe09e7350a957f5e39e9a";
async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}
async function run(command, args, options = {}) {
  await new Promise((done, fail) => {
    const child = spawn(command, args, {
      cwd: root,
      stdio: "inherit",
      ...options,
    });
    child.on("error", fail);
    child.on("exit", (code) =>
      code === 0
        ? done()
        : fail(new Error(`${command} exited with code ${code}.`)),
    );
  });
}
async function hasCommand(command) {
  return new Promise((resolve) => {
    const child = spawn(command, ["--version"], { stdio: "ignore" });
    child.on("error", () => resolve(false));
    child.on("exit", (code) => resolve(code === 0));
  });
}
await mkdir(runtime, { recursive: true });
if (!(await exists(resolve(runtime, "llama.cpp/CMakeLists.txt")))) {
  console.log(`Downloading official llama.cpp source at ${revision}…`);
  const response = await fetch(
    `https://codeload.github.com/ggml-org/llama.cpp/tar.gz/${revision}`,
  );
  if (!response.ok)
    throw new Error(`Source download failed: HTTP ${response.status}`);
  await writeFile(
    resolve(runtime, "llama.cpp.tar.gz"),
    Buffer.from(await response.arrayBuffer()),
  );
  await run("tar", [
    "-xzf",
    resolve(runtime, "llama.cpp.tar.gz"),
    "-C",
    runtime,
  ]);
  await rename(
    resolve(runtime, `llama.cpp-${revision}`),
    resolve(runtime, "llama.cpp"),
  );
}
let cmake = "cmake";
if (!(await hasCommand(cmake))) {
  cmake = resolve(runtime, "tools/cmake/data/bin/cmake");
  if (!(await exists(cmake))) {
    console.log("Installing CMake into the project runtime folder…");
    await run("python3", [
      "-m",
      "pip",
      "install",
      "--target",
      resolve(runtime, "tools"),
      "cmake==4.4.4",
      "--disable-pip-version-check",
    ]);
  }
}
await run(cmake, [
  "-S",
  resolve(root, "native"),
  "-B",
  resolve(runtime, "build"),
  "-DCMAKE_BUILD_TYPE=Release",
]);
await run(cmake, [
  "--build",
  resolve(runtime, "build"),
  "--target",
  "tokenscope-tracer",
  "--parallel",
  String(Math.min(6, availableParallelism())),
]);
await writeFile(
  resolve(runtime, "tracer-build.json"),
  JSON.stringify({ revision, builtAt: new Date().toISOString() }, null, 2) +
    "\n",
);
console.log("\nLayer tracing is ready. Open Layer explorer in TokenScope.");
