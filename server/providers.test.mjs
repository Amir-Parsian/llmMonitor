import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import {
  connection,
  normalizeTokens,
  alignTokens,
  generate,
  listModels,
} from "./providers.mjs";
import { createAppServer } from "./index.mjs";

async function withServer(handler, run) {
  const server = createServer(handler);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    return await run(`http://127.0.0.1:${server.address().port}`);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}
const reply = (res, data) => {
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(data));
};
async function body(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  return JSON.parse(Buffer.concat(chunks).toString());
}

test("connections allow local servers and reject arbitrary remote destinations", () => {
  assert.equal(connection().candidateLimit, 20);
  assert.equal(connection({ provider: "llamacpp" }).candidateLimit, 100);
  for (const baseUrl of [
    "https://example.com",
    "http://127.0.0.1.example.com",
    "file:///tmp/model",
    "http://user:password@localhost:8080",
    "http://localhost:8080/private",
  ])
    assert.throws(() => connection({ baseUrl }));
});

test("log probabilities retain original probability mass and the chosen token outside top N", () => {
  const [token] = normalizeTokens([
    {
      token: " rare",
      logprob: Math.log(0.01),
      top_logprobs: [
        { token: " likely", logprob: Math.log(0.6) },
        { token: " other", logprob: Math.log(0.2) },
      ],
    },
  ]);
  assert.equal(token.candidates.length, 3);
  assert.equal(token.candidates[2].selected, true);
  assert.ok(
    Math.abs(
      token.candidates.reduce((sum, c) => sum + c.probability, 0) - 0.81,
    ) < 1e-10,
  );
  assert.equal(
    normalizeTokens([{ token: "x", logprob: null }])[0].probability,
    null,
  );
  assert.equal(
    normalizeTokens([{ token: "x", logprob: 1 }])[0].probability,
    null,
  );
});

test("UTF-8 byte fragments and non-display tokens preserve exact output offsets", () => {
  const records = [
    { token: "Hi", bytes: [72, 105], logprob: 0 },
    { token: " fragment", bytes: [240, 159], logprob: 0 },
    { token: " fragment2", bytes: [152, 128], logprob: 0 },
    { token: "<eos>", bytes: [60, 101, 111, 115, 62], logprob: 0 },
  ];
  const tokens = alignTokens(normalizeTokens(records), "Hi😀");
  assert.deepEqual(
    tokens.map((t) => [t.start, t.end]),
    [
      [0, 2],
      [2, 4],
      [4, 6],
      [null, null],
    ],
  );
});

test("reasoning tokens cannot get attached to matching words in the visible answer", () => {
  const tokens = normalizeTokens([
    { token: "blue", logprob: 0 },
    { token: " thought ", logprob: 0 },
    { token: "The sky is blue.", logprob: 0 },
    { token: "<eos>", logprob: 0 },
  ]);
  assert.deepEqual(
    alignTokens(tokens, "The sky is blue.").map((t) => [t.start, t.end]),
    [
      [null, null],
      [null, null],
      [0, 16],
      [null, null],
    ],
  );
  assert.ok(
    alignTokens(tokens, "Different response").every((t) => t.start === null),
  );
});

test("distinct UTF-8 fragments with identical display text remain separate candidates", () => {
  const [token] = normalizeTokens([
    {
      token: "�",
      bytes: [240, 159],
      logprob: Math.log(0.5),
      top_logprobs: [
        { token: "�", bytes: [240, 159], logprob: Math.log(0.5) },
        { token: "�", bytes: [152, 128], logprob: Math.log(0.2) },
      ],
    },
  ]);
  assert.equal(token.candidates.length, 2);
  assert.deepEqual(
    token.candidates.map((c) => c.selected),
    [true, false],
  );
});

test("Ollama uses real logprobs, disables thinking, discovers text models and deduplicates runners", async () => {
  await withServer(
    async (req, res) => {
      if (req.url === "/api/tags")
        return reply(res, {
          models: [
            { name: "local", capabilities: ["completion"] },
            { name: "local", capabilities: ["completion"] },
            { name: "image-only", capabilities: ["image"] },
          ],
        });
      assert.equal(req.url, "/api/chat");
      const input = await body(req);
      assert.equal(input.logprobs, true);
      assert.equal(input.top_logprobs, 20);
      assert.equal(input.think, false);
      assert.equal(input.stream, false);
      assert.equal(input.options.seed, 42);
      reply(res, {
        message: { content: "Hi!" },
        logprobs: [
          {
            token: "Hi",
            logprob: Math.log(0.8),
            top_logprobs: [
              { token: "Hi", logprob: Math.log(0.8) },
              { token: "Hello", logprob: Math.log(0.1) },
            ],
          },
          { token: "!", logprob: Math.log(0.9) },
        ],
        eval_count: 2,
        eval_duration: 1e9,
        done_reason: "stop",
      });
    },
    async (baseUrl) => {
      assert.deepEqual(
        (await listModels({ baseUrl })).models.map((m) => m.name),
        ["local"],
      );
      const result = await generate({
        baseUrl,
        model: "local",
        prompt: "Greet me",
      });
      assert.equal(result.text, "Hi!");
      assert.equal(result.tokens[0].candidates.length, 2);
      assert.equal(result.warning, null);
      assert.equal(result.tokensPerSecond, 2);
      assert.deepEqual(
        result.tokens.map((t) => t.start),
        [0, 2],
      );
      assert.equal(result.demo, false);
    },
  );
});

test("missing logprobs produce an honest warning instead of invented candidates", async () => {
  await withServer(
    (req, res) => reply(res, { message: { content: "Plain output" } }),
    async (baseUrl) => {
      const result = await generate({
        baseUrl,
        model: "old-model",
        prompt: "Hello",
      });
      assert.equal(result.tokens.length, 0);
      assert.match(result.warning, /did not return/);
      assert.equal(result.text, "Plain output");
    },
  );
});

test("llama.cpp applies the model chat template and requests 100 native candidates", async () => {
  await withServer(
    async (req, res) => {
      const input = await body(req);
      if (req.url === "/apply-template") {
        assert.equal(input.messages[0].content, "Hello");
        return reply(res, { prompt: "<user>Hello<assistant>" });
      }
      assert.equal(req.url, "/completion");
      assert.equal(input.prompt, "<user>Hello<assistant>");
      assert.equal(input.n_probs, 100);
      assert.equal(input.post_sampling_probs, false);
      const top_logprobs = Array.from({ length: 100 }, (_, i) => ({
        id: i,
        token: `candidate${i}`,
        logprob: Math.log(1 / 200),
      }));
      reply(res, {
        content: "candidate0",
        probs: [
          {
            id: 0,
            token: "candidate0",
            logprob: Math.log(1 / 200),
            top_logprobs,
          },
        ],
        timings: { predicted_per_second: 18 },
        stop_type: "limit",
      });
    },
    async (baseUrl) => {
      const result = await generate({
        baseUrl,
        provider: "llamacpp",
        model: "gguf",
        prompt: "Hello",
        topN: 100,
      });
      assert.equal(result.tokens[0].candidates.length, 100);
      assert.equal(result.tokensPerSecond, 18);
      assert.equal(result.finishReason, "limit");
    },
  );
});

test("invalid controls and exceeding a provider limit are rejected before inference", async () => {
  for (const options of [
    { topN: 100 },
    { temperature: -1 },
    { temperature: null },
    { maxTokens: 0 },
    { seed: -1 },
    { prompt: "" },
    { model: "" },
  ]) {
    const input = { prompt: "Hello", model: "local", ...options };
    // null is deliberately treated as default, just like an omitted control.
    if (options.temperature === null) continue;
    await assert.rejects(generate(input), (error) => error.status === 400);
  }
});

test("API rejects cross-origin generation, malformed JSON and unsupported methods", async () => {
  const server = createAppServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const baseUrl = `http://127.0.0.1:${server.address().port}`;
  try {
    assert.equal(
      (
        await fetch(baseUrl + "/api/generate", {
          method: "POST",
          headers: {
            Origin: "https://remote.example",
            "Content-Type": "application/json",
          },
          body: "{}",
        })
      ).status,
      403,
    );
    assert.equal(
      (
        await fetch(baseUrl + "/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{broken",
        })
      ).status,
      400,
    );
    assert.equal(
      (await fetch(baseUrl + "/api/generate", { method: "POST", body: "{}" }))
        .status,
      415,
    );
    assert.equal((await fetch(baseUrl + "/api/no-route")).status, 404);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});
