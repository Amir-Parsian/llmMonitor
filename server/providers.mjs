export class ApiError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.status = status;
  }
}

export function connection(input = {}) {
  const provider = input.provider ?? "ollama";
  if (!["ollama", "llamacpp"].includes(provider))
    throw new ApiError("Choose Ollama or llama.cpp.");
  let url;
  try {
    url = new URL(
      input.baseUrl ||
        (provider === "ollama"
          ? "http://127.0.0.1:11434"
          : "http://127.0.0.1:8080"),
    );
  } catch {
    throw new ApiError("Enter a valid local server URL.");
  }
  if (
    !["http:", "https:"].includes(url.protocol) ||
    !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    !["", "/"].includes(url.pathname)
  ) {
    throw new ApiError(
      "Use a local server address, such as http://127.0.0.1:11434.",
    );
  }
  return {
    provider,
    baseUrl: url.origin,
    candidateLimit: provider === "ollama" ? 20 : 100,
  };
}

async function upstream(baseUrl, path, { signal, body } = {}) {
  let response;
  try {
    response = await fetch(baseUrl + path, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
      signal: signal
        ? AbortSignal.any([signal, AbortSignal.timeout(300_000)])
        : AbortSignal.timeout(8_000),
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new ApiError(
      `Cannot reach ${baseUrl}. Start your local server and reconnect. ${error.name === "TimeoutError" ? "The request timed out." : ""}`,
      502,
    );
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError(
      "The model server returned an unreadable response.",
      502,
    );
  }
  if (!response.ok || data.error) {
    const detail =
      typeof data.error === "string"
        ? data.error
        : data.error?.message || response.statusText;
    throw new ApiError(`Model server: ${detail}`, 502);
  }
  return data;
}

export async function listModels(input) {
  const config = connection(input);
  const data = await upstream(
    config.baseUrl,
    config.provider === "ollama" ? "/api/tags" : "/v1/models",
  );
  const source =
    config.provider === "ollama"
      ? (data.models || []).filter(
          (m) => !m.capabilities || m.capabilities.includes("completion"),
        )
      : data.data || [];
  const models = [
    ...new Map(
      source.map((m) => [
        m.name || m.id,
        {
          name: m.name || m.id,
          size: m.details?.parameter_size || "",
          quantization: m.details?.quantization_level || "",
        },
      ]),
    ).values(),
  ];
  return { ...config, models };
}

function probability(record) {
  if (
    typeof record.prob === "number" &&
    Number.isFinite(record.prob) &&
    record.prob >= 0 &&
    record.prob <= 1
  )
    return record.prob;
  if (typeof record.logprob === "number" && record.logprob <= 0)
    return Math.exp(record.logprob);
  return null;
}

function candidate(record) {
  const p = probability(record);
  return {
    text: record.token ?? record.tok_str ?? record.content ?? "",
    probability: p,
    logprob:
      Number.isFinite(record.logprob) && record.logprob <= 0
        ? record.logprob
        : p
          ? Math.log(p)
          : null,
    bytes: Array.isArray(record.bytes)
      ? record.bytes
      : Array.from(
          Buffer.from(record.token ?? record.tok_str ?? record.content ?? ""),
        ),
    ...(record.id !== undefined ? { id: record.id } : {}),
  };
}

export function normalizeTokens(records = []) {
  return records.map((record, index) => {
    const chosen = candidate(record);
    const alternatives = (
      record.top_logprobs ||
      record.top_probs ||
      record.probs ||
      []
    )
      .map(candidate)
      .filter((c) => c.probability !== null);
    const key = (c) =>
      c.id !== undefined ? `id:${c.id}` : `bytes:${c.bytes.join(",")}`;
    const unique = [...new Map(alternatives.map((c) => [key(c), c])).values()];
    const reportedChosen = unique.find((c) => key(c) === key(chosen));
    if (chosen.probability === null && reportedChosen) {
      chosen.probability = reportedChosen.probability;
      chosen.logprob = reportedChosen.logprob;
    }
    if (
      chosen.probability !== null &&
      !unique.some((c) => key(c) === key(chosen))
    )
      unique.push(chosen);
    unique.sort((a, b) => b.probability - a.probability);
    return {
      index,
      ...chosen,
      bytes: Array.isArray(record.bytes)
        ? record.bytes
        : Array.from(Buffer.from(chosen.text)),
      candidates: unique.map((c, i) => ({
        ...c,
        rank: i + 1,
        selected: key(c) === key(chosen),
      })),
    };
  });
}

// Byte offsets preserve the exact response even when a UTF-8 character spans tokens.
// Stop and reasoning tokens remain inspectable, without being attached to unrelated words.
export function alignTokens(tokens, text) {
  const output = Buffer.from(text);
  const source = Buffer.concat(tokens.map((token) => Buffer.from(token.bytes)));
  const responseOffset = output.length ? source.lastIndexOf(output) : -1;
  let cursor = 0;
  return tokens.map((token) => {
    const start = cursor - responseOffset;
    cursor += token.bytes.length;
    const end = cursor - responseOffset;
    if (
      responseOffset < 0 ||
      !token.bytes.length ||
      end <= 0 ||
      start >= output.length
    )
      return { ...token, start: null, end: null };
    return {
      ...token,
      start: Math.max(0, start),
      end: Math.min(output.length, end),
    };
  });
}

export async function generate(input, signal) {
  const config = connection(input);
  if (
    typeof input.prompt !== "string" ||
    !input.prompt.trim() ||
    input.prompt.length > 32000
  )
    throw new ApiError("Enter a prompt between 1 and 32,000 characters.");
  if (
    typeof input.model !== "string" ||
    !input.model.trim() ||
    input.model.length > 256
  )
    throw new ApiError("Select an installed model.");
  const temperature = input.temperature ?? 0.7;
  const maxTokens = input.maxTokens ?? 256;
  const topN = input.topN ?? config.candidateLimit;
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2)
    throw new ApiError("Temperature must be between 0 and 2.");
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 2048)
    throw new ApiError("Maximum tokens must be between 1 and 2,048.");
  if (!Number.isInteger(topN) || topN < 1 || topN > config.candidateLimit)
    throw new ApiError(
      `This provider supports between 1 and ${config.candidateLimit} candidates.`,
    );
  const seed = input.seed ?? 42;
  if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647)
    throw new ApiError("Seed must be a nonnegative 32-bit integer.");
  const started = performance.now();
  let data, text, records, tokensPerSecond, finishReason;
  if (config.provider === "ollama") {
    data = await upstream(config.baseUrl, "/api/chat", {
      signal,
      body: {
        model: input.model,
        messages: [{ role: "user", content: input.prompt }],
        stream: false,
        think: false,
        logprobs: true,
        top_logprobs: topN,
        options: { temperature, num_predict: maxTokens, seed },
      },
    });
    text = data.message?.content || "";
    records = data.logprobs;
    tokensPerSecond =
      data.eval_duration > 0
        ? data.eval_count / (data.eval_duration / 1e9)
        : null;
    finishReason = data.done_reason;
  } else {
    // Use the server's chat template, then its native completion endpoint for n_probs > 20.
    const template = await upstream(config.baseUrl, "/apply-template", {
      signal,
      body: {
        model: input.model,
        messages: [{ role: "user", content: input.prompt }],
        add_generation_prompt: true,
      },
    });
    if (typeof template.prompt !== "string")
      throw new ApiError(
        "llama.cpp did not return a chat template. Use a recent llama-server build.",
        502,
      );
    data = await upstream(config.baseUrl, "/completion", {
      signal,
      body: {
        model: input.model,
        prompt: template.prompt,
        stream: false,
        n_predict: maxTokens,
        temperature,
        seed,
        n_probs: topN,
        post_sampling_probs: false,
      },
    });
    text = data.content || "";
    records = data.probs || data.completion_probabilities;
    tokensPerSecond = data.timings?.predicted_per_second ?? null;
    finishReason = data.stop_type;
  }
  const tokens = alignTokens(
    normalizeTokens(Array.isArray(records) ? records : []),
    text,
  );
  const hasProbabilities = tokens.some((t) => t.probability !== null);
  let warning = null;
  if (!hasProbabilities)
    warning =
      "This model server did not return token probabilities. The response is shown as plain text. Try a newer Ollama version or another model / runner. Probabilities cannot be reconstructed from text alone.";
  else if (text && !tokens.some((t) => t.start !== null))
    warning =
      "The server's token bytes could not be matched to the response. The original text is preserved; switch to Tokens to inspect the recorded data separately.";
  else if (!tokens.some((t) => t.candidates.some((c) => !c.selected)))
    warning =
      "The server returned chosen-token probabilities without alternatives. Try another model or a newer model server to see the other candidates.";
  const actualTopN = Math.max(
    0,
    ...tokens.map(
      (t) =>
        t.candidates.filter((c) => !c.selected).length +
        (t.candidates.some((c) => c.selected) ? 1 : 0),
    ),
  );
  return {
    id: crypto.randomUUID(),
    createdAt: new Date().toISOString(),
    provider: config.provider,
    model: input.model,
    prompt: input.prompt,
    temperature,
    seed,
    maxTokens,
    requestedTopN: topN,
    actualTopN,
    text,
    tokens,
    duration: (performance.now() - started) / 1000,
    tokensPerSecond,
    finishReason,
    probabilityBasis:
      "Provider-reported probabilities; sampling settings can change the final selection distribution.",
    warning,
    demo: false,
  };
}
