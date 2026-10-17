import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, FormEvent, MouseEvent } from "react";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Copy,
  Download,
  FlaskConical,
  Layers3,
  LoaderCircle,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  Square,
  Terminal,
  Workflow,
  X,
  Zap,
} from "lucide-react";
import type { Exploration, Model, Settings, Token } from "./types";
import { makeDemo } from "./demo";
import { pct, visible } from "./format";
import { useTokenHoverPreview } from "./TokenHoverPreview";
import LayerExplorer from "./LayerExplorer";

const defaults: Settings = {
  provider: "ollama",
  baseUrl: "http://127.0.0.1:11434",
  temperature: 0.7,
  maxTokens: 256,
  topN: 20,
  seed: 42,
};
function stored<T>(key: string, fallback: T): T {
  try {
    return JSON.parse(localStorage.getItem(key) || "null") ?? fallback;
  } catch {
    return fallback;
  }
}
function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* An exploration is still usable when browser storage is full. */
  }
}
const hue = (p: number | null) =>
  p === null
    ? "#a8adb9"
    : p > 0.7
      ? "#5baf98"
      : p > 0.35
        ? "#a79cde"
        : "#e4aa60";
const samples = [
  "Why is the sky blue? Explain in three sentences.",
  "Write the opening of a mystery in three sentences.",
  "Explain quantum computing in three sentences.",
];

function TokenText({
  result,
  selected,
  setSelected,
  mode,
  heatmap,
}: {
  result: Exploration;
  selected: number;
  setSelected: (n: number) => void;
  mode: string;
  heatmap: boolean;
}) {
  const hover = useTokenHoverPreview(`${result.id}:${mode}`, setSelected);
  function choose(index: number) {
    hover.dismiss();
    setSelected(index);
  }
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const bytes = encoder.encode(result.text);
  if (!result.tokens.length)
    return (
      <p className="response-plain">
        {result.text || "The model returned an empty response."}
      </p>
    );
  if (mode === "tokens")
    return (
      <>
        <div className="token-chips">
          {result.tokens.map((t) => (
            <button
              key={t.index}
              className={`token-chip ${selected === t.index ? "selected" : ""}`}
              style={{ "--token-color": hue(t.probability) } as CSSProperties}
              {...hover.triggerProps(t)}
              onClick={() => choose(t.index)}
              aria-label={`Inspect token ${t.index + 1}: ${visible(t.text)}`}
            >
              {visible(t.text)}
            </button>
          ))}
        </div>
        {hover.preview}
      </>
    );
  const pieces = [];
  let cursor = 0;
  for (const t of result.tokens) {
    if (t.start === null || t.end === null) continue;
    if (t.start > cursor)
      pieces.push(
        <span key={`gap-${cursor}`}>
          {decoder.decode(bytes.slice(cursor, t.start))}
        </span>,
      );
    // A single Unicode code point may span several tokens. Extend the rendered
    // character once; each contributing token is accessible in the Tokens view.
    if (t.end <= cursor) continue;
    let end = t.end;
    while (end < bytes.length && (bytes[end] & 0xc0) === 0x80) end++;
    pieces.push(
      <span
        role="button"
        tabIndex={0}
        key={t.index}
        className={`word-token ${selected === t.index ? "selected" : ""} ${heatmap ? "heatmap" : ""}`}
        style={{ "--token-color": hue(t.probability) } as CSSProperties}
        {...hover.triggerProps(t)}
        onClick={() => choose(t.index)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            choose(t.index);
          }
        }}
        aria-label={`Inspect token ${t.index + 1}: ${visible(t.text)}`}
      >
        {decoder.decode(bytes.slice(Math.max(cursor, t.start), end))}
      </span>,
    );
    cursor = end;
  }
  if (cursor < bytes.length)
    pieces.push(<span key="tail">{decoder.decode(bytes.slice(cursor))}</span>);
  return (
    <>
      <div className="response-text">{pieces}</div>
      {hover.preview}
    </>
  );
}

function ProbabilityChart({
  tokens,
  selected,
  onSelect,
}: {
  tokens: Token[];
  selected: number;
  onSelect: (i: number) => void;
}) {
  const width = 720,
    height = 106,
    pad = 10;
  const points = tokens.map((t, i) => [
    pad + (i / Math.max(1, tokens.length - 1)) * (width - pad * 2),
    8 + (1 - (t.probability ?? 0)) * (height - 20),
  ]);
  const line = points.map(([x, y]) => `${x},${y}`).join(" ");
  return (
    <div className="chart-wrap">
      <div className="axis-labels">
        <span>100%</span>
        <span>50%</span>
        <span>0%</span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="group"
        aria-label="Selected token probability across the response"
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id="area" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#8a7de0" stopOpacity=".23" />
            <stop offset="100%" stopColor="#8a7de0" stopOpacity=".01" />
          </linearGradient>
        </defs>
        {[8, 51, 94].map((y) => (
          <line
            key={y}
            x1="0"
            y1={y}
            x2={width}
            y2={y}
            stroke="#edeef3"
            strokeDasharray="4 5"
          />
        ))}
        <polygon
          points={`${pad},${height} ${line} ${width - pad},${height}`}
          fill="url(#area)"
        />
        <polyline
          points={line}
          fill="none"
          stroke="#a295e2"
          strokeWidth="1.8"
          vectorEffect="non-scaling-stroke"
        />
        {points[selected] && (
          <>
            <line
              x1={points[selected][0]}
              x2={points[selected][0]}
              y1="0"
              y2={height}
              stroke="#7562cb"
              strokeDasharray="3 4"
            />
            <circle
              cx={points[selected][0]}
              cy={points[selected][1]}
              r="4.3"
              fill="#7761cf"
              stroke="white"
              strokeWidth="2"
            />
          </>
        )}
        {points.map(([x], i) => (
          <rect
            key={i}
            x={x - width / tokens.length / 2}
            width={Math.max(8, width / tokens.length)}
            y="0"
            height={height}
            fill="transparent"
            tabIndex={0}
            role="button"
            aria-label={`Select token ${i + 1}, ${pct(tokens[i].probability)}`}
            onClick={() => onSelect(i)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect(i);
              }
            }}
          >
            <title>
              {visible(tokens[i].text)} · {pct(tokens[i].probability)}
            </title>
          </rect>
        ))}
      </svg>
    </div>
  );
}

export default function App() {
  const [view, setView] = useState<"tokens" | "layers">(() =>
    new URLSearchParams(window.location.search).get("view") === "layers"
      ? "layers"
      : "tokens",
  );
  function changeView(value: "tokens" | "layers") {
    setView(value);
    const url = new URL(window.location.href);
    if (value === "layers") url.searchParams.set("view", "layers");
    else url.searchParams.delete("view");
    window.history.replaceState(null, "", url);
  }
  const [settings, setSettings] = useState<Settings>(() => ({
    ...defaults,
    ...stored("tokenscope.settings", {}),
  }));
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");
  const [connectionState, setConnectionState] = useState<
    "loading" | "online" | "offline"
  >("loading");
  const [connectionError, setConnectionError] = useState("");
  const [result, setResult] = useState<Exploration | null>(makeDemo);
  const [prompt, setPrompt] = useState(
    "Why is the sky blue? Explain it simply.",
  );
  const [history, setHistory] = useState<Exploration[]>(() =>
    stored<Exploration[]>("tokenscope.history", [])
      .filter(
        (r) => r && Array.isArray(r.tokens) && typeof r.prompt === "string",
      )
      .slice(0, 12),
  );
  const [selected, setSelected] = useState(33);
  const [mode, setMode] = useState("text");
  const [heatmap, setHeatmap] = useState(true);
  const [candidateCount, setCandidateCount] = useState(10);
  const [search, setSearch] = useState("");
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const [modal, setModal] = useState<"settings" | "help" | null>(null);
  const [copied, setCopied] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const generationRevision = useRef(0);
  const connectRevision = useRef(0);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const inspectorRef = useRef<HTMLElement>(null);
  const token = result?.tokens[selected];
  const candidates = useMemo(
    () =>
      (token?.candidates || []).filter((c) =>
        c.text.toLowerCase().includes(search.toLowerCase()),
      ),
    [token, search],
  );
  const coverage =
    token?.candidates.reduce((n, c) => n + c.probability, 0) ?? 0;
  const probabilities =
    result?.tokens.filter((t) => t.probability !== null) || [];
  const avg = probabilities.length
    ? probabilities.reduce((n, t) => n + t.probability!, 0) /
      probabilities.length
    : null;

  const connect = useCallback(async (config: Settings) => {
    const revision = ++connectRevision.current;
    setConnectionState("loading");
    setConnectionError("");
    try {
      const response = await fetch(
        "/api/models?" +
          new URLSearchParams({
            provider: config.provider,
            baseUrl: config.baseUrl,
          }),
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      if (revision !== connectRevision.current) return;
      setModels(data.models);
      setModel((previous) =>
        data.models.some((m: Model) => m.name === previous)
          ? previous
          : data.models[0]?.name || "",
      );
      setConnectionState("online");
    } catch (e) {
      if (revision !== connectRevision.current) return;
      setModels([]);
      setModel("");
      setConnectionState("offline");
      setConnectionError((e as Error).message);
    }
  }, []);
  useEffect(() => {
    connect(settings);
  }, [settings.provider, settings.baseUrl, connect]);
  useEffect(() => {
    save("tokenscope.settings", settings);
  }, [settings]);
  useEffect(() => {
    if (!running) return;
    const start = Date.now();
    setElapsed(0);
    const timer = setInterval(
      () => setElapsed((Date.now() - start) / 1000),
      100,
    );
    return () => clearInterval(timer);
  }, [running]);
  useEffect(() => () => controller.current?.abort(), []);

  function select(index: number, reveal = false) {
    setSelected(index);
    setSearch("");
    if (reveal && window.matchMedia("(max-width: 760px)").matches) {
      inspectorRef.current?.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
        block: "start",
      });
    }
  }
  async function generate(e?: FormEvent) {
    e?.preventDefault();
    if (running || !prompt.trim() || !model || connectionState !== "online")
      return;
    const revision = ++generationRevision.current;
    const requestController = new AbortController();
    controller.current = requestController;
    setRunning(true);
    setError("");
    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: requestController.signal,
        body: JSON.stringify({ ...settings, model, prompt }),
      });
      const data = await response.json();
      if (revision !== generationRevision.current) return;
      if (!response.ok) throw new Error(data.error);
      setResult(data);
      select(0);
      setHistory((previous) => {
        const next = [data, ...previous].slice(0, 12);
        save("tokenscope.history", next);
        return next;
      });
    } catch (e) {
      if (revision !== generationRevision.current) return;
      if ((e as Error).name === "AbortError")
        setError(
          "Generation stopped. Your previous exploration is still available.",
        );
      else setError((e as Error).message);
    } finally {
      if (revision === generationRevision.current) {
        setRunning(false);
        controller.current = null;
      }
    }
  }
  function stopGeneration(event: MouseEvent<HTMLButtonElement>) {
    // React can replace this control with the submit button during the click.
    // Cancel its native default action before that replacement can submit again.
    event.preventDefault();
    generationRevision.current++;
    controller.current?.abort();
    controller.current = null;
    setRunning(false);
    setError(
      "Generation stopped. Your previous exploration is still available.",
    );
  }
  function showDemo() {
    if (running) return;
    changeView("tokens");
    setResult(makeDemo());
    setPrompt("Why is the sky blue? Explain it simply.");
    select(33);
    setError("");
  }
  function exportResult() {
    if (!result) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `tokenscope-${result.demo ? "demo" : result.id.slice(0, 8)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function copy() {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setError(
        "Clipboard is unavailable in this browser. Use Export to save the response.",
      );
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a href="/" className="brand" aria-label="TokenScope home">
          <span className="brand-mark">
            <Workflow size={23} />
          </span>
          <span>
            token<span className="brand-light">scope</span>
            <sup>LOCAL</sup>
          </span>
        </a>
        <button
          className="new-button"
          disabled={running}
          onClick={() => {
            changeView("tokens");
            setResult(null);
            setPrompt("");
            setError("");
            promptRef.current?.focus();
          }}
        >
          <Plus size={17} /> New exploration <span>↗</span>
        </button>
        <div className="nav-label">WORKSPACE</div>
        <button
          className={`nav-item primary-nav ${view === "tokens" ? "active" : ""}`}
          onClick={() => {
            changeView("tokens");
            promptRef.current?.focus();
          }}
        >
          <Layers3 size={18} /> Token explorer{" "}
          {view === "tokens" && <span className="nav-dot" />}
        </button>
        <button
          className={`nav-item primary-nav ${view === "layers" ? "active" : ""}`}
          onClick={() => changeView("layers")}
        >
          <Activity size={18} /> Layer explorer{" "}
          {view === "layers" && <span className="nav-dot" />}
        </button>
        <button className="nav-item" onClick={showDemo} disabled={running}>
          <FlaskConical size={18} /> Example playground
        </button>
        <div className="nav-label recent-label">
          RECENT EXPLORATIONS{" "}
          <span>{history.length.toString().padStart(2, "0")}</span>
        </div>
        <div className="history-list">
          {history.length ? (
            history.map((item) => (
              <button
                key={item.id}
                className={`history-item ${result?.id === item.id ? "current" : ""}`}
                disabled={running}
                onClick={() => {
                  changeView("tokens");
                  setResult(item);
                  setPrompt(item.prompt);
                  select(0);
                  setError("");
                }}
              >
                <span className="history-bullet" />
                <span>{item.prompt}</span>
              </button>
            ))
          ) : (
            <p className="history-empty">
              Your discoveries will
              <br />
              feel at home here.
            </p>
          )}
        </div>
        <div className="sidebar-bottom">
          <div className="local-note">
            <span className="local-note-icon">
              <ShieldCheck size={20} />
            </span>
            <strong>Your model. Your machine.</strong>
            <p>
              Prompts and responses stay
              <br />
              between you and your local LLM.
            </p>
            <span className="local-mini">
              <span /> LOCAL BY DESIGN
            </span>
          </div>
          <button className="nav-item" onClick={() => setModal("help")}>
            <BookOpen size={17} /> How it works <ArrowUpRight size={14} />
          </button>
          <button className="nav-item" onClick={() => setModal("settings")}>
            <Settings2 size={17} /> Connection settings
          </button>
          <div className="sidebar-footer">
            <span className="tiny-brand">ts</span>
            <div>
              TokenScope<span>Local model observatory</span>
            </div>
            <span className="version">v1.0</span>
          </div>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div className="breadcrumb">
            <Layers3 size={16} />
            <span>Workspace</span>
            <ChevronRight size={13} />
            <strong>
              {view === "layers" ? "Layer explorer" : "Token explorer"}
            </strong>
          </div>
          <button
            className={`connection-badge ${connectionState}`}
            onClick={() => setModal("settings")}
          >
            <span className="status-dot" />
            {connectionState === "online"
              ? `${settings.provider === "ollama" ? "Ollama" : "llama.cpp"} connected`
              : connectionState === "loading"
                ? "Connecting…"
                : "Connect local model"}
            <ChevronDown size={12} />
          </button>
        </header>
        <div className="workspace">
          <div hidden={view !== "layers"}>
            <LayerExplorer defaultModel={model} />
          </div>
          <div hidden={view !== "tokens"}>
            <div className="page-heading">
              <div>
                <div className="eyebrow">
                  <span /> A LITTLE CLOSER TO THE MODEL
                </div>
                <h1>
                  Inside the next token<span>.</span>
                </h1>
                <p>See what your model chose. Discover what it almost said.</p>
              </div>
              <button
                className="secondary-button help-button"
                onClick={() => setModal("help")}
              >
                <CircleHelp size={16} /> How it works <ArrowUpRight size={14} />
              </button>
            </div>

            <form className="prompt-card" onSubmit={generate}>
              <div className="section-heading">
                <div>
                  <span className="step-label">01</span>
                  <h2>Start with a question</h2>
                </div>
                <span className="subtle-label">
                  A prompt opens up possibilities
                </span>
              </div>
              <textarea
                ref={promptRef}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Ask your local model anything…"
                aria-label="Your prompt"
                maxLength={32000}
                disabled={running}
                onKeyDown={(e) => {
                  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                    e.preventDefault();
                    generate();
                  }
                }}
              />
              <div className="prompt-examples">
                <span>Try a little curiosity</span>
                {samples.map((s, i) => (
                  <button
                    type="button"
                    key={s}
                    disabled={running}
                    onClick={() => {
                      setPrompt(s);
                      promptRef.current?.focus();
                    }}
                  >
                    {i === 0 ? "☀" : i === 1 ? "✧" : "⌘"}{" "}
                    {i === 0
                      ? "The blue sky"
                      : i === 1
                        ? "A little mystery"
                        : "Quantum, simply"}
                  </button>
                ))}
              </div>
              <div className="prompt-controls">
                <div className="model-control">
                  <span className="model-icon">
                    <Layers3 size={16} />
                  </span>
                  <select
                    aria-label="Local model"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    disabled={running || !models.length}
                  >
                    <option value="" disabled>
                      {connectionState === "loading"
                        ? "Finding models…"
                        : !models.length
                          ? "No model connected"
                          : "Select model"}
                    </option>
                    {models.map((m) => (
                      <option key={m.name} value={m.name}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={13} />
                  {model && models.find((m) => m.name === model)?.size && (
                    <span className="model-size">
                      {models.find((m) => m.name === model)?.size}
                    </span>
                  )}
                </div>
                <div className="generation-controls">
                  <label
                    className="temperature-control"
                    title="Higher temperature makes token selection more varied."
                  >
                    <span>Temperature</span>
                    <input
                      aria-label="Temperature"
                      type="range"
                      min="0"
                      max="2"
                      step=".1"
                      value={settings.temperature}
                      disabled={running}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          temperature: +e.target.value,
                        })
                      }
                    />
                    <output>{settings.temperature.toFixed(1)}</output>
                  </label>
                  {running ? (
                    <button
                      key="stop"
                      className="primary-button stop-button"
                      type="button"
                      onClick={stopGeneration}
                    >
                      <Square size={13} /> Stop{" "}
                      <span>{elapsed.toFixed(1)}s</span>
                    </button>
                  ) : (
                    <button
                      key="generate"
                      type="submit"
                      className="primary-button"
                      disabled={
                        !prompt.trim() || !model || connectionState !== "online"
                      }
                    >
                      <Sparkles size={15} /> Generate <ArrowRight size={16} />
                    </button>
                  )}
                </div>
              </div>
            </form>
            {connectionState === "offline" && (
              <div className="connection-notice">
                <Terminal size={16} />
                <span>
                  Start Ollama with <code>ollama serve</code>, then connect to
                  explore your models.
                </span>
                <button onClick={() => connect(settings)}>
                  <RefreshCw size={13} /> Reconnect
                </button>
              </div>
            )}
            {connectionState === "online" && !models.length && (
              <div className="notice warning">
                No text models found. Download one with{" "}
                <code>ollama pull gemma4:e4B</code>, then reconnect in settings.
              </div>
            )}
            {error && (
              <div className="notice error" role="alert">
                <CircleHelp size={16} />
                <span>{error}</span>
                <button aria-label="Dismiss error" onClick={() => setError("")}>
                  <X size={15} />
                </button>
              </div>
            )}
            {running && (
              <div className="notice generating" role="status">
                <LoaderCircle className="spin" size={17} />
                <span>
                  Looking inside <strong>{model}</strong>… The first run may
                  take a moment to load the model.
                </span>
                <span>{elapsed.toFixed(1)}s</span>
              </div>
            )}

            {result ? (
              <>
                <div className="exploration-heading">
                  <div>
                    <span className="step-label">02</span>
                    <h2>Follow the possibilities</h2>
                    <span
                      className={`source-badge ${result.demo ? "demo" : ""}`}
                    >
                      <span />
                      {result.demo ? "Illustrative demo" : "Local generation"}
                    </span>
                  </div>
                  <button className="text-button" onClick={exportResult}>
                    <Download size={14} /> Export exploration
                  </button>
                </div>
                {result.warning && (
                  <div className="notice warning" role="status">
                    <CircleHelp size={17} />
                    <span>{result.warning}</span>
                  </div>
                )}
                {["length", "limit"].includes(result.finishReason || "") && (
                  <div className="notice warning" role="status">
                    <CircleHelp size={17} />
                    <span>
                      The response reached the output token limit. Increase
                      “Maximum output tokens” in settings for a longer response.
                    </span>
                  </div>
                )}
                <div className="exploration-grid">
                  <div className="response-column">
                    <section className="response-card">
                      <div className="response-toolbar">
                        <div className="response-label">
                          <span className="response-icon">
                            <Sparkles size={15} />
                          </span>
                          <strong>Model response</strong>
                        </div>
                        <div className="response-actions">
                          <div
                            className="segmented"
                            aria-label="Response display"
                          >
                            <button
                              className={mode === "text" ? "active" : ""}
                              onClick={() => setMode("text")}
                            >
                              Text
                            </button>
                            <button
                              className={mode === "tokens" ? "active" : ""}
                              onClick={() => setMode("tokens")}
                            >
                              Tokens
                            </button>
                          </div>
                          <button
                            className="icon-button copy-button"
                            aria-label="Copy response"
                            onClick={copy}
                          >
                            {copied ? <Check size={15} /> : <Copy size={15} />}
                          </button>
                        </div>
                      </div>
                      <div className="response-body">
                        <div className="response-source">
                          <span className="mini-dot" />
                          {result.demo ? "EXAMPLE RESPONSE" : result.model}
                          <span>·</span>
                          {result.demo
                            ? "Synthetic probabilities"
                            : new Date(result.createdAt).toLocaleTimeString(
                                [],
                                {
                                  hour: "2-digit",
                                  minute: "2-digit",
                                },
                              )}
                        </div>
                        <TokenText
                          result={result}
                          selected={selected}
                          setSelected={(index) => select(index, true)}
                          mode={mode}
                          heatmap={heatmap}
                        />
                      </div>
                      <div className="response-footer">
                        <span>
                          <span className="pointer-symbol">↖</span> Hover to
                          preview · click to inspect
                        </span>
                        <button
                          className="heatmap-toggle"
                          aria-pressed={heatmap}
                          onClick={() => setHeatmap(!heatmap)}
                        >
                          <span className={`switch ${heatmap ? "on" : ""}`} />
                          <span>Probability colors</span>
                        </button>
                      </div>
                      <div className="color-legend">
                        <span>Less likely</span>
                        <div className="legend-gradient" />
                        <span>More likely</span>
                        <span className="legend-spacer" />
                        <span className="legend-note">
                          Probability of the chosen token
                        </span>
                      </div>
                    </section>
                    <div className="metrics">
                      <div>
                        <span>
                          <Layers3 size={13} /> Generated tokens
                        </span>
                        <strong>
                          {result.tokens.length || "—"}
                          <small>tokens</small>
                        </strong>
                      </div>
                      <div>
                        <span>
                          <Activity size={13} /> Mean probability
                        </span>
                        <strong>
                          {pct(avg)}
                          <small>chosen tokens</small>
                        </strong>
                      </div>
                      <div>
                        <span>
                          <Zap size={13} /> Generation speed
                        </span>
                        <strong>
                          {result.tokensPerSecond?.toFixed(1) || "—"}
                          <small>tokens / sec</small>
                        </strong>
                      </div>
                    </div>
                    {probabilities.length > 0 && (
                      <section className="chart-card">
                        <div className="chart-heading">
                          <div>
                            <Activity size={15} />
                            <h3>The probability trail</h3>
                          </div>
                          <span>ONE STEP AT A TIME</span>
                        </div>
                        <ProbabilityChart
                          tokens={result.tokens}
                          selected={selected}
                          onSelect={(index) => select(index, true)}
                        />
                        <div className="chart-footer">
                          <span>Token 1</span>
                          <span className="chart-key">
                            <span /> Chosen token probability
                          </span>
                          <span>Token {result.tokens.length}</span>
                        </div>
                      </section>
                    )}
                    <div className="learning-note">
                      <span className="learning-icon">
                        <Layers3 size={18} />
                      </span>
                      <div>
                        <strong>A word can hold more than one token.</strong>
                        <p>
                          Models choose tokens: words, word pieces, and
                          punctuation. Switch to Tokens to see the exact
                          boundaries.
                        </p>
                      </div>
                      <button
                        aria-label="Learn about tokens"
                        onClick={() => setModal("help")}
                      >
                        <ArrowUpRight size={17} />
                      </button>
                    </div>
                  </div>

                  <aside
                    className="inspector"
                    ref={inspectorRef}
                    aria-label="Token candidate inspector"
                  >
                    <div className="inspector-heading">
                      <div>
                        <span className="inspector-dot" />
                        <h2>Token inspector</h2>
                      </div>
                      <span className="live-label">
                        {result.demo ? "DEMO" : "RECORDED"}
                      </span>
                    </div>
                    {token ? (
                      <>
                        <div className="chosen-token">
                          <div className="token-position">
                            <span>
                              TOKEN {String(selected + 1).padStart(2, "0")}{" "}
                              <span className="of">
                                / {result.tokens.length}
                              </span>
                            </span>
                            <div className="token-nav">
                              <button
                                aria-label="Previous token"
                                disabled={selected === 0}
                                onClick={() => select(selected - 1)}
                              >
                                <ChevronLeft size={15} />
                              </button>
                              <button
                                aria-label="Next token"
                                disabled={selected === result.tokens.length - 1}
                                onClick={() => select(selected + 1)}
                              >
                                <ChevronRight size={15} />
                              </button>
                            </div>
                          </div>
                          <div className="chosen-token-main">
                            <code>{visible(token.text)}</code>
                            <span className="selected-badge">
                              <Check size={10} /> Selected
                            </span>
                          </div>
                          <div className="selected-probability">
                            <strong>{pct(token.probability)}</strong>
                            <span>reported probability</span>
                          </div>
                        </div>
                        <div className="candidate-toolbar">
                          <h3>What else was possible?</h3>
                          <select
                            aria-label="Number of candidates to display"
                            value={candidateCount}
                            onChange={(e) => setCandidateCount(+e.target.value)}
                          >
                            {[5, 10, 20, 100].map((n) => (
                              <option key={n} value={n}>
                                Top {n}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="candidate-search">
                          <Search size={14} />
                          <input
                            aria-label="Search candidates"
                            placeholder="Find a candidate…"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                          />
                          {search && (
                            <button
                              aria-label="Clear candidate search"
                              onClick={() => setSearch("")}
                            >
                              <X size={12} />
                            </button>
                          )}
                        </div>
                        <div className="candidate-columns">
                          <span>RANK / TOKEN</span>
                          <span>PROBABILITY</span>
                        </div>
                        <div className="candidate-list">
                          {candidates.length ? (
                            candidates
                              .slice(0, candidateCount)
                              .map((candidate, i) => (
                                <div
                                  className={`candidate-row ${candidate.selected ? "chosen" : ""}`}
                                  key={`${candidate.id ?? candidate.text}-${i}`}
                                  title={`Log probability: ${candidate.logprob?.toFixed(6) ?? "Unavailable"}`}
                                >
                                  <span
                                    className="candidate-fill"
                                    style={{
                                      width: `${Math.min(100, candidate.probability * 100)}%`,
                                    }}
                                  />
                                  <span className="rank">
                                    {String(candidate.rank).padStart(2, "0")}
                                  </span>
                                  <code>{visible(candidate.text)}</code>
                                  {candidate.selected && <Check size={12} />}
                                  <span className="candidate-pct">
                                    {pct(candidate.probability)}
                                  </span>
                                </div>
                              ))
                          ) : (
                            <div className="candidate-empty">
                              {search
                                ? "No candidates match your search."
                                : "This server did not provide candidate probabilities."}
                            </div>
                          )}
                        </div>
                        <div className="candidate-summary">
                          <span>
                            Showing{" "}
                            {Math.min(candidateCount, candidates.length)} of{" "}
                            {token.candidates.length} returned
                          </span>
                          <span>· = space</span>
                        </div>
                        {token.candidates.length > 0 && (
                          <div className="coverage-box">
                            <div>
                              <span>Returned probability mass</span>
                              <strong>{pct(Math.min(1, coverage))}</strong>
                            </div>
                            <div className="coverage-track">
                              <span
                                style={{
                                  width: `${Math.min(100, coverage * 100)}%`,
                                }}
                              />
                            </div>
                            <p>
                              {pct(Math.max(0, 1 - coverage))} remains outside
                              the returned candidates.
                            </p>
                          </div>
                        )}
                        {candidateCount > token.candidates.length && (
                          <div className="limit-note">
                            {result.provider === "ollama"
                              ? "Ollama returns up to 20 candidates per token. Connect llama.cpp in settings to request up to 100."
                              : "The server may return fewer candidates than requested."}
                          </div>
                        )}
                        <div className="context-box">
                          <span className="context-label">
                            THE CONTEXT BEFORE THIS TOKEN
                          </span>
                          <p>
                            {result.tokens
                              .slice(Math.max(0, selected - 10), selected)
                              .map((t) => t.text)
                              .join("") || "Beginning of the response"}
                            <span className="context-cursor" />
                          </p>
                        </div>
                      </>
                    ) : (
                      <div className="inspector-empty">
                        <Search size={26} />
                        <h3>Meet the possibilities.</h3>
                        <p>
                          Candidate probabilities will appear here when your
                          model returns them.
                        </p>
                      </div>
                    )}
                  </aside>
                </div>
                {result.demo && (
                  <div className="demo-disclaimer">
                    <FlaskConical size={13} />
                    <span>
                      You’re exploring an illustrative example. Generate with
                      your local model to see real probabilities.
                    </span>
                    <button onClick={() => promptRef.current?.focus()}>
                      Make it yours <ArrowRight size={13} />
                    </button>
                  </div>
                )}
              </>
            ) : (
              <div className="empty-state">
                <span>
                  <Workflow size={36} />
                </span>
                <h2>A whole world of next words.</h2>
                <p>
                  Ask a question above. Then explore every token
                  <br />
                  and the possibilities behind it.
                </p>
                <button className="secondary-button" onClick={showDemo}>
                  <FlaskConical size={15} /> Explore an example
                </button>
              </div>
            )}
          </div>
          <footer className="workspace-footer">
            <span>
              <ShieldCheck size={12} /> Local inference. Open possibilities.
            </span>
            <span>
              Built for the curious mind{" "}
              <span className="footer-sparkle">✳</span>
            </span>
          </footer>
        </div>
      </main>

      {modal && (
        <div
          className="modal-backdrop"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setModal(null);
          }}
        >
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-title"
            onKeyDown={(e) => {
              if (e.key === "Escape") setModal(null);
              if (e.key === "Tab") {
                const controls = Array.from(
                  e.currentTarget.querySelectorAll<HTMLElement>(
                    "button:not(:disabled), input:not(:disabled), select:not(:disabled), a[href]",
                  ),
                ).filter((el) => el.offsetParent !== null);
                const first = controls[0],
                  last = controls[controls.length - 1];
                if (e.shiftKey && document.activeElement === first) {
                  e.preventDefault();
                  last?.focus();
                } else if (!e.shiftKey && document.activeElement === last) {
                  e.preventDefault();
                  first?.focus();
                }
              }
            }}
          >
            <button
              autoFocus
              className="modal-close icon-button"
              aria-label="Close dialog"
              onClick={() => setModal(null)}
            >
              <X size={19} />
            </button>
            {modal === "settings" ? (
              <>
                <span className="modal-symbol">
                  <Settings2 size={24} />
                </span>
                <h2 id="modal-title">Connect your model.</h2>
                <p className="modal-intro">
                  A small window into the LLM on your machine.
                </p>
                <div className="settings-provider">
                  <button
                    className={settings.provider === "ollama" ? "active" : ""}
                    disabled={running}
                    onClick={() =>
                      setSettings({
                        ...settings,
                        provider: "ollama",
                        baseUrl: "http://127.0.0.1:11434",
                        topN: 20,
                      })
                    }
                  >
                    Ollama <span>Up to 20 candidates</span>
                  </button>
                  <button
                    className={settings.provider === "llamacpp" ? "active" : ""}
                    disabled={running}
                    onClick={() =>
                      setSettings({
                        ...settings,
                        provider: "llamacpp",
                        baseUrl: "http://127.0.0.1:8080",
                        topN: 100,
                      })
                    }
                  >
                    llama.cpp <span>Up to 100 candidates</span>
                  </button>
                </div>
                <label className="setting-field">
                  Local server URL
                  <input
                    value={settings.baseUrl}
                    disabled={running}
                    onChange={(e) =>
                      setSettings({ ...settings, baseUrl: e.target.value })
                    }
                    placeholder="http://127.0.0.1:11434"
                  />
                </label>
                <div className="setting-grid">
                  <label className="setting-field">
                    Candidates per token
                    <select
                      value={settings.topN}
                      disabled={running}
                      onChange={(e) =>
                        setSettings({ ...settings, topN: +e.target.value })
                      }
                    >
                      {(settings.provider === "ollama"
                        ? [5, 10, 20]
                        : [5, 10, 20, 50, 100]
                      ).map((n) => (
                        <option key={n}>{n}</option>
                      ))}
                    </select>
                  </label>
                  <label className="setting-field">
                    Maximum output tokens
                    <input
                      type="number"
                      min="1"
                      max="2048"
                      step="1"
                      value={settings.maxTokens}
                      disabled={running}
                      onChange={(e) =>
                        setSettings({ ...settings, maxTokens: +e.target.value })
                      }
                    />
                  </label>
                </div>
                <label className="setting-field">
                  Random seed
                  <input
                    type="number"
                    min="0"
                    max="2147483647"
                    step="1"
                    value={settings.seed}
                    disabled={running}
                    onChange={(e) =>
                      setSettings({ ...settings, seed: +e.target.value })
                    }
                  />
                  <small>
                    A fixed seed helps compare runs. Exact repeatability depends
                    on the model server.
                  </small>
                </label>
                <div className={`settings-status ${connectionState}`}>
                  <span className="status-dot" />
                  <span>
                    {connectionState === "online"
                      ? `${models.length} text model${models.length === 1 ? "" : "s"} available`
                      : connectionState === "loading"
                        ? "Connecting to local server…"
                        : connectionError}
                  </span>
                </div>
                <div className="setup-command">
                  <Terminal size={15} />
                  <code>
                    {settings.provider === "ollama"
                      ? "ollama serve"
                      : "llama-server -m /path/to/model.gguf --port 8080"}
                  </code>
                </div>
                <div className="modal-footer">
                  <button
                    className="secondary-button"
                    disabled={running || connectionState === "loading"}
                    onClick={() => connect(settings)}
                  >
                    <RefreshCw size={14} /> Reconnect
                  </button>
                  <button
                    className="primary-button"
                    onClick={() => setModal(null)}
                  >
                    Back to exploring <ArrowRight size={15} />
                  </button>
                </div>
              </>
            ) : (
              <>
                <span className="modal-symbol">
                  <BookOpen size={24} />
                </span>
                <h2 id="modal-title">A peek behind the prediction.</h2>
                <p className="modal-intro">
                  Every response is a sequence of small choices.
                </p>
                <div className="explanation">
                  <h3>01 · Tokens, rather than whole words</h3>
                  <p>
                    A token can be a word, part of a word, punctuation, or a
                    byte fragment. Click the response to inspect a token, or
                    switch to Tokens to see every boundary.
                  </p>
                  <h3>02 · A distribution of possibilities</h3>
                  <p>
                    Given the prompt and previous tokens, the model assigns
                    scores to the vocabulary. These become probabilities.
                    Sampling settings influence which token is selected; the
                    most likely token is not always the one chosen.
                  </p>
                  <h3>03 · Probabilities recorded during generation</h3>
                  <p>
                    TokenScope reads probabilities from your model server.
                    Ollama supports up to 20 alternatives; llama.cpp supports
                    requesting 100. The list is a snapshot at that exact
                    position, with the original preceding tokens.
                  </p>
                  <h3>04 · Read the numbers carefully</h3>
                  <p>
                    We display exp(log probability) as reported by the provider,
                    without rescaling the visible candidates to 100%. Sampling
                    can alter the final selection distribution. These values
                    measure token likelihood, not factual accuracy. The chart
                    shows the chosen token’s reported probability at each step.
                  </p>
                  <p>
                    Each generation is independent. Recent explorations are
                    stored in this browser; use Export to save a full record.
                    Plain text from an earlier chat cannot reveal its original
                    candidate probabilities.
                  </p>
                  <div className="docs-links">
                    <a
                      href="https://docs.ollama.com/api/chat"
                      target="_blank"
                      rel="noreferrer"
                    >
                      Ollama API <ArrowUpRight size={13} />
                    </a>
                    <a
                      href="https://github.com/ggml-org/llama.cpp/tree/master/tools/server"
                      target="_blank"
                      rel="noreferrer"
                    >
                      llama.cpp server <ArrowUpRight size={13} />
                    </a>
                  </div>
                </div>
                <button
                  className="primary-button full-width"
                  onClick={() => setModal(null)}
                >
                  Let’s explore <ArrowRight size={15} />
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
