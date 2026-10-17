import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, MouseEvent } from "react";
import {
  Activity,
  ArrowDown,
  ArrowRight,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Cpu,
  Download,
  Layers3,
  LoaderCircle,
  RefreshCw,
  ScanLine,
  ShieldCheck,
  Square,
  Terminal,
  Workflow,
  Zap,
} from "lucide-react";
import type {
  LayerTrace,
  TracingStatus,
  TraceStep,
  LayerCandidate,
} from "./layerTypes";
import { pct, visible } from "./format";
import "./layers.css";

function exportTrace(trace: LayerTrace) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(trace)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `tokenscope-layers-${trace.id.slice(0, 8)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const compact = (number: number) =>
  Math.abs(number) >= 1000
    ? `${(number / 1000).toFixed(1)}k`
    : number.toFixed(2);
function CandidateRows({
  candidates,
  chosen,
}: {
  candidates: LayerCandidate[];
  chosen?: number;
}) {
  return (
    <div className="layer-candidates">
      {candidates.map((candidate, index) => (
        <div
          className={`candidate-row ${candidate.id === chosen ? "chosen" : ""}`}
          key={candidate.id}
        >
          <span
            className="candidate-fill"
            style={{ width: `${candidate.probability * 100}%` }}
          />
          <span className="rank">{String(index + 1).padStart(2, "0")}</span>
          <code>{visible(candidate.text)}</code>
          {candidate.id === chosen && <Check size={11} />}
          <span className="candidate-pct">{pct(candidate.probability)}</span>
        </div>
      ))}
    </div>
  );
}

function ActivationMap({
  step,
  selected,
  onSelect,
  hiddenSize,
}: {
  step: TraceStep;
  selected: number;
  onSelect: (layer: number) => void;
  hiddenSize: number;
}) {
  const values = step.layers
    .flatMap((layer) => layer.activation.bins.map(Math.abs))
    .sort((a, b) => a - b);
  const scale = values[Math.floor(values.length * 0.98)] || 1;
  const width = 716,
    rowHeight = 15,
    left = 42,
    columns = step.layers[0]?.activation.bins.length || 64;
  return (
    <>
      <div className="activation-scroll">
        <svg
          viewBox={`0 0 ${width} ${step.layers.length * rowHeight + 42}`}
          role="group"
          aria-label="Measured hidden activation heatmap"
        >
          <text x="0" y="10" className="heatmap-axis">
            BLOCK
          </text>
          <text x={left} y="10" className="heatmap-axis">
            CHANNEL 1
          </text>
          <text x={width - 68} y="10" className="heatmap-axis">
            {hiddenSize.toLocaleString()}
          </text>
          {step.layers.map((layer, row) => (
            <g
              key={layer.index}
              role="button"
              tabIndex={0}
              aria-label={`Inspect block ${layer.index + 1}`}
              onClick={() => onSelect(layer.index)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(layer.index);
                }
              }}
            >
              <rect
                x="0"
                y={row * rowHeight + 22}
                width={width}
                height={rowHeight}
                rx="3"
                fill={selected === layer.index ? "#eee7fa" : "transparent"}
              />
              <text
                x="8"
                y={row * rowHeight + 32}
                className={`heatmap-row-label ${selected === layer.index ? "selected" : ""}`}
              >
                {String(layer.index + 1).padStart(2, "0")}
              </text>
              {layer.activation.bins.map((value, col) => (
                <rect
                  key={col}
                  x={left + (col * (width - left)) / columns}
                  y={row * rowHeight + 23}
                  width={(width - left) / columns - 1}
                  height={rowHeight - 2}
                  rx="1"
                  fill={value < 0 ? "#439c8d" : "#8060bf"}
                  fillOpacity={
                    0.08 + 0.85 * Math.min(1, Math.abs(value) / scale)
                  }
                >
                  <title>{`Block ${layer.index + 1} · channels ${Math.floor((col * hiddenSize) / columns) + 1}–${Math.floor(((col + 1) * hiddenSize) / columns)} · mean ${value.toFixed(5)}`}</title>
                </rect>
              ))}
            </g>
          ))}
        </svg>
      </div>
      <div className="activation-legend">
        <span>
          <i className="negative" /> Negative
        </span>
        <span>64 channel averages · full vectors in export</span>
        <span>
          Positive <i className="positive" />
        </span>
      </div>
    </>
  );
}

export default function LayerExplorer({
  defaultModel,
}: {
  defaultModel: string;
}) {
  const [status, setStatus] = useState<TracingStatus | null>(null);
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState(
    "Why is the sky blue? Answer in one sentence.",
  );
  const [maxTokens, setMaxTokens] = useState(4);
  const [trace, setTrace] = useState<LayerTrace | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [layerIndex, setLayerIndex] = useState(0);
  const [head, setHead] = useState(-1);
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const controller = useRef<AbortController | null>(null);
  const revision = useRef(0);
  const step = trace?.steps[stepIndex];
  const layer = step?.layers.find((l) => l.index === layerIndex);
  const predictions = step?.lens.layers?.find(
    (l) => l.layer === layerIndex,
  )?.candidates;
  const context = trace?.contextTokens.slice(0, step?.contextLength || 0) || [];
  const attention = useMemo(() => {
    if (!layer?.attention.length) return [];
    if (head >= 0) return layer.attention[head] || [];
    return layer.attention[0].map(
      (_, i) =>
        layer.attention.reduce((sum, row) => sum + (row[i] || 0), 0) /
        layer.attention.length,
    );
  }, [layer, head]);
  const topAttention = attention
    .map((weight, index) => ({ weight, index, token: context[index] }))
    .filter((item) => item.token)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 8);
  const entropy = attention.reduce(
    (sum, p) => (p > 0 ? sum - p * Math.log(p) : sum),
    0,
  );
  const entropyPct =
    attention.length > 1 ? entropy / Math.log(attention.length) : 0;

  async function refresh() {
    try {
      const response = await fetch("/api/tracing/status");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      setStatus(data);
      setError("");
      setModel((previous) =>
        data.models.some((m: { name: string }) => m.name === previous)
          ? previous
          : data.models.find(
              (m: { name: string; fitsMemory: boolean }) =>
                m.name === (defaultModel || "gemma4:e4B") && m.fitsMemory,
            )?.name ||
            data.models.find((m: { fitsMemory: boolean }) => m.fitsMemory)
              ?.name ||
            "",
      );
    } catch (e) {
      setError((e as Error).message);
    }
  }
  useEffect(() => {
    refresh();
    return () => controller.current?.abort();
  }, []);
  useEffect(() => {
    if (!running) return;
    const start = Date.now();
    const timer = setInterval(
      () => setElapsed((Date.now() - start) / 1000),
      100,
    );
    return () => clearInterval(timer);
  }, [running]);
  async function run(event?: FormEvent) {
    event?.preventDefault();
    if (running || !prompt.trim() || !model || !status?.ready) return;
    const runRevision = ++revision.current;
    const request = new AbortController();
    controller.current = request;
    setRunning(true);
    setError("");
    setElapsed(0);
    setPhase("Starting the local tracing engine");
    try {
      const response = await fetch("/api/tracing/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, prompt, maxTokens }),
        signal: request.signal,
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error);
      }
      if (!response.body)
        throw new Error("This browser cannot read the trace stream.");
      const reader = response.body.getReader(),
        decoder = new TextDecoder();
      let pending = "",
        received = false;
      function eventLine(line: string) {
        if (!line.trim() || runRevision !== revision.current) return;
        const event = JSON.parse(line);
        if (event.type === "error") throw new Error(event.error);
        if (event.type === "progress")
          setPhase(
            `${event.stage}${event.step >= 0 ? ` · step ${event.step + 1} / ${maxTokens}` : ""}`,
          );
        if (event.type === "result") {
          setTrace(event.data);
          setStepIndex(0);
          setLayerIndex(0);
          setHead(-1);
          received = true;
        }
      }
      while (true) {
        const { value, done } = await reader.read();
        pending += decoder.decode(value, { stream: !done });
        let end;
        while ((end = pending.indexOf("\n")) >= 0) {
          const line = pending.slice(0, end);
          pending = pending.slice(end + 1);
          eventLine(line);
        }
        if (done) break;
      }
      if (pending.trim()) eventLine(pending);
      if (!received && runRevision === revision.current)
        throw new Error("The tracing engine ended without a result.");
    } catch (e) {
      if (runRevision === revision.current)
        setError(
          (e as Error).name === "AbortError"
            ? "Trace stopped."
            : (e as Error).message,
        );
    } finally {
      if (runRevision === revision.current) {
        setRunning(false);
        controller.current = null;
      }
    }
  }
  function stop(event: MouseEvent) {
    event.preventDefault();
    revision.current++;
    controller.current?.abort();
    controller.current = null;
    setRunning(false);
    setError("Trace stopped. Your previous measurements are still available.");
  }

  return (
    <div className="layer-explorer">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span /> FROM INPUT TO POSSIBILITY
          </div>
          <h1>
            Follow the signal<span>.</span>
          </h1>
          <p>
            Look inside the layers. See the measurements behind the next token.
          </p>
        </div>
        <span className="layer-mode-badge">
          <ScanLine size={14} /> Layer explorer
        </span>
      </div>
      <div className="layer-trust-note">
        <ShieldCheck size={17} />
        <span>
          <strong>Real measurements, separate run.</strong> Traces use your
          existing Ollama GGUF weights in an instrumented llama.cpp engine.
          Ollama’s API does not expose these internal tensors.
        </span>
      </div>
      <form className="prompt-card layer-prompt" onSubmit={run}>
        <div className="section-heading">
          <div>
            <span className="step-label">01</span>
            <h2>Ask a short question</h2>
          </div>
          <span className="subtle-label">
            Greedy decoding · CPU tracing · up to 8 steps
          </span>
        </div>
        <textarea
          aria-label="Layer trace prompt"
          placeholder="What would you like to look inside?"
          value={prompt}
          maxLength={1200}
          disabled={running}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
              e.preventDefault();
              run();
            }
          }}
        />
        <div className="prompt-examples">
          <span>A small question, many layers</span>
          <button
            type="button"
            disabled={running}
            onClick={() => setPrompt("Say hello in one sentence.")}
          >
            A simple hello
          </button>
          <button
            type="button"
            disabled={running}
            onClick={() => setPrompt("The capital of France is")}
          >
            A familiar fact
          </button>
          <button
            type="button"
            disabled={running}
            onClick={() =>
              setPrompt("Why is the sky blue? Answer in one sentence.")
            }
          >
            The blue sky
          </button>
        </div>
        <div className="prompt-controls">
          <div className="model-control">
            <span className="model-icon">
              <Cpu size={16} />
            </span>
            <select
              aria-label="Trace model"
              value={model}
              disabled={running || !status?.models.length}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="" disabled>
                {status ? "Choose a GGUF model" : "Finding local GGUF models…"}
              </option>
              {status?.models.map((m) => (
                <option key={m.name} value={m.name} disabled={!m.fitsMemory}>
                  {m.name} · {(m.size / 1e9).toFixed(1)} GB
                  {m.fitsMemory ? "" : " · exceeds memory budget"}
                </option>
              ))}
            </select>
          </div>
          <div className="layer-run-controls">
            <label>
              Steps{" "}
              <select
                aria-label="Trace generation steps"
                value={maxTokens}
                disabled={running}
                onChange={(e) => setMaxTokens(+e.target.value)}
              >
                {[1, 2, 4, 8].map((n) => (
                  <option key={n}>{n}</option>
                ))}
              </select>
            </label>
            {running ? (
              <button
                key="stop-trace"
                type="button"
                className="primary-button stop-button"
                onClick={stop}
              >
                <Square size={13} /> Stop <span>{elapsed.toFixed(1)}s</span>
              </button>
            ) : (
              <button
                key="run-trace"
                type="submit"
                className="primary-button"
                disabled={!status?.ready || !model || !prompt.trim()}
              >
                <ScanLine size={15} /> Trace layers <ArrowRight size={15} />
              </button>
            )}
          </div>
        </div>
      </form>
      {!status && !error && (
        <div className="notice generating">
          <LoaderCircle size={16} className="spin" /> Checking the tracing
          engine…
        </div>
      )}
      {status && !status.ready && (
        <div className="layer-setup">
          <Terminal size={21} />
          <div>
            <strong>Build the tracing engine once.</strong>
            <p>
              Run <code>npm run setup:tracer</code> in the project folder. It
              builds the pinned llama.cpp engine locally; your model weights
              stay where they are.
            </p>
          </div>
          <button className="secondary-button" onClick={refresh}>
            <RefreshCw size={13} /> Check again
          </button>
        </div>
      )}
      {status?.ready && !status.models.length && (
        <div className="notice warning">
          No local GGUF text model was found. Download one with Ollama, then
          refresh the model list. Custom model locations can be set with
          OLLAMA_MODELS.
        </div>
      )}
      {error && (
        <div className="notice error" role="alert">
          <CircleHelp size={16} />
          <span>{error}</span>
          <button className="text-button" onClick={() => setError("")}>
            Dismiss
          </button>
        </div>
      )}
      {running && (
        <div className="notice generating" role="status">
          <LoaderCircle size={17} className="spin" />
          <span>{phase}</span>
          <span>{elapsed.toFixed(1)}s</span>
        </div>
      )}
      {!trace ? (
        <div className="layer-empty">
          <div className="layer-empty-stack">
            {[0, 1, 2, 3, 4].map((n) => (
              <span
                key={n}
                style={{
                  transform: `translateY(${n * 10}px) rotate(-18deg)`,
                  opacity: 1 - n * 0.12,
                }}
              >
                <Layers3 size={26} />
              </span>
            ))}
          </div>
          <h2>Every layer leaves a signal.</h2>
          <p>
            Trace a few generation steps to explore measured activations,
            <br />
            attention heads, and intermediate token predictions.
          </p>
          <div className="layer-empty-features">
            <span>
              <Activity size={15} /> Hidden activations
            </span>
            <span>
              <ScanLine size={15} /> Attention maps
            </span>
            <span>
              <Zap size={15} /> Prediction trail
            </span>
          </div>
          <span className="layer-empty-footnote">
            No simulated layer activity is shown.
          </span>
        </div>
      ) : (
        step &&
        layer && (
          <>
            <div className="exploration-heading">
              <div>
                <span className="step-label">02</span>
                <h2>Inside this generation</h2>
                <span className="source-badge">
                  <span /> Measured trace
                </span>
              </div>
              <button
                className="text-button"
                onClick={() => exportTrace(trace)}
              >
                <Download size={14} /> Export trace
              </button>
            </div>
            <div className="layer-summary">
              <div>
                <Layers3 size={15} />
                <strong>{trace.layerCount}</strong>
                <span>transformer blocks</span>
              </div>
              <div>
                <Activity size={15} />
                <strong>{trace.hiddenSize.toLocaleString()}</strong>
                <span>hidden dimensions</span>
              </div>
              <div>
                <ScanLine size={15} />
                <strong>
                  {Math.max(...step.layers.map((l) => l.attention.length))}
                </strong>
                <span>attention heads</span>
              </div>
              <div>
                <Cpu size={15} />
                <strong>{trace.duration.toFixed(1)}s</strong>
                <span>capture + projections</span>
              </div>
            </div>
            <section className="layer-step-card">
              <div className="layer-section-title">
                <div>
                  <Workflow size={16} />
                  <h2>One prediction at a time</h2>
                </div>
                <span>{trace.model} · GREEDY</span>
              </div>
              <div className="layer-step-controls">
                <button
                  className="icon-button"
                  aria-label="Previous trace step"
                  disabled={!stepIndex}
                  onClick={() => setStepIndex(stepIndex - 1)}
                >
                  <ChevronLeft size={17} />
                </button>
                <div className="layer-step-tokens">
                  {trace.steps.map((s, i) => (
                    <button
                      key={i}
                      aria-label={`Inspect trace step ${i + 1}`}
                      className={stepIndex === i ? "active" : ""}
                      onClick={() => setStepIndex(i)}
                    >
                      <small>{String(i + 1).padStart(2, "0")}</small>
                      <code>
                        {s.token.isStop ? "[end]" : visible(s.token.text)}
                      </code>
                    </button>
                  ))}
                </div>
                <button
                  className="icon-button"
                  aria-label="Next trace step"
                  disabled={stepIndex === trace.steps.length - 1}
                  onClick={() => setStepIndex(stepIndex + 1)}
                >
                  <ChevronRight size={17} />
                </button>
              </div>
              <div className="layer-step-caption">
                <span>
                  Predicting <code>{visible(step.token.text)}</code> from
                  context position <strong>{step.contextLength - 1}</strong> ·{" "}
                  {step.contextLength} context tokens
                </span>
                <span>
                  {trace.finishReason === "limit"
                    ? "Captured prefix · configured step limit reached"
                    : "Generation reached its stop token"}
                </span>
              </div>
            </section>
            <div className="layer-analysis-grid">
              <section className="layer-panel activation-panel">
                <div className="layer-section-title">
                  <div>
                    <Activity size={16} />
                    <h2>The residual stream</h2>
                  </div>
                  <span>CLICK A ROW TO INSPECT</span>
                </div>
                <p className="layer-panel-intro">
                  Measured block outputs at the last context position. Each
                  column averages a group of hidden dimensions.
                </p>
                <ActivationMap
                  step={step}
                  selected={layerIndex}
                  onSelect={setLayerIndex}
                  hiddenSize={trace.hiddenSize}
                />
              </section>
              <aside className="layer-detail-column">
                <section className="layer-panel">
                  <div className="layer-section-title">
                    <div>
                      <Layers3 size={16} />
                      <h2>Block {layerIndex + 1}</h2>
                    </div>
                    <div className="token-nav">
                      <button
                        aria-label="Previous block"
                        disabled={layerIndex === 0}
                        onClick={() => setLayerIndex(layerIndex - 1)}
                      >
                        <ChevronLeft size={14} />
                      </button>
                      <button
                        aria-label="Next block"
                        disabled={layerIndex === trace.layerCount - 1}
                        onClick={() => setLayerIndex(layerIndex + 1)}
                      >
                        <ChevronRight size={14} />
                      </button>
                    </div>
                  </div>
                  <div className="layer-stats">
                    <div>
                      <span>Activation RMS</span>
                      <strong>{compact(layer.activation.rms)}</strong>
                    </div>
                    <div>
                      <span>Similarity to previous</span>
                      <strong>
                        {layer.activation.cosineToPrevious === null
                          ? "—"
                          : layer.activation.cosineToPrevious.toFixed(3)}
                      </strong>
                    </div>
                    <div>
                      <span>Minimum value</span>
                      <strong>{compact(layer.activation.min)}</strong>
                    </div>
                    <div>
                      <span>Maximum value</span>
                      <strong>{compact(layer.activation.max)}</strong>
                    </div>
                  </div>
                  <div className="layer-detail-note">
                    RMS measures activation magnitude. Similarity is the cosine
                    between adjacent block outputs.
                  </div>
                </section>
                <section className="layer-panel">
                  <div className="layer-section-title">
                    <div>
                      <Zap size={16} />
                      <h2>Projected candidates</h2>
                    </div>
                    <span className="layer-diagnostic">DIAGNOSTIC</span>
                  </div>
                  <p className="layer-panel-intro">
                    The model’s final readout applied to block {layerIndex + 1}.
                    These are untuned logit-lens estimates, not decisions made
                    by this block.
                  </p>
                  {predictions ? (
                    <CandidateRows
                      candidates={predictions}
                      chosen={step.token.id}
                    />
                  ) : (
                    <p className="layer-panel-intro">
                      {step.lens.reason ||
                        "Intermediate projections are unavailable for this architecture."}
                    </p>
                  )}
                </section>
                <section className="layer-panel">
                  <div className="layer-section-title">
                    <div>
                      <ArrowRight size={16} />
                      <h2>Actual final candidates</h2>
                    </div>
                  </div>
                  <CandidateRows
                    candidates={step.candidates.slice(0, 5)}
                    chosen={step.token.id}
                  />
                  <div className="layer-detail-note">
                    Original softmax probabilities from the final model output.
                    Greedy decoding chooses the highest score.
                  </div>
                </section>
              </aside>
            </div>
            <section className="layer-panel attention-panel">
              <div className="layer-section-title">
                <div>
                  <ScanLine size={16} />
                  <h2>Where block {layerIndex + 1} attends</h2>
                </div>
                <label>
                  Inspect{" "}
                  <select
                    aria-label="Attention head"
                    value={head}
                    onChange={(e) => setHead(+e.target.value)}
                  >
                    <option value={-1}>Average of heads</option>
                    {layer.attention.map((_, h) => (
                      <option key={h} value={h}>
                        Head {h + 1}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <p className="layer-panel-intro">
                Attention from the last context position to earlier positions,
                including the model’s chat-template tokens. Hover a cell for its
                exact weight.
              </p>
              {layer.attention.length ? (
                <div className="attention-layout">
                  <div className="attention-grid-scroll">
                    <svg
                      width={Math.max(400, context.length * 26 + 60)}
                      height={layer.attention.length * 22 + 96}
                      role="group"
                      aria-label={`Measured attention weights for block ${layerIndex + 1}`}
                    >
                      {layer.attention.map((weights, h) => (
                        <g key={h}>
                          <text
                            x="0"
                            y={h * 22 + 16}
                            className="attention-head-label"
                          >
                            H{h + 1}
                          </text>
                          {weights.map((p, i) => (
                            <rect
                              key={i}
                              x={i * 26 + 36}
                              y={h * 22 + 2}
                              width="24"
                              height="19"
                              rx="3"
                              fill="#8262c4"
                              fillOpacity={
                                (head === -1 || head === h ? 1 : 0.4) *
                                (0.05 + 0.95 * Math.sqrt(p))
                              }
                            >
                              <title>{`Head ${h + 1} · position ${i} · ${visible(context[i]?.text || "")} · ${(p * 100).toFixed(5)}% · weight ${p.toFixed(8)}`}</title>
                            </rect>
                          ))}
                        </g>
                      ))}
                      {context.map((token, i) => (
                        <text
                          key={i}
                          transform={`translate(${i * 26 + 45},${layer.attention.length * 22 + 13}) rotate(50)`}
                          className="attention-token-label"
                        >
                          {visible(token.text).slice(0, 9)}
                        </text>
                      ))}
                    </svg>
                  </div>
                  <div className="attention-ranking">
                    <h3>
                      {head < 0 ? "Across all heads" : `Head ${head + 1}`}{" "}
                      <span>TOP WEIGHTS</span>
                    </h3>
                    {topAttention.map((item) => (
                      <div key={item.index}>
                        <span className="attention-position">{item.index}</span>
                        <code title={visible(item.token.text)}>
                          {visible(item.token.text)}
                        </code>
                        <strong>{pct(item.weight)}</strong>
                      </div>
                    ))}
                    <div className="attention-entropy">
                      <span>Normalized attention entropy</span>
                      <strong>{pct(entropyPct)}</strong>
                    </div>
                    <p>
                      Low entropy means attention is concentrated on fewer
                      positions.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="layer-panel-intro">
                  This model did not expose separate attention tensors. Its
                  measured block outputs are available above.
                </div>
              )}
            </section>
            <section className="layer-panel prediction-trail">
              <div className="layer-section-title">
                <div>
                  <Zap size={16} />
                  <h2>How the projected prediction changes</h2>
                </div>
                <span>UNTUNED LOGIT LENS</span>
              </div>
              <p className="layer-panel-intro">
                Each row shows a block’s top projected token. Select a row to
                inspect that block’s activations and attention.
              </p>
              {step.lens.available ? (
                <div className="prediction-trail-list">
                  {step.lens.layers.map((item) => (
                    <button
                      key={item.layer}
                      className={item.layer === layerIndex ? "active" : ""}
                      onClick={() => setLayerIndex(item.layer)}
                      aria-label={`Inspect prediction at block ${item.layer + 1}`}
                    >
                      <span>
                        Block {String(item.layer + 1).padStart(2, "0")}
                      </span>
                      <code>{visible(item.candidates[0]?.text || "")}</code>
                      <i
                        style={{
                          width: `${item.candidates[0]?.probability * 100}%`,
                        }}
                      />
                      <strong>
                        {pct(item.candidates[0]?.probability ?? null)}
                      </strong>
                      {item.candidates[0]?.id === step.token.id && (
                        <Check size={12} />
                      )}
                    </button>
                  ))}
                </div>
              ) : (
                <p className="layer-panel-intro">{step.lens.reason}</p>
              )}
            </section>
            <div className="layer-provenance">
              <ShieldCheck size={15} />
              <div>
                <strong>Measured tensors, with an explicit source.</strong>
                <p>
                  {trace.notice} Captured{" "}
                  {new Date(trace.createdAt).toLocaleString()}. Full hidden
                  vectors, attention matrices, model digest, and engine revision
                  are included in the export.
                </p>
                <p>Prompt format: {trace.templateSource}.</p>
                {step.lens.available && (
                  <p>
                    Final readout check: maximum logit difference{" "}
                    <code>
                      {step.lens.finalReadoutMaxLogitError?.toExponential(2)}
                    </code>
                    . Intermediate projections and attention weights help
                    inspect computations; they do not provide a complete
                    explanation of an answer.
                  </p>
                )}
              </div>
            </div>
          </>
        )
      )}
    </div>
  );
}
