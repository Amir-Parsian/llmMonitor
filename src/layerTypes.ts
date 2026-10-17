export type LayerCandidate = { id: number; text: string; probability: number };
export type LayerMeasurement = {
  index: number;
  activation: {
    values: number[];
    bins: number[];
    rms: number;
    mean: number;
    min: number;
    max: number;
    cosineToPrevious: number | null;
  };
  attention: number[][];
};
export type TraceStep = {
  index: number;
  contextLength: number;
  token: LayerCandidate & { isStop: boolean };
  candidates: LayerCandidate[];
  layers: LayerMeasurement[];
  lens: {
    available: boolean;
    reason?: string;
    finalReadoutMaxLogitError?: number;
    layers: { layer: number; candidates: LayerCandidate[] }[];
  };
};
export type LayerTrace = {
  id: string;
  model: string;
  prompt: string;
  modelDigest: string;
  engine: string;
  revision: string;
  mode: string;
  sampling: string;
  layerCount: number;
  hiddenSize: number;
  promptTokenCount: number;
  templateSource: string;
  steps: TraceStep[];
  text: string;
  duration: number;
  finishReason: string;
  createdAt: string;
  notice: string;
  contextTokens: {
    id: number;
    text: string;
    position: number;
    source: string;
  }[];
};
export type TracingStatus = {
  ready: boolean;
  busy: boolean;
  models: { name: string; size: number; digest: string; fitsMemory: boolean }[];
  revision: string;
  setupCommand: string;
};
