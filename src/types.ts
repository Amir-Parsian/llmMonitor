export type Provider = "ollama" | "llamacpp";
export type Candidate = {
  text: string;
  probability: number;
  logprob: number | null;
  id?: number;
  rank: number;
  selected: boolean;
};
export type Token = {
  index: number;
  text: string;
  probability: number | null;
  logprob: number | null;
  bytes: number[];
  candidates: Candidate[];
  start: number | null;
  end: number | null;
};
export type Exploration = {
  id: string;
  createdAt: string;
  provider: Provider;
  model: string;
  prompt: string;
  text: string;
  tokens: Token[];
  temperature: number;
  seed: number;
  maxTokens: number;
  requestedTopN: number;
  actualTopN: number;
  duration: number;
  tokensPerSecond: number | null;
  finishReason?: string;
  warning: string | null;
  demo: boolean;
  probabilityBasis: string;
};
export type Settings = {
  provider: Provider;
  baseUrl: string;
  temperature: number;
  maxTokens: number;
  topN: number;
  seed: number;
};
export type Model = { name: string; size: string; quantization: string };
