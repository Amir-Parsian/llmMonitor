import type { Exploration } from "./types";

const pieces = [
  "The",
  " sky",
  " looks",
  " blue",
  " because",
  " of",
  " the",
  " way",
  " sunlight",
  " interacts",
  " with",
  " our",
  " atmosphere",
  ".\n\n",
  "Sun",
  "light",
  " contains",
  " all",
  " the",
  " colors",
  " of",
  " the",
  " rainbow",
  ".",
  " As",
  " it",
  " passes",
  " through",
  " the",
  " atmosphere",
  ",",
  " tiny",
  " molecules",
  " scatter",
  " shorter",
  " blue",
  " wavelengths",
  " more",
  " than",
  " longer",
  " red",
  " wavelengths",
  ".",
  " This",
  " process",
  " is",
  " called",
  " Ray",
  "leigh",
  " scattering",
  ".\n\n",
  "That",
  " scattered",
  " blue",
  " light",
  " reaches",
  " your",
  " eyes",
  " from",
  " every",
  " direction",
  " —",
  " giving",
  " the",
  " sky",
  " its",
  " familiar",
  " blue",
  " color",
  ".",
];
const words = [
  " the",
  " a",
  " blue",
  " light",
  " our",
  " tiny",
  " sunlight",
  " sky",
  " air",
  " more",
  " shorter",
  " all",
  " of",
  " to",
  " this",
  " in",
  " from",
  " each",
  " when",
  " because",
];
export function makeDemo(): Exploration {
  let offset = 0;
  const tokens = pieces.map((text, index) => {
    const p =
      index === 33
        ? 0.426
        : Math.min(0.985, 0.26 + ((index * 37 + 23) % 73) / 100);
    const alternateWords =
      index === 33
        ? [
            " reflect",
            " absorb",
            " redirect",
            " spread",
            " disperse",
            " deflect",
            " interact",
            " bend",
            " filter",
            " transmit",
            " refract",
            " bounce",
            " emit",
            " split",
            " distribute",
            " diffuse",
            " separate",
            " change",
            " move",
          ]
        : words.filter((w) => w !== text).slice(0, 19);
    const weights = alternateWords.map((_, i) => Math.exp(-i * 0.34));
    const total = weights.reduce((a, b) => a + b, 0);
    const candidates = [
      { text, probability: p, logprob: Math.log(p), selected: true, rank: 1 },
      ...alternateWords.map((word, i) => ({
        text: word,
        probability: ((1 - p) * 0.87 * weights[i]) / total,
        logprob: Math.log(((1 - p) * 0.87 * weights[i]) / total),
        selected: false,
        rank: i + 2,
      })),
    ]
      .sort((a, b) => b.probability - a.probability)
      .map((c, i) => ({ ...c, rank: i + 1 }));
    const bytes = Array.from(new TextEncoder().encode(text));
    const start = offset;
    offset += bytes.length;
    return {
      index,
      text,
      probability: p,
      logprob: Math.log(p),
      bytes,
      candidates,
      start,
      end: offset,
    };
  });
  return {
    id: "demo",
    createdAt: new Date().toISOString(),
    provider: "ollama",
    model: "Illustrative example",
    prompt: "Why is the sky blue? Explain it simply.",
    text: pieces.join(""),
    tokens,
    temperature: 0.7,
    seed: 42,
    maxTokens: 256,
    requestedTopN: 20,
    actualTopN: 20,
    duration: 0,
    tokensPerSecond: null,
    warning: null,
    demo: true,
    probabilityBasis:
      "Illustrative probabilities, created to demonstrate the interface.",
  };
}
