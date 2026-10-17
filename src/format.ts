export const pct = (value: number | null) =>
  value === null
    ? "Unavailable"
    : value === 0
      ? "0%"
      : value < 0.0001
        ? "< 0.01%"
        : `${(value * 100).toFixed(value < 0.01 ? 2 : 1)}%`;

export const visible = (text: string) =>
  text.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "⇥") || "∅";
