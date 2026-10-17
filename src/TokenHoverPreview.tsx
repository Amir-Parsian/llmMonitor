import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import type { FocusEvent, PointerEvent } from "react";
import { createPortal } from "react-dom";
import { ArrowUpRight, Check, X } from "lucide-react";
import { pct, visible } from "./format";
import type { Token } from "./types";

type Preview = { token: Token; anchor: HTMLElement; anchorRect: DOMRect };

export function useTokenHoverPreview(
  resetKey: string,
  onPin: (index: number) => void,
) {
  const [active, setActive] = useState<Preview | null>(null);
  const [position, setPosition] = useState<{
    left: number;
    top: number;
  } | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const opening = useRef<number | undefined>(undefined);
  const closing = useRef<number | undefined>(undefined);
  const id = useId();

  function cancelTimers() {
    window.clearTimeout(opening.current);
    window.clearTimeout(closing.current);
  }

  function dismiss() {
    cancelTimers();
    setActive(null);
    setPosition(null);
  }

  function open(
    token: Token,
    anchor: HTMLElement,
    delay: number,
    point?: { x: number; y: number },
  ) {
    cancelTimers();
    // A leading space can wrap onto a different line from its token. Anchor
    // the preview to the fragment under the pointer, rather than both lines.
    const fragments = Array.from(anchor.getClientRects());
    const anchorRect =
      (point
        ? fragments.find(
            (rect) =>
              point.x >= rect.left &&
              point.x <= rect.right &&
              point.y >= rect.top &&
              point.y <= rect.bottom,
          )
        : fragments.find((rect) => rect.width > 1)) ??
      anchor.getBoundingClientRect();
    opening.current = window.setTimeout(() => {
      setPosition(null);
      setActive({ token, anchor, anchorRect });
    }, delay);
  }

  function leave() {
    window.clearTimeout(opening.current);
    window.clearTimeout(closing.current);
    // Keep the preview available while the pointer crosses the small gap to it.
    closing.current = window.setTimeout(dismiss, 180);
  }

  function blur(event: FocusEvent<HTMLElement>) {
    const next = event.relatedTarget as Node | null;
    if (
      next &&
      (cardRef.current?.contains(next) || active?.anchor.contains(next))
    )
      return;
    leave();
  }

  useEffect(() => {
    dismiss();
    return cancelTimers;
  }, [resetKey]);

  useLayoutEffect(() => {
    if (!active || !cardRef.current) return;
    if (!active.anchor.isConnected) {
      dismiss();
      return;
    }
    const anchor = active.anchorRect;
    const card = cardRef.current.getBoundingClientRect();
    const gutter = 12;
    const below = window.innerHeight - anchor.bottom - gutter - 8;
    const above = anchor.top - gutter - 8;
    const top =
      below >= card.height || below >= above
        ? anchor.bottom + 8
        : anchor.top - card.height - 8;
    setPosition({
      left: Math.max(
        gutter,
        Math.min(
          anchor.left,
          document.documentElement.clientWidth - card.width - gutter,
        ),
      ),
      top: Math.max(
        gutter,
        Math.min(top, window.innerHeight - card.height - gutter),
      ),
    });
  }, [active]);

  useEffect(() => {
    if (!active) return;
    function keydown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        if (cardRef.current?.contains(document.activeElement))
          active?.anchor.focus();
        dismiss();
      }
    }
    function outside(event: Event) {
      const target = event.target as Node;
      if (
        !cardRef.current?.contains(target) &&
        !active?.anchor.contains(target)
      )
        dismiss();
    }
    function scroll(event: Event) {
      if (!cardRef.current?.contains(event.target as Node)) dismiss();
    }
    document.addEventListener("keydown", keydown);
    document.addEventListener("pointerdown", outside);
    document.addEventListener("scroll", scroll, true);
    window.addEventListener("resize", dismiss);
    return () => {
      document.removeEventListener("keydown", keydown);
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("scroll", scroll, true);
      window.removeEventListener("resize", dismiss);
    };
  }, [active]);

  const triggerProps = (token: Token) => ({
    onPointerEnter: (event: PointerEvent<HTMLElement>) => {
      if (event.pointerType !== "touch")
        open(token, event.currentTarget, 90, {
          x: event.clientX,
          y: event.clientY,
        });
    },
    onPointerLeave: leave,
    onFocus: (event: FocusEvent<HTMLElement>) =>
      open(token, event.currentTarget, 0),
    onBlur: blur,
    "aria-haspopup": "dialog" as const,
    "aria-expanded": active?.token.index === token.index,
    "aria-controls": active?.token.index === token.index ? id : undefined,
  });

  const preview =
    active &&
    createPortal(
      <div
        id={id}
        ref={cardRef}
        className="token-hover-preview"
        role="dialog"
        aria-label={`Candidates for token ${active.token.index + 1}: ${visible(active.token.text)}`}
        style={{
          left: position?.left ?? 0,
          top: position?.top ?? 0,
          visibility: position ? "visible" : "hidden",
        }}
        onPointerEnter={cancelTimers}
        onPointerLeave={leave}
        onFocusCapture={cancelTimers}
        onBlurCapture={blur}
      >
        <div className="hover-preview-heading">
          <span>
            TOKEN {String(active.token.index + 1).padStart(2, "0")} ·{" "}
            {active.token.candidates.length} CANDIDATES
          </span>
          <button aria-label="Close token preview" onClick={dismiss}>
            <X size={14} />
          </button>
        </div>
        <div className="hover-preview-selected">
          <code>{visible(active.token.text)}</code>
          <strong>{pct(active.token.probability)}</strong>
        </div>
        <div className="hover-preview-labels">
          <span>CANDIDATE TOKEN</span>
          <span>PROBABILITY</span>
        </div>
        <div
          className="hover-preview-candidates"
          role="list"
          aria-label="Candidate probabilities"
          tabIndex={0}
        >
          {active.token.candidates.length ? (
            active.token.candidates.map((candidate, index) => (
              <div
                role="listitem"
                className={`candidate-row ${candidate.selected ? "chosen" : ""}`}
                key={index}
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
                {candidate.selected && (
                  <Check size={12} aria-label="Chosen token" />
                )}
                <span className="candidate-pct">
                  {pct(candidate.probability)}
                </span>
              </div>
            ))
          ) : (
            <p className="hover-preview-empty">
              This server did not return candidate probabilities.
            </p>
          )}
        </div>
        <div className="hover-preview-footer">
          <span>
            · = space ·{" "}
            {active.token.candidates.length
              ? "Scroll for all candidates"
              : "No probabilities available"}
          </span>
          <button
            onClick={() => {
              onPin(active.token.index);
              dismiss();
            }}
          >
            Pin to inspector <ArrowUpRight size={12} />
          </button>
        </div>
      </div>,
      document.body,
    );

  return { triggerProps, preview, dismiss };
}
