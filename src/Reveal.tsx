import { useEffect, useRef, useState, type ReactNode } from "react";

const reduced = () => typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

/** True once the element has scrolled into view (and stays true). */
function useInView<T extends Element>(): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    if (!ref.current || typeof IntersectionObserver === "undefined" || reduced()) return setSeen(true);
    const io = new IntersectionObserver(([e]) => e.isIntersecting && (setSeen(true), io.disconnect()), { threshold: 0.15 });
    io.observe(ref.current);
    return () => io.disconnect();
  }, []);
  return [ref, seen];
}

export function Reveal({ children, className = "", delay = 0, from = "up" }: { children: ReactNode; className?: string; delay?: number; from?: "up" | "left" | "right" | "zoom" }) {
  const [ref, seen] = useInView<HTMLDivElement>();
  return (
    <div ref={ref} className={`rv rv-${from} ${seen ? "in" : ""} ${className}`} style={{ transitionDelay: `${delay}ms` }}>
      {children}
    </div>
  );
}

export function Count({ to, suffix = "" }: { to: number; suffix?: string }) {
  const [ref, seen] = useInView<HTMLSpanElement>();
  const [n, setN] = useState(reduced() ? to : 0);
  useEffect(() => {
    if (!seen || reduced()) return setN(to);
    const t0 = performance.now();
    let raf = requestAnimationFrame(function tick(t) {
      const k = Math.min(1, (t - t0) / 1400);
      setN(Math.round(to * (1 - Math.pow(1 - k, 3))));
      if (k < 1) raf = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(raf);
  }, [seen, to]);
  return <span ref={ref}>{n}{suffix}</span>;
}

/** Writes scroll position to CSS variables: --sy (px) for parallax, --p (0..1) for the progress bar. */
export function useScrollVars() {
  useEffect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      const d = document.documentElement;
      d.style.setProperty("--sy", String(Math.round(scrollY)));
      d.style.setProperty("--p", String(Math.min(1, scrollY / Math.max(1, d.scrollHeight - innerHeight))));
    };
    const on = () => { if (!raf) raf = requestAnimationFrame(update); };
    update();
    addEventListener("scroll", on, { passive: true });
    addEventListener("resize", on);
    return () => { removeEventListener("scroll", on); removeEventListener("resize", on); cancelAnimationFrame(raf); };
  }, []);
}
