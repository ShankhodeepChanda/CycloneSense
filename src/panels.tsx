import type { ReactNode } from "react";

const PATHS: Record<string, string> = {
  dash: "M3 3h7v9H3z M14 3h7v5h-7z M14 12h7v9h-7z M3 16h7v5H3z",
  map: "M9 3 3 5v16l6-2 6 2 6-2V3l-6 2z M9 3v16 M15 5v16",
  storm: "M12 12m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0 M12 4a8 8 0 0 1 8 8 M12 20a8 8 0 0 1-8-8 M12 1a11 11 0 0 1 11 11 M12 23A11 11 0 0 1 1 12",
  clock: "M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0-18 0 M12 7v5l3 2",
  temp: "M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z",
  flask: "M10 2v7.5L4 19a2 2 0 0 0 1.7 3h12.6a2 2 0 0 0 1.7-3l-6-9.5V2 M8.5 2h7",
  shield: "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z",
  search: "M11 11m-7 0a7 7 0 1 0 14 0a7 7 0 1 0-14 0 M21 21l-4.3-4.3",
  play: "M6 4l14 8-14 8z",
  pause: "M6 4h4v16H6z M14 4h4v16h-4z",
  layers: "M12 2 2 7l10 5 10-5z M2 17l10 5 10-5 M2 12l10 5 10-5",
  bell: "M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9 M13.7 21a2 2 0 0 1-3.4 0",
  skip: "M5 4l10 8-10 8z M19 5v14",
};

export const Icon = ({ name, size = 18 }: { name: string; size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d={PATHS[name]} />
  </svg>
);

export const Card = ({ id, title, icon, aside, i = 0, className = "", children }: { id?: string; title: string; icon?: string; aside?: ReactNode; i?: number; className?: string; children: ReactNode }) => (
  <section id={id} className={`card ${className}`} style={{ ["--i" as string]: i }}>
    <header className="card-h">
      <h2>{icon && <Icon name={icon} size={16} />}{title}</h2>
      {aside}
    </header>
    {children}
  </section>
);

/** Ring gauge: value out of max, with a label. `max` is the scale's top, shown in the caption. */
export function Gauge({ value, max, text, label, caption, color }: { value: number; max: number; text: string; label: string; caption: string; color: string }) {
  const r = 30, c = 2 * Math.PI * r, k = Math.max(0, Math.min(1, value / max));
  return (
    <div className="gauge" role="img" aria-label={`${label}: ${text}. ${caption}`}>
      <svg viewBox="0 0 80 80">
        <circle cx="40" cy="40" r={r} fill="none" stroke="var(--line)" strokeWidth="7" />
        <circle cx="40" cy="40" r={r} fill="none" stroke={color} strokeWidth="7" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - k)} transform="rotate(-90 40 40)" className="arc" />
        <text x="40" y="45" textAnchor="middle" className="g-val">{text}</text>
      </svg>
      <b>{label}</b>
      <small>{caption}</small>
    </div>
  );
}

/** Hourly pressure line with a marker at "now". Axis shows the real min and max of the series. */
export function PressureChart({ series, nowIndex }: { series: number[]; nowIndex: number }) {
  const ok = series.filter((v) => Number.isFinite(v));
  if (ok.length < 4) return <p className="muted">No pressure series available.</p>;
  const lo = Math.min(...ok), hi = Math.max(...ok), pad = Math.max(0.5, (hi - lo) * 0.15);
  const W = 300, H = 110, x = (i: number) => 8 + (i / (series.length - 1)) * (W - 16), y = (v: number) => 10 + (1 - (v - (lo - pad)) / (hi - lo + 2 * pad)) * (H - 30);
  const pts = series.map((v, i) => (Number.isFinite(v) ? `${x(i).toFixed(1)},${y(v).toFixed(1)}` : null)).filter(Boolean) as string[];
  const nx = x(nowIndex);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={`Surface pressure from ${Math.round(lo)} to ${Math.round(hi)} hectopascals over 48 hours`}>
      <rect x={nx} y="6" width={W - 8 - nx} height={H - 26} fill="var(--blue-soft)" opacity=".7" />
      <polyline points={pts.join(" ")} fill="none" stroke="var(--blue)" strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
      <line x1={nx} x2={nx} y1="6" y2={H - 20} stroke="var(--ink)" strokeDasharray="3 3" />
      <circle cx={nx} cy={y(series[nowIndex])} r="4" fill="var(--blue)" stroke="#fff" strokeWidth="2" />
      <text x="8" y={H - 4} className="ax">-24 h</text>
      <text x={nx} y={H - 4} textAnchor="middle" className="ax">now</text>
      <text x={W - 8} y={H - 4} textAnchor="end" className="ax">+24 h</text>
      <text x="10" y="18" className="ax">{hi.toFixed(0)} hPa</text>
      <text x="10" y={H - 24} className="ax">{lo.toFixed(0)} hPa</text>
    </svg>
  );
}
