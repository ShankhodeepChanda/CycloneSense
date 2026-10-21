import { useCallback, useEffect, useState } from "react";
import type { ActiveCycloneResult, ActiveCyclone } from "../server/gdacs";
import { LiveMap } from "./LiveMap";
import { LAYERS, iso, resolveNrt, slotAt, type Variant } from "./gibs";
import { Count, Reveal, useScrollVars } from "./Reveal";

const utcDay = (d: Date) => d.toISOString().slice(0, 10);
// Today's daily mosaic only exists once the daytime pass over India has been processed.
const defaultDay = () => utcDay(new Date(Date.now() - (new Date().getUTCHours() < 10 ? 86_400_000 : 0)));
const ALERT_TEXT: Record<string, string> = { Red: "Red alert", Orange: "Orange alert", Green: "Green" };

const arm = (k: number) => {
  const pts: string[] = [];
  for (let i = 0; i <= 60; i++) {
    const t = i / 60, a = k * (Math.PI * 2 / 5) + t * 2.8 * Math.PI, r = 22 + t * 172;
    pts.push(`${(200 + r * Math.cos(a)).toFixed(1)},${(200 + r * Math.sin(a)).toFixed(1)}`);
  }
  return "M" + pts.join("L");
};

function Storm() {
  return (
    <div className="storm-wrap" aria-hidden="true">
      <svg className="storm" viewBox="0 0 400 400">
        <defs>
          <radialGradient id="glow"><stop offset="0" stopColor="#67e8f9" stopOpacity=".35" /><stop offset="1" stopColor="#67e8f9" stopOpacity="0" /></radialGradient>
          <linearGradient id="armg" x1="0" x2="1"><stop offset="0" stopColor="#e0f2fe" /><stop offset="1" stopColor="#38bdf8" stopOpacity=".2" /></linearGradient>
        </defs>
        <circle cx="200" cy="200" r="198" fill="url(#glow)" />
        {[0, 1, 2, 3, 4].map((k) => (
          <g key={k}>
            <path d={arm(k)} fill="none" stroke="#7dd3fc" strokeOpacity=".07" strokeWidth="26" />
            <path d={arm(k)} fill="none" stroke="url(#armg)" strokeOpacity=".7" strokeWidth="3" strokeLinecap="round" />
          </g>
        ))}
        <g className="inner">
          {[120, 80, 46].map((r) => <circle key={r} cx="200" cy="200" r={r} fill="none" stroke="#bae6fd" strokeOpacity=".18" strokeDasharray="3 9" />)}
        </g>
        <circle className="eye" cx="200" cy="200" r="15" fill="#06121f" stroke="#e0f2fe" strokeOpacity=".8" strokeWidth="2" />
      </svg>
    </div>
  );
}

export function App() {
  useScrollVars();
  const [solid, setSolid] = useState(false);
  const [layerId, setLayerId] = useState(LAYERS[0].id);
  const [day, setDay] = useState(defaultDay);
  const [nrt, setNrt] = useState<{ variant: Variant; delayMin: number } | null | "loading">("loading");
  const [extra, setExtra] = useState(0); // minutes stepped back from the newest working image
  const [feed, setFeed] = useState<ActiveCycloneResult | null>(null);
  const [focus, setFocus] = useState<ActiveCyclone | null>(null);
  const [blank, setBlank] = useState(false);

  const layer = LAYERS.find((l) => l.id === layerId)!;
  const resolved = nrt !== "loading" && nrt !== null ? nrt : null;
  const variant = layer.kind === "nrt" && resolved ? resolved.variant : layer.variants[0];
  const when = slotAt((resolved?.delayMin ?? 60) + extra);
  const time = layer.kind === "daily" ? day : iso(when);
  const onTileProblem = useCallback((p: boolean) => setBlank(p), []);
  const nioList = feed?.cyclones.filter((c) => c.nio) ?? [];
  const elsewhere = feed?.cyclones.filter((c) => !c.nio) ?? [];

  useEffect(() => {
    const on = () => setSolid(scrollY > 40);
    on();
    addEventListener("scroll", on, { passive: true });
    return () => removeEventListener("scroll", on);
  }, []);

  useEffect(() => {
    resolveNrt(LAYERS.find((l) => l.kind === "nrt")!).then(setNrt);
  }, []);

  useEffect(() => {
    const load = () =>
      fetch("/api/active-cyclones")
        .then((r) => r.json())
        .then(setFeed)
        .catch(() => setFeed({ ok: false, fetchedAt: null, cyclones: [], source: "" }));
    load();
    const t = setInterval(load, 10 * 60_000);
    return () => clearInterval(t);
  }, []);

  const show = (c: ActiveCyclone) => {
    setFocus(c);
    document.getElementById("map")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const pill = (on: boolean) => `rounded-full border px-4 py-2 text-sm font-medium transition ${on ? "border-cyan-700 bg-cyan-700 text-white shadow" : "border-slate-300 bg-white hover:bg-slate-100"}`;

  return (
    <>
      <div className="progress" aria-hidden="true" />
      <nav className={`nav ${solid ? "solid" : ""}`} aria-label="Main">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <a href="#top" className="flex items-center gap-2 text-lg font-bold tracking-tight">
            <svg width="26" height="26" viewBox="0 0 400 400" aria-hidden="true"><path d={arm(0)} fill="none" stroke="currentColor" strokeWidth="26" strokeLinecap="round" /><path d={arm(2)} fill="none" stroke="currentColor" strokeWidth="26" strokeLinecap="round" /></svg>
            CycloneSense
          </a>
          <div className="hidden gap-6 text-sm font-medium sm:flex">
            <a href="#map">Live map</a><a href="#how">How it works</a><a href="#data">Data and model</a><a href="#safety">Safety</a>
          </div>
        </div>
      </nav>

      <header id="top" className="hero">
        <Storm />
        <div className="hero-copy mx-auto w-full max-w-6xl px-4 pt-20">
          <div className="hero-in max-w-2xl">
            <span className="chip-live"><i className="dot" />
              {feed === null ? "Checking live cyclone feed…" : !feed.ok ? "Live cyclone feed unavailable" : `${nioList.length} active near India · ${elsewhere.length} elsewhere`}
            </span>
            <h1 className="mt-5">Watch cyclones <span className="grad">build over India's seas.</span></h1>
            <p className="mt-5 max-w-xl text-lg text-sky-100/90">
              Live NASA and Himawari satellite imagery of the Bay of Bengal and Arabian Sea, with every active tropical cyclone on the map. Free, no sign-in.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <a href="#map" className="btn btn-primary">Open the live map ↓</a>
              <a href="#how" className="btn btn-ghost">How it works</a>
            </div>
          </div>
        </div>
        <div className="hint" aria-hidden="true" />
      </header>

      <section className="mx-auto max-w-6xl px-4 py-16" aria-label="At a glance">
        <div className="grid grid-cols-2 gap-6 text-center md:grid-cols-4">
          {[[<Count to={3} />, "satellite layers"], [<Count to={10} />, "minute Himawari refresh"], [<Count to={20} />, "automated tests"], [<>70/15/15</>, "split by storm ID"]].map(([n, l], i) => (
            <Reveal key={i} delay={i * 110}><div className="stat">{n}</div><div className="mt-1 text-sm text-slate-600">{l}</div></Reveal>
          ))}
        </div>
      </section>

      <section id="map" className="mx-auto max-w-6xl px-4 pb-20">
        <Reveal><h2 className="text-3xl font-extrabold tracking-tight">Live satellite map</h2>
          <p className="mt-2 max-w-2xl text-slate-600">Pick a layer, step through time, and tap a cyclone to fly to it. Click the map once to zoom with your scroll wheel.</p></Reveal>
        <div className="mt-6 grid gap-5 lg:grid-cols-[1fr_330px]">
          <Reveal from="zoom">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              {LAYERS.map((l) => (
                <button key={l.id} aria-pressed={l.id === layerId} onClick={() => setLayerId(l.id)} className={pill(l.id === layerId)}>
                  {l.label}{l.kind === "nrt" && (nrt === "loading" ? " (checking…)" : nrt === null ? " (unavailable)" : "")}
                </button>
              ))}
            </div>
            <div className="h-[62vh] min-h-[380px] overflow-hidden rounded-[18px] shadow-xl ring-1 ring-slate-900/10">
              <LiveMap layer={layer} variant={variant} time={time} cyclones={feed?.cyclones ?? []} focus={focus} onTileProblem={onTileProblem} />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
              {layer.kind === "daily" ? (
                <label className="flex items-center gap-2">Date (UTC)
                  <input type="date" value={day} max={utcDay(new Date())} onChange={(e) => e.target.value && setDay(e.target.value)} className="rounded border border-slate-300 bg-white px-2 py-1" />
                </label>
              ) : (
                <>
                  <span>Satellite time: {when.toISOString().slice(11, 16)} UTC / {when.toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit" })} IST</span>
                  <button onClick={() => setExtra((e) => e + 10)} className="rounded border border-slate-300 bg-white px-3 py-1 hover:bg-slate-100">Earlier</button>
                  <button onClick={() => setExtra((e) => Math.max(30 - (resolved?.delayMin ?? 60), e - 10))} className="rounded border border-slate-300 bg-white px-3 py-1 hover:bg-slate-100">Later</button>
                </>
              )}
            </div>
            <p className="mt-2 text-sm text-slate-600">{layer.note}</p>
            {layer.kind === "nrt" && nrt === "loading" && <p role="status" className="mt-2 rounded bg-sky-50 p-2 text-sm text-sky-900">Finding the latest Himawari image…</p>}
            {layer.kind === "nrt" && nrt === null && <p role="status" className="mt-2 rounded bg-amber-50 p-2 text-sm text-amber-900">The near real-time layer is not answering right now. Try the daily layers.</p>}
            {blank && (resolved || layer.kind === "daily") && (
              <p role="status" className="mt-2 rounded bg-amber-50 p-2 text-sm text-amber-900">
                This image is not available yet. {layer.kind === "daily" ? "Pick the previous day." : "Press Earlier to step back 10 minutes."}
              </p>
            )}
          </Reveal>

          <Reveal from="right" delay={150}>
            <aside className="space-y-4">
              <section className="card" aria-live="polite">
                <h3 className="text-lg font-bold">Active cyclones</h3>
                {feed === null && <p className="mt-2 text-sm text-slate-600">Loading…</p>}
                {feed && !feed.ok && (
                  <p className="mt-2 text-sm text-amber-900">The live cyclone feed could not be reached. Please check the official IMD bulletin below.
                    {feed.error && <span className="mt-1 block text-xs text-slate-500">Reason: {feed.error}</span>}</p>
                )}
                {feed?.ok && nioList.length === 0 && <p className="mt-2 text-sm text-slate-600">No active tropical cyclones are reported in the Bay of Bengal or Arabian Sea right now.</p>}
                {nioList.length > 0 && <List items={nioList} onPick={show} />}
                {feed?.ok && elsewhere.length > 0 && (
                  <details className="mt-3">
                    <summary className="cursor-pointer text-sm font-medium text-cyan-800">Elsewhere in the world ({elsewhere.length})</summary>
                    <List items={elsewhere} onPick={show} />
                  </details>
                )}
                {feed?.ok && (
                  <p className="mt-3 text-xs text-slate-500">Source: {feed.source}. {feed.stale ? "Showing the last successful update. " : ""}
                    Checked {feed.fetchedAt ? new Date(feed.fetchedAt).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata" }) + " IST" : ""}.</p>
                )}
              </section>
            </aside>
          </Reveal>
        </div>
      </section>

      <section id="how" className="bg-white py-20">
        <div className="mx-auto max-w-6xl px-4">
          <Reveal><h2 className="text-3xl font-extrabold tracking-tight">How it works</h2>
            <p className="mt-2 max-w-2xl text-slate-600">Three steps from raw satellite data to something you can read at a glance.</p></Reveal>
          <div className="mt-10 grid gap-6 md:grid-cols-3">
            {[["1", "Satellites look down", "NASA VIIRS and MODIS pass over once a day. Japan's Himawari-9 sends a new infrared image every 10 minutes."],
              ["2", "Storms are tracked", "GDACS, the UN and EU alert system, reports every active tropical cyclone. We place each one on the map."],
              ["3", "You see it live", "Layers, time controls and storm markers in one page. No account and no API key."]].map(([n, h, b], i) => (
              <Reveal key={n} delay={i * 160}>
                <div className="card h-full"><div className="num">{n}</div><h3 className="mt-4 text-xl font-bold">{h}</h3><p className="mt-2 text-slate-600">{b}</p></div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section id="data" className="mx-auto max-w-6xl px-4 py-20">
        <Reveal><h2 className="text-3xl font-extrabold tracking-tight">The research behind it</h2></Reveal>
        <div className="mt-8 grid gap-6 md:grid-cols-2">
          <Reveal from="left"><div className="card h-full">
            <h3 className="text-xl font-bold">An open dataset</h3>
            <p className="mt-2 text-slate-600">NOAA IBTrACS storm records joined with NASA MODIS satellite chips, split by storm ID so no storm appears in both training and test data. Built for Kaggle.</p>
          </div></Reveal>
          <Reveal from="right" delay={120}><div className="card h-full">
            <h3 className="text-xl font-bold">An intensity model</h3>
            <p className="mt-2 text-slate-600">A ResNet-18 learns to estimate maximum wind speed from a satellite chip and is scored against a predict-the-average baseline. Results will be published after the full training run.</p>
          </div></Reveal>
        </div>
      </section>

      <section id="safety" className="bg-slate-900 py-16 text-white">
        <Reveal className="mx-auto max-w-3xl px-4 text-center">
          <h2 className="text-3xl font-extrabold tracking-tight">Official warnings come first</h2>
          <p className="mt-3 text-slate-300">CycloneSense is a research project, not a warning system. For warnings and evacuation advice, follow the India Meteorological Department and your state disaster authority.</p>
          <a href="https://mausam.imd.gov.in/" target="_blank" rel="noreferrer" className="btn btn-primary mt-6">IMD cyclone bulletins ↗</a>
        </Reveal>
      </section>

      <footer className="mx-auto max-w-6xl px-4 py-8 text-xs text-slate-500">
        Imagery: NASA GIBS / EOSDIS, JMA Himawari. Cyclone positions: GDACS. Coastlines: Natural Earth.
      </footer>
    </>
  );
}

function List({ items, onPick }: { items: ActiveCyclone[]; onPick: (c: ActiveCyclone) => void }) {
  return (
    <ul className="mt-2 space-y-2">
      {items.map((c) => (
        <li key={c.id}>
          <button onClick={() => onPick(c)} className="w-full rounded-xl border border-slate-200 p-2.5 text-left transition hover:border-cyan-500 hover:bg-cyan-50">
            <span className="font-semibold">{c.name}</span>
            <span className="block text-sm text-slate-600">
              {ALERT_TEXT[c.alert] ?? c.alert}{c.windKt ? ` · ${c.windKt} kt winds` : ""} · {c.lat.toFixed(1)}°{c.lat >= 0 ? "N" : "S"} {Math.abs(c.lon).toFixed(1)}°{c.lon >= 0 ? "E" : "W"}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
