import { useCallback, useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import type { ActiveCycloneResult, ActiveCyclone } from "../server/gdacs";
import { LiveMap, PLACES } from "./LiveMap";
import { LAYERS, iso, resolveNrt, slotAt, type Variant } from "./gibs";
import { fetchWeather, type Weather } from "./weather";
import { Card, Gauge, Icon, PressureChart } from "./panels";
import { WindEstimator } from "./WindEstimator";

const DAY = 86_400_000;
const utcDay = (d: Date) => d.toISOString().slice(0, 10);
// Today's daily mosaic only exists once the daytime pass over India has been processed.
const defaultDay = () => utcDay(new Date(Date.now() - (new Date().getUTCHours() < 10 ? DAY : 0)));
const ALERT_TEXT: Record<string, string> = { Red: "Red alert", Orange: "Orange alert", Green: "Green" };
const ALERT_COLOR: Record<string, string> = { Red: "var(--red)", Orange: "var(--orange)", Green: "var(--green)" };
const CITIES = PLACES.filter((p) => p[3] === "city");
const NAV: [string, string, string][] = [["top", "Dashboard", "dash"], ["map", "Live map", "map"], ["alerts", "Cyclones", "storm"], ["timeline", "Timeline", "clock"], ["weather", "Weather", "temp"], ["research", "Research", "flask"], ["safety", "Safety", "shield"]];
const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
const when = (s: string | null) => (s ? new Date(/Z|[+-]\d\d:?\d\d$/.test(s) ? s : s + "Z").toLocaleString("en-IN", { timeZone: "UTC", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hour12: false }) + " UTC" : "unknown");
const kmh = (kt: number) => Math.round(kt * 1.852);

function useReveal() {
  const ref = useRef<HTMLDivElement>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    if (!ref.current || typeof IntersectionObserver === "undefined" || matchMedia("(prefers-reduced-motion: reduce)").matches) return setSeen(true);
    const io = new IntersectionObserver(([e]) => e.isIntersecting && (setSeen(true), io.disconnect()), { threshold: 0.2 });
    io.observe(ref.current);
    return () => io.disconnect();
  }, []);
  return [ref, seen] as const;
}
function Reveal({ children, delay = 0 }: { children: ReactNode; delay?: number }) {
  const [ref, seen] = useReveal();
  return <div ref={ref} className={`rv ${seen ? "in" : ""}`} style={{ transitionDelay: `${delay}ms` }}>{children}</div>;
}

export function App() {
  const [feed, setFeed] = useState<ActiveCycloneResult | null>(null);
  const [nrt, setNrt] = useState<{ variant: Variant; delayMin: number } | null | "loading">("loading");
  const [layerId, setLayerId] = useState(LAYERS[0].id);
  const [step, setStep] = useState(0); // frames back from the newest image
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [opacity, setOpacity] = useState(1);
  const [showCoast, setShowCoast] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showStorms, setShowStorms] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<{ center: [number, number]; zoom: number; n: number }>({ center: [20, 128], zoom: 3, n: 0 });
  const [blank, setBlank] = useState(false);
  const [city, setCity] = useState("Chennai");
  const [wx, setWx] = useState<{ s: "loading" } | { s: "ok"; w: Weather } | { s: "err"; msg: string }>({ s: "loading" });
  const [clock, setClock] = useState(() => new Date());
  const [query, setQuery] = useState("");
  const [miss, setMiss] = useState(false);
  const [active, setActive] = useState("top");
  const [layersOpen] = useState(() => innerWidth > 1400); // starts closed on smaller screens so the map stays visible
  const n = useRef(0);
  const fly = useCallback((center: [number, number], zoom: number) => setView({ center, zoom, n: ++n.current }), []);

  const nioList = feed?.cyclones.filter((c) => c.nio) ?? [];
  const elsewhere = feed?.cyclones.filter((c) => !c.nio) ?? [];
  const all = feed?.cyclones ?? [];
  const sel = all.find((c) => c.id === selectedId) ?? nioList[0] ?? elsewhere[0] ?? null;

  // Without a working Himawari layer, fall back to true colour rather than an empty map.
  const wanted = LAYERS.find((l) => l.id === layerId)!;
  const layer = wanted.kind === "nrt" && nrt === null ? LAYERS[0] : wanted;
  const resolved = nrt !== "loading" && nrt !== null ? nrt : null;
  const isNrt = layer.kind === "nrt";
  const maxStep = isNrt ? 36 : 7;
  const delayMin = (resolved?.delayMin ?? 60) + step * 10;
  const frameDay = utcDay(new Date(Date.parse(defaultDay()) - step * DAY));
  const time = isNrt ? iso(slotAt(delayMin)) : frameDay;
  const variant = isNrt && resolved ? resolved.variant : layer.variants[0];
  const under = utcDay(new Date(Date.parse(isNrt ? defaultDay() : frameDay) - DAY));
  const frameLabel = isNrt
    ? `${slotAt(delayMin).toISOString().slice(11, 16)} UTC / ${slotAt(delayMin).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false })} IST`
    : `${frameDay} (UTC)`;
  const onTileProblem = useCallback((p: boolean) => setBlank(p), []);

  useEffect(() => { resolveNrt(LAYERS.find((l) => l.kind === "nrt")!).then(setNrt); }, []);
  useEffect(() => {
    const load = () => fetch("/api/active-cyclones").then((r) => r.json()).then(setFeed)
      .catch(() => setFeed({ ok: false, fetchedAt: null, cyclones: [], source: "" }));
    load();
    const t = setInterval(load, 10 * 60_000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => { const t = setInterval(() => setClock(new Date()), 30_000); return () => clearInterval(t); }, []);
  useEffect(() => { const t = setTimeout(() => fly([17, 82], 4), 300); return () => clearTimeout(t); }, [fly]); // the opening fly-in

  // Weather for the chosen city, refreshed every 15 minutes.
  useEffect(() => {
    const p = CITIES.find((c) => c[0] === city)!;
    const ctl = new AbortController();
    const load = () => fetchWeather(p[1], p[2], ctl.signal).then((w) => setWx({ s: "ok", w })).catch((e) => !ctl.signal.aborted && setWx({ s: "err", msg: String(e?.message ?? e).slice(0, 120) }));
    setWx({ s: "loading" });
    load();
    const t = setInterval(load, 15 * 60_000);
    return () => { ctl.abort(); clearInterval(t); };
  }, [city]);

  // Timeline playback: oldest frame to newest, then loop.
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setStep((s) => (s <= 0 ? maxStep : s - 1)), 1100 / speed);
    return () => clearInterval(t);
  }, [playing, speed, maxStep]);

  // Scroll-spy: of the sections whose top has passed 40% of the viewport, the one closest to that line is active.
  useEffect(() => {
    const on = () => {
      let cur = "top", best = -Infinity;
      for (const [id] of NAV) {
        const top = document.getElementById(id)?.getBoundingClientRect().top ?? Infinity;
        if (scrollY > 120 && top < innerHeight * 0.4 && top > best) { best = top; cur = id; }
      }
      setActive(cur);
    };
    on();
    addEventListener("scroll", on, { passive: true });
    return () => removeEventListener("scroll", on);
  }, []);

  const pick = (c: ActiveCyclone) => { setSelectedId(c.id); fly([c.lat, c.lon], c.nio ? 6 : 5); document.getElementById("map")?.scrollIntoView({ behavior: "smooth", block: "start" }); };
  const chooseLayer = (id: string) => { setLayerId(id); setStep(0); setPlaying(false); };
  const search = (e: FormEvent) => {
    e.preventDefault();
    const q = query.trim().toLowerCase();
    if (!q) return;
    const storm = all.find((c) => c.name.toLowerCase().includes(q));
    const place = PLACES.find((p) => p[0].toLowerCase().includes(q));
    setMiss(!storm && !place);
    if (storm) return pick(storm);
    if (place) { if (place[3] === "city") setCity(place[0]); fly([place[1], place[2]], place[3] === "city" ? 7 : 5); document.getElementById("map")?.scrollIntoView({ behavior: "smooth", block: "start" }); }
  };

  const strongest = [...all].sort((a, b) => (b.windKt ?? 0) - (a.windKt ?? 0))[0];
  const alertCount = (a: string) => all.filter((c) => c.alert === a).length;
  const status =
    feed === null ? "Checking the live cyclone feed…"
    : !feed.ok ? "The live cyclone feed is unavailable right now. Check the IMD bulletin."
    : (nioList.length ? `${nioList.length} active in the Bay of Bengal or Arabian Sea: ${nioList.map((c) => c.name).join(", ")}.` : "No active cyclone in the Bay of Bengal or Arabian Sea right now.")
      + (elsewhere.length ? ` ${elsewhere.length} elsewhere in the world, strongest ${strongest.name}${strongest.windKt ? ` at ${strongest.windKt} kt` : ""}.` : "");
  const w = wx.s === "ok" ? wx.w : null;

  return (
    <div className="app">
      <a href="#map" className="skip">Skip to the map</a>
      <aside className="side">
        <a href="#top" className="logo"><svg width="30" height="30" viewBox="0 0 40 40" aria-hidden="true"><circle cx="20" cy="20" r="19" fill="var(--blue)" /><path d="M20 20m-3 0a3 3 0 1 0 6 0 3 3 0 1 0-6 0 M20 8a12 12 0 0 1 12 12 M20 32A12 12 0 0 1 8 20" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" /></svg>CycloneSense</a>
        <nav aria-label="Sections">
          {NAV.map(([id, label, icon]) => (
            <a key={id} href={`#${id}`} aria-current={active === id ? "true" : undefined}><Icon name={icon} />{label}</a>
          ))}
        </nav>
        <p className="side-note">A research project, not an official warning system.</p>
      </aside>

      <div className="main">
        <header className="top" id="top">
          <div className="title"><h1>Live cyclone dashboard</h1><p>Bay of Bengal and Arabian Sea, with active storms worldwide</p></div>
          <form onSubmit={search} role="search" className="search">
            <Icon name="search" size={16} />
            <input value={query} onChange={(e) => { setQuery(e.target.value); setMiss(false); }} list="places" placeholder="Search a city, sea or storm" aria-label="Search a city, sea or storm" />
            <datalist id="places">{[...PLACES.map((p) => p[0]), ...all.map((c) => c.name)].map((x) => <option key={x} value={x} />)}</datalist>
            {miss && <span className="miss" role="status">No match</span>}
          </form>
          <div className="live" title="Time of the last clock update"><i className="dot" />LIVE<b>{clock.toISOString().slice(11, 16)} UTC</b></div>
          <a className="btn" href="https://mausam.imd.gov.in/" target="_blank" rel="noreferrer"><Icon name="bell" size={16} />Official warnings</a>
        </header>

        <p className="summary card" style={{ ["--i" as string]: 0 }}><b>Situation summary</b> {status} <small>Generated from the live feed, not a forecast.</small></p>

        <div className="grid">
          <Card id="map" title="Live satellite map" icon="map" i={1} className="map-card" aside={<span className="pill">{layer.label}</span>}>
            <div className="map-wrap">
              <LiveMap layer={layer} variant={variant} under={under} time={time} opacity={opacity} showCoast={showCoast} showLabels={showLabels} showStorms={showStorms}
                cyclones={all} selectedId={sel?.id ?? null} onSelect={setSelectedId} view={view} onTileProblem={onTileProblem} />
              <details className="layers" open={layersOpen}>
                <summary><Icon name="layers" size={16} />Layers</summary>
                <div role="radiogroup" aria-label="Satellite layer">
                  {LAYERS.map((l) => (
                    <label key={l.id} className={l.id === layer.id ? "on" : ""}>
                      <input type="radio" name="layer" checked={l.id === layer.id} onChange={() => chooseLayer(l.id)} />
                      <span>{l.label}{l.kind === "nrt" && (nrt === "loading" ? " (checking)" : nrt === null ? " (unavailable)" : "")}</span>
                    </label>
                  ))}
                </div>
                <label className="slider">Imagery opacity<input type="range" min="0.2" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} /></label>
                <label className="chk"><input type="checkbox" checked={showCoast} onChange={(e) => setShowCoast(e.target.checked)} />Coastlines</label>
                <label className="chk"><input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />City and sea names</label>
                <label className="chk"><input type="checkbox" checked={showStorms} onChange={(e) => setShowStorms(e.target.checked)} />Cyclone markers</label>
              </details>
              {(blank || (wanted.kind === "nrt" && nrt === null)) && (
                <p className="banner" role="status">
                  {wanted.kind === "nrt" && nrt === null ? "The near real-time layer is not answering, so true colour is shown." : "This image is not available yet. Use the timeline to step back."}
                </p>
              )}
              {sel && (
                <div className="storm" aria-live="polite">
                  <div className="s-head"><b>{sel.name}</b><span className="alert" style={{ background: ALERT_COLOR[sel.alert] ?? "var(--mute)" }}>{ALERT_TEXT[sel.alert] ?? sel.alert}</span></div>
                  <dl>
                    <dt>Location</dt><dd>{sel.nio ? "Bay of Bengal / Arabian Sea" : "Outside the North Indian Ocean"}</dd>
                    <dt>Wind</dt><dd>{sel.windKt ? `${sel.windKt} kt (${kmh(sel.windKt)} km/h)` : "not reported"}</dd>
                    <dt>Position</dt><dd>{Math.abs(sel.lat).toFixed(1)}°{sel.lat >= 0 ? "N" : "S"} {Math.abs(sel.lon).toFixed(1)}°{sel.lon >= 0 ? "E" : "W"}</dd>
                    <dt>Started</dt><dd>{when(sel.from)}</dd>
                    <dt>Updated</dt><dd>{when(sel.updated)}</dd>
                  </dl>
                  <div className="s-act"><button onClick={() => fly([sel.lat, sel.lon], 6)}>Fly to storm</button>{sel.url && <a href={sel.url} target="_blank" rel="noreferrer">GDACS report</a>}</div>
                </div>
              )}
            </div>
            <p className="note">{layer.note} Dimmer areas are filled in from the previous day's true colour. Click the map once to zoom with the scroll wheel.</p>
          </Card>

          <div className="rail">
            <Card id="alerts" title="Storm alerts" icon="bell" i={2} aside={<span className="pill">{all.length} active</span>}>
              {feed === null && <p className="muted">Loading…</p>}
              {feed && !feed.ok && <p className="warn">The live cyclone feed could not be reached. Please check the IMD bulletin.{feed.error && <small> Reason: {feed.error}</small>}</p>}
              {feed?.ok && all.length === 0 && <p className="muted">No active tropical cyclones are reported anywhere right now.</p>}
              {feed?.ok && nioList.length === 0 && all.length > 0 && <p className="muted">None in the Bay of Bengal or Arabian Sea.</p>}
              <ul className="alerts">
                {[...nioList, ...elsewhere].map((c) => (
                  <li key={c.id}>
                    <button onClick={() => pick(c)} aria-pressed={sel?.id === c.id}>
                      <i style={{ background: ALERT_COLOR[c.alert] ?? "var(--mute)" }} />
                      <span><b>{c.name}</b><small>{ALERT_TEXT[c.alert] ?? c.alert}{c.windKt ? `, ${c.windKt} kt` : ""}{c.nio ? ", near India" : ""}</small></span>
                    </button>
                  </li>
                ))}
              </ul>
              {feed?.ok && <p className="note">Source: {feed.source}. {feed.stale ? "Last successful update. " : ""}Colours are GDACS humanitarian levels, not IMD categories.</p>}
            </Card>

            <Card id="weather" title="Current conditions" icon="temp" i={3} aside={
              <select value={city} onChange={(e) => setCity(e.target.value)} aria-label="City">{CITIES.map((c) => <option key={c[0]}>{c[0]}</option>)}</select>}>
              {wx.s === "loading" && <p className="muted">Loading weather…</p>}
              {wx.s === "err" && <p className="warn">The weather service could not be reached. <small>Reason: {wx.msg}</small></p>}
              {w && (
                <>
                  <div className="now"><b>{Math.round(w.temp)}°C</b>
                    <ul><li>Humidity {Math.round(w.humidity)}%</li><li>Pressure {Math.round(w.pressure)} hPa</li><li>Wind {Math.round(w.windKmh)} km/h from {COMPASS[Math.round(w.windDir / 45) % 8]}</li><li>Gusts {Math.round(w.gustKmh)} km/h</li></ul></div>
                  <h3>Pressure, 48 hours</h3>
                  <PressureChart series={w.series} nowIndex={w.nowIndex} />
                  {w.tendency3h !== null && (
                    <p className={`tend ${w.tendency3h <= -3 ? "fast" : ""}`}>
                      {Math.abs(w.tendency3h).toFixed(1)} hPa {w.tendency3h < 0 ? "fall" : "rise"} in the last 3 hours.
                      {w.tendency3h <= -3 && " A fall this fast can mean a storm or front is approaching. Check official bulletins."}
                    </p>
                  )}
                  <p className="note">Weather data by <a href="https://open-meteo.com/" target="_blank" rel="noreferrer">Open-Meteo.com</a>, model values at {city}, updated {w.time.replace("T", " ")} UTC.</p>
                </>
              )}
            </Card>
          </div>

          <div className="bottom">
            <Card id="timeline" title="Timeline playback" icon="clock" i={4}>
              <div className="tl">
                <button className="round" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"}><Icon name={playing ? "pause" : "play"} size={18} /></button>
                <button className="round ghost" onClick={() => { setStep(0); setPlaying(false); }} aria-label="Jump to latest image" title="Latest"><Icon name="skip" size={16} /></button>
                <div className="seg" role="group" aria-label="Playback speed">{[1, 2, 4].map((s) => <button key={s} aria-pressed={speed === s} onClick={() => setSpeed(s)}>{s}x</button>)}</div>
                <output className="frame">{frameLabel}</output>
              </div>
              <input className="scrub" type="range" min="0" max={maxStep} value={maxStep - step} aria-label="Image time" onChange={(e) => { setStep(maxStep - +e.target.value); setPlaying(false); }} />
              <div className="ticks"><span>{isNrt ? "6 hours earlier" : "7 days earlier"}</span><span>Latest</span></div>
              <p className="note">{isNrt ? "Each step is 10 minutes." : "Each step is one day."} Frames load as they play, so the first loop can be slow.</p>
            </Card>

            <Card title="Live indicators" icon="dash" i={5}>
              <div className="gauges">
                <Gauge value={strongest?.windKt ?? 0} max={180} text={strongest?.windKt ? String(strongest.windKt) : "–"} label="Strongest wind" caption="knots, scale to 180" color="var(--blue)" />
                <Gauge value={nioList.length} max={5} text={String(nioList.length)} label="Near India" caption="active storms" color="var(--orange)" />
                <Gauge value={all.length} max={10} text={String(all.length)} label="Worldwide" caption="active storms" color="var(--teal)" />
                <Gauge value={isNrt ? delayMin : step} max={isNrt ? 360 : 7} text={isNrt ? `${delayMin}m` : `${step}d`} label="Image age" caption={isNrt ? "behind real time" : "days back"} color="var(--violet)" />
              </div>
            </Card>

            <Card title="Storm strength" icon="storm" i={6} aside={<span className="pill">{alertCount("Red")} red · {alertCount("Orange")} orange · {alertCount("Green")} green</span>}>
              {all.length === 0 ? <p className="muted">No active storms to compare.</p> : (
                <ul className="bars">
                  {[...all].sort((a, b) => (b.windKt ?? 0) - (a.windKt ?? 0)).map((c) => (
                    <li key={c.id}><span>{c.name}</span><div><i style={{ width: `${Math.min(100, ((c.windKt ?? 0) / 180) * 100)}%`, background: ALERT_COLOR[c.alert] ?? "var(--mute)" }} /></div><b>{c.windKt ? `${c.windKt} kt` : "n/a"}</b></li>
                  ))}
                </ul>
              )}
            </Card>
          </div>
        </div>

        <section id="research" className="about">
          <Reveal><h2>The research</h2>
            <p>CycloneSense joins NOAA IBTrACS storm records with NASA MODIS satellite chips and keeps all observations from each storm in one split. Model A estimates wind speed from infrared imagery. This research estimate is not an official intensity advisory.</p></Reveal>
          <Reveal delay={80}><WindEstimator /></Reveal>
          <Reveal delay={120}><h2 id="safety">Official warnings come first</h2>
            <p>CycloneSense is a research project, not a warning system. For warnings and evacuation advice, follow the India Meteorological Department and your state disaster authority.</p>
            <p><a href="https://mausam.imd.gov.in/" target="_blank" rel="noreferrer">IMD weather and cyclone bulletins</a></p></Reveal>
        </section>
        <footer className="credits">Imagery: NASA GIBS / EOSDIS, JMA Himawari. Cyclone positions: GDACS. Weather: Open-Meteo.com (CC BY 4.0). Coastlines: Natural Earth.</footer>
      </div>
    </div>
  );
}
