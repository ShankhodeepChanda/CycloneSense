import { useCallback, useEffect, useRef, useState } from "react";
import type { ActiveCycloneResult, ActiveCyclone } from "../server/gdacs";
import { LiveMap } from "./LiveMap";
import { LAYERS, iso, resolveNrt, slotAt, type Variant } from "./gibs";

const utcDay = (d: Date) => d.toISOString().slice(0, 10);
// Today's daily mosaic only exists once the daytime pass over India has been processed.
const defaultDay = () => utcDay(new Date(Date.now() - (new Date().getUTCHours() < 10 ? 86_400_000 : 0)));
const ALERT_TEXT: Record<string, string> = { Red: "Red alert", Orange: "Orange alert", Green: "Green" };
type LL = [number, number];

// The page is a story told on one sticky map: each chapter picks a layer and a camera.
const CHAPTERS: { id: string; rail: string; layer: string; center: LL; zoom: number; color: string }[] = [
  { id: "top", rail: "Start", layer: "viirs", center: [17, 82], zoom: 4, color: "var(--kelp)" },
  { id: "seas", rail: "Two seas", layer: "viirs", center: [15, 70], zoom: 5, color: "var(--kelp)" },
  { id: "clouds", rail: "Cloud tops", layer: "modis-ir", center: [15, 88], zoom: 5, color: "var(--signal)" },
  { id: "tenmin", rail: "Every 10 minutes", layer: "himawari", center: [14, 92], zoom: 4, color: "var(--sky)" },
  { id: "live", rail: "Active now", layer: "himawari", center: [16, 82], zoom: 4, color: "var(--signal)" },
  { id: "explore", rail: "Explore", layer: "viirs", center: [16, 82], zoom: 4, color: "var(--foam)" },
];
const LAST = CHAPTERS.length - 1;

function useActiveChapter() {
  const [active, setActive] = useState(0);
  useEffect(() => {
    const io = new IntersectionObserver(
      (es) => es.forEach((e) => e.isIntersecting && setActive(CHAPTERS.findIndex((c) => c.id === e.target.id))),
      { rootMargin: "-45% 0px -45% 0px" },
    );
    CHAPTERS.forEach((c) => { const n = document.getElementById(c.id); if (n) io.observe(n); });
    return () => io.disconnect();
  }, []);
  return active;
}

export function App() {
  const active = useActiveChapter();
  const [layerId, setLayerId] = useState(LAYERS[0].id);
  const [day, setDay] = useState(defaultDay);
  const [nrt, setNrt] = useState<{ variant: Variant; delayMin: number } | null | "loading">("loading");
  const [extra, setExtra] = useState(0); // minutes stepped back from the newest working image
  const [feed, setFeed] = useState<ActiveCycloneResult | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [view, setView] = useState<{ center: LL; zoom: number; n: number }>({ center: [20, 128], zoom: 3, n: 0 });
  const [blank, setBlank] = useState(false);
  const n = useRef(0);

  const nioList = feed?.cyclones.filter((c) => c.nio) ?? [];
  const elsewhere = feed?.cyclones.filter((c) => !c.nio) ?? [];
  const resolved = nrt !== "loading" && nrt !== null ? nrt : null;
  // Without a working Himawari layer, fall back to true colour rather than an empty map.
  const wanted = LAYERS.find((l) => l.id === layerId)!;
  const layer = wanted.kind === "nrt" && nrt === null ? LAYERS[0] : wanted;
  const variant = layer.kind === "nrt" && resolved ? resolved.variant : layer.variants[0];
  const when = slotAt((resolved?.delayMin ?? 60) + extra);
  const time = layer.kind === "daily" ? day : iso(when);
  const onTileProblem = useCallback((p: boolean) => setBlank(p), []);
  const fly = (center: LL, zoom: number) => setView({ center, zoom, n: ++n.current });

  useEffect(() => { resolveNrt(LAYERS.find((l) => l.kind === "nrt")!).then(setNrt); }, []);
  const refreshFeed = useCallback(async () => {
    setRefreshing(true);
    try {
      const response = await fetch("/api/active-cyclones", { cache: "no-store" });
      if (!response.ok) throw new Error(`Feed request failed (${response.status})`);
      setFeed(await response.json());
    } catch {
      setFeed({ ok: false, fetchedAt: null, cyclones: [], source: "" });
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    refreshFeed();
    const t = setInterval(refreshFeed, 10 * 60_000);
    return () => clearInterval(t);
  }, [refreshFeed]);

  // New chapter: switch layer and move the camera. In "Active now", head for the strongest storm near India.
  const target = nioList[0];
  useEffect(() => {
    const c = CHAPTERS[active];
    setLayerId(c.layer);
    setExtra(0);
    if (c.id === "live" && target) fly([target.lat, target.lon], 6);
    else fly(c.center, c.zoom);
  }, [active, target?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const go = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  const chap = CHAPTERS[active];
  const strongest = elsewhere[0];
  const status =
    feed === null ? "Checking the live cyclone feed…"
    : !feed.ok ? "The live cyclone feed is unavailable right now. Check the IMD bulletin below."
    : (nioList.length ? `${nioList.length} active in the Bay of Bengal or Arabian Sea: ${nioList.map((c) => c.name).join(", ")}.` : "No active cyclone in the Bay of Bengal or Arabian Sea right now.")
      + (strongest ? ` ${elsewhere.length} elsewhere in the world, strongest ${strongest.name}${strongest.windKt ? ` at ${strongest.windKt} kt` : ""}.` : "");

  return (
    <>
      <a href="#explore" className="skip">Skip to the map controls</a>
      <header className="brand">CycloneSense</header>

      <nav className="rail" aria-label="Chapters">
        {CHAPTERS.map((c, i) => (
          <button key={c.id} onClick={() => go(c.id)} aria-label={c.rail} aria-current={i === active} title={c.rail} className={i === active ? "on" : ""} />
        ))}
      </nav>

      <div className="story">
        <div className={`stage ${active === LAST ? "live" : ""}`}>
          <LiveMap layer={layer} variant={variant} under={utcDay(new Date(Date.parse(day) - 86_400_000))} time={time} cyclones={feed?.cyclones ?? []} view={view} interactive={active === LAST} onTileProblem={onTileProblem} />
          <aside className="hud" aria-live="polite">
            <b>{layer.label}</b>
            <span>{layer.kind === "daily" ? `${day} (UTC)` : `${when.toISOString().slice(11, 16)} UTC / ${when.toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit" })} IST`}</span>
            <span className="feed-status"><i className={feed?.ok ? "dot" : "dot off"} />{feed?.ok ? `Feed checked ${feed.fetchedAt ? new Date(feed.fetchedAt).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit" }) : ""} IST` : feed === null ? "Connecting to live feed" : "Feed offline"}<button className="refresh" type="button" onClick={refreshFeed} disabled={refreshing} aria-label={refreshing ? "Refreshing cyclone feed" : "Refresh cyclone feed"}>{refreshing ? "Updating…" : "Refresh"}</button></span>
          </aside>
        </div>

        <main className="chapters">
          <section id="top" className="ch end">
            <Panel on={active === 0} color={chap.color} wide>
              <h1>Cyclones over India's seas, as they happen.</h1>
              <p className="lede">{status}</p>
              <p className="hint">Scroll to follow the satellites.</p>
            </Panel>
          </section>

          <section id="seas" className="ch">
            <Panel on={active === 1} color={CHAPTERS[1].color}>
              <h2>Two seas feed India's cyclones</h2>
              <p>The Arabian Sea borders Gujarat, Maharashtra, Goa, Karnataka and Kerala. The Bay of Bengal faces Odisha, Andhra Pradesh, Tamil Nadu and West Bengal. A cyclone that reaches India forms over one of them.</p>
              <p className="note">NASA VIIRS true colour, one daytime pass each day.</p>
            </Panel>
          </section>

          <section id="clouds" className="ch">
            <Panel on={active === 2} color={CHAPTERS[2].color}>
              <h2>Read the cloud tops</h2>
              <p>Thermal infrared shows how cold the clouds are. Colder, higher cloud tops usually mean a stronger storm, and the swirl of a cyclone shows up even at night.</p>
              <p className="note">NASA MODIS Terra cloud-top temperature, daytime pass.</p>
            </Panel>
          </section>

          <section id="tenmin" className="ch">
            <Panel on={active === 3} color={CHAPTERS[3].color}>
              <h2>A new image every 10 minutes</h2>
              <p>Japan's Himawari-9 satellite scans this region every 10 minutes. The map shows the newest image NASA has processed, usually an hour or so behind real time.</p>
              {nrt === null && <p className="warn" role="status">The near real-time layer is not answering right now, so the map shows true colour instead.</p>}
            </Panel>
          </section>

          <section id="live" className="ch">
            <Panel on={active === 4} color={CHAPTERS[4].color}>
              <h2>Active right now</h2>
              {feed === null && <p>Loading…</p>}
              {feed && !feed.ok && <p className="warn">The live cyclone feed could not be reached. Please check the official IMD bulletin below.{feed.error && <small> Reason: {feed.error}</small>}</p>}
              {feed?.ok && nioList.length === 0 && <p>No active tropical cyclones are reported in the Bay of Bengal or Arabian Sea right now.</p>}
              {nioList.length > 0 && <Storms items={nioList} onPick={(c) => (go("explore"), fly([c.lat, c.lon], 6))} />}
              {elsewhere.length > 0 && (
                <>
                  <h3>Elsewhere in the world</h3>
                  <Storms items={elsewhere} onPick={(c) => (go("explore"), fly([c.lat, c.lon], 5))} />
                </>
              )}
              {feed?.ok && <p className="note">Source: {feed.source}. {feed.stale ? "Showing the last successful update. " : ""}Alert colours are GDACS humanitarian levels, not IMD categories.</p>}
            </Panel>
          </section>

          <section id="explore" className="ch">
            <Panel on={active === LAST} color={CHAPTERS[LAST].color}>
              <h2>Explore the map</h2>
              <p>Drag to pan. Click the map once to zoom with your scroll wheel.</p>
              <div className="pills" role="group" aria-label="Satellite layer">
                {LAYERS.map((l) => (
                  <button key={l.id} aria-pressed={l.id === layer.id} onClick={() => { setLayerId(l.id); setExtra(0); }}>
                    {l.label}{l.kind === "nrt" && (nrt === "loading" ? " (checking)" : nrt === null ? " (unavailable)" : "")}
                  </button>
                ))}
              </div>
              <div className="row">
                {layer.kind === "daily" ? (
                  <label>Date (UTC) <input type="date" value={day} max={utcDay(new Date())} onChange={(e) => e.target.value && setDay(e.target.value)} /></label>
                ) : (
                  <>
                    <button onClick={() => setExtra((e) => e + 10)}>Earlier</button>
                    <button onClick={() => setExtra((e) => Math.max(30 - (resolved?.delayMin ?? 60), e - 10))}>Later</button>
                  </>
                )}
              </div>
              <p className="note">{layer.note} Dimmer areas are filled in from the previous day's true colour.</p>
              {blank && <p className="warn" role="status">This image is not available yet. {layer.kind === "daily" ? "Pick the previous day." : "Press Earlier to step back 10 minutes."}</p>}
            </Panel>
          </section>
        </main>
      </div>

      <footer className="outro">
        <div>
          <h2>The research</h2>
          <p>CycloneSense also builds an open dataset: NOAA IBTrACS storm records joined with NASA MODIS satellite chips, split by storm so no storm is in both training and test data. A ResNet-18 estimates wind speed from a chip and is scored against a predict-the-average baseline. Results will be published after the full training run.</p>
        </div>
        <div>
          <h2>Official warnings come first</h2>
          <p>CycloneSense is a research project, not a warning system. For warnings and evacuation advice, follow the India Meteorological Department and your state disaster authority.</p>
          <p><a href="https://mausam.imd.gov.in/" target="_blank" rel="noreferrer">IMD weather and cyclone bulletins</a></p>
        </div>
        <p className="credits">Imagery: NASA GIBS / EOSDIS, JMA Himawari. Cyclone positions: GDACS. Coastlines: Natural Earth.</p>
      </footer>
    </>
  );
}

function Panel({ on, color, wide, children }: { on: boolean; color: string; wide?: boolean; children: React.ReactNode }) {
  return <div className={`panel ${wide ? "wide" : ""}`} data-on={on} style={{ ["--rule" as string]: color }}>{children}</div>;
}

function Storms({ items, onPick }: { items: ActiveCyclone[]; onPick: (c: ActiveCyclone) => void }) {
  return (
    <ul className="storms">
      {items.map((c) => (
        <li key={c.id}>
          <button onClick={() => onPick(c)}>
            <b>{c.name}</b>
            <span>{ALERT_TEXT[c.alert] ?? c.alert}{c.windKt ? `, ${c.windKt} kt winds` : ""}, {Math.abs(c.lat).toFixed(1)}°{c.lat >= 0 ? "N" : "S"} {Math.abs(c.lon).toFixed(1)}°{c.lon >= 0 ? "E" : "W"}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
