import { useCallback, useEffect, useState } from "react";
import type { ActiveCycloneResult, ActiveCyclone } from "../server/gdacs";
import { LAYERS, LiveMap } from "./LiveMap";

const utcDay = (d: Date) => d.toISOString().slice(0, 10);
// Today's daily mosaic only exists once the daytime pass over India has been processed.
const defaultDay = () => utcDay(new Date(Date.now() - (new Date().getUTCHours() < 10 ? 86_400_000 : 0)));

function nrtTime(delayMin: number): Date {
  return new Date(Math.floor((Date.now() - delayMin * 60_000) / 600_000) * 600_000);
}

const ALERT_TEXT: Record<string, string> = { Red: "Red alert", Orange: "Orange alert", Green: "Green" };

export function App() {
  const [layerId, setLayerId] = useState(LAYERS[0].id);
  const [day, setDay] = useState(defaultDay);
  const [delay, setDelay] = useState(60); // minutes behind "now" for the near real-time layer
  const [feed, setFeed] = useState<ActiveCycloneResult | null>(null);
  const [focus, setFocus] = useState<ActiveCyclone | null>(null);
  const [blank, setBlank] = useState(false);

  const layer = LAYERS.find((l) => l.id === layerId)!;
  const when = nrtTime(delay);
  const time = layer.kind === "daily" ? day : when.toISOString().replace(/\.\d{3}Z$/, "Z");
  const onTileProblem = useCallback((p: boolean) => setBlank(p), []);

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

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b bg-white px-4 py-3">
        <h1 className="text-xl font-semibold">CycloneSense</h1>
        <p className="text-sm text-slate-600">Live satellite view of the Bay of Bengal and Arabian Sea</p>
      </header>

      <main className="mx-auto grid max-w-6xl gap-4 p-4 lg:grid-cols-[1fr_320px]">
        <section aria-label="Satellite map">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            {LAYERS.map((l) => (
              <button
                key={l.id}
                aria-pressed={l.id === layerId}
                onClick={() => setLayerId(l.id)}
                className={`rounded-full border px-3 py-1.5 text-sm ${
                  l.id === layerId ? "border-sky-700 bg-sky-700 text-white" : "border-slate-300 bg-white hover:bg-slate-100"
                }`}
              >
                {l.label}
              </button>
            ))}
          </div>

          <div className="h-[60vh] min-h-[360px] overflow-hidden rounded-lg border bg-slate-200">
            <LiveMap layer={layer} time={time} cyclones={feed?.cyclones ?? []} focus={focus} onTileProblem={onTileProblem} />
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
            {layer.kind === "daily" ? (
              <label className="flex items-center gap-2">
                Date (UTC)
                <input type="date" value={day} max={utcDay(new Date())} onChange={(e) => e.target.value && setDay(e.target.value)}
                  className="rounded border border-slate-300 px-2 py-1" />
              </label>
            ) : (
              <>
                <span>
                  Satellite time: {when.toISOString().slice(11, 16)} UTC /{" "}
                  {when.toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit" })} IST
                </span>
                <button onClick={() => setDelay((d) => d + 10)} className="rounded border border-slate-300 bg-white px-2 py-1 hover:bg-slate-100">Earlier</button>
                <button onClick={() => setDelay((d) => Math.max(30, d - 10))} className="rounded border border-slate-300 bg-white px-2 py-1 hover:bg-slate-100">Later</button>
              </>
            )}
          </div>
          <p className="mt-2 text-sm text-slate-600">{layer.note}</p>
          {blank && (
            <p role="status" className="mt-2 rounded bg-amber-50 p-2 text-sm text-amber-900">
              This image is not available yet. {layer.kind === "daily" ? "Pick the previous day." : "Press Earlier to step back 10 minutes."}
            </p>
          )}
        </section>

        <aside className="space-y-4">
          <section className="rounded-lg border bg-white p-4" aria-live="polite">
            <h2 className="font-semibold">Active cyclones</h2>
            {feed === null && <p className="mt-2 text-sm text-slate-600">Loading…</p>}
            {feed && !feed.ok && (
              <p className="mt-2 text-sm text-amber-900">
                The live cyclone feed could not be reached. Please check the official IMD bulletin below.
              </p>
            )}
            {feed?.ok && feed.cyclones.length === 0 && (
              <p className="mt-2 text-sm text-slate-600">No active tropical cyclones are reported in the North Indian Ocean right now.</p>
            )}
            {feed?.ok && feed.cyclones.length > 0 && (
              <ul className="mt-2 space-y-2">
                {feed.cyclones.map((c) => (
                  <li key={c.id}>
                    <button onClick={() => setFocus(c)} className="w-full rounded border border-slate-200 p-2 text-left hover:bg-slate-50">
                      <span className="font-medium">{c.name}</span>
                      <span className="block text-sm text-slate-600">
                        {ALERT_TEXT[c.alert] ?? c.alert}
                        {c.windKt ? ` · ${c.windKt} kt winds` : ""} · {c.lat.toFixed(1)}°N {c.lon.toFixed(1)}°E
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
            {feed?.ok && (
              <p className="mt-3 text-xs text-slate-500">
                Source: {feed.source}. {feed.stale ? "Showing the last successful update. " : ""}
                Checked {feed.fetchedAt ? new Date(feed.fetchedAt).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata" }) + " IST" : ""}.
              </p>
            )}
          </section>

          <section className="rounded-lg border bg-white p-4">
            <h2 className="font-semibold">Official warnings</h2>
            <p className="mt-2 text-sm text-slate-600">
              For warnings and evacuation advice, follow the India Meteorological Department and your state disaster authority.
            </p>
            <a href="https://mausam.imd.gov.in/" target="_blank" rel="noreferrer" className="mt-2 inline-block text-sm text-sky-700 underline">
              IMD weather and cyclone bulletins
            </a>
          </section>
        </aside>
      </main>

      <footer className="mx-auto max-w-6xl px-4 pb-6 text-xs text-slate-500">
        CycloneSense is a research project and is not an official warning system. Imagery: NASA GIBS / EOSDIS, JMA Himawari.
        Cyclone positions: GDACS.
      </footer>
    </div>
  );
}
