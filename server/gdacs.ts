// Turns a GDACS event-list GeoJSON into the active tropical cyclones near India.
// GDACS (UN/EC Global Disaster Alert and Coordination System) is a real, keyless feed.
// Parsing is defensive on purpose: unknown fields are ignored, never invented.

export interface ActiveCyclone {
  id: string;
  name: string;
  alert: string; // Green | Orange | Red (GDACS humanitarian alert level, not an IMD category)
  windKt: number | null;
  lat: number;
  lon: number;
  from: string | null;
  updated: string | null;
  url: string | null;
}

// North Indian Ocean: Arabian Sea + Bay of Bengal (+ margin)
const NIO = { minLat: 0, maxLat: 32, minLon: 40, maxLon: 100 };

export function parseActiveCyclones(geojson: any): ActiveCyclone[] {
  const features: any[] = Array.isArray(geojson?.features) ? geojson.features : [];
  const byEvent = new Map<string, ActiveCyclone>();

  for (const f of features) {
    const p = f?.properties ?? {};
    if (p.eventtype !== "TC" || String(p.iscurrent) !== "true") continue;
    if (f?.geometry?.type !== "Point") continue;
    // A TC event also carries track/cone/first/last points; only the centroid is the current position.
    if (p.Class && p.Class !== "Point_Centroid") continue;

    const [lon, lat] = f.geometry.coordinates ?? [];
    if (typeof lat !== "number" || typeof lon !== "number") continue;
    if (lat < NIO.minLat || lat > NIO.maxLat || lon < NIO.minLon || lon > NIO.maxLon) continue;

    const kmh = Number(p.severitydata?.severity);
    const id = String(p.eventid ?? `${p.eventname}-${lat}-${lon}`);
    byEvent.set(id, {
      id,
      name: String(p.eventname || p.name || "Unnamed system"),
      alert: String(p.alertlevel || "Green"),
      windKt: Number.isFinite(kmh) && kmh > 0 ? Math.round(kmh / 1.852) : null,
      lat,
      lon,
      from: p.fromdate ?? null,
      updated: p.datemodified ?? null,
      url: p.url?.report ?? p.url?.details ?? null,
    });
  }
  return [...byEvent.values()].sort((a, b) => (b.windKt ?? 0) - (a.windKt ?? 0));
}

const GDACS_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/MAP?eventtypes=TC";
const TTL_MS = 10 * 60 * 1000;
let cache: { at: number; data: ActiveCyclone[] } | null = null;

export interface ActiveCycloneResult {
  ok: boolean;
  stale?: boolean;
  fetchedAt: string | null;
  cyclones: ActiveCyclone[];
  source: string;
}

export async function getActiveCyclones(): Promise<ActiveCycloneResult> {
  const source = "GDACS (gdacs.org)";
  if (cache && Date.now() - cache.at < TTL_MS) {
    return { ok: true, fetchedAt: new Date(cache.at).toISOString(), cyclones: cache.data, source };
  }
  try {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 10_000);
    const res = await fetch(GDACS_URL, { signal: ctl.signal, headers: { accept: "application/json" } });
    clearTimeout(timer);
    if (!res.ok) throw new Error(`GDACS responded ${res.status}`);
    cache = { at: Date.now(), data: parseActiveCyclones(await res.json()) };
    return { ok: true, fetchedAt: new Date(cache.at).toISOString(), cyclones: cache.data, source };
  } catch (err) {
    // Serve the last good answer (clearly marked) rather than nothing; never fabricate one.
    if (cache) return { ok: true, stale: true, fetchedAt: new Date(cache.at).toISOString(), cyclones: cache.data, source };
    console.error("active cyclone feed failed:", err);
    return { ok: false, fetchedAt: null, cyclones: [], source };
  }
}
