// Per-event GDACS track geometry.
//
// The event list carries only the current centroid. The per-event geometry endpoint
// carries the whole track: Line_Line_N two-point segments that chain into the ordered
// centres, one Point_Polygon_Point_N per centre whose `polygonlabel` is its
// "DD/MM HH:MM UTC" timestamp, and Poly_Cones for GDACS's own forecast cone. `todate`
// is where the observed track ends and GDACS's forecast begins.
//
// Parsing stays defensive, like the event list: a field GDACS stops sending becomes
// null, never a guess.

export interface TrackPoint {
  lat: number;
  lon: number;
  time: string | null; // ISO UTC, or null when the label was unreadable
  observed: boolean; // false once past `todate`: GDACS's forecast, not an observation
}

export interface CycloneTrack {
  eventId: string;
  name: string;
  asOf: string | null; // `todate`: the latest observation GDACS holds
  points: TrackPoint[];
  observed: TrackPoint[];
  forecast: TrackPoint[];
  cone: [number, number][] | null; // GDACS's cone as a [lat, lon] ring
}

const TTL_MS = 10 * 60 * 1000;

const emptyTrack = (): CycloneTrack => ({
  eventId: "",
  name: "",
  asOf: null,
  points: [],
  observed: [],
  forecast: [],
  cone: null,
});

function classIndex(value: unknown, prefix: string): number | null {
  const text = String(value ?? "");
  if (!text.startsWith(prefix)) return null;
  const rest = text.slice(prefix.length);
  return /^\d+$/.test(rest) ? Number(rest) : null;
}

/** "21/10 15:00 UTC" carries no year. GDACS dates it from the event, so take the year
 *  from `fromdate` and roll forward when that would place the label before the event. */
export function labelToIso(label: unknown, fromdate: unknown): string | null {
  const match = /^(\d{1,2})\/(\d{1,2})\s+(\d{1,2}):(\d{2})/.exec(String(label ?? "").trim());
  if (!match) return null;
  const day = Number(match[1]);
  const month = Number(match[2]);
  const hour = Number(match[3]);
  const minute = Number(match[4]);
  if (month < 1 || month > 12 || day < 1 || day > 31 || hour > 23) return null;
  const start = Date.parse(String(fromdate ?? "") + "Z");
  const year = Number.isFinite(start) ? new Date(start).getUTCFullYear() : new Date().getUTCFullYear();
  let stamp = Date.UTC(year, month - 1, day, hour, minute);
  if (!Number.isFinite(stamp)) return null;
  if (Number.isFinite(start) && stamp < start - 7 * 86_400_000) {
    stamp = Date.UTC(year + 1, month - 1, day, hour, minute); // the event crossed New Year
  }
  return new Date(stamp).toISOString();
}

const isLonLat = (point: unknown): point is [number, number] =>
  Array.isArray(point) && typeof point[0] === "number" && typeof point[1] === "number";

/** Mean of a polygon ring: for a GDACS point-radii polygon this is its centre. */
function ringCentre(coordinates: unknown): [number, number] | null {
  const outer = Array.isArray(coordinates) ? (coordinates as unknown[])[0] : null;
  const ring = (Array.isArray(outer) ? outer : []).filter(isLonLat);
  if (!ring.length) return null;
  const lon = ring.reduce((sum, p) => sum + p[0], 0) / ring.length;
  const lat = ring.reduce((sum, p) => sum + p[1], 0) / ring.length;
  return [lat, lon];
}

function ringToLatLng(coordinates: unknown): [number, number][] | null {
  const outer = Array.isArray(coordinates) ? (coordinates as unknown[])[0] : null;
  const ring = (Array.isArray(outer) ? outer : []).filter(isLonLat).map((p) => [p[1], p[0]] as [number, number]);
  return ring.length ? ring : null;
}

/** Turn one GDACS getgeometry response into an ordered track split at `todate`. */
export function parseCycloneTrack(geojson: any): CycloneTrack {
  const features: any[] = Array.isArray(geojson?.features) ? geojson.features : [];
  if (!features.length) return emptyTrack();

  const first = features[0]?.properties ?? {};
  const asOfMs = Date.parse(String(first.todate ?? "") + "Z");
  const asOf = Number.isFinite(asOfMs) ? new Date(asOfMs).toISOString() : null;

  const segments = features
    .map((feature) => ({ index: classIndex(feature?.properties?.Class, "Line_Line_"), geometry: feature?.geometry }))
    .filter((segment) => segment.index !== null && segment.geometry?.type === "LineString")
    .sort((a, b) => (a.index as number) - (b.index as number));

  // Chain the segments: the first contributes both ends, each later one its far end.
  let centres: [number, number][] = [];
  for (const segment of segments) {
    const pairs = (Array.isArray(segment.geometry.coordinates) ? segment.geometry.coordinates : [])
      .filter(isLonLat)
      .map((point: [number, number]) => [point[1], point[0]] as [number, number]);
    if (!pairs.length) continue;
    centres.push(...(centres.length ? pairs.slice(1) : pairs));
  }

  const labelled = features
    .map((feature) => ({
      index: classIndex(feature?.properties?.Class, "Point_Polygon_Point_"),
      properties: feature?.properties,
      geometry: feature?.geometry,
    }))
    .filter((point) => point.index !== null)
    .sort((a, b) => (a.index as number) - (b.index as number));

  if (!centres.length) {
    // No line segments: fall back to the centre of each point's radii polygon.
    centres = labelled.map((point) => ringCentre(point.geometry?.coordinates)).filter(Boolean) as [number, number][];
  }

  // An unreadable label inherits the previous point's side of `todate` rather than
  // silently becoming a forecast point.
  let observed = true;
  const points: TrackPoint[] = centres.map(([lat, lon], index) => {
    const time = labelToIso(labelled[index]?.properties?.polygonlabel, first.fromdate);
    if (time !== null && asOf !== null) observed = Date.parse(time) <= Date.parse(asOf);
    return { lat, lon, time, observed };
  });

  const coneFeature = features.find(
    (feature) => feature?.properties?.Class === "Poly_Cones" && feature?.geometry?.type === "Polygon",
  );

  return {
    eventId: String(first.eventid ?? ""),
    name: String(first.eventname || first.name || "Unnamed system"),
    asOf,
    points,
    observed: points.filter((point) => point.observed),
    forecast: points.filter((point) => !point.observed),
    cone: coneFeature ? ringToLatLng(coneFeature.geometry.coordinates) : null,
  };
}

const geometryUrl = (eventId: string, episodeId: string | number | null) =>
  "https://www.gdacs.org/gdacsapi/api/polygons/getgeometry?eventtype=TC&eventid=" +
  encodeURIComponent(eventId) +
  (episodeId === null || episodeId === undefined ? "" : "&episodeid=" + encodeURIComponent(String(episodeId)));

const trackCache = new Map<string, { at: number; track: CycloneTrack }>();

export interface CycloneTrackResult {
  ok: boolean;
  stale?: boolean;
  fetchedAt: string | null;
  track: CycloneTrack | null;
  source: string;
  error?: string;
}

/** Fetch one storm's track geometry, cached for the same interval as the event list. */
export async function getCycloneTrack(
  eventId: string,
  episodeId: string | number | null = null,
): Promise<CycloneTrackResult> {
  const source = "GDACS (gdacs.org)";
  const key = eventId + ":" + (episodeId ?? "");
  const cached = trackCache.get(key);
  if (cached && Date.now() - cached.at < TTL_MS) {
    return { ok: true, fetchedAt: new Date(cached.at).toISOString(), track: cached.track, source };
  }
  try {
    const ctl = new AbortController();
    // GDACS geometry regularly takes 6-8 s for a storm with a full track and cone, so a
    // 10 s budget aborted it whenever the event-list fetch was in flight at the same time.
    const timer = setTimeout(() => ctl.abort(), 25_000);
    const res = await fetch(geometryUrl(eventId, episodeId), {
      signal: ctl.signal,
      headers: { accept: "application/json", "user-agent": "CycloneSense/1.0 (research project)" },
    });
    clearTimeout(timer);
    if (!res.ok) throw new Error("GDACS geometry responded " + res.status);
    const track = parseCycloneTrack(await res.json());
    const at = Date.now();
    trackCache.set(key, { at, track });
    return { ok: true, fetchedAt: new Date(at).toISOString(), track, source };
  } catch (err) {
    // Serve the last good answer (clearly marked) rather than nothing; never invent one.
    if (cached) {
      return { ok: true, stale: true, fetchedAt: new Date(cached.at).toISOString(), track: cached.track, source };
    }
    console.error("cyclone track geometry failed:", err);
    return {
      ok: false,
      fetchedAt: null,
      track: null,
      source,
      error: String((err as Error)?.message ?? err).slice(0, 200),
    };
  }
}
