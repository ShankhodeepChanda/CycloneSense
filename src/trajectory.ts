// Client side of the trajectory forecast: the API's shape, the geometry the map draws,
// and the one fetch that connects them.
//
// The cone is built from each waypoint's own uncertainty_radius_km, which the API takes
// from the checkpoint's measured per-lead validation error. Nothing here invents a width.

const KM_PER_DEGREE = 111.195;

export interface Waypoint {
  forecast_hour: number;
  lat: number;
  lon: number;
  wind_kt: number;
  pressure_hpa: number;
  uncertainty_radius_km: number;
  /** null where the shipped Eastern Hemisphere coastline has no coverage. */
  dist2land_km: number | null;
}

export interface LandfallEstimate {
  projected: boolean;
  location: string | null;
  lat: number | null;
  lon: number | null;
  eta_hours: number | null;
  note: string;
  threshold_km: number;
}

export interface TrajectoryForecast {
  status: string;
  cyclone_id: string;
  region: string;
  forecast_generated_at: string;
  source: "trained-sequence-model" | "physics-extrapolation" | string;
  model: string;
  /** Which inference path served it: "numpy" in deployment, "torch" locally. */
  runtime?: "numpy" | "torch" | "none" | string;
  checkpoint: string | null;
  observations_used: number;
  inputs_imputed: string[];
  current_metrics: { lat: number; lon: number; wind_kt: number; pressure_hpa: number; dist2land_km: number | null };
  waypoints: Waypoint[];
  landfall_estimate: LandfallEstimate;
  summary: {
    peak_wind_kt: number;
    min_pressure_hpa: number;
    peak_pressure_hpa: number;
    pressure_change_hpa: number;
    wind_change_kt: number;
    max_deepening_hpa: number;
    intensity_class: string;
    forecast_track_km: number;
  };
  validation: { lead_track_rmse_km: number[]; metrics: Record<string, unknown> };
  disclaimer: string;
}

export interface Observation {
  lat: number;
  lon: number;
  wind_kt?: number;
  pressure_hpa?: number;
  timestamp?: string;
}

/** Leads the map labels and the panel lists. */
export const WAYPOINT_HIGHLIGHTS = [12, 24, 48];

/** Degrees of latitude and longitude spanned by a distance at a given latitude. */
export function kmToDegrees(km: number, lat: number): { dLat: number; dLon: number } {
  const dLat = km / KM_PER_DEGREE;
  // Clamp the cosine so a near-polar position widens the cone instead of dividing by zero.
  const shrink = Math.max(Math.cos((lat * Math.PI) / 180), 0.01);
  return { dLat, dLon: dLat / shrink };
}

type LatLng = [number, number];

/**
 * A cone of uncertainty around the forecast path.
 *
 * Each anchor (the current position, then every waypoint) is offset to both sides of the
 * local track direction by its own radius, so the shape follows a curving track instead
 * of being a straight wedge. The ring runs up the left edge and back down the right.
 */
export function conePolygon(origin: LatLng, waypoints: Waypoint[]): LatLng[] {
  if (!waypoints.length) return [];
  const anchors: { lat: number; lon: number; radius: number }[] = [
    { lat: origin[0], lon: origin[1], radius: 0 },
    ...waypoints.map((w) => ({ lat: w.lat, lon: w.lon, radius: Math.max(0, w.uncertainty_radius_km) })),
  ];

  const left: LatLng[] = [];
  const right: LatLng[] = [];
  for (let i = 0; i < anchors.length; i++) {
    const previous = anchors[Math.max(0, i - 1)];
    const next = anchors[Math.min(anchors.length - 1, i + 1)];
    // Local heading, in degrees, with longitude scaled so the perpendicular is a true
    // right angle on the ground rather than on an unprojected grid.
    const scale = Math.max(Math.cos((anchors[i].lat * Math.PI) / 180), 0.01);
    let headingLat = next.lat - previous.lat;
    let headingLon = (next.lon - previous.lon) * scale;
    if (Math.hypot(headingLat, headingLon) < 1e-9) {
      // A forecast that predicts no motion still has uncertainty, and the panel quotes it
      // in km. Without a heading there is no perpendicular, so spread the cone east-west
      // rather than drawing a zero-area ring over a figure of hundreds of km.
      headingLat = 1;
      headingLon = 0;
    }
    const length = Math.hypot(headingLat, headingLon);
    const { dLat, dLon } = kmToDegrees(anchors[i].radius, anchors[i].lat);
    // Perpendicular to the heading: (-headingLon, headingLat) normalised.
    const offsetLat = (-headingLon / length) * dLat;
    const offsetLon = (headingLat / length) * dLon;
    left.push([anchors[i].lat + offsetLat, anchors[i].lon + offsetLon]);
    right.push([anchors[i].lat - offsetLat, anchors[i].lon - offsetLon]);
  }

  return [...left, ...right.reverse(), left[0]];
}

type Stamped<T> = { point: T; index: number; at: number };

/**
 * Oldest first: timestamped points in time order, then any untimed ones in arrival order.
 *
 * Comparing a finite time against NaN and falling back to the index is not a consistent
 * total order, and V8 is free to order such an array arbitrarily once it passes ten
 * elements. Ranking untimed points after timed ones keeps the comparator transitive.
 */
function chronological<T extends { time?: string | null }>(points: T[]): Stamped<T>[] {
  const stamped: Stamped<T>[] = points.map((point, index) => ({
    point,
    index,
    at: point.time ? Date.parse(point.time) : NaN,
  }));
  stamped.sort((a, b) => {
    const aTimed = Number.isFinite(a.at);
    const bTimed = Number.isFinite(b.at);
    if (aTimed && bTimed) return a.at - b.at || a.index - b.index;
    if (aTimed !== bTimed) return aTimed ? -1 : 1;
    return a.index - b.index;
  });
  return stamped;
}

/** Track points as Leaflet lat/lng pairs, oldest first. */
export function trackLatLngs(points: { lat: number; lon: number; time?: string | null }[]): LatLng[] {
  return chronological(points).map(({ point }) => [point.lat, point.lon] as LatLng);
}

/** Hemisphere-correct position, so a Southern or Western storm is not labelled N and E. */
export function formatPosition(lat: number, lon: number): string {
  const ns = lat < 0 ? "S" : "N";
  const ew = lon < 0 ? "W" : "E";
  return `${Math.abs(lat).toFixed(1)}\u00b0${ns} ${Math.abs(lon).toFixed(1)}\u00b0${ew}`;
}

/** Plain words for a signed pressure change, including the zero case. */
export function describePressureChange(hpa: number): string {
  if (Math.abs(hpa) < 0.5) return "no net change";
  return hpa > 0 ? "filling, so weakening" : "deepening";
}

/** One sentence for the landfall panel, preferring the engine's own explanation. */
export function describeLandfall(landfall: LandfallEstimate): string {
  if (!landfall.projected) return landfall.note;
  if (landfall.eta_hours === null || landfall.eta_hours < 1) return landfall.note;
  return `Projected landfall: ${landfall.location} in about ${Math.round(landfall.eta_hours)} hours.`;
}

/** True when the forecaster is being used outside the basin it was trained on. */
export function isOutsideHomeBasin(region: string): boolean {
  return region.startsWith("Outside");
}

/**
 * Build the forecaster's input from an observed track.
 *
 * GDACS reports one wind figure for the event, not one per fix, so that figure goes only
 * on the latest observation. The API derives the rest and says it did; attaching the
 * current wind to every past fix would be inventing a history.
 */
export function observationsFromTrack(
  points: { lat: number; lon: number; time?: string | null }[],
  currentWindKt: number | null,
  steps = 4,
): Observation[] {
  const window = chronological(points)
    .map(({ point }) => point)
    .slice(-steps);
  return window.map((point, index) => {
    const observation: Observation = { lat: point.lat, lon: point.lon };
    if (point.time) observation.timestamp = point.time;
    if (index === window.length - 1 && currentWindKt !== null && Number.isFinite(currentWindKt)) {
      observation.wind_kt = currentWindKt;
    }
    return observation;
  });
}

function endpoint(): string {
  const configured = import.meta.env?.VITE_API_URL?.trim().replace(/\/$/, "");
  return configured ? `${configured}/predict/trajectory` : "/api/predict/trajectory";
}

/** Ask the engine for a forecast. Throws with the API's own message on a refusal. */
export async function fetchTrajectory(
  cycloneId: string,
  observations: Observation[],
  signal?: AbortSignal,
): Promise<TrajectoryForecast> {
  const response = await fetch(endpoint(), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ cyclone_id: cycloneId, observations }),
    signal,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail =
      typeof payload?.detail === "string"
        ? payload.detail
        : Array.isArray(payload?.detail) && payload.detail[0]?.msg
          ? String(payload.detail[0].msg)
          : "The forecast service could not use these observations.";
    throw new Error(detail);
  }
  return payload as TrajectoryForecast;
}

// ---------------------------------------------------------------------------
// Hindcast demo
//
// GDACS reports no storm for most of the year, which hides the whole forecast. Replaying
// a past storm shows it, and shows it better: a historical case has a known outcome, so
// the map can draw the forecast next to the track the storm actually took and the error
// is something the viewer reads rather than something the page asserts.
// ---------------------------------------------------------------------------

const EARTH_RADIUS_KM = 6371.0;

export interface DemoFix {
  time: string;
  lat: number;
  lon: number;
  wind_kt: number | null;
  pressure_hpa: number | null;
  dist2land_km: number | null;
  category: string | null;
}

export interface DemoStorm {
  id: string;
  name: string;
  /** "trained-on" means its error is in-sample; "held-out" means the model never saw it. */
  provenance: "trained-on" | "held-out" | "unknown" | string;
  season: number;
  basin: string;
  peak_wind_kt: number;
  min_pressure_hpa: number;
  fixes: DemoFix[];
  /** Index of the first fix that reached the coast, or null if it never did. */
  landfall_index: number | null;
  default_start_index: number;
  usable_start_range: [number, number];
}

export interface DemoStormSet {
  source: string;
  note: string;
  provenance_note: string;
  input_fixes: number;
  future_fixes: number;
  step_hours: number;
  storms: DemoStorm[];
}

export interface HindcastScore {
  leads: number[];
  /** Great-circle distance from forecast to outcome at each lead; null past the record. */
  errorKm: (number | null)[];
  windErrorKt: (number | null)[];
  meanErrorKm: number | null;
}

/** Great-circle distance in kilometres. */
export function haversineKm(aLat: number, aLon: number, bLat: number, bLon: number): number {
  const toRad = Math.PI / 180;
  const dLat = (bLat - aLat) * toRad;
  const dLon = (bLon - aLon) * toRad;
  const inner =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(aLat * toRad) * Math.cos(bLat * toRad) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.min(1, Math.sqrt(inner)));
}

/** The observations the model forecasts from: the fixes ending at `startIndex`. */
export function hindcastObservations(storm: DemoStorm, startIndex: number, inputFixes = 4): Observation[] {
  const end = Math.max(0, Math.min(startIndex, storm.fixes.length - 1));
  return storm.fixes.slice(Math.max(0, end - inputFixes + 1), end + 1).map((fix) => {
    const observation: Observation = { lat: fix.lat, lon: fix.lon, timestamp: fix.time };
    if (fix.wind_kt !== null) observation.wind_kt = fix.wind_kt;
    if (fix.pressure_hpa !== null) observation.pressure_hpa = fix.pressure_hpa;
    return observation;
  });
}

/** What the storm actually did next: the fixes after `startIndex`. */
export function hindcastActual(storm: DemoStorm, startIndex: number, futureFixes = 8): DemoFix[] {
  return storm.fixes.slice(startIndex + 1, startIndex + 1 + futureFixes);
}

/** Score a forecast against the outcome, lead by lead. */
export function scoreHindcast(waypoints: Waypoint[], actual: DemoFix[]): HindcastScore {
  const leads = waypoints.map((waypoint) => waypoint.forecast_hour);
  const errorKm = waypoints.map((waypoint, index) => {
    const outcome = actual[index];
    return outcome ? haversineKm(waypoint.lat, waypoint.lon, outcome.lat, outcome.lon) : null;
  });
  const windErrorKt = waypoints.map((waypoint, index) => {
    const outcome = actual[index];
    return outcome && outcome.wind_kt !== null ? waypoint.wind_kt - outcome.wind_kt : null;
  });
  const scored = errorKm.filter((km): km is number => km !== null);
  return {
    leads,
    errorKm,
    windErrorKt,
    meanErrorKm: scored.length ? scored.reduce((sum, km) => sum + km, 0) / scored.length : null,
  };
}

/** Load the committed demo storms. */
export async function fetchDemoStorms(signal?: AbortSignal): Promise<DemoStormSet> {
  const response = await fetch("/demo-storms.json", { signal });
  if (!response.ok) throw new Error("Could not load the demo storms.");
  return (await response.json()) as DemoStormSet;
}

/** How a hindcast result should be read, given whether the model fitted that storm. */
export function describeProvenance(provenance: string): string {
  if (provenance === "held-out")
    return "The model never saw this storm in training, so this error is the honest one.";
  if (provenance === "trained-on")
    return "The model was trained on this storm, so this error is in-sample and flatters it.";
  return "Whether the model trained on this storm is not recorded.";
}
