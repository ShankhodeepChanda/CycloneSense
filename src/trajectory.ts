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
    const headingLat = next.lat - previous.lat;
    const headingLon = (next.lon - previous.lon) * scale;
    const length = Math.hypot(headingLat, headingLon) || 1;
    const { dLat, dLon } = kmToDegrees(anchors[i].radius, anchors[i].lat);
    // Perpendicular to the heading: (-headingLon, headingLat) normalised.
    const offsetLat = (-headingLon / length) * dLat;
    const offsetLon = (headingLat / length) * dLon;
    left.push([anchors[i].lat + offsetLat, anchors[i].lon + offsetLon]);
    right.push([anchors[i].lat - offsetLat, anchors[i].lon - offsetLon]);
  }

  return [...left, ...right.reverse(), left[0]];
}

/** Track points as Leaflet lat/lng pairs, oldest first. */
export function trackLatLngs(points: { lat: number; lon: number; time?: string | null }[]): LatLng[] {
  const timed = points.map((point, index) => ({ point, index, at: point.time ? Date.parse(point.time) : NaN }));
  // Points without a timestamp keep their arrival order; timestamped ones sort by time.
  timed.sort((a, b) => {
    if (Number.isFinite(a.at) && Number.isFinite(b.at)) return a.at - b.at;
    return a.index - b.index;
  });
  return timed.map(({ point }) => [point.lat, point.lon] as LatLng);
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
  const ordered = points
    .map((point, index) => ({ point, index, at: point.time ? Date.parse(point.time) : NaN }))
    .sort((a, b) => (Number.isFinite(a.at) && Number.isFinite(b.at) ? a.at - b.at : a.index - b.index))
    .map(({ point }) => point);

  const window = ordered.slice(-steps);
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
