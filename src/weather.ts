// Current conditions + 48 h surface-pressure series for a point, from Open-Meteo (keyless, CC BY 4.0).
export interface Weather {
  time: string;
  temp: number;
  humidity: number;
  pressure: number;
  windKmh: number;
  windDir: number;
  gustKmh: number;
  series: number[]; // hourly pressure, hPa: 24 h back to 24 h ahead
  nowIndex: number; // position of "now" in series
  tendency3h: number | null; // hPa change over the last 3 h
}

const num = (x: unknown, what: string) => {
  if (typeof x !== "number" || !Number.isFinite(x)) throw new Error(`weather data missing ${what}`);
  return x;
};

export function parseWeather(j: any): Weather {
  const c = j?.current;
  const h = j?.hourly;
  if (!c || !Array.isArray(h?.time) || !Array.isArray(h?.surface_pressure)) throw new Error("unexpected weather response");
  const hour = String(c.time).slice(0, 13) + ":00";
  const idx = h.time.indexOf(hour);
  const series: number[] = h.surface_pressure.map((p: unknown) => (typeof p === "number" ? p : NaN));
  const nowIndex = idx >= 0 ? idx : Math.min(24, series.length - 1);
  const back = series[nowIndex - 3], now = series[nowIndex];
  return {
    time: String(c.time),
    temp: num(c.temperature_2m, "temperature"),
    humidity: num(c.relative_humidity_2m, "humidity"),
    pressure: num(c.surface_pressure, "pressure"),
    windKmh: num(c.wind_speed_10m, "wind"),
    windDir: num(c.wind_direction_10m, "wind direction"),
    gustKmh: num(c.wind_gusts_10m, "gusts"),
    series,
    nowIndex,
    tendency3h: Number.isFinite(back) && Number.isFinite(now) ? now - back : null,
  };
}

export async function fetchWeather(lat: number, lon: number, signal?: AbortSignal): Promise<Weather> {
  const url = "https://api.open-meteo.com/v1/forecast?latitude=" + lat + "&longitude=" + lon +
    "&current=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m" +
    "&hourly=surface_pressure&past_hours=24&forecast_hours=24&timezone=UTC";
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`weather service responded ${r.status}`);
  return parseWeather(await r.json());
}
