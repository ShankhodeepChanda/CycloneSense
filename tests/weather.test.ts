// Run: npm run test:web
// The sample is hand-written to the Open-Meteo layout as I understand it (fixture, not live data).
import test from "node:test";
import assert from "node:assert/strict";
import { parseWeather } from "../src/weather";

const times = Array.from({ length: 48 }, (_, i) => `2026-10-21T${String(i % 24).padStart(2, "0")}:00`);
const sample = (over: any = {}) => ({
  current: { time: "2026-10-21T12:15", temperature_2m: 31.2, relative_humidity_2m: 70, surface_pressure: 1005.5, wind_speed_10m: 18, wind_direction_10m: 225, wind_gusts_10m: 30, ...over },
  hourly: { time: times.map((t, i) => (i < 24 ? t.replace("2026-10-21", "2026-10-20") : t)), surface_pressure: Array.from({ length: 48 }, (_, i) => 1010 - i * 0.1) },
});

test("parses current conditions and finds now in the series", () => {
  const w = parseWeather(sample());
  assert.equal(w.temp, 31.2);
  assert.equal(w.series.length, 48);
  assert.equal(w.nowIndex, 36); // 2026-10-21T12:00
  assert.ok(Math.abs((w.tendency3h as number) + 0.3) < 1e-9);
});

test("rejects missing or malformed fields instead of inventing numbers", () => {
  assert.throws(() => parseWeather(null));
  assert.throws(() => parseWeather(sample({ surface_pressure: null })));
  assert.throws(() => parseWeather({ current: sample().current, hourly: {} }));
});
