import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { ActiveCyclone } from "../server/gdacs";
import { LAYERS, tileUrl, type SatLayer, type Variant } from "./gibs";

export const PLACES: [string, number, number, "city" | "sea"][] = [
  ["Bay of Bengal", 14, 88, "sea"], ["Arabian Sea", 15, 64, "sea"], ["Indian Ocean", 2, 78, "sea"],
  ["New Delhi", 28.61, 77.21, "city"], ["Mumbai", 19.08, 72.88, "city"], ["Kolkata", 22.57, 88.36, "city"],
  ["Chennai", 13.08, 80.27, "city"], ["Visakhapatnam", 17.69, 83.22, "city"], ["Kochi", 9.93, 76.27, "city"],
  ["Bhubaneswar", 20.3, 85.82, "city"], ["Colombo", 6.93, 79.86, "city"], ["Dhaka", 23.81, 90.41, "city"],
  ["Karachi", 24.86, 67.01, "city"], ["Muscat", 23.59, 58.41, "city"], ["Yangon", 16.84, 96.15, "city"], ["Port Blair", 11.62, 92.73, "city"],
];
// VIIRS JPEG tiles paint no-data areas opaque black. Redraw them on a canvas with black made transparent so the layer below shows through.
const KeyedTiles = (L.TileLayer as any).extend({
  createTile(this: L.TileLayer, coords: L.Coords, done: L.DoneCallback) {
    const tile = document.createElement("canvas");
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      tile.width = img.width; tile.height = img.height;
      const c = tile.getContext("2d")!;
      c.drawImage(img, 0, 0);
      try {
        const d = c.getImageData(0, 0, tile.width, tile.height), p = d.data, w = tile.width, h = tile.height;
        for (let i = 0; i < p.length; i += 4) if (p[i] <= 14 && p[i + 1] <= 14 && p[i + 2] <= 14) p[i + 3] = 0;
        // JPEG leaves a dark fringe next to the black: peel up to 2 pixels of near-black off every cleared edge.
        for (let pass = 0; pass < 2; pass++) {
          const kill: number[] = [];
          for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            const o = (y * w + x) * 4;
            if (p[o + 3] === 0 || Math.max(p[o], p[o + 1], p[o + 2]) > 60) continue;
            if ((x > 0 && p[o - 1] === 0) || (x < w - 1 && p[o + 7] === 0) || (y > 0 && p[o - w * 4 + 3] === 0) || (y < h - 1 && p[o + w * 4 + 3] === 0)) kill.push(o + 3);
          }
          for (const k of kill) p[k] = 0;
        }
        c.putImageData(d, 0, 0);
      } catch { /* shortcut: if the browser blocks pixel access the tile stays as drawn, black gaps included */ }
      done(undefined, tile);
    };
    img.onerror = () => done(new Error("tile failed"), tile);
    img.src = this.getTileUrl(coords);
    return tile;
  },
});
const tiles = (x: Variant, time: string, opts: L.TileLayerOptions): L.TileLayer =>
  new (x.key ? KeyedTiles : L.TileLayer)(tileUrl(x, time), { maxNativeZoom: x.maxNative, maxZoom: 10, updateWhenZooming: false, ...opts });

const ALERT_COLOR: Record<string, string> = { Red: "#ef4444", Orange: "#f97316", Green: "#22c55e" };

interface Props {
  layer: SatLayer;
  variant: Variant;
  under: string; // day (YYYY-MM-DD) of the dimmed true-colour layer that fills gaps
  time: string; // YYYY-MM-DD for daily layers, ISO UTC for near real-time
  opacity: number;
  showCoast: boolean;
  showLabels: boolean;
  showStorms: boolean;
  cyclones: ActiveCyclone[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  view: { center: [number, number]; zoom: number; n: number }; // n changes to re-trigger the same view
  onTileProblem: (problem: boolean) => void;
}

export function LiveMap({ layer, variant, under, time, opacity, showCoast, showLabels, showStorms, cyclones, selectedId, onSelect, view, onTileProblem }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const sat = useRef<L.TileLayer | null>(null);
  const base = useRef<L.TileLayer | null>(null);
  const stale = useRef<L.TileLayer[]>([]);
  const coast = useRef<L.GeoJSON | null>(null);
  const markers = useRef<L.LayerGroup | null>(null);
  const labels = useRef<L.LayerGroup | null>(null);
  const op = useRef(opacity);
  op.current = opacity;

  useEffect(() => {
    const m = L.map(el.current!, { center: [20, 128], zoom: 3, minZoom: 3, maxZoom: 10, worldCopyJump: true, scrollWheelZoom: false, zoomControl: false });
    L.control.zoom({ position: "bottomright" }).addTo(m);
    // The page scrolls, so wheel-zoom only switches on after a click and off again when the pointer leaves.
    m.on("click", () => m.scrollWheelZoom.enable());
    m.getContainer().addEventListener("mouseleave", () => m.scrollWheelZoom.disable());
    m.getContainer().title = "Click the map once to zoom with the scroll wheel";
    labels.current = L.layerGroup();
    for (const [name, lat, lon, kind] of PLACES) {
      L.marker([lat, lon], { interactive: false, keyboard: false, icon: L.divIcon({ className: "", html: `<span class="cy-label ${kind}">${name}</span>`, iconSize: [0, 0] }) }).addTo(labels.current);
    }
    fetch("/coast.json").then((r) => r.json()).then((g) => {
      if (map.current) coast.current = L.geoJSON(g, { interactive: false, style: { color: "#fff", weight: 1.1, opacity: 0.85 } });
    }).catch(() => {}); // coastline is decoration; the map works without it
    markers.current = L.layerGroup().addTo(m);
    map.current = m;
    return () => { m.remove(); map.current = null; };
  }, []);

  const toggle = (g: L.Layer | null, on: boolean) => {
    const m = map.current;
    if (!m || !g) return;
    if (on && !m.hasLayer(g)) g.addTo(m);
    if (!on && m.hasLayer(g)) g.remove();
  };
  useEffect(() => {
    // The coastline file loads asynchronously, so retry until it exists.
    const t = setInterval(() => { if (coast.current) { coast.current.setStyle({ color: layer.coast }); toggle(coast.current, showCoast); clearInterval(t); } }, 150);
    coast.current?.setStyle({ color: layer.coast });
    toggle(coast.current, showCoast);
    return () => clearInterval(t);
  }, [layer, showCoast]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => toggle(labels.current, showLabels), [showLabels]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { sat.current?.setOpacity(opacity); }, [opacity]);

  useEffect(() => {
    const m = map.current;
    if (!m) return;
    // Keep the old image on screen until the new one has loaded, then drop it: no blank map while tiles arrive.
    const old = sat.current;
    if (old) stale.current.push(old);
    let errors = 0;
    onTileProblem(false);
    const next = tiles(variant, time, { zIndex: 2, opacity: op.current, attribution: "Imagery: NASA GIBS / EOSDIS" })
      .on("tileerror", () => { errors += 1; if (errors === 6) onTileProblem(true); })
      .on("load", () => { for (const l of stale.current.splice(0)) if (l !== next) l.remove(); })
      .addTo(m);
    sat.current = next;
  }, [variant, time, onTileProblem]);

  // Dimmed previous-day true colour underneath, so gaps in the top layer show land and sea instead of nothing.
  useEffect(() => {
    if (!map.current) return;
    base.current?.remove();
    base.current = tiles({ ...LAYERS[0].variants[0], maxNative: 5 }, under, { zIndex: 1, opacity: 0.6 }).addTo(map.current);
  }, [under]);

  useEffect(() => {
    markers.current?.clearLayers();
    if (!showStorms) return;
    for (const c of cyclones) {
      const color = ALERT_COLOR[c.alert] ?? "#94a3b8";
      const mk = L.marker([c.lat, c.lon], {
        zIndexOffset: c.id === selectedId ? 2000 : c.nio ? 1000 : 0,
        icon: L.divIcon({ className: "", iconSize: [34, 34], iconAnchor: [17, 17],
          html: `<div class="cy-pin${c.nio ? " nio" : ""}${c.id === selectedId ? " sel" : ""}" style="--c:${color}"><i></i><b></b></div>` }),
      }).on("click", () => onSelect(c.id)).addTo(markers.current!);
      const name = document.createElement("span"); // textContent, so a feed name can never inject HTML
      name.textContent = c.name;
      mk.bindTooltip(name, { permanent: true, direction: "right", offset: [14, 0] });
    }
  }, [cyclones, selectedId, showStorms, onSelect]);

  useEffect(() => {
    const m = map.current;
    if (!m) return;
    const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
    m.flyTo(view.center, view.zoom, { animate: !calm, duration: 2 });
  }, [view]);

  return <div ref={el} role="application" aria-label="Satellite map of the Indian Ocean" className="cy-map h-full w-full" />;
}
