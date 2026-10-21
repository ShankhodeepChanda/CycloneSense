import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { ActiveCyclone } from "../server/gdacs";
import { tileUrl, type SatLayer, type Variant } from "./gibs";

const PLACES: [string, number, number, "city" | "sea"][] = [
  ["Bay of Bengal", 14, 88, "sea"], ["Arabian Sea", 15, 64, "sea"], ["Indian Ocean", 2, 78, "sea"],
  ["New Delhi", 28.61, 77.21, "city"], ["Mumbai", 19.08, 72.88, "city"], ["Kolkata", 22.57, 88.36, "city"],
  ["Chennai", 13.08, 80.27, "city"], ["Visakhapatnam", 17.69, 83.22, "city"], ["Kochi", 9.93, 76.27, "city"],
  ["Bhubaneswar", 20.3, 85.82, "city"], ["Colombo", 6.93, 79.86, "city"], ["Dhaka", 23.81, 90.41, "city"],
  ["Karachi", 24.86, 67.01, "city"], ["Muscat", 23.59, 58.41, "city"], ["Yangon", 16.84, 96.15, "city"], ["Port Blair", 11.62, 92.73, "city"],
];
const ALERT_COLOR: Record<string, string> = { Red: "#ef4444", Orange: "#f97316", Green: "#22c55e" };

interface Props {
  layer: SatLayer;
  variant: Variant;
  time: string; // YYYY-MM-DD for daily layers, ISO UTC for near real-time
  cyclones: ActiveCyclone[];
  view: { center: [number, number]; zoom: number; n: number }; // n changes to re-trigger the same view
  interactive: boolean; // false while the page story scrolls past, true in the explore chapter
  onTileProblem: (problem: boolean) => void;
}

export function LiveMap({ layer, variant, time, cyclones, view, interactive, onTileProblem }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const sat = useRef<L.TileLayer | null>(null);
  const coast = useRef<L.GeoJSON | null>(null);
  const markers = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    const m = L.map(el.current!, { center: [20, 128], zoom: 3, minZoom: 3, maxZoom: 10, worldCopyJump: true, scrollWheelZoom: false, zoomControl: false });
    L.control.zoom({ position: "bottomright" }).addTo(m);
    // The page scrolls, so wheel-zoom only switches on after a click and off again when the pointer leaves.
    m.on("click", () => m.scrollWheelZoom.enable());
    m.getContainer().addEventListener("mouseleave", () => m.scrollWheelZoom.disable());
    for (const [name, lat, lon, kind] of PLACES) {
      L.marker([lat, lon], { interactive: false, keyboard: false, icon: L.divIcon({ className: "", html: `<span class="cy-label ${kind}">${name}</span>`, iconSize: [0, 0] }) }).addTo(m);
    }
    fetch("/coast.json").then((r) => r.json()).then((g) => {
      if (map.current) coast.current = L.geoJSON(g, { interactive: false, style: { color: "#fff", weight: 1.1, opacity: 0.85 } }).addTo(map.current);
    }).catch(() => {}); // coastline is decoration; the map works without it
    markers.current = L.layerGroup().addTo(m);
    map.current = m;
    return () => { m.remove(); map.current = null; };
  }, []);

  useEffect(() => {
    coast.current?.setStyle({ color: layer.coast });
  }, [layer]);

  useEffect(() => {
    if (!map.current) return;
    sat.current?.remove();
    let errors = 0;
    onTileProblem(false);
    sat.current = L.tileLayer(tileUrl(variant, time), { maxNativeZoom: variant.maxNative, maxZoom: 10, opacity: 0, attribution: "Imagery: NASA GIBS / EOSDIS", className: "cy-imagery" })
      .on("load", () => sat.current?.setOpacity(1))
      .on("tileerror", () => { errors += 1; if (errors === 6) onTileProblem(true); })
      .addTo(map.current);
  }, [variant, time, onTileProblem]);

  useEffect(() => {
    markers.current?.clearLayers();
    for (const c of cyclones) {
      const color = ALERT_COLOR[c.alert] ?? "#94a3b8";
      const mk = L.marker([c.lat, c.lon], {
        zIndexOffset: c.nio ? 1000 : 0,
        icon: L.divIcon({ className: "", iconSize: [34, 34], iconAnchor: [17, 17],
          html: `<div class="cy-pin${c.nio ? " nio" : ""}" style="--c:${color}"><i></i><b></b></div>` }),
      }).addTo(markers.current!);
      const name = document.createElement("span"); // textContent, so a feed name can never inject HTML
      name.textContent = c.name;
      mk.bindTooltip(name, { permanent: true, direction: "right", offset: [14, 0] });
    }
  }, [cyclones]);

  useEffect(() => {
    const m = map.current;
    if (!m) return;
    const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
    // Shift the target away from the text panel (left on desktop, bottom on phones) so the storm is not hidden behind it.
    const wide = innerWidth > 900;
    const at = m.project(view.center, view.zoom).add(wide ? [-Math.min(240, innerWidth * 0.18), 0] : [0, innerHeight * 0.18]);
    m.flyTo(m.unproject(at, view.zoom), view.zoom, { animate: !calm, duration: 2.4 });
  }, [view]);

  useEffect(() => {
    const m = map.current;
    if (!m) return;
    for (const h of [m.dragging, m.touchZoom, m.doubleClickZoom, m.boxZoom, m.keyboard]) interactive ? h.enable() : h.disable();
    if (!interactive) m.scrollWheelZoom.disable();
  }, [interactive]);

  return <div ref={el} role="application" aria-label="Satellite map of the Indian Ocean" className="cy-map h-full w-full" />;
}
