import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { ActiveCyclone } from "../server/gdacs";

export interface SatLayer {
  id: string;
  label: string;
  gibs: string; // GIBS layer id
  tms: string; // tile matrix set, which also sets the finest native zoom
  ext: "jpg" | "png";
  maxNative: number;
  kind: "daily" | "nrt";
  note: string;
}

// Layer ids/zoom levels follow the NASA GIBS WMTS catalogue. Check them against
// https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/1.0.0/WMTSCapabilities.xml if a layer ever shows blank.
export const LAYERS: SatLayer[] = [
  { id: "viirs", label: "Daily true colour", gibs: "VIIRS_SNPP_CorrectedReflectance_TrueColor", tms: "GoogleMapsCompatible_Level9",
    ext: "jpg", maxNative: 9, kind: "daily",
    note: "NASA VIIRS, one daytime pass per day (about 1:30 pm local time). Best for seeing cloud structure." },
  { id: "modis-ir", label: "Daily thermal IR", gibs: "MODIS_Terra_Brightness_Temp_Band31_Day", tms: "GoogleMapsCompatible_Level7",
    ext: "png", maxNative: 7, kind: "daily",
    note: "NASA MODIS Terra cloud-top temperature, daytime pass. Colder (higher) clouds are usually the stronger storms." },
  { id: "himawari", label: "Near real-time IR", gibs: "Himawari_AHI_Band13_Clean_Infrared", tms: "GoogleMapsCompatible_Level6",
    ext: "png", maxNative: 6, kind: "nrt",
    note: "Japan's Himawari-9, updated every 10 minutes. Sees the Bay of Bengal well; the Arabian Sea is near the edge of its view." },
];

export function tileUrl(l: SatLayer, time: string): string {
  return `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/${l.gibs}/default/${time}/${l.tms}/{z}/{y}/{x}.${l.ext}`;
}

const ALERT_COLOR: Record<string, string> = { Red: "#dc2626", Orange: "#ea580c", Green: "#16a34a" };

interface Props {
  layer: SatLayer;
  time: string; // YYYY-MM-DD for daily layers, ISO UTC for near real-time
  cyclones: ActiveCyclone[];
  focus: ActiveCyclone | null;
  onTileProblem: (problem: boolean) => void;
}

export function LiveMap({ layer, time, cyclones, focus, onTileProblem }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const sat = useRef<L.TileLayer | null>(null);
  const markers = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    const m = L.map(el.current!, { center: [16, 80], zoom: 4, minZoom: 3, maxZoom: 10, worldCopyJump: true });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      zIndex: 1, maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
    }).addTo(m);
    // Keyless NASA GIBS reference overlays (coastlines + place names). If their tiles fail, drop the overlay rather than show broken tiles.
    for (const [id, z] of [["Coastlines_15m", 4], ["Reference_Labels_15m", 5]] as const) {
      let errors = 0;
      const o = L.tileLayer(`https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/${id}/default/GoogleMapsCompatible_Level9/{z}/{y}/{x}.png`, {
        zIndex: z, maxNativeZoom: 9, maxZoom: 10, attribution: "NASA GIBS",
      }).on("tileerror", () => {
        errors += 1;
        if (errors === 8) o.remove();
      });
      o.addTo(m);
    }
    markers.current = L.layerGroup().addTo(m);
    map.current = m;
    return () => {
      m.remove();
      map.current = null;
    };
  }, []);

  useEffect(() => {
    if (!map.current) return;
    sat.current?.remove();
    let errors = 0;
    onTileProblem(false);
    sat.current = L.tileLayer(tileUrl(layer, time), {
      zIndex: 3, maxNativeZoom: layer.maxNative, maxZoom: 10, attribution: "Imagery: NASA GIBS / EOSDIS",
    })
      .on("tileerror", () => {
        errors += 1;
        if (errors === 6) onTileProblem(true);
      })
      .addTo(map.current);
  }, [layer, time, onTileProblem]);

  useEffect(() => {
    markers.current?.clearLayers();
    for (const c of cyclones) {
      L.circleMarker([c.lat, c.lon], {
        radius: 10, weight: 3, color: ALERT_COLOR[c.alert] ?? "#475569", fillColor: "#ffffff", fillOpacity: 0.85,
      })
        .bindTooltip(c.name, { permanent: true, direction: "right", offset: [10, 0] })
        .addTo(markers.current!);
    }
  }, [cyclones]);

  useEffect(() => {
    if (focus) map.current?.flyTo([focus.lat, focus.lon], 6);
  }, [focus]);

  return <div ref={el} role="application" aria-label="Satellite map of the Indian Ocean" className="h-full w-full rounded-lg" />;
}
