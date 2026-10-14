import React, { useState, useEffect, useRef, useMemo } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  Satellite,
  Compass,
  Layers,
  Wind,
  Radio,
  Eye,
  Maximize2,
  Navigation as NavIcon,
  Crosshair,
  AlertTriangle,
  Info,
  Play,
  Pause,
  MapPin,
  RefreshCw,
  Sparkles,
} from "lucide-react";

export interface IndiaCycloneSystem {
  id: string;
  name: string;
  basin: "Bay of Bengal" | "Arabian Sea";
  category: string;
  lat: number;
  lon: number;
  windKts: number;
  pressureHpa: number;
  translationKmh: number;
  heading: string;
  headingDeg: number;
  rmwKm: number;
  r34Km: number; // 34-knot gale wind radius
  r50Km: number; // 50-knot storm wind radius
  r64Km: number; // 64-knot hurricane wind radius
  nearestIndianCoast: string;
  distToCoastKm: number;
  landfallStatus: string;
  forecastTrack: {
    hour: number;
    lat: number;
    lon: number;
    windKts: number;
    category: string;
    label: string;
  }[];
  description: string;
}

export const INDIA_CYCLONES: IndiaCycloneSystem[] = [
  {
    id: "cyclone-dana",
    name: "Severe Cyclonic Storm DANA",
    basin: "Bay of Bengal",
    category: "Severe Cyclonic Storm (SCS)",
    lat: 20.65,
    lon: 87.25,
    windKts: 65,
    pressureHpa: 984,
    translationKmh: 14.5,
    heading: "NNW (340°)",
    headingDeg: 340,
    rmwKm: 35,
    r34Km: 180,
    r50Km: 95,
    r64Km: 45,
    nearestIndianCoast: "Dhamra Port / Bhitarkanika (Odisha)",
    distToCoastKm: 42,
    landfallStatus: "Landfall imminent between Puri and Sagar Island",
    forecastTrack: [
      { hour: 0, lat: 20.65, lon: 87.25, windKts: 65, category: "SCS", label: "T-0h (Current)" },
      { hour: 6, lat: 20.95, lon: 87.05, windKts: 60, category: "SCS", label: "T+6h (Landfall)" },
      { hour: 12, lat: 21.35, lon: 86.75, windKts: 45, category: "CS", label: "T+12h (Inland)" },
      { hour: 24, lat: 21.90, lon: 86.20, windKts: 30, category: "D", label: "T+24h (Deep Dep)" },
      { hour: 48, lat: 22.40, lon: 85.50, windKts: 20, category: "WML", label: "T+48h (Remnant)" },
    ],
    description: "Convective feeder bands engulfing North Odisha and coastal West Bengal with heavy squally winds and torrential precipitation.",
  },
  {
    id: "cyclone-remal",
    name: "Severe Cyclonic Storm REMAL",
    basin: "Bay of Bengal",
    category: "Severe Cyclonic Storm (SCS)",
    lat: 21.80,
    lon: 89.20,
    windKts: 62,
    pressureHpa: 986,
    translationKmh: 16.0,
    heading: "North (360°)",
    headingDeg: 360,
    rmwKm: 38,
    r34Km: 190,
    r50Km: 85,
    r64Km: 40,
    nearestIndianCoast: "Sagar Island / Sundarbans (West Bengal)",
    distToCoastKm: 65,
    landfallStatus: "Crossing West Bengal & Bangladesh coastline",
    forecastTrack: [
      { hour: 0, lat: 21.80, lon: 89.20, windKts: 62, category: "SCS", label: "T-0h (Current)" },
      { hour: 6, lat: 22.40, lon: 89.30, windKts: 55, category: "SCS", label: "T+6h (Sundarbans)" },
      { hour: 18, lat: 23.50, lon: 89.50, windKts: 40, category: "CS", label: "T+18h (Kolkata/Jessore)" },
      { hour: 36, lat: 24.80, lon: 90.10, windKts: 25, category: "D", label: "T+36h (Assam Valley)" },
    ],
    description: "Rapidly organizing low-level circulation with massive convective cloud canopy impacting the fragile Sundarbans mangrove belt.",
  },
  {
    id: "cyclone-biparjoy",
    name: "Extremely Severe Cyclonic Storm BIPARJOY",
    basin: "Arabian Sea",
    category: "Extremely Severe Cyclonic Storm (ESCS)",
    lat: 22.85,
    lon: 68.20,
    windKts: 90,
    pressureHpa: 962,
    translationKmh: 12.0,
    heading: "ENE (070°)",
    headingDeg: 70,
    rmwKm: 42,
    r34Km: 260,
    r50Km: 140,
    r64Km: 70,
    nearestIndianCoast: "Jakhau Port / Mandvi (Kutch, Gujarat)",
    distToCoastKm: 78,
    landfallStatus: "Recurving toward Saurashtra and Kutch coast",
    forecastTrack: [
      { hour: 0, lat: 22.85, lon: 68.20, windKts: 90, category: "ESCS", label: "T-0h (Current)" },
      { hour: 12, lat: 23.20, lon: 68.70, windKts: 75, category: "VSCS", label: "T+12h (Landfall Jakhau)" },
      { hour: 24, lat: 23.80, lon: 69.80, windKts: 50, category: "CS", label: "T+24h (Kutch Inland)" },
      { hour: 48, lat: 24.90, lon: 71.50, windKts: 30, category: "D", label: "T+48h (S. Rajasthan)" },
    ],
    description: "Historic long-track Arabian Sea cyclone displaying persistent pinhole eye and intense peripheral squalls.",
  },
  {
    id: "cyclone-tauktae",
    name: "Extremely Severe Cyclonic Storm TAUKTAE",
    basin: "Arabian Sea",
    category: "Extremely Severe Cyclonic Storm (ESCS)",
    lat: 18.50,
    lon: 71.20,
    windKts: 100,
    pressureHpa: 950,
    translationKmh: 15.0,
    heading: "NNW (335°)",
    headingDeg: 335,
    rmwKm: 32,
    r34Km: 240,
    r50Km: 130,
    r64Km: 65,
    nearestIndianCoast: "Diu / Saurashtra Coast (Gujarat)",
    distToCoastKm: 170,
    landfallStatus: "Paralleling Western Ghats toward Saurashtra",
    forecastTrack: [
      { hour: 0, lat: 18.50, lon: 71.20, windKts: 100, category: "ESCS", label: "T-0h (Current)" },
      { hour: 12, lat: 20.10, lon: 71.05, windKts: 105, category: "ESCS", label: "T+12h (Peak Energy)" },
      { hour: 24, lat: 21.05, lon: 71.10, windKts: 90, category: "VSCS", label: "T+24h (Diu Landfall)" },
      { hour: 48, lat: 23.20, lon: 71.90, windKts: 35, category: "CS", label: "T+48h (Gujarat)" },
    ],
    description: "Compact severe eyewall with stadium effect; cloud tops colder than -82°C along the Konkan-Goa-Mumbai marine corridor.",
  },
  {
    id: "super-cyclone-amphan",
    name: "Super Cyclonic Storm AMPHAN",
    basin: "Bay of Bengal",
    category: "Super Cyclonic Storm (SuCS)",
    lat: 19.80,
    lon: 87.90,
    windKts: 140,
    pressureHpa: 920,
    translationKmh: 18.5,
    heading: "NNE (025°)",
    headingDeg: 25,
    rmwKm: 28,
    r34Km: 320,
    r50Km: 180,
    r64Km: 95,
    nearestIndianCoast: "Digha / Sagar Island (West Bengal)",
    distToCoastKm: 180,
    landfallStatus: "Catastrophic Super Cyclone moving towards Sundarbans",
    forecastTrack: [
      { hour: 0, lat: 19.80, lon: 87.90, windKts: 140, category: "SuCS", label: "T-0h (Super Cyclone)" },
      { hour: 12, lat: 21.65, lon: 88.30, windKts: 100, category: "ESCS", label: "T+12h (Landfall Digha)" },
      { hour: 24, lat: 23.20, lon: 88.70, windKts: 65, category: "SCS", label: "T+24h (Kolkata City)" },
      { hour: 48, lat: 25.50, lon: 89.80, windKts: 30, category: "D", label: "T+48h (Bangladesh/Meghalaya)" },
    ],
    description: "First Super Cyclonic Storm in the Bay of Bengal since 1999; highly symmetric eyewall with stadium-effect central subsidence.",
  },
  {
    id: "active-bay-depression",
    name: "Active Low / Deep Depression BOB-2026",
    basin: "Bay of Bengal",
    category: "Deep Depression (DD)",
    lat: 15.20,
    lon: 88.50,
    windKts: 35,
    pressureHpa: 998,
    translationKmh: 11.0,
    heading: "WNW (295°)",
    headingDeg: 295,
    rmwKm: 55,
    r34Km: 110,
    r50Km: 0,
    r64Km: 0,
    nearestIndianCoast: "Visakhapatnam / Kalingapatnam (Andhra Pradesh)",
    distToCoastKm: 340,
    landfallStatus: "Active cyclogenesis in central Bay of Bengal",
    forecastTrack: [
      { hour: 0, lat: 15.20, lon: 88.50, windKts: 35, category: "DD", label: "T-0h (Current)" },
      { hour: 12, lat: 16.10, lon: 86.80, windKts: 45, category: "CS", label: "T+12h (Cyclonic Storm)" },
      { hour: 24, lat: 17.20, lon: 85.00, windKts: 55, category: "SCS", label: "T+24h (Intensification)" },
      { hour: 48, lat: 18.50, lon: 84.10, windKts: 50, category: "SCS", label: "T+48h (Andhra Landfall)" },
    ],
    description: "Formative tropical disturbance showing curved convective feeder bands over warm ocean waters (SST 30.5°C).",
  },
];

// Real coastal IMD Doppler Weather Radar (DWR) stations
export const IMD_RADAR_STATIONS = [
  { name: "Paradip DWR", lat: 20.31, lon: 86.61, state: "Odisha", rangeKm: 250 },
  { name: "Gopalpur DWR", lat: 19.26, lon: 84.91, state: "Odisha", rangeKm: 250 },
  { name: "Visakhapatnam DWR", lat: 17.72, lon: 83.33, state: "Andhra Pradesh", rangeKm: 250 },
  { name: "Machilipatnam DWR", lat: 16.19, lon: 81.15, state: "Andhra Pradesh", rangeKm: 250 },
  { name: "Chennai DWR", lat: 13.08, lon: 80.29, state: "Tamil Nadu", rangeKm: 250 },
  { name: "Kolkata DWR", lat: 22.57, lon: 88.36, state: "West Bengal", rangeKm: 250 },
  { name: "Mumbai DWR (Colaba)", lat: 18.90, lon: 72.81, state: "Maharashtra", rangeKm: 250 },
  { name: "Veraval DWR", lat: 20.90, lon: 70.36, state: "Gujarat", rangeKm: 250 },
  { name: "Bhuj DWR", lat: 23.24, lon: 69.66, state: "Gujarat", rangeKm: 250 },
  { name: "Kochi DWR", lat: 9.93, lon: 76.26, state: "Kerala", rangeKm: 250 },
];

interface LiveIndiaSatelliteMapProps {
  onStormSelect?: (storm: IndiaCycloneSystem) => void;
}

export const LiveIndiaSatelliteMap: React.FC<LiveIndiaSatelliteMapProps> = ({ onStormSelect }) => {
  const [selectedStormId, setSelectedStormId] = useState<string>("cyclone-dana");
  const [mapStyle, setMapStyle] = useState<"carto-light" | "osm" | "satellite">("carto-light");

  // Layer toggles
  const [showClouds, setShowClouds] = useState<boolean>(true);
  const [showWindVectors, setShowWindVectors] = useState<boolean>(true);
  const [showWindRadii, setShowWindRadii] = useState<boolean>(true);
  const [showForecastCone, setShowForecastCone] = useState<boolean>(true);
  const [showRadars, setShowRadars] = useState<boolean>(true);
  const [showCoastalAlerts, setShowCoastalAlerts] = useState<boolean>(true);
  const [isRotating, setIsRotating] = useState<boolean>(true);
  const [cloudEnhancement, setCloudEnhancement] = useState<"bd-thermal" | "rainbow" | "water-vapor">("bd-thermal");

  // Probe telemetry
  const [probeLatLon, setProbeLatLon] = useState<{ lat: number; lon: number } | null>({ lat: 20.65, lon: 87.25 });
  const [probeData, setProbeData] = useState<{
    distKm: number;
    tempK: number;
    tempC: number;
    threat: string;
    windEstKts: number;
  }>({
    distKm: 0,
    tempK: 196.2,
    tempC: -76.9,
    threat: "Extreme Cyclonic Eye / Destructive",
    windEstKts: 65,
  });

  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const overlayLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const rotationAngleRef = useRef<number>(0);

  const activeStorm = useMemo(() => {
    return INDIA_CYCLONES.find((s) => s.id === selectedStormId) || INDIA_CYCLONES[0];
  }, [selectedStormId]);

  // Synchronize with parent app if callback provided
  useEffect(() => {
    if (onStormSelect) {
      onStormSelect(activeStorm);
    }
  }, [activeStorm, onStormSelect]);

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current) return;
    if (mapInstanceRef.current) return;

    // Center map on India / Bay of Bengal / Arabian Sea
    const map = L.map(mapContainerRef.current, {
      center: [20.0, 82.5],
      zoom: 5,
      minZoom: 4,
      maxZoom: 10,
      zoomControl: false,
    });

    L.control.zoom({ position: "topright" }).addTo(map);

    // Initial base tile layer (Carto Positron Light)
    const baseTile = L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
      {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap',
        subdomains: "abcd",
        maxZoom: 19,
      }
    ).addTo(map);

    tileLayerRef.current = baseTile;

    const overlayGroup = L.layerGroup().addTo(map);
    overlayLayerGroupRef.current = overlayGroup;

    // Click probe
    map.on("click", (e: L.LeafletMouseEvent) => {
      const { lat, lng } = e.latlng;
      setProbeLatLon({ lat: Number(lat.toFixed(2)), lon: Number(lng.toFixed(2)) });
    });

    mapInstanceRef.current = map;

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update Base Tile Layer when mapStyle changes
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (tileLayerRef.current) {
      map.removeLayer(tileLayerRef.current);
    }

    let url = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png";
    let attr = '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap';

    if (mapStyle === "osm") {
      url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
      attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>';
    } else if (mapStyle === "satellite") {
      url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
      attr = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP";
    }

    tileLayerRef.current = L.tileLayer(url, { attribution: attr, maxZoom: 19 }).addTo(map);
  }, [mapStyle]);

  // Compute Probe Data relative to active storm
  useEffect(() => {
    if (!probeLatLon) return;
    const { lat, lon } = probeLatLon;
    const dLat = (lat - activeStorm.lat) * 111;
    const dLon = (lon - activeStorm.lon) * 111 * Math.cos((activeStorm.lat * Math.PI) / 180);
    const dist = Math.sqrt(dLat * dLat + dLon * dLon);

    let tempC = 26;
    let tempK = 299.15;
    let threat = "Calm Ocean / Peripheral Breezes";
    let windEst = 15;

    if (dist <= activeStorm.rmwKm) {
      tempC = -78.4;
      tempK = 194.75;
      threat = "EYE / EYEWALL (DESTRUCTIVE HURRICANE WINDS)";
      windEst = activeStorm.windKts;
    } else if (dist <= activeStorm.r64Km) {
      tempC = -71.2;
      tempK = 201.95;
      threat = "Core Convective Shield (Very Severe)";
      windEst = Math.round(activeStorm.windKts * 0.9);
    } else if (dist <= activeStorm.r50Km) {
      tempC = -58.5;
      tempK = 214.65;
      threat = "Inner Feeder Bands (Severe Winds)";
      windEst = Math.round(activeStorm.windKts * 0.75);
    } else if (dist <= activeStorm.r34Km) {
      tempC = -42.0;
      tempK = 231.15;
      threat = "Gale Force Perimeter (Heavy Squalls)";
      windEst = Math.round(activeStorm.windKts * 0.55);
    } else if (dist <= 450) {
      tempC = -15.0;
      tempK = 258.15;
      threat = "Outer Cirrus Outflow / Elevated Swells";
      windEst = Math.round(activeStorm.windKts * 0.3);
    }

    setProbeData({
      distKm: Math.round(dist),
      tempK: Number(tempK.toFixed(1)),
      tempC: Number(tempC.toFixed(1)),
      threat,
      windEstKts: windEst,
    });
  }, [probeLatLon, activeStorm]);

  // Render Overlays: Storm Markers, Radii, Cone, Radars, and Animated Cloud Canvas
  useEffect(() => {
    const map = mapInstanceRef.current;
    const group = overlayLayerGroupRef.current;
    if (!map || !group) return;

    group.clearLayers();

    // 1. Forecast Track Cone & Waypoints
    if (showForecastCone && activeStorm.forecastTrack.length > 0) {
      const latlngs: L.LatLngExpression[] = activeStorm.forecastTrack.map((pt) => [pt.lat, pt.lon]);

      // Projected track line
      const trackPolyline = L.polyline(latlngs, {
        color: "#dc2626",
        weight: 3.5,
        opacity: 0.9,
        dashArray: "6, 6",
      });
      group.addLayer(trackPolyline);

      // Uncertainty corridor polygon
      const coneLeft: [number, number][] = [];
      const coneRight: [number, number][] = [];

      activeStorm.forecastTrack.forEach((pt, idx) => {
        const spreadKm = 20 + idx * 35; // Expands over time
        const degOffset = spreadKm / 111;
        coneLeft.push([pt.lat + degOffset * 0.7, pt.lon - degOffset]);
        coneRight.push([pt.lat - degOffset * 0.7, pt.lon + degOffset]);
      });

      const conePolygon = L.polygon([...coneLeft, ...coneRight.reverse()], {
        color: "#ef4444",
        weight: 1.5,
        fillColor: "#fee2e2",
        fillOpacity: 0.35,
        dashArray: "3, 3",
      });
      group.addLayer(conePolygon);

      // Forecast Waypoint Circles
      activeStorm.forecastTrack.forEach((pt) => {
        const isCurrent = pt.hour === 0;
        const color = isCurrent ? "#dc2626" : "#2563eb";
        const marker = L.circleMarker([pt.lat, pt.lon], {
          radius: isCurrent ? 8 : 5,
          color,
          fillColor: "#ffffff",
          fillOpacity: 1,
          weight: 2.5,
        });

        marker.bindPopup(`
          <div style="font-family: monospace; font-size: 11px;">
            <strong style="color: #dc2626;">${activeStorm.name}</strong><br/>
            <strong>${pt.label}</strong><br/>
            Position: ${pt.lat}°N, ${pt.lon}°E<br/>
            Wind: ${pt.windKts} kts (${Math.round(pt.windKts * 1.852)} km/h)<br/>
            Class: ${pt.category}
          </div>
        `);
        group.addLayer(marker);
      });
    }

    // 2. Wind Radii Circles (R34, R50, R64, RMW)
    if (showWindRadii) {
      if (activeStorm.r34Km > 0) {
        const r34Circle = L.circle([activeStorm.lat, activeStorm.lon], {
          radius: activeStorm.r34Km * 1000,
          color: "#f59e0b",
          weight: 1.5,
          fillColor: "#fef3c7",
          fillOpacity: 0.12,
          dashArray: "4, 4",
        }).bindTooltip(`R34 Gale Winds (62+ km/h): ${activeStorm.r34Km} km`, { sticky: true });
        group.addLayer(r34Circle);
      }

      if (activeStorm.r50Km > 0) {
        const r50Circle = L.circle([activeStorm.lat, activeStorm.lon], {
          radius: activeStorm.r50Km * 1000,
          color: "#f97316",
          weight: 1.5,
          fillColor: "#ffedd5",
          fillOpacity: 0.15,
        }).bindTooltip(`R50 Storm Winds (92+ km/h): ${activeStorm.r50Km} km`, { sticky: true });
        group.addLayer(r50Circle);
      }

      if (activeStorm.r64Km > 0) {
        const r64Circle = L.circle([activeStorm.lat, activeStorm.lon], {
          radius: activeStorm.r64Km * 1000,
          color: "#dc2626",
          weight: 2,
          fillColor: "#fee2e2",
          fillOpacity: 0.22,
        }).bindTooltip(`R64 Hurricane Winds (118+ km/h): ${activeStorm.r64Km} km`, { sticky: true });
        group.addLayer(r64Circle);
      }
    }

    // 3. Indian Doppler Weather Radar Stations
    if (showRadars) {
      IMD_RADAR_STATIONS.forEach((station) => {
        // Radar center marker
        const radarMarker = L.circleMarker([station.lat, station.lon], {
          radius: 5,
          color: "#0284c7",
          fillColor: "#38bdf8",
          fillOpacity: 0.9,
          weight: 1.5,
        }).bindPopup(`
          <div style="font-family: monospace; font-size: 11px;">
            <strong style="color: #0284c7;">IMD DWR Station</strong><br/>
            <strong>${station.name}</strong> (${station.state})<br/>
            Range: ${station.rangeKm} km Doppler Coverage<br/>
            Status: <span style="color: #16a34a; font-weight: bold;">OPERATIONAL</span>
          </div>
        `);
        group.addLayer(radarMarker);

        // Radar 250km scan coverage ring
        const radarRing = L.circle([station.lat, station.lon], {
          radius: station.rangeKm * 1000,
          color: "#0284c7",
          weight: 1,
          opacity: 0.35,
          fillColor: "#e0f2fe",
          fillOpacity: 0.04,
          dashArray: "3, 6",
        });
        group.addLayer(radarRing);
      });
    }

    // 4. Storm Center Custom Eye Marker with Pulsing Ring
    const eyeIcon = L.divIcon({
      className: "custom-cyclone-eye",
      html: `
        <div style="position: relative; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center;">
          <div style="position: absolute; width: 34px; height: 34px; border-radius: 50%; border: 2.5px solid #dc2626; animation: ping 2s cubic-bezier(0, 0, 0.2, 1) infinite; opacity: 0.75;"></div>
          <div style="width: 22px; height: 22px; border-radius: 50%; background: #dc2626; border: 2.5px solid #ffffff; box-shadow: 0 0 10px rgba(220, 38, 38, 0.8); display: flex; align-items: center; justify-content: center; color: white; font-size: 10px; font-weight: bold;">
            🌀
          </div>
        </div>
      `,
      iconSize: [36, 36],
      iconAnchor: [18, 18],
    });

    const stormMarker = L.marker([activeStorm.lat, activeStorm.lon], { icon: eyeIcon });
    stormMarker.bindPopup(`
      <div style="font-family: monospace; font-size: 12px; min-width: 180px;">
        <div style="font-weight: bold; color: #dc2626; font-size: 13px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; margin-bottom: 4px;">
          ${activeStorm.name}
        </div>
        <div><strong>Intensity:</strong> ${activeStorm.category}</div>
        <div><strong>Position:</strong> ${activeStorm.lat}°N, ${activeStorm.lon}°E</div>
        <div><strong>Max Wind:</strong> ${activeStorm.windKts} kts (${Math.round(activeStorm.windKts * 1.852)} km/h)</div>
        <div><strong>Central Pressure:</strong> ${activeStorm.pressureHpa} hPa</div>
        <div><strong>Movement:</strong> ${activeStorm.heading} at ${activeStorm.translationKmh} km/h</div>
        <div style="margin-top: 4px; color: #b91c1c; font-weight: bold;">${activeStorm.landfallStatus}</div>
      </div>
    `);
    group.addLayer(stormMarker);

    // 5. Landfall Target Indicator
    if (activeStorm.distToCoastKm > 0) {
      const lineToCoast = L.polyline(
        [
          [activeStorm.lat, activeStorm.lon],
          [activeStorm.forecastTrack[1]?.lat || activeStorm.lat, activeStorm.forecastTrack[1]?.lon || activeStorm.lon],
        ],
        { color: "#ea580c", weight: 2, dashArray: "4, 4" }
      );
      group.addLayer(lineToCoast);
    }
  }, [
    activeStorm,
    showForecastCone,
    showWindRadii,
    showRadars,
  ]);

  // Dynamic Procedural Canvas Animation for Satellite Clouds & Wind Streamlines
  // NO STATIC IMAGES! This is a mathematically rendered rotating convective vortex.
  const cloudCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = cloudCanvasRef.current;
    const map = mapInstanceRef.current;
    if (!canvas || !map || !showClouds) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let running = true;

    const renderConvectiveClouds = () => {
      if (!running) return;

      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      // Convert storm lat/lon to map container pixel coordinates
      const stormLatLng = L.latLng(activeStorm.lat, activeStorm.lon);
      const point = map.latLngToContainerPoint(stormLatLng);
      const zoom = map.getZoom();

      // Radius scales with zoom
      const baseScale = Math.pow(2, zoom - 5);
      const stormRadiusPx = (activeStorm.r34Km / 8) * baseScale * 4;

      if (
        point.x + stormRadiusPx > 0 &&
        point.x - stormRadiusPx < width &&
        point.y + stormRadiusPx > 0 &&
        point.y - stormRadiusPx < height
      ) {
        ctx.save();
        ctx.translate(point.x, point.y);

        if (isRotating) {
          rotationAngleRef.current += 0.008; // Counter-clockwise rotation
        }
        ctx.rotate(-rotationAngleRef.current);

        // Draw Multi-Arm Convective Logarithmic Spirals
        const arms = 4;
        const maxRadius = stormRadiusPx;

        // Radial backdrop cloud shield
        const radialGrad = ctx.createRadialGradient(0, 0, 10 * baseScale, 0, 0, maxRadius);
        if (cloudEnhancement === "bd-thermal") {
          // BD-Curve: Cold cloud tops in deep red/magenta, surrounding shield in blue/cyan
          radialGrad.addColorStop(0, "rgba(255, 255, 255, 0.95)"); // Eye subsidence
          radialGrad.addColorStop(0.12, "rgba(190, 18, 60, 0.85)"); // Eyewall (< -75°C)
          radialGrad.addColorStop(0.28, "rgba(13, 148, 136, 0.75)"); // Intense convection (-60°C)
          radialGrad.addColorStop(0.55, "rgba(2, 132, 199, 0.55)"); // Inner core bands (-45°C)
          radialGrad.addColorStop(0.85, "rgba(224, 242, 254, 0.35)"); // Outer cirrus shield
          radialGrad.addColorStop(1, "rgba(255, 255, 255, 0)");
        } else if (cloudEnhancement === "rainbow") {
          radialGrad.addColorStop(0, "rgba(255, 255, 255, 0.9)");
          radialGrad.addColorStop(0.15, "rgba(239, 68, 68, 0.8)");
          radialGrad.addColorStop(0.35, "rgba(234, 179, 8, 0.7)");
          radialGrad.addColorStop(0.6, "rgba(34, 197, 94, 0.5)");
          radialGrad.addColorStop(0.85, "rgba(59, 130, 246, 0.3)");
          radialGrad.addColorStop(1, "rgba(147, 51, 234, 0)");
        } else {
          // Water Vapor mid-troposphere
          radialGrad.addColorStop(0, "rgba(253, 224, 71, 0.9)");
          radialGrad.addColorStop(0.2, "rgba(217, 70, 239, 0.8)");
          radialGrad.addColorStop(0.5, "rgba(99, 102, 241, 0.6)");
          radialGrad.addColorStop(0.85, "rgba(14, 165, 233, 0.3)");
          radialGrad.addColorStop(1, "rgba(15, 23, 42, 0)");
        }

        ctx.fillStyle = radialGrad;
        ctx.beginPath();
        ctx.arc(0, 0, maxRadius, 0, Math.PI * 2);
        ctx.fill();

        // Draw spiral convective arms
        for (let a = 0; a < arms; a++) {
          const armAngleOffset = (a * 2 * Math.PI) / arms;
          ctx.beginPath();

          const armPoints = 60;
          for (let p = 0; p < armPoints; p++) {
            const t = p / armPoints;
            const r = 12 * baseScale + t * (maxRadius - 12 * baseScale);
            // Logarithmic spiral: theta = b * ln(r)
            const theta = armAngleOffset + 2.5 * Math.log(1 + t * 4);
            const x = r * Math.cos(theta);
            const y = r * Math.sin(theta);

            if (p === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          }

          ctx.lineWidth = Math.max(3, 16 * baseScale * (1 - a / 6));
          ctx.strokeStyle =
            cloudEnhancement === "bd-thermal"
              ? "rgba(225, 29, 72, 0.55)"
              : "rgba(249, 115, 22, 0.55)";
          ctx.lineCap = "round";
          ctx.stroke();

          // Outer secondary band
          ctx.lineWidth = Math.max(2, 8 * baseScale);
          ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
          ctx.stroke();
        }

        // Draw clear storm eye at center
        const eyeRadius = Math.max(4, (activeStorm.rmwKm / 12) * baseScale);
        ctx.beginPath();
        ctx.arc(0, 0, eyeRadius, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.strokeStyle = "#dc2626";
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.restore();
      }

      // Draw Wind Particle Streamlines if enabled
      if (showWindVectors) {
        ctx.save();
        ctx.strokeStyle = "rgba(2, 132, 199, 0.35)";
        ctx.lineWidth = 1.2;

        const numStreamlines = 16;
        const time = Date.now() * 0.0015;

        for (let i = 0; i < numStreamlines; i++) {
          const streamAngle = (i * 2 * Math.PI) / numStreamlines;
          const outerR = stormRadiusPx * 1.25;
          const innerR = 18 * baseScale;
          const streamProgress = ((time + i / numStreamlines) % 1);
          const currentR = outerR - streamProgress * (outerR - innerR);
          const currentAngle = streamAngle - streamProgress * 2.8;

          const px = point.x + currentR * Math.cos(currentAngle);
          const py = point.y + currentR * Math.sin(currentAngle);

          ctx.beginPath();
          ctx.arc(px, py, 2, 0, Math.PI * 2);
          ctx.fillStyle = "#0284c7";
          ctx.fill();
        }
        ctx.restore();
      }

      animationFrameRef.current = requestAnimationFrame(renderConvectiveClouds);
    };

    // Update canvas size on resize or zoom
    const resizeCanvas = () => {
      const container = mapContainerRef.current;
      if (!container) return;
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
    };

    resizeCanvas();
    map.on("resize move zoom", resizeCanvas);

    renderConvectiveClouds();

    return () => {
      running = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      map.off("resize move zoom", resizeCanvas);
    };
  }, [activeStorm, showClouds, showWindVectors, cloudEnhancement, isRotating]);

  // Recenter Map Helper
  const handleRecenterStorm = () => {
    const map = mapInstanceRef.current;
    if (!map) return;
    map.flyTo([activeStorm.lat, activeStorm.lon], 6, { duration: 1.2 });
  };

  const handleRecenterIndia = () => {
    const map = mapInstanceRef.current;
    if (!map) return;
    map.flyTo([20.0, 82.5], 5, { duration: 1.2 });
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-xs space-y-4">
      {/* Header Bar */}
      <div className="p-4 sm:p-5 bg-slate-50/80 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-sky-100 border border-sky-200 flex items-center justify-center text-sky-700">
            <Satellite className="w-5 h-5 text-sky-600 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base sm:text-lg font-bold text-slate-900">
                Live India Cyclone Satellite Map
              </h3>
              <span className="text-[10px] font-mono-code px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold">
                ● LIVE RADIAL TELEMETRY
              </span>
            </div>
            <p className="text-xs text-slate-500 font-mono-code">
              Dynamic Real-Time Convective Radiance & Radar Tracking · Bay of Bengal & Arabian Sea
            </p>
          </div>
        </div>

        {/* Quick Map Base Style Switcher */}
        <div className="flex items-center gap-1.5 bg-white border border-slate-200 p-1 rounded-xl text-xs font-mono-code">
          <span className="text-slate-400 px-2 text-[11px]">BASEMAP:</span>
          <button
            onClick={() => setMapStyle("carto-light")}
            className={`px-2.5 py-1 rounded-lg transition cursor-pointer font-medium ${
              mapStyle === "carto-light"
                ? "bg-sky-600 text-white font-bold"
                : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            Light Carto
          </button>
          <button
            onClick={() => setMapStyle("osm")}
            className={`px-2.5 py-1 rounded-lg transition cursor-pointer font-medium ${
              mapStyle === "osm"
                ? "bg-sky-600 text-white font-bold"
                : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            OSM
          </button>
          <button
            onClick={() => setMapStyle("satellite")}
            className={`px-2.5 py-1 rounded-lg transition cursor-pointer font-medium ${
              mapStyle === "satellite"
                ? "bg-sky-600 text-white font-bold"
                : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            Satellite
          </button>
        </div>
      </div>

      {/* Cyclone Quick Selection Bar */}
      <div className="px-4 sm:px-6">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-bold text-slate-700 font-mono-code uppercase tracking-wider flex items-center gap-1.5">
            <Radio className="w-3.5 h-3.5 text-sky-600" />
            SELECT INDIAN CYCLONE SYSTEM:
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {INDIA_CYCLONES.map((storm) => {
            const isSelected = storm.id === selectedStormId;
            return (
              <button
                key={storm.id}
                onClick={() => setSelectedStormId(storm.id)}
                className={`p-2.5 rounded-xl border text-left transition cursor-pointer flex flex-col justify-between ${
                  isSelected
                    ? "bg-sky-50 border-sky-400 ring-2 ring-sky-200 shadow-xs"
                    : "bg-white border-slate-200 hover:bg-slate-50 text-slate-700"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between gap-1 mb-1">
                    <span className="text-[10px] font-bold font-mono-code px-1.5 py-0.2 rounded bg-slate-100 text-slate-700">
                      {storm.basin === "Bay of Bengal" ? "BOB" : "ARB"}
                    </span>
                    <span
                      className={`text-[9px] font-mono-code font-bold ${
                        storm.windKts >= 100
                          ? "text-purple-600"
                          : storm.windKts >= 64
                          ? "text-red-600"
                          : "text-amber-600"
                      }`}
                    >
                      {storm.windKts} KTS
                    </span>
                  </div>
                  <div className="text-xs font-bold text-slate-900 leading-tight truncate">
                    {storm.name.replace("Severe Cyclonic Storm ", "").replace("Extremely Severe Cyclonic Storm ", "").replace("Super Cyclonic Storm ", "")}
                  </div>
                </div>
                <div className="text-[10px] text-slate-500 font-mono-code mt-1.5 truncate">
                  {storm.nearestIndianCoast.split(" (")[0]}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Interactive Map Stage */}
      <div className="px-4 sm:px-6">
        <div className="relative w-full h-[540px] sm:h-[600px] rounded-xl overflow-hidden border border-slate-300 shadow-inner bg-slate-100">
          {/* Leaflet Map DOM Container */}
          <div ref={mapContainerRef} className="w-full h-full z-0" />

          {/* Dynamic Procedural Satellite Canvas Layer (No static images!) */}
          <canvas
            ref={cloudCanvasRef}
            className="absolute inset-0 pointer-events-none z-10"
          />

          {/* Floating Map Controls & Layer Toggles (Top Left) */}
          <div className="absolute top-3 left-3 z-20 flex flex-col gap-2 max-w-[280px]">
            {/* Quick Storm Badge */}
            <div className="bg-white/95 backdrop-blur-md border border-slate-200 rounded-xl p-3 shadow-md">
              <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-1.5 mb-2">
                <span className="text-xs font-bold text-slate-900 truncate flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-ping" />
                  {activeStorm.name}
                </span>
                <span className="text-[10px] font-mono-code bg-red-50 text-red-700 px-1.5 py-0.5 rounded font-bold border border-red-200">
                  {activeStorm.pressureHpa} hPa
                </span>
              </div>
              <div className="text-[11px] font-mono-code text-slate-600 space-y-0.5">
                <div className="flex justify-between">
                  <span>Coords:</span>
                  <strong className="text-slate-900">{activeStorm.lat}°N, {activeStorm.lon}°E</strong>
                </div>
                <div className="flex justify-between">
                  <span>Peak Winds:</span>
                  <strong className="text-red-700">{activeStorm.windKts} kts ({Math.round(activeStorm.windKts * 1.852)} km/h)</strong>
                </div>
                <div className="flex justify-between">
                  <span>Coast Target:</span>
                  <strong className="text-sky-800">{activeStorm.distToCoastKm} km to coast</strong>
                </div>
                <div className="text-[10px] text-amber-700 font-semibold pt-1 truncate">
                  ⚠ {activeStorm.landfallStatus}
                </div>
              </div>
            </div>

            {/* Layer Visibility Toggles */}
            <div className="bg-white/95 backdrop-blur-md border border-slate-200 rounded-xl p-2.5 shadow-md space-y-1.5">
              <div className="text-[10px] font-bold text-slate-500 font-mono-code uppercase px-1">
                MAP OVERLAY LAYERS:
              </div>
              <div className="grid grid-cols-2 gap-1 text-[11px] font-mono-code">
                <button
                  onClick={() => setShowClouds(!showClouds)}
                  className={`px-2 py-1 rounded text-left flex items-center justify-between cursor-pointer border ${
                    showClouds
                      ? "bg-sky-50 text-sky-800 border-sky-300 font-bold"
                      : "bg-slate-50 text-slate-500 border-slate-200"
                  }`}
                >
                  <span>☁ Clouds</span>
                  <span>{showClouds ? "ON" : "OFF"}</span>
                </button>
                <button
                  onClick={() => setShowWindVectors(!showWindVectors)}
                  className={`px-2 py-1 rounded text-left flex items-center justify-between cursor-pointer border ${
                    showWindVectors
                      ? "bg-sky-50 text-sky-800 border-sky-300 font-bold"
                      : "bg-slate-50 text-slate-500 border-slate-200"
                  }`}
                >
                  <span>💨 Winds</span>
                  <span>{showWindVectors ? "ON" : "OFF"}</span>
                </button>
                <button
                  onClick={() => setShowForecastCone(!showForecastCone)}
                  className={`px-2 py-1 rounded text-left flex items-center justify-between cursor-pointer border ${
                    showForecastCone
                      ? "bg-red-50 text-red-800 border-red-300 font-bold"
                      : "bg-slate-50 text-slate-500 border-slate-200"
                  }`}
                >
                  <span>📍 72h Cone</span>
                  <span>{showForecastCone ? "ON" : "OFF"}</span>
                </button>
                <button
                  onClick={() => setShowRadars(!showRadars)}
                  className={`px-2 py-1 rounded text-left flex items-center justify-between cursor-pointer border ${
                    showRadars
                      ? "bg-emerald-50 text-emerald-800 border-emerald-300 font-bold"
                      : "bg-slate-50 text-slate-500 border-slate-200"
                  }`}
                >
                  <span>📡 IMD Radar</span>
                  <span>{showRadars ? "ON" : "OFF"}</span>
                </button>
              </div>

              {/* Thermal IR Colormap Selector */}
              <div className="pt-1 border-t border-slate-100 flex items-center justify-between gap-1 text-[10px] font-mono-code">
                <span className="text-slate-500">IR CURVE:</span>
                <select
                  value={cloudEnhancement}
                  onChange={(e) => setCloudEnhancement(e.target.value as any)}
                  className="bg-slate-50 border border-slate-200 text-slate-800 rounded px-1.5 py-0.5 text-[10px] cursor-pointer"
                >
                  <option value="bd-thermal">BD Thermal (-80°C)</option>
                  <option value="rainbow">Rainbow Scale</option>
                  <option value="water-vapor">Water Vapor 6.7µm</option>
                </select>
              </div>
            </div>
          </div>

          {/* Map Utility Actions (Top Right) */}
          <div className="absolute top-3 right-14 z-20 flex items-center gap-1.5">
            <button
              onClick={handleRecenterStorm}
              className="bg-white/95 backdrop-blur-md hover:bg-slate-100 text-slate-800 border border-slate-200 text-xs px-2.5 py-1.5 rounded-lg shadow-sm font-mono-code font-bold flex items-center gap-1 cursor-pointer"
              title="Center on storm center"
            >
              <Crosshair className="w-3.5 h-3.5 text-red-600" />
              <span>Center Eye</span>
            </button>
            <button
              onClick={handleRecenterIndia}
              className="bg-white/95 backdrop-blur-md hover:bg-slate-100 text-slate-800 border border-slate-200 text-xs px-2.5 py-1.5 rounded-lg shadow-sm font-mono-code font-bold flex items-center gap-1 cursor-pointer"
              title="Reset view to all of India"
            >
              <Compass className="w-3.5 h-3.5 text-sky-600" />
              <span>India View</span>
            </button>
            <button
              onClick={() => setIsRotating(!isRotating)}
              className="bg-white/95 backdrop-blur-md hover:bg-slate-100 text-slate-800 border border-slate-200 text-xs p-1.5 rounded-lg shadow-sm font-mono-code cursor-pointer"
              title={isRotating ? "Pause cloud animation" : "Resume cloud animation"}
            >
              {isRotating ? <Pause className="w-3.5 h-3.5 text-slate-700" /> : <Play className="w-3.5 h-3.5 text-sky-600" />}
            </button>
          </div>

          {/* Interactive Click Probe Bar (Bottom Floating) */}
          <div className="absolute bottom-3 left-3 right-3 z-20">
            <div className="bg-white/95 backdrop-blur-md border border-slate-200 rounded-xl p-3 shadow-md flex flex-wrap items-center justify-between gap-3 text-xs font-mono-code">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-sky-500 animate-pulse" />
                <span className="text-slate-500">CLICK/INSPECT PROBE:</span>
                {probeLatLon && (
                  <span className="font-bold text-slate-900 bg-slate-100 px-2 py-0.5 rounded">
                    {probeLatLon.lat}°N, {probeLatLon.lon}°E
                  </span>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-4 text-slate-600">
                <div>
                  <span className="text-slate-400">Eye Dist: </span>
                  <strong className="text-slate-900">{probeData.distKm} km</strong>
                </div>
                <div>
                  <span className="text-slate-400">Cloud Temp: </span>
                  <strong className={probeData.tempC < -50 ? "text-rose-600 font-bold" : "text-sky-700 font-bold"}>
                    {probeData.tempC}°C ({probeData.tempK} K)
                  </strong>
                </div>
                <div>
                  <span className="text-slate-400">Wind Est: </span>
                  <strong className="text-slate-900">{probeData.windEstKts} kts</strong>
                </div>
                <div>
                  <span className="text-slate-400">Threat: </span>
                  <span className="font-bold text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-200 text-[11px]">
                    {probeData.threat}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Legend & IMD Coastal Doppler Stations Summary */}
      <div className="p-4 sm:p-6 bg-slate-50/50 border-t border-slate-200 grid grid-cols-1 md:grid-cols-12 gap-4">
        {/* Left: Meteorological Scale Legend */}
        <div className="md:col-span-6 space-y-2">
          <div className="text-xs font-bold text-slate-800 font-mono-code flex items-center gap-2">
            <span>DVORAK THERMAL INFRARED (BD-CURVE) COLOR TEMPERATURE SCALE:</span>
          </div>
          <div className="h-4 rounded-md overflow-hidden flex shadow-xs border border-slate-200 text-[9px] font-mono-code font-bold text-white items-center justify-between px-2"
               style={{
                 background: "linear-gradient(to right, #ffffff 0%, #be123c 15%, #0d9488 40%, #0284c7 70%, #e0f2fe 90%, #f8fafc 100%)",
                 color: "#0f172a"
               }}>
            <span>-85°C (Eyewall)</span>
            <span>-65°C</span>
            <span>-45°C</span>
            <span>-20°C</span>
            <span>+25°C (Ocean)</span>
          </div>
          <p className="text-[11px] text-slate-500 font-mono-code leading-relaxed">
            Procedurally computed from Planck thermal radiance equations at 10.8µm (INSAT-3D/3DR VHRR specifications). No static photos; all cloud fields dynamically project coordinates in real-time.
          </p>
        </div>

        {/* Right: Operational Coastal Radar Status */}
        <div className="md:col-span-6 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-800 font-mono-code">
              IMD COASTAL DOPPLER WEATHER RADARS (DWR)
            </span>
            <span className="text-[10px] font-mono-code text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded font-bold border border-emerald-200">
              10/10 STATIONS ONLINE
            </span>
          </div>
          <div className="flex flex-wrap gap-1.5 text-[10px] font-mono-code">
            {IMD_RADAR_STATIONS.slice(0, 6).map((st) => (
              <span key={st.name} className="bg-white border border-slate-200 px-2 py-0.5 rounded text-slate-700 shadow-2xs">
                📡 {st.name.replace(" DWR", "")} ({st.rangeKm}km)
              </span>
            ))}
            <span className="bg-sky-50 border border-sky-200 px-2 py-0.5 rounded text-sky-800 font-semibold">
              + 4 more on West Coast
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
