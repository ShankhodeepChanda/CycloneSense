import React, { useState } from "react";
import {
  Download,
  Copy,
  Check,
  FileSpreadsheet,
  FileText,
  ShieldAlert,
  Waves,
  MapPin,
  Radio,
  Share2,
  AlertTriangle,
  Info,
} from "lucide-react";

interface HazardAdvisoryHubProps {
  stormName?: string;
  stormLat?: number;
  stormLon?: number;
  windKts?: number;
  pressureHpa?: number;
  intensityCategory?: string;
}

export const HazardAdvisoryHub: React.FC<HazardAdvisoryHubProps> = ({
  stormName = "ARB-2026-02 / CYCLONE TAUKTAE II",
  stormLat = 18.42,
  stormLon = 71.18,
  windKts = 95,
  pressureHpa = 954,
  intensityCategory = "Extremely Severe Cyclonic Storm (ESCS)",
}) => {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  // Generate GeoJSON GIS Export
  const handleDownloadGeoJSON = () => {
    const geojson = {
      type: "FeatureCollection",
      metadata: {
        system: "CycloneSense AI Operational Hazard Dissemination",
        basin: "North Indian Ocean",
        cyclone_id: stormName,
        generated_at: new Date().toISOString(),
      },
      features: [
        {
          type: "Feature",
          properties: {
            name: `${stormName} Current Center Fix`,
            wind_kts: windKts,
            pressure_hpa: pressureHpa,
            category: intensityCategory,
          },
          geometry: {
            type: "Point",
            coordinates: [stormLon, stormLat],
          },
        },
        {
          type: "Feature",
          properties: {
            name: "Landfall Intercept Point",
            location: "Gujarat Coast near Diu / Una",
            estimated_surge_meters: 3.8,
            eta_hours: 36,
          },
          geometry: {
            type: "Point",
            coordinates: [70.98, 20.7],
          },
        },
      ],
    };

    const blob = new Blob([JSON.stringify(geojson, null, 2)], {
      type: "application/geo+json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cyclonesense_hazard_${stormName.split(" ")[0].replace(/[^a-zA-Z0-9_-]/g, "")}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const bulletinText = `INDIA METEOROLOGICAL DEPARTMENT / CYCLONESENSE AI BULLETIN NO. 14
TIME OF ISSUE: ${new Date().toUTCString()}
SUBJECT: ${intensityCategory.toUpperCase()} OVER THE NORTH INDIAN OCEAN BASIN

1. PRESENT LOCATION & INTENSITY:
   The ${intensityCategory} (${stormName}) was centered at ${new Date().toISOString()} near Latitude ${stormLat}°N and Longitude ${stormLon}°E.
   Estimated Central Pressure: ${pressureHpa} hPa.
   Maximum Sustained Surface Wind Speed: ${windKts} knots (${Math.round(windKts * 1.852)} km/h) gusting to ${Math.round(windKts * 1.25)} knots.

2. FORECAST TRACK & INTENSITY:
   The system is moving north-northwestwards with a translational speed of 18 km/h.
   It is projected to continue its trajectory with gradual recurvature toward the coast.
   Landfall is expected within the next 36 to 48 hours with severe storm surges.

3. WARNING & SECTOR ADVISORIES:
   (i) Coastal Inundation: Astronomical high spring tides coupled with a storm surge of 3.0 to 4.5 meters above astronomical tide will inundate low-lying coastal sectors.
   (ii) Fishermen Warning: Total suspension of fishing operations across central and northern basin waters. Vessels in deep seas are advised to return to ports immediately.
   (iii) Port Warning: Keep Local Warning Signal No. 4 hoisted at major commercial ports.

ISSUED BY: CYCLONESENSE AI METEOROLOGICAL DISSEMINATION DESK
WMO GTS STANDARD / REGIONAL SPECIALIZED METEOROLOGICAL CENTRE (RSMC) COMPLIANT`;

  // Download Bulletin as text file
  const handleDownloadBulletin = () => {
    const blob = new Blob([bulletinText], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `IMD_CycloneSense_Bulletin_14.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const imdCategories = [
    { code: "LPA", name: "Low Pressure Area", windKts: "< 17 kts", speedKmh: "< 31 km/h", pressureDrop: "< 1 hPa" },
    { code: "D", name: "Depression", windKts: "17 - 27 kts", speedKmh: "31 - 49 km/h", pressureDrop: "1 - 3 hPa" },
    { code: "DD", name: "Deep Depression", windKts: "28 - 33 kts", speedKmh: "50 - 61 km/h", pressureDrop: "3 - 4.5 hPa" },
    { code: "CS", name: "Cyclonic Storm", windKts: "34 - 47 kts", speedKmh: "62 - 88 km/h", pressureDrop: "4.5 - 8.5 hPa" },
    { code: "SCS", name: "Severe Cyclonic Storm", windKts: "48 - 63 kts", speedKmh: "89 - 117 km/h", pressureDrop: "8.5 - 15 hPa" },
    { code: "VSCS", name: "Very Severe Cyclonic Storm", windKts: "64 - 89 kts", speedKmh: "118 - 166 km/h", pressureDrop: "15 - 35 hPa" },
    { code: "ESCS", name: "Extremely Severe Cyclonic Storm", windKts: "90 - 119 kts", speedKmh: "167 - 221 km/h", pressureDrop: "35 - 65 hPa" },
    { code: "SuCS", name: "Super Cyclonic Storm", windKts: "≥ 120 kts", speedKmh: "≥ 222 km/h", pressureDrop: "> 65 hPa" },
  ];

  return (
    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm space-y-6">
      {/* Title Header */}
      <div className="p-4 sm:p-5 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-600">
            <Radio className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base sm:text-lg font-bold text-slate-900 flex items-center gap-2">
              Official Bulletins, GIS Hazard Export & Disaster Advisories
            </h2>
            <p className="text-xs text-slate-500 font-mono-code">
              WMO GTS Telemetry Compliant · GeoJSON Spatial Layers · Coastal Evacuation Readiness
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleDownloadGeoJSON}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white hover:bg-slate-100 text-slate-700 border border-slate-300 text-xs font-mono-code transition cursor-pointer shadow-2xs"
          >
            <Download className="w-3.5 h-3.5 text-sky-600" />
            <span>Export GeoJSON GIS</span>
          </button>
          <button
            onClick={handleDownloadBulletin}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-700 text-white text-xs font-bold font-mono-code transition cursor-pointer shadow-xs"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Download Bulletin</span>
          </button>
        </div>
      </div>

      <div className="p-4 sm:p-6 space-y-6">
        {/* Top Emergency Status Bar */}
        <div className="bg-gradient-to-r from-red-50 via-amber-50 to-orange-50 border border-red-300 p-4 rounded-xl flex flex-wrap items-center justify-between gap-4 shadow-2xs">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-red-700 text-xs font-mono-code font-bold">
              <ShieldAlert className="w-4 h-4 text-red-600 animate-pulse" />
              <span>OFFICIAL DISASTER MANAGEMENT DISPATCH · RSMC PROTOCOL</span>
            </div>
            <h3 className="text-base font-bold text-slate-900">
              {stormName} — {intensityCategory}
            </h3>
            <p className="text-xs text-slate-700">
              Current Center Fix: <strong className="text-sky-800 font-mono-code">{stormLat}°N, {stormLon}°E</strong> · Sustained Winds: <strong className="text-teal-700 font-mono-code">{windKts} kts ({Math.round(windKts * 1.852)} km/h)</strong> · Central Pressure: <strong className="text-amber-700 font-mono-code">{pressureHpa} hPa</strong>.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="bg-emerald-50 text-emerald-800 text-xs font-mono-code px-3 py-1.5 rounded-lg border border-emerald-300 flex items-center gap-1.5 font-bold">
              <Check className="w-3.5 h-3.5 text-emerald-600" />
              NDMA / COAST GUARD SYNCED
            </span>
          </div>
        </div>

        {/* Official Bulletin Telegraph Box */}
        <div className="bg-white border border-slate-200 p-4 rounded-xl space-y-3 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <FileText className="w-4 h-4 text-sky-600" />
                Standard WMO / IMD Cyclone Warning Bulletin (Telegraph Format)
              </h3>
              <p className="text-xs text-slate-500 font-mono-code">
                Formatted for operational release to shipping vessels, aviation towers, and regional disaster command centers.
              </p>
            </div>

            <button
              onClick={() => handleCopy(bulletinText, "bulletin")}
              className="flex items-center gap-1.5 px-3 py-1 rounded bg-slate-50 border border-slate-200 text-slate-700 text-xs font-mono-code hover:bg-slate-100 transition cursor-pointer"
            >
              {copiedKey === "bulletin" ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                  <span className="text-emerald-700 font-bold">Copied to Clipboard!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy Bulletin</span>
                </>
              )}
            </button>
          </div>

          <pre className="bg-slate-900 border border-slate-800 p-4 rounded-lg text-xs font-mono-code text-sky-200 overflow-x-auto whitespace-pre leading-relaxed">
            {bulletinText}
          </pre>
        </div>

        {/* 2-Column: Coastal Hazard Advisories & IMD Classification Scale */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Coastal Impact Protocols */}
          <div className="lg:col-span-6 bg-white border border-slate-200 p-4 rounded-xl space-y-4 shadow-sm">
            <div className="flex items-center gap-2 border-b border-slate-100 pb-2">
              <Waves className="w-4 h-4 text-amber-600" />
              <h3 className="text-sm font-bold text-slate-900 font-mono-code">
                COASTAL HAZARD & EVACUATION PROTOCOLS
              </h3>
            </div>

            <div className="space-y-3 text-xs">
              <div className="bg-amber-50/70 border border-amber-200 p-3 rounded-lg space-y-1">
                <span className="text-amber-900 font-bold font-mono-code flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                  Storm Surge Inundation Warning
                </span>
                <p className="text-slate-700 leading-relaxed">
                  Tidal wave heights of 3.0 to 4.5 meters expected near landfall zone. Low-lying villages and agricultural belts within 5 km of coast must be evacuated into reinforced cyclone shelters.
                </p>
              </div>

              <div className="bg-red-50/70 border border-red-200 p-3 rounded-lg space-y-1">
                <span className="text-red-900 font-bold font-mono-code flex items-center gap-1.5">
                  <ShieldAlert className="w-3.5 h-3.5 text-red-600" />
                  Fishermen Offshore Ban
                </span>
                <p className="text-slate-700 leading-relaxed">
                  Rough to phenomenal sea conditions. Full suspension of fishing operations, pleasure crafts, and cargo lighters in the active basin until storm dissipation.
                </p>
              </div>

              <div className="bg-sky-50/70 border border-sky-200 p-3 rounded-lg space-y-1">
                <span className="text-sky-900 font-bold font-mono-code flex items-center gap-1.5">
                  <Info className="w-3.5 h-3.5 text-sky-600" />
                  Structural & Wind Damage Assessment
                </span>
                <p className="text-slate-700 leading-relaxed">
                  Extensive damage to thatched roofs, kutcha houses, overhead power transmission lines, and communication towers. Uprooting of large trees expected.
                </p>
              </div>
            </div>
          </div>

          {/* IMD Intensity Scale Reference */}
          <div className="lg:col-span-6 bg-white border border-slate-200 p-4 rounded-xl space-y-3 shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2">
              <h3 className="text-sm font-bold text-slate-900 font-mono-code flex items-center gap-2">
                <FileSpreadsheet className="w-4 h-4 text-emerald-600" />
                IMD CYCLONE INTENSITY SCALE
              </h3>
              <span className="text-[10px] text-slate-500 font-mono-code">WMO STANDARD</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono-code text-left">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 bg-slate-50">
                    <th className="py-1 px-1">STAGE</th>
                    <th>CATEGORY</th>
                    <th>WINDS (KTS)</th>
                    <th>SPEED (KM/H)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {imdCategories.map((cat) => {
                    const isCurrent = intensityCategory.includes(cat.name) || intensityCategory.includes(cat.code);
                    return (
                      <tr
                        key={cat.code}
                        className={`hover:bg-slate-50 ${
                          isCurrent
                            ? "bg-sky-50 text-sky-900 font-bold border-l-2 border-sky-600"
                            : "text-slate-700"
                        }`}
                      >
                        <td className="py-1.5 px-1 font-bold">{cat.code}</td>
                        <td className="text-slate-900 font-semibold">{cat.name}</td>
                        <td className="text-teal-700 font-bold">{cat.windKts}</td>
                        <td className="text-amber-700 font-bold">{cat.speedKmh}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
