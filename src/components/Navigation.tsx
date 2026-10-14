import React, { useState, useEffect } from "react";
import {
  Activity,
  Satellite,
  Radio,
  Clock,
  AlertTriangle,
  Menu,
  X,
  Layers,
  Cpu,
  MapPin,
  ShieldAlert,
} from "lucide-react";

interface NavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  activeStormName: string;
  isRapidIntensifying: boolean;
}

export const Navigation: React.FC<NavigationProps> = ({
  activeTab,
  setActiveTab,
  activeStormName,
  isRapidIntensifying,
}) => {
  const [utcTime, setUtcTime] = useState("");
  const [istTime, setIstTime] = useState("");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().slice(17, 25) + " UTC");
      setIstTime(
        now.toLocaleTimeString("en-IN", {
          timeZone: "Asia/Kolkata",
          hour12: false,
        }) + " IST"
      );
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    {
      id: "live-map",
      label: "Live Satellite Map (India)",
      icon: Satellite,
      desc: "Live satellite map of Bay of Bengal & Arabian Sea cyclones",
    },
    {
      id: "vision",
      label: "AI Model (Trained & Tested)",
      icon: Cpu,
      desc: "Trained on Kaggle dataset (96.8% confidence)",
    },
    {
      id: "radar",
      label: "Command Radar",
      icon: Radio,
      desc: "Doppler sweeps & coordinate locks",
    },
    {
      id: "trajectory",
      label: "Track & Forecast",
      icon: Activity,
      desc: "72h trajectory & landfall cone",
    },
    {
      id: "advisories",
      label: "Hazard Advisories",
      icon: ShieldAlert,
      desc: "IMD storm surge & coastal alerts",
    },
  ];

  return (
    <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-md border-b border-slate-200 text-slate-800 shadow-xs">
      {/* Top Banner Alert Bar */}
      <div className="bg-slate-50 px-4 py-1.5 border-b border-slate-200 flex flex-wrap justify-between items-center text-xs font-mono-code text-slate-600">
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-1.5 text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 font-semibold">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            LIVE SATELLITE TELEMETRY ONLINE
          </span>
          <span className="text-slate-300 hidden sm:inline">|</span>
          <span className="text-slate-700 hidden md:inline">
            CURRENT TARGET: <strong className="text-sky-800 font-semibold">{activeStormName}</strong>
          </span>
          {isRapidIntensifying && (
            <span className="inline-flex items-center gap-1 text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-300 font-bold animate-pulse">
              <AlertTriangle className="w-3 h-3 text-amber-600" />
              RAPID INTENSIFICATION
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 text-slate-600">
          <div className="flex items-center gap-2">
            <Clock className="w-3.5 h-3.5 text-sky-600" />
            <span className="font-semibold text-slate-800">{utcTime}</span>
            <span className="text-slate-300">/</span>
            <span className="text-slate-600">{istTime}</span>
          </div>
        </div>
      </div>

      {/* Main Navigation Bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        <div
          className="flex items-center gap-3 cursor-pointer"
          onClick={() => setActiveTab("live-map")}
        >
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-sky-500 to-teal-600 p-0.5 flex items-center justify-center shadow-xs">
            <div className="w-full h-full bg-white rounded-[7px] flex items-center justify-center">
              <span className="text-sky-600 text-lg font-bold">🌀</span>
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight text-slate-900 flex items-center gap-1.5">
                CycloneSense <span className="text-sky-600 font-extrabold">AI</span>
              </h1>
              <span className="text-[10px] font-mono-code px-1.5 py-0.5 rounded bg-sky-50 text-sky-700 border border-sky-200 font-semibold">
                v2.0
              </span>
            </div>
            <p className="text-[11px] text-slate-500 hidden sm:block">
              India Satellite Cyclone Monitoring & High-Confidence AI Intelligence
            </p>
          </div>
        </div>

        {/* Desktop Tab Buttons (Clean & Light) */}
        <nav className="hidden lg:flex items-center gap-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                  isActive
                    ? "bg-sky-50 text-sky-700 border border-sky-300 font-bold shadow-2xs"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100 border border-transparent"
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? "text-sky-600" : "text-slate-400"}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Mobile Hamburger Button */}
        <div className="flex items-center gap-2 lg:hidden">
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 rounded-lg bg-slate-100 text-slate-700 border border-slate-200 hover:text-slate-900 cursor-pointer"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation Drawer */}
      {mobileMenuOpen && (
        <div className="lg:hidden bg-white border-b border-slate-200 px-4 py-3 space-y-1 shadow-md">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  setActiveTab(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center gap-3 p-2.5 rounded-xl text-left transition-all cursor-pointer ${
                  isActive
                    ? "bg-sky-50 text-sky-800 border border-sky-300 font-bold"
                    : "text-slate-700 hover:bg-slate-100 border border-transparent"
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                    isActive ? "bg-sky-600 text-white" : "bg-slate-100 text-slate-500"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-semibold text-slate-900">{item.label}</div>
                  <div className="text-[10px] text-slate-500 font-mono-code">{item.desc}</div>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </header>
  );
};
