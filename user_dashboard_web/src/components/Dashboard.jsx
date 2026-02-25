import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Shield, User, LogOut, Activity, TrendingUp } from 'lucide-react';
import MetricsCard from './MetricsCard';
import HistoryChart from './HistoryChart';
import RadarSweep from './RadarSweep';
import HeatmapVisualizer from './HeatmapVisualizer';

const POLLING_INTERVAL = 2000;

const Dashboard = ({ username, onLogout }) => {
    const [latest, setLatest] = useState(null);
    const [history, setHistory] = useState([]);
    const [error, setError] = useState(null);
    const [profileOpen, setProfileOpen] = useState(false);
    const [heatmapExpanded, setHeatmapExpanded] = useState(false);

    // ── Escape key to close modal ────────────────────────────────────────────
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') setHeatmapExpanded(false);
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    // ── Data polling ─────────────────────────────────────────────────────────
    useEffect(() => {
        const fetchData = async () => {
            try {
                const [latestRes, historyRes] = await Promise.all([
                    axios.get('http://localhost:8000/api/data/latest'),
                    axios.get('http://localhost:8000/api/data/history'),
                ]);
                if (latestRes.data.available) {
                    setLatest(latestRes.data);
                    setError(null);
                } else {
                    setError('Telemetry bridge offline. Awaiting sensor sync...');
                }
                setHistory(historyRes.data);
            } catch {
                setError('Connection to command center lost.');
            }
        };
        fetchData();
        const interval = setInterval(fetchData, POLLING_INTERVAL);
        return () => clearInterval(interval);
    }, []);

    // ── Scroll-reveal via IntersectionObserver (fires once) ──────────────────
    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('in-view');
                        observer.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.12 }
        );
        const timer = setTimeout(() => {
            document.querySelectorAll('.scroll-reveal').forEach((el) => observer.observe(el));
        }, 80);
        return () => {
            clearTimeout(timer);
            observer.disconnect();
        };
    }, []);

    const formatTime = (ts) =>
        ts
            ? new Date(ts).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
            })
            : '--:--:--';

    const riskColor = {
        LOW: 'text-green-400',
        MEDIUM: 'text-yellow-400',
        HIGH: 'text-orange-400',
        CRITICAL: 'text-red-500',
    }[latest?.risk_level] || 'text-gray-400';

    return (
        <div className="db-root">
            {/* Heatmap Backdrop Overlay */}
            <div
                className={`heatmap-modal-backdrop ${heatmapExpanded ? 'modal-active' : ''}`}
                onClick={() => setHeatmapExpanded(false)}
            />

            {/* ── HEADER ──────────────────────────────────────────────────── */}
            <header className="db-header">
                {/* Left: Logo */}
                <div className="flex items-center gap-3">
                    <Shield size={24} className="text-white" strokeWidth={1.5} />
                    <h1 className="text-lg font-semibold tracking-widest text-white uppercase">
                        SENTINEL <span className="font-light text-[var(--text-muted)]">LIVE</span>
                    </h1>
                </div>

                {/* Right: Network Sync + Profile */}
                <div className="flex items-center gap-6">
                    {/* Network Synchronization */}
                    <div className="hidden md:flex flex-col items-end">
                        <p className="text-[0.58rem] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)] opacity-70 mb-0.5">
                            Network Synchronization
                        </p>
                        <div className="flex items-center gap-3">
                            <span className="font-mono text-sm font-light text-white tracking-widest">
                                {formatTime(latest?.timestamp)}
                            </span>
                            <div className="flex items-center gap-1.5">
                                <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_6px_rgba(34,197,94,0.8)]" />
                                <span className="text-[0.62rem] text-green-400 font-semibold tracking-wider uppercase">
                                    Connected
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* User Profile */}
                    <div className="relative" style={{ zIndex: 60 }}>
                        <button
                            onClick={() => setProfileOpen(!profileOpen)}
                            className="hover:bg-white/5 p-1.5 rounded-full transition-colors"
                        >
                            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-gray-700 to-gray-900 border border-gray-600 flex items-center justify-center">
                                <User size={17} className="text-gray-300" />
                            </div>
                        </button>
                        {profileOpen && (
                            <div className="profile-dropdown">
                                <div className="px-5 py-4">
                                    <p className="text-[10px] font-semibold text-[var(--accent-yellow)] uppercase tracking-[0.15em] mb-1.5">
                                        Active Operator
                                    </p>
                                    <p className="text-sm text-white font-medium truncate">{username}</p>
                                </div>
                                <div className="mx-3 h-px bg-[var(--border-color)]" />
                                <div className="py-1.5">
                                    <button
                                        onClick={onLogout}
                                        className="w-full text-left px-5 py-3 text-sm text-red-400 hover:bg-red-500/8 hover:text-red-300 transition-colors flex items-center gap-3 rounded-lg"
                                    >
                                        <LogOut size={14} /> Terminate Session
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </header>

            {/* ── MAIN BODY ────────────────────────────────────────────────── */}
            <main className="db-body">

                {/* Error bar */}
                {error && (
                    <div className="db-error-bar scroll-reveal">
                        <p className="text-red-300/90 text-sm font-medium flex items-center gap-3">
                            <Activity size={15} className="animate-pulse" /> {error}
                        </p>
                    </div>
                )}

                {/* ── LEFT COLUMN ─────────────────────────────────────────── */}
                <div className="flex flex-col gap-6 w-full">
                    {/* Card 1 — Command Overview */}
                    <div className="db-card scroll-reveal flex flex-col justify-between">
                        <div>
                            <p className="db-card-label">Command Overview</p>
                            <p className="db-card-sub mb-5">
                                High-level spatial intelligence and real-time risk aggregation.
                            </p>
                        </div>

                        <div>
                            {/* Risk badge */}
                            <div className="flex items-center gap-3 mb-5">
                                <RadarSweep />
                                <div>
                                    <p className="text-[0.62rem] uppercase tracking-widest text-[var(--text-muted)] mb-0.5">
                                        Real-Time Risk
                                    </p>
                                    <p className={`text-2xl font-bold tracking-tight leading-none ${riskColor}`}>
                                        {latest?.risk_level || 'UNKNOWN'}
                                    </p>
                                </div>
                            </div>

                            {/* People count */}
                            <div className="border-t border-[var(--border-color)] pt-4">
                                <p className="text-[0.62rem] uppercase tracking-widest text-[var(--text-muted)] mb-1">
                                    Detected Persons
                                </p>
                                <p className="text-3xl font-bold text-white">
                                    {latest?.people_count ?? '—'}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Card 3 — Crowd Density Trend */}
                    <div className="db-card scroll-reveal delay-200 flex flex-col h-[260px]">
                        <div className="flex items-center gap-2.5 mb-2">
                            <TrendingUp size={18} className="text-[var(--accent-yellow)]" strokeWidth={1.5} />
                            <p className="db-card-label mb-0">Crowd Density Trend</p>
                        </div>
                        <p className="db-card-sub mb-3">
                            Historical crowd fluctuation over the last 50 data points.
                        </p>
                        <div className="flex-1 min-h-[120px]">
                            <HistoryChart data={history} />
                        </div>
                    </div>
                </div>

                {/* ── RIGHT COLUMN ────────────────────────────────────────── */}
                <div className="flex flex-col gap-6 w-full h-full">
                    {/* Card 2 — Spatial Heatmap (DOMINANT) */}
                    <div className="db-card scroll-reveal delay-150 flex flex-col h-[520px]">
                        <div className="flex justify-between items-start mb-4">
                            <div>
                                <p className="db-card-label">Spatial Heatmap</p>
                                <p className="db-card-sub">
                                    Live density estimation across monitored zones.
                                </p>
                            </div>
                            {/* Legend */}
                            <div className="flex items-center gap-4 bg-[var(--background-darker)]/40 px-3 py-2 rounded-lg border border-[var(--border-color)]">
                                {[
                                    { color: 'bg-red-500', shadow: 'shadow-[0_0_6px_rgba(239,68,68,0.6)]', label: 'Critical' },
                                    { color: 'bg-yellow-400', shadow: 'shadow-[0_0_6px_rgba(250,204,21,0.6)]', label: 'Elevated' },
                                    { color: 'bg-blue-500', shadow: 'shadow-[0_0_6px_rgba(59,130,246,0.6)]', label: 'Nominal' },
                                ].map(({ color, shadow, label }) => (
                                    <div key={label} className="flex items-center gap-2">
                                        <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${color} ${shadow}`} />
                                        <span className="text-[0.65rem] text-[var(--text-secondary)] uppercase tracking-wider font-semibold">{label}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Heatmap Container */}
                        <div
                            className={`w-full flex-1 relative overflow-hidden rounded-lg border border-white/10 bg-black/40 p-6 shadow-[inset_0_0_40px_rgba(255,255,255,0.02)] hv-container-base ${heatmapExpanded ? 'hv-container-expanded' : ''}`}
                            onClick={() => { if (!heatmapExpanded) setHeatmapExpanded(true); }}
                        >
                            {heatmapExpanded && (
                                <button className="hv-close-btn" onClick={(e) => { e.stopPropagation(); setHeatmapExpanded(false); }}>
                                    ✕
                                </button>
                            )}
                            <HeatmapVisualizer />
                        </div>
                    </div>

                    {/* Card 4 — Stats row */}
                    <div className="db-stats-container scroll-reveal delay-300">
                        {[
                            {
                                label: 'Current Density',
                                value: latest?.people_count != null ? `${latest.people_count}` : '—',
                                unit: 'persons',
                                color: 'text-white',
                            },
                            {
                                label: 'Alert Level',
                                value: latest?.risk_level || '—',
                                unit: 'status',
                                color: riskColor,
                            },
                            {
                                label: 'Last Sync',
                                value: formatTime(latest?.timestamp),
                                unit: 'local time',
                                color: 'text-blue-300',
                            },
                        ].map(({ label, value, unit, color }) => (
                            <div key={label} className="db-stat-card flex flex-col justify-center h-full">
                                <p className="text-[0.6rem] uppercase tracking-widest text-[var(--text-muted)] mb-1.5">
                                    {label}
                                </p>
                                <p className={`text-2xl font-bold font-mono ${color} truncate`}>{value}</p>
                                <p className="text-[0.6rem] text-[var(--text-muted)] mt-1 opacity-60 uppercase tracking-widest">{unit}</p>
                            </div>
                        ))}
                    </div>
                </div>

            </main>
        </div>
    );
};

export default Dashboard;
