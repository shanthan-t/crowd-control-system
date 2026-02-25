import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { motion, AnimatePresence, LayoutGroup } from 'framer-motion';
import {
    Play, Square, Wifi, WifiOff, Monitor, Mic, Users,
    AlertTriangle, CheckCircle, Loader, Camera, Radio
} from 'lucide-react';

const API = 'http://localhost:8000';
const STREAM_URL = `${API}/api/monitor/stream`;

const SOURCE_OPTIONS = [
    { value: '0', label: 'Webcam' },
    { value: 'rtsp', label: 'IP Camera' },
];

const riskStyle = {
    LOW: { color: 'text-green-400', bg: 'bg-green-500/10', border: 'border-green-500/25' },
    MEDIUM: { color: 'text-yellow-400', bg: 'bg-yellow-500/10', border: 'border-yellow-500/25' },
    HIGH: { color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/25' },
};

const safetyStyle = {
    SAFE: 'text-green-400',
    CAUTION: 'text-yellow-400',
    DANGEROUS: 'text-red-400',
};

/* Pill shared-layout transition */
const pillTransition = {
    duration: 0.36,
    ease: [0.22, 1, 0.36, 1],
};

/* Text zoom keyframes — camera app style */
const textZoom = {
    scale: [1, 1.08, 1],
    transition: {
        duration: 0.22,
        ease: 'easeOut',
        times: [0, 0.45, 1],
    },
};

/* ── Segmented Control Component ──────────────────────────────── */
const SegmentedControl = ({ options, value, onChange, layoutId, disabled }) => (
    <div className="seg-ctrl">
        {options.map(opt => (
            <button
                key={opt.value}
                className="seg-ctrl-item"
                onClick={() => onChange(opt.value)}
                disabled={disabled}
                type="button"
            >
                {value === opt.value && (
                    <motion.div
                        className="seg-ctrl-pill"
                        layoutId={layoutId}
                        transition={pillTransition}
                    />
                )}
                {value === opt.value ? (
                    <motion.span
                        className="seg-ctrl-label seg-ctrl-label--active"
                        key={`active-${opt.value}`}
                        animate={textZoom}
                    >
                        {opt.label}
                    </motion.span>
                ) : (
                    <span className="seg-ctrl-label seg-ctrl-label--inactive">
                        {opt.label}
                    </span>
                )}
            </button>
        ))}
    </div>
);

const AdminLiveMonitor = () => {
    // Config state
    const [sourceType, setSourceType] = useState('0');
    const [customUrl, setCustomUrl] = useState('');

    // Runtime state
    const [isRunning, setIsRunning] = useState(false);
    const [loading, setLoading] = useState(false);
    const [streamReady, setStreamReady] = useState(false);
    const [metrics, setMetrics] = useState({
        people: 0, risk: 'LOW', safety: 'SAFE', audio: 'NORMAL', skeleton: false,
    });

    const pollRef = useRef(null);
    const imgRef = useRef(null);
    const [streamKey, setStreamKey] = useState(0);

    // ── Poll status ──────────────────────────────────────────────────────
    const pollStatus = useCallback(async () => {
        try {
            const { data } = await axios.get(`${API}/api/monitor/status`);
            setIsRunning(data.running);
            setMetrics({
                people: data.people ?? 0,
                risk: data.risk ?? 'LOW',
                safety: data.safety ?? 'SAFE',
                audio: data.audio ?? 'NORMAL',
                skeleton: data.skeleton ?? false,
            });
        } catch { /* backend not yet up */ }
    }, []);

    useEffect(() => {
        pollStatus();
    }, [pollStatus]);

    useEffect(() => {
        if (isRunning) {
            pollRef.current = setInterval(pollStatus, 1500);
        } else {
            clearInterval(pollRef.current);
        }
        return () => clearInterval(pollRef.current);
    }, [isRunning, pollStatus]);

    // ── Controls ─────────────────────────────────────────────────────────
    const handleStart = async () => {
        setLoading(true);
        const source = sourceType === 'rtsp' ? customUrl : sourceType;
        try {
            await axios.post(`${API}/api/monitor/start`, { source, model: 'yolov8n-pose.pt' });
            setStreamReady(false);
            setStreamKey(k => k + 1);
            setIsRunning(true);
        } catch (err) {
            console.error('[Monitor] Start error', err);
        } finally {
            setLoading(false);
        }
    };

    const handleStop = async () => {
        setLoading(true);
        try {
            await axios.post(`${API}/api/monitor/stop`);
            setIsRunning(false);
            setStreamReady(false);
        } catch (err) {
            console.error('[Monitor] Stop error', err);
        } finally {
            setLoading(false);
        }
    };

    // ── Derived UI values ─────────────────────────────────────────────────
    const rs = riskStyle[metrics.risk] || riskStyle.LOW;
    const riskBadge = `monitor-metric-value ${rs.color}`;

    return (
        <div className="admin-content-area">

            {/* Section header */}
            <div className="admin-section-header">
                <h2 className="admin-section-title">Live Monitor</h2>
            </div>

            <div className="monitor-layout">

                {/* ── LEFT: Video + Metrics ──────────────────────────────── */}
                <div className="monitor-main">

                    {/* Metrics bar */}
                    <div className="monitor-metrics-bar">
                        <div className="monitor-metric">
                            <div>
                                <p className="monitor-metric-label">People</p>
                                <p className="monitor-metric-value font-mono">{metrics.people}</p>
                            </div>
                        </div>
                        <div className="monitor-metric">
                            <div>
                                <p className="monitor-metric-label">Risk</p>
                                <span className={riskBadge}>{metrics.risk}</span>
                            </div>
                        </div>
                        <div className="monitor-metric">
                            <div>
                                <p className="monitor-metric-label">Safety</p>
                                <p className={`monitor-metric-value ${safetyStyle[metrics.safety] || 'text-white'}`}>
                                    {metrics.safety}
                                </p>
                            </div>
                        </div>
                        <div className="monitor-metric">
                            <div>
                                <p className="monitor-metric-label">Audio</p>
                                <p className={`monitor-metric-value ${metrics.audio === 'PANIC' ? 'text-red-400' : 'text-green-400'}`}>
                                    {metrics.audio}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Video container */}
                    <div className={`monitor-video-container ${isRunning ? 'monitor-video-active' : ''}`}>
                        {isRunning ? (
                            <>
                                {!streamReady && (
                                    <div className="monitor-video-overlay">
                                        <Loader size={28} className="animate-spin text-[var(--accent-yellow)]" />
                                        <p className="text-xs text-[var(--text-muted)] mt-2">Initializing feed...</p>
                                    </div>
                                )}
                                <img
                                    key={streamKey}
                                    ref={imgRef}
                                    src={STREAM_URL}
                                    alt="Live Detection Feed"
                                    className="monitor-video-img"
                                    onLoad={() => setStreamReady(true)}
                                    onError={() => setStreamReady(false)}
                                />
                            </>
                        ) : (
                            <div className="monitor-video-placeholder">
                                <Camera size={44} className="text-[var(--text-muted)] opacity-20 mb-4" />
                                <p className="text-sm text-[var(--text-muted)] opacity-35 font-normal">
                                    Start the system to begin live feed
                                </p>
                            </div>
                        )}
                    </div>

                </div>

                {/* ── RIGHT: Config Sidebar ──────────────────────────────── */}
                <LayoutGroup>
                    <div className="monitor-sidebar">

                        {/* Input Source */}
                        <div className="db-card">
                            <div className="sidebar-section-title">
                                <Camera size={14} />
                                Input Source
                            </div>
                            <SegmentedControl
                                options={SOURCE_OPTIONS}
                                value={sourceType}
                                onChange={setSourceType}
                                layoutId="source-pill"
                                disabled={isRunning}
                            />
                            <AnimatePresence>
                                {sourceType === 'rtsp' && (
                                    <motion.div
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: 'auto' }}
                                        exit={{ opacity: 0, height: 0 }}
                                        transition={{ duration: 0.22, ease: 'easeOut' }}
                                        style={{ overflow: 'hidden' }}
                                    >
                                        <input
                                            type="text"
                                            className="w-full bg-[rgba(255,255,255,0.03)] border border-[rgba(255,255,255,0.06)] rounded-xl px-3 py-2.5 text-[13px] text-white focus:outline-none focus:border-[rgba(10,132,255,0.4)] transition-colors mt-3 font-mono"
                                            placeholder="rtsp://... or http://IP:8080/video"
                                            value={customUrl}
                                            onChange={e => setCustomUrl(e.target.value)}
                                            disabled={isRunning}
                                        />
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>

                        {/* System Controls */}
                        <div className="db-card" style={{ gap: '16px', display: 'flex', flexDirection: 'column' }}>
                            <div className="sidebar-section-title" style={{ marginBottom: 0 }}>
                                System Controls
                            </div>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                <motion.button
                                    className="sys-btn-start"
                                    onClick={handleStart}
                                    disabled={isRunning || loading}
                                    whileTap={{ scale: 0.97 }}
                                    transition={{ duration: 0.12 }}
                                >
                                    {loading && !isRunning
                                        ? <Loader size={15} className="animate-spin" />
                                        : <Play size={15} />}
                                    START SYSTEM
                                </motion.button>

                                <motion.button
                                    className="sys-btn-stop"
                                    onClick={handleStop}
                                    disabled={!isRunning || loading}
                                    whileTap={{ scale: 0.97 }}
                                    transition={{ duration: 0.12 }}
                                >
                                    {loading && isRunning
                                        ? <Loader size={15} className="animate-spin" />
                                        : <Square size={15} />}
                                    STOP SYSTEM
                                </motion.button>
                            </div>

                            {/* Status chip */}
                            <div className={`sys-status-chip ${isRunning ? 'sys-status-chip--running' : 'sys-status-chip--stopped'}`}>
                                {isRunning
                                    ? <><Wifi size={13} /><span>System Running</span><span style={{ marginLeft: 'auto' }}>●</span></>
                                    : <><WifiOff size={13} /><span>System Stopped</span><span style={{ marginLeft: 'auto' }}>●</span></>}
                            </div>
                        </div>

                    </div>
                </LayoutGroup>
            </div>
        </div>
    );
};

export default AdminLiveMonitor;



