import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Zap, AlertTriangle, AlertCircle, Info,
    TrendingUp, TrendingDown, Minus, Users,
    Camera, Activity, Shield, Loader, Radio,
    Bell, CheckCircle, X, MapPin, UserPlus
} from 'lucide-react';

const API = '';

const SEVERITY_CONFIG = {
    CRITICAL: { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.08)', border: 'rgba(239, 68, 68, 0.18)', icon: AlertTriangle },
    WARNING: { color: '#eab308', bg: 'rgba(234, 179, 8, 0.06)', border: 'rgba(234, 179, 8, 0.15)', icon: AlertCircle },
    INFO: { color: '#3b82f6', bg: 'rgba(59, 130, 246, 0.06)', border: 'rgba(59, 130, 246, 0.12)', icon: Info },
};

const RISK_COLORS = {
    LOW: '#22c55e',
    MEDIUM: '#eab308',
    HIGH: '#f97316',
    CRITICAL: '#ef4444',
};

const TrendIcon = ({ trend }) => {
    if (trend === 'RISING') return <TrendingUp size={14} style={{ color: '#ef4444' }} />;
    if (trend === 'FALLING') return <TrendingDown size={14} style={{ color: '#22c55e' }} />;
    return <Minus size={14} style={{ color: 'rgba(255,255,255,0.3)' }} />;
};

/* ── Parse staff count from recommendation text ──────────────────── */
const parseStaffCount = (staffAction) => {
    if (!staffAction) return null;
    const match = staffAction.match(/\+(\d+)/);
    return match ? parseInt(match[1], 10) : null;
};

/* ── Recommendation Card (with Deploy button) ────────────────────── */
const RecCard = ({ rec, onDeploy }) => {
    const config = SEVERITY_CONFIG[rec.severity] || SEVERITY_CONFIG.INFO;
    const Icon = config.icon;
    const deployCount = parseStaffCount(rec.staff_action);

    return (
        <motion.div
            className="ai-rec-card"
            style={{
                background: config.bg,
                borderColor: config.border,
            }}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 8 }}
            transition={{ duration: 0.22 }}
        >
            <div className="ai-rec-header">
                <div className="ai-rec-badge" style={{ background: config.border, color: config.color }}>
                    <Icon size={11} />
                    {rec.severity}
                </div>
                <div className="ai-rec-confidence">
                    <div className="ai-rec-conf-bar">
                        <motion.div
                            className="ai-rec-conf-fill"
                            style={{ background: config.color }}
                            initial={{ width: 0 }}
                            animate={{ width: `${rec.confidence * 100}%` }}
                            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                        />
                    </div>
                    <span className="ai-rec-conf-text">{Math.round(rec.confidence * 100)}%</span>
                </div>
            </div>
            <p className="ai-rec-title">{rec.title}</p>
            <p className="ai-rec-detail">{rec.detail}</p>

            {/* Deploy button — only when staff deployment is recommended */}
            {deployCount && onDeploy && (
                <button
                    className="deploy-btn"
                    onClick={() => onDeploy(deployCount, rec.camera_id)}
                >
                    <UserPlus size={14} />
                    Deploy Staff (+{deployCount})
                </button>
            )}
        </motion.div>
    );
};


/* ── AdminTacticalPanel ──────────────────────────────────────────── */
const AdminTacticalPanel = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [cameras, setCameras] = useState([]);
    const [streamReady, setStreamReady] = useState(false);
    const [streamKey, setStreamKey] = useState(0);
    const pollRef = useRef(null);

    // ── Dispatch state (simple, deterministic) ────────────────────────
    const [dispatchEvent, setDispatchEvent] = useState(null);
    const [showDispatchModal, setShowDispatchModal] = useState(false);
    const [pendingDeploy, setPendingDeploy] = useState(null); // { count, cameraId }

    // ── Poll AI recommendations ───────────────────────────────────────
    useEffect(() => {
        const fetchRecs = async () => {
            try {
                const { data: result } = await axios.get(`${API}/api/ai/recommendations`);
                setData(result);
                setLoading(false);
            } catch {
                setLoading(false);
            }
        };
        fetchRecs();
        pollRef.current = setInterval(fetchRecs, 3000);
        return () => clearInterval(pollRef.current);
    }, []);

    // ── Poll camera list for stream URLs ──────────────────────────────
    useEffect(() => {
        let cancelled = false;
        const fetchCameras = async () => {
            try {
                const { data: result } = await axios.get(`${API}/api/cameras/list`);
                if (!cancelled) setCameras(result.cameras || []);
            } catch { /* backend down */ }
        };
        fetchCameras();
        const interval = setInterval(fetchCameras, 3000);
        return () => { cancelled = true; clearInterval(interval); };
    }, []);

    // Get running cameras for stream
    const runningCameras = cameras.filter(c => c.running);
    const primaryCam = runningCameras[0] || null;
    const streamUrl = primaryCam
        ? `${API}/api/cameras/stream/${primaryCam.camera_id}`
        : null;

    // Reset stream when primary camera changes
    const prevCamId = useRef(null);
    useEffect(() => {
        if (primaryCam?.camera_id !== prevCamId.current) {
            prevCamId.current = primaryCam?.camera_id || null;
            setStreamReady(false);
            setStreamKey(k => k + 1);
        }
    }, [primaryCam?.camera_id]);

    // Use zone_assessments from AI engine (the correct key)
    const zones = data?.zone_assessments || [];

    const riskColor = RISK_COLORS[data?.risk_level] || RISK_COLORS.LOW;
    const riskScore = data?.risk_score ?? 0;
    const riskPct = Math.round(riskScore * 100);

    // ── Dispatch handlers (deterministic, user-initiated) ─────────────
    const handleDeployClick = (count, cameraId) => {
        setPendingDeploy({ count, cameraId });
        setShowDispatchModal(true);
    };

    const handleConfirmDispatch = async () => {
        if (!pendingDeploy) return;

        const roomName = 'Room 1';
        const newDispatch = {
            id: crypto.randomUUID(),
            room: roomName,
            requiredStaff: pendingDeploy.count,
            csi: data?.csi?.crowd_safety_index ?? 0,
            count: data?.total_people ?? 0,
            createdAt: Date.now(),
            status: 'ACTIVE',
        };

        setDispatchEvent(newDispatch);
        setShowDispatchModal(false);
        setPendingDeploy(null);

        // Broadcast to backend → staff SSE stream picks it up
        try {
            await axios.post(`${API}/api/dispatch/confirm`, {
                dispatch_id: newDispatch.id,
            });
        } catch {
            // Backend dispatch may not match this ID — create fresh
            // The SSE stream will broadcast the active dispatch
        }
    };

    const handleCancelModal = () => {
        setShowDispatchModal(false);
        setPendingDeploy(null);
    };

    const resolveDispatch = () => {
        setDispatchEvent(null);
    };

    // ── Poll dispatch status from backend to sync acknowledgements ────
    useEffect(() => {
        if (!dispatchEvent || dispatchEvent.status === 'RESOLVED') return;
        const interval = setInterval(async () => {
            try {
                const { data: result } = await axios.get(`${API}/api/dispatch/status`);
                if (result.dispatch && result.dispatch.status === 'assigned') {
                    setDispatchEvent(prev => prev ? {
                        ...prev,
                        status: 'ACKNOWLEDGED',
                        assignedTo: result.dispatch.assigned_to?.name || 'Staff',
                    } : null);
                }
            } catch { /* ignore */ }
        }, 3000);
        return () => clearInterval(interval);
    }, [dispatchEvent]);

    return (
        <div className="admin-content-area">

            {/* Section header */}
            <div className="admin-section-header">
                <h2 className="admin-section-title">AI Tactical Intelligence</h2>
                <p className="admin-section-sub">
                    Automated operational recommendations powered by real-time analysis
                </p>
            </div>

            {/* ── Live stream + Risk gauge row ─────────────────────── */}
            <div className="ai-tactical-grid">

                {/* Left Column: Live Stream */}
                <div className="ai-stream-panel">
                    <div className="ai-stream-container">
                        <div className="ai-stream-header">
                            <Radio size={13} style={{ color: streamUrl ? '#22c55e' : 'rgba(255,255,255,0.2)' }} />
                            <span>Live Feed</span>
                            {primaryCam && (
                                <span className="ai-stream-cam-label">{primaryCam.label}</span>
                            )}
                        </div>
                        <div className="ai-stream-viewport">
                            {streamUrl ? (
                                <>
                                    {!streamReady && (
                                        <div className="ai-stream-loader">
                                            <Loader size={22} className="animate-spin" style={{ color: 'rgba(255,255,255,0.3)' }} />
                                            <span>Connecting to stream…</span>
                                        </div>
                                    )}
                                    <img
                                        key={streamKey}
                                        src={streamUrl}
                                        alt="Live Camera"
                                        className="ai-stream-img"
                                        onLoad={() => setStreamReady(true)}
                                        onError={() => setStreamReady(false)}
                                    />
                                </>
                            ) : (
                                <div className="ai-stream-offline">
                                    <Camera size={28} style={{ color: 'rgba(255,255,255,0.08)' }} />
                                    <span>
                                        {cameras.length === 0
                                            ? 'No cameras added'
                                            : 'All cameras stopped'}
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Right Column: Risk Score on Top, Widgets Below */}
                <div className="ai-stats-panel">

                    {/* Risk Score Gauge */}
                    <div className="ai-risk-gauge">
                        <div className="ai-gauge-header">
                            <Shield size={16} style={{ color: riskColor }} />
                            <span className="ai-gauge-label">Composite Risk Score</span>
                        </div>

                        <div className="ai-gauge-score" style={{ color: riskColor }}>
                            {loading ? '—' : riskPct}
                            <span className="ai-gauge-pct">%</span>
                        </div>

                        <div className="ai-gauge-bar-track">
                            <motion.div
                                className="ai-gauge-bar-fill"
                                style={{ background: `linear-gradient(90deg, #22c55e, ${riskColor})` }}
                                animate={{ width: loading ? '0%' : `${riskPct}%` }}
                                transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                            />
                        </div>

                        <div className="ai-gauge-footer">
                            <div className="ai-gauge-level" style={{ color: riskColor }}>
                                {data?.risk_level || '—'}
                            </div>
                            <div className="ai-gauge-trend">
                                <TrendIcon trend={data?.trend} />
                                <span className="ai-gauge-trend-text">
                                    {data?.trend || 'STABLE'}
                                    {data?.trend_slope != null && data.trend_slope !== 0 && (
                                        <span className="ai-gauge-slope">
                                            {data.trend_slope > 0 ? '+' : ''}{data.trend_slope}/s
                                        </span>
                                    )}
                                </span>
                            </div>
                        </div>

                        <div className="ai-gauge-total">
                            <Users size={13} />
                            <span>{data?.total_people ?? 0} persons detected</span>
                        </div>
                    </div>

                    {/* Lower Right Grid: Camera Overview & Recommendations */}
                    <div className="ai-secondary-widgets">

                        {/* Camera Zone Summary */}
                        <div className="ai-cam-summary">
                            <div className="ai-cam-summary-header">
                                <Camera size={14} />
                                <span>Camera Overview</span>
                                <span className="ai-recs-count">{cameras.length}</span>
                            </div>
                            {cameras.length === 0 ? (
                                <div className="ai-cam-empty">No cameras added</div>
                            ) : (
                                <div className="ai-cam-list">
                                    {cameras.map(cam => {
                                        const zone = zones.find(z => z.zone_id === cam.camera_id);
                                        const densityPct = zone?.density
                                            ? Math.min((zone.density / 10) * 100, 100)
                                            : 0;
                                        return (
                                            <div key={cam.camera_id} className="ai-cam-row">
                                                <div className="ai-cam-row-left">
                                                    <div className={`ai-cam-dot ${cam.running ? 'ai-cam-dot--on' : 'ai-cam-dot--off'}`} />
                                                    <span className="ai-cam-name">{cam.label || cam.source_url}</span>
                                                </div>
                                                <div className="ai-cam-row-right">
                                                    <span className="ai-cam-count">{cam.people || 0}p</span>
                                                    <div className="ai-cam-density-bar">
                                                        <motion.div
                                                            className="ai-cam-density-fill"
                                                            style={{
                                                                background: densityPct >= 80 ? '#ef4444'
                                                                    : densityPct >= 50 ? '#eab308'
                                                                        : '#22c55e'
                                                            }}
                                                            animate={{ width: `${densityPct}%` }}
                                                            transition={{ duration: 0.5 }}
                                                        />
                                                    </div>
                                                    <span className="ai-cam-pct">{Math.round(densityPct)}%</span>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>

                        {/* Recommendations — Deploy button embedded in cards */}
                        <div className="ai-recs-section ai-recs-section--inline">
                            <div className="ai-recs-header">
                                <Activity size={15} />
                                <span>Active Recommendations</span>
                                <span className="ai-recs-count">{data?.recommendations?.length || 0}</span>
                            </div>
                            <div className="ai-recs-list">
                                <AnimatePresence mode="popLayout">
                                    {(data?.recommendations || []).map(rec => (
                                        <RecCard
                                            key={rec.id}
                                            rec={rec}
                                            onDeploy={dispatchEvent ? null : handleDeployClick}
                                        />
                                    ))}
                                </AnimatePresence>
                            </div>
                        </div>

                    </div>
                </div>

            </div>

            {/* ── Active Deployment Panel (persistent until resolved) ─── */}
            <AnimatePresence>
                {dispatchEvent && (
                    <motion.div
                        className="dispatch-active-card"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{ duration: 0.3 }}
                    >
                        <div className="dispatch-active-header">
                            <div className="dispatch-active-icon">
                                <AlertTriangle size={18} />
                            </div>
                            <div className="dispatch-active-info">
                                <h4 className="dispatch-active-title">STAFF DISPATCHED</h4>
                                <div className="dispatch-active-meta">
                                    <span>Room: <strong>{dispatchEvent.room}</strong></span>
                                    <span>Required: <strong>+{dispatchEvent.requiredStaff}</strong></span>
                                    <span>CSI: <strong>{dispatchEvent.csi}</strong></span>
                                </div>
                            </div>
                            <div className={`dispatch-active-status dispatch-active-status--${dispatchEvent.status.toLowerCase()}`}>
                                {dispatchEvent.status === 'ACTIVE' && (
                                    <><Bell size={12} className="animate-pulse" /> ACTIVE</>
                                )}
                                {dispatchEvent.status === 'ACKNOWLEDGED' && (
                                    <><CheckCircle size={12} /> ACKNOWLEDGED{dispatchEvent.assignedTo ? ` — ${dispatchEvent.assignedTo}` : ''}</>
                                )}
                            </div>
                        </div>
                        <div className="dispatch-active-actions">
                            <span className="dispatch-active-time">
                                Dispatched {new Date(dispatchEvent.createdAt).toLocaleTimeString()}
                            </span>
                            <button className="dispatch-resolve-btn" onClick={resolveDispatch}>
                                <CheckCircle size={13} />
                                Resolve
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── Dispatch Confirmation Modal ───────────────────────── */}
            <AnimatePresence>
                {showDispatchModal && pendingDeploy && (
                    <motion.div
                        className="ai-dispatch-overlay"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <motion.div
                            className="ai-dispatch-modal"
                            initial={{ scale: 0.92, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.92, opacity: 0 }}
                            transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
                        >
                            <div className="ai-dispatch-modal-header">
                                <AlertTriangle size={22} style={{ color: '#ef4444' }} />
                                <div>
                                    <h3 className="ai-dispatch-modal-title">Confirm Staff Deployment</h3>
                                    <p className="ai-dispatch-modal-sub">
                                        CSI: <strong>{data?.csi?.crowd_safety_index ?? '—'}</strong> · Room: <strong>Room 1</strong> · Count: <strong>{data?.total_people ?? 0}</strong>
                                    </p>
                                </div>
                                <button className="ai-dispatch-close" onClick={handleCancelModal}>
                                    <X size={18} />
                                </button>
                            </div>

                            <div className="ai-dispatch-rec">
                                Deploy <strong>+{pendingDeploy.count}</strong> additional staff to active zone
                            </div>

                            <div className="ai-dispatch-actions">
                                <button
                                    className="ai-dispatch-btn ai-dispatch-btn--confirm"
                                    onClick={handleConfirmDispatch}
                                >
                                    Dispatch Now
                                </button>
                                <button
                                    className="ai-dispatch-btn ai-dispatch-btn--cancel"
                                    onClick={handleCancelModal}
                                >
                                    Cancel
                                </button>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>

        </div>
    );
};

export default AdminTacticalPanel;
