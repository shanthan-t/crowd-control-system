import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { motion, AnimatePresence, LayoutGroup } from 'framer-motion';
import {
    Play, Square, Wifi, WifiOff, Monitor, Users,
    AlertTriangle, Loader, Camera, Plus, X, Trash2,
    Radio, PlayCircle, StopCircle, Zap
} from 'lucide-react';

const API = '';

const riskColors = {
    LOW: '#22c55e',
    MEDIUM: '#eab308',
    HIGH: '#ef4444',
};

/* ── CameraCard — individual camera tile in the grid ─────────────── */
const CameraCard = ({ camera, onStart, onStop, onRemove }) => {
    const [streamReady, setStreamReady] = useState(false);
    const [confirmRemove, setConfirmRemove] = useState(false);
    const [actionLoading, setActionLoading] = useState(false);
    const streamUrl = `${API}/api/cameras/stream/${camera.camera_id}`;
    const [streamKey, setStreamKey] = useState(0);

    // Reset stream when camera starts
    useEffect(() => {
        if (camera.running) {
            setStreamReady(false);
            setStreamKey(k => k + 1);
        }
    }, [camera.running]);

    const handleAction = async (action) => {
        setActionLoading(true);
        try {
            await action();
        } finally {
            setActionLoading(false);
        }
    };

    const riskColor = riskColors[camera.risk] || riskColors.LOW;

    return (
        <motion.div
            className="cam-card"
            layout
            initial={{ opacity: 0, scale: 0.95, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -8 }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
        >
            {/* Card Header */}
            <div className="cam-card-header">
                <div className="cam-card-title-row">
                    <div className={`cam-status-dot ${camera.running ? 'cam-status-dot--on' : 'cam-status-dot--off'}`} />
                    <span className="cam-card-label">{camera.label || camera.source_url}</span>
                </div>
                <div className="cam-card-actions">
                    {camera.running ? (
                        <button
                            className="cam-action-btn cam-action-btn--stop"
                            onClick={() => handleAction(() => onStop(camera.camera_id))}
                            disabled={actionLoading}
                            title="Stop"
                        >
                            {actionLoading ? <Loader size={13} className="animate-spin" /> : <Square size={13} />}
                        </button>
                    ) : (
                        <button
                            className="cam-action-btn cam-action-btn--start"
                            onClick={() => handleAction(() => onStart(camera.camera_id))}
                            disabled={actionLoading}
                            title="Start"
                        >
                            {actionLoading ? <Loader size={13} className="animate-spin" /> : <Play size={13} />}
                        </button>
                    )}
                    {!confirmRemove ? (
                        <button
                            className="cam-action-btn cam-action-btn--remove"
                            onClick={() => setConfirmRemove(true)}
                            title="Remove"
                        >
                            <Trash2 size={13} />
                        </button>
                    ) : (
                        <button
                            className="cam-action-btn cam-action-btn--confirm-remove"
                            onClick={() => { setConfirmRemove(false); onRemove(camera.camera_id); }}
                            onBlur={() => setConfirmRemove(false)}
                            title="Click again to confirm"
                        >
                            <X size={13} />
                        </button>
                    )}
                </div>
            </div>

            {/* Video Preview */}
            <div className={`cam-card-preview ${camera.running ? 'cam-card-preview--active' : ''}`}>
                {camera.running ? (
                    <>
                        {!streamReady && (
                            <div className="cam-card-overlay">
                                <Loader size={22} className="animate-spin" style={{ color: 'rgba(255,255,255,0.4)' }} />
                                <span className="cam-card-overlay-text">Connecting...</span>
                            </div>
                        )}
                        <img
                            key={streamKey}
                            src={streamUrl}
                            alt="Camera feed"
                            className="cam-card-img"
                            onLoad={() => setStreamReady(true)}
                            onError={() => setStreamReady(false)}
                        />
                    </>
                ) : (
                    <div className="cam-card-placeholder">
                        <Camera size={28} style={{ color: 'rgba(255,255,255,0.12)' }} />
                        <span className="cam-card-placeholder-text">
                            {camera.error || 'Stopped'}
                        </span>
                    </div>
                )}
            </div>

            {/* Metrics Footer */}
            {camera.running && (
                <motion.div
                    className="cam-card-metrics"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                >
                    <div className="cam-metric">
                        <Users size={12} />
                        <span className="cam-metric-val">{camera.people ?? 0}</span>
                    </div>
                    <div className="cam-metric">
                        <div className="cam-risk-dot" style={{ background: riskColor }} />
                        <span className="cam-metric-val" style={{ color: riskColor }}>
                            {camera.risk || 'LOW'}
                        </span>
                    </div>
                    <div className="cam-metric">
                        <span className="cam-metric-label">{camera.safety || 'SAFE'}</span>
                    </div>
                </motion.div>
            )}

            {/* Source URL footer */}
            <div className="cam-card-source">
                {camera.source_url === '0' ? 'Webcam' : camera.source_url}
            </div>
        </motion.div>
    );
};


/* ── AdminLiveMonitor — main multi-camera monitor page ───────────── */
const AdminLiveMonitor = () => {
    const [cameras, setCameras] = useState([]);
    const [newUrl, setNewUrl] = useState('');
    const [newLabel, setNewLabel] = useState('');
    const [addLoading, setAddLoading] = useState(false);
    const [batchLoading, setBatchLoading] = useState(false);
    const pollRef = useRef(null);

    // ── Poll camera list ──────────────────────────────────────────────
    const pollCameras = useCallback(async () => {
        try {
            const { data } = await axios.get(`${API}/api/cameras/list`);
            setCameras(data.cameras || []);
        } catch { /* backend not up */ }
    }, []);

    useEffect(() => {
        pollCameras();
        pollRef.current = setInterval(pollCameras, 3000);
        return () => clearInterval(pollRef.current);
    }, [pollCameras]);

    // ── Camera actions ────────────────────────────────────────────────
    const addCamera = async (sourceUrl, label) => {
        setAddLoading(true);
        try {
            const { data } = await axios.post(`${API}/api/cameras/add`, {
                source_url: sourceUrl,
                label: label || '',
            });
            if (data.camera_id) {
                // Auto-start the new camera
                await axios.post(`${API}/api/cameras/start/${data.camera_id}`);
            }
            await pollCameras();
        } catch (err) {
            console.error('[Monitor] Add error', err);
        } finally {
            setAddLoading(false);
        }
    };

    const handleAddSubmit = async (e) => {
        e.preventDefault();
        if (!newUrl.trim()) return;
        await addCamera(newUrl.trim(), newLabel.trim());
        setNewUrl('');
        setNewLabel('');
    };

    const handleAddWebcam = async () => {
        await addCamera('0', 'Webcam');
    };

    const startCamera = async (cameraId) => {
        try {
            await axios.post(`${API}/api/cameras/start/${cameraId}`);
            await pollCameras();
        } catch (err) {
            console.error('[Monitor] Start error', err);
        }
    };

    const stopCamera = async (cameraId) => {
        try {
            await axios.post(`${API}/api/cameras/stop/${cameraId}`);
            await pollCameras();
        } catch (err) {
            console.error('[Monitor] Stop error', err);
        }
    };

    const removeCamera = async (cameraId) => {
        try {
            await axios.post(`${API}/api/cameras/remove/${cameraId}`);
            await pollCameras();
        } catch (err) {
            console.error('[Monitor] Remove error', err);
        }
    };

    const startAll = async () => {
        setBatchLoading(true);
        try {
            await axios.post(`${API}/api/cameras/start-all`);
            await pollCameras();
        } finally { setBatchLoading(false); }
    };

    const stopAll = async () => {
        setBatchLoading(true);
        try {
            await axios.post(`${API}/api/cameras/stop-all`);
            await pollCameras();
        } finally { setBatchLoading(false); }
    };

    // ── Derived state ─────────────────────────────────────────────────
    const runningCount = cameras.filter(c => c.running).length;
    const totalCount = cameras.length;

    // Determine grid class based on camera count
    const getGridClass = () => {
        if (totalCount === 0) return 'cam-grid cam-grid--empty';
        if (totalCount === 1) return 'cam-grid cam-grid--1';
        if (totalCount === 2) return 'cam-grid cam-grid--2';
        if (totalCount <= 4) return 'cam-grid cam-grid--4';
        return 'cam-grid cam-grid--many';
    };

    return (
        <div className="admin-content-area">

            {/* Section header */}
            <div className="admin-section-header">
                <h2 className="admin-section-title">Live Monitor</h2>
                <p className="admin-section-sub">
                    Add and manage multiple camera streams simultaneously.
                </p>
            </div>

            {/* ── Add Camera Bar ─────────────────────────────────────── */}
            <div className="cam-add-bar">
                <form className="cam-add-form" onSubmit={handleAddSubmit}>
                    <div className="cam-add-inputs">
                        <input
                            type="text"
                            className="cam-add-url"
                            placeholder="rtsp://... or http://IP:8080/video"
                            value={newUrl}
                            onChange={e => setNewUrl(e.target.value)}
                            disabled={addLoading}
                        />
                        <input
                            type="text"
                            className="cam-add-label"
                            placeholder="Label (optional)"
                            value={newLabel}
                            onChange={e => setNewLabel(e.target.value)}
                            disabled={addLoading}
                        />
                    </div>
                    <div className="cam-add-buttons">
                        <motion.button
                            type="submit"
                            className="cam-add-btn"
                            disabled={!newUrl.trim() || addLoading}
                            whileTap={{ scale: 0.96 }}
                        >
                            {addLoading
                                ? <Loader size={14} className="animate-spin" />
                                : <Plus size={14} />}
                            Add IP Camera
                        </motion.button>
                        <motion.button
                            type="button"
                            className="cam-add-webcam-btn"
                            onClick={handleAddWebcam}
                            disabled={addLoading}
                            whileTap={{ scale: 0.96 }}
                        >
                            <Camera size={14} />
                            Webcam
                        </motion.button>
                    </div>
                </form>
            </div>

            {/* ── System Controls ────────────────────────────────────── */}
            {totalCount > 0 && (
                <div className="cam-system-bar">
                    <div className="cam-system-status">
                        {runningCount > 0 ? (
                            <>
                                <Wifi size={14} style={{ color: '#22c55e' }} />
                                <span>{runningCount} of {totalCount} running</span>
                            </>
                        ) : (
                            <>
                                <WifiOff size={14} style={{ color: 'var(--text-muted)', opacity: 0.5 }} />
                                <span>{totalCount} camera{totalCount !== 1 ? 's' : ''} added</span>
                            </>
                        )}
                    </div>
                    <div className="cam-system-actions">
                        <motion.button
                            className="cam-batch-btn cam-batch-btn--start"
                            onClick={startAll}
                            disabled={batchLoading || runningCount === totalCount}
                            whileTap={{ scale: 0.96 }}
                        >
                            <PlayCircle size={14} />
                            Start All
                        </motion.button>
                        <motion.button
                            className="cam-batch-btn cam-batch-btn--stop"
                            onClick={stopAll}
                            disabled={batchLoading || runningCount === 0}
                            whileTap={{ scale: 0.96 }}
                        >
                            <StopCircle size={14} />
                            Stop All
                        </motion.button>
                    </div>
                </div>
            )}

            {/* ── Camera Grid ────────────────────────────────────────── */}
            <LayoutGroup>
                <div className={getGridClass()}>
                    <AnimatePresence mode="popLayout">
                        {cameras.map(cam => (
                            <CameraCard
                                key={cam.camera_id}
                                camera={cam}
                                onStart={startCamera}
                                onStop={stopCamera}
                                onRemove={removeCamera}
                            />
                        ))}
                    </AnimatePresence>

                    {totalCount === 0 && (
                        <motion.div
                            className="cam-empty-state"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.2 }}
                        >
                            <Monitor size={44} style={{ color: 'rgba(255,255,255,0.08)' }} />
                            <p className="cam-empty-title">No cameras added</p>
                            <p className="cam-empty-sub">
                                Use the bar above to add an IP camera or your webcam.
                            </p>
                        </motion.div>
                    )}
                </div>
            </LayoutGroup>

        </div>
    );
};

export default AdminLiveMonitor;
