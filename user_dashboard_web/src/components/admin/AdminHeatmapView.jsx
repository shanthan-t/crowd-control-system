/**
 * AdminHeatmapView.jsx — Spatial Intelligence Page
 * =================================================
 *
 * Enterprise-grade density analytics dashboard.
 *
 * Layout:
 *   Left (70%) — Blueprint + density overlay (HeatmapCanvas)
 *   Right (30%) — Analytics panel:
 *     - Active detections count
 *     - Peak density zone location
 *     - Average occupancy score
 *     - Temporal stability indicator
 *     - Camera selector (multi-cam ready)
 *     - Density legend
 */

import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
    Activity, MapPin, Gauge, BarChart3,
    Radio, Camera, TrendingUp
} from 'lucide-react';
import HeatmapCanvas from '../shared/HeatmapCanvas';


const AdminHeatmapView = () => {
    // Calibration metadata
    const [calibrated, setCalibrated] = useState(false);
    const [calibMeta, setCalibMeta] = useState({});

    // Camera list
    const [cameras, setCameras] = useState([]);
    const [selectedCamera, setSelectedCamera] = useState('');

    // Live density analytics
    const [liveData, setLiveData] = useState({});

    // ── Fetch calibration status ─────────────────────────────────
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const resp = await axios.get('/api/calibration/current');
                if (!cancelled && resp.data.calibrated) {
                    setCalibrated(true);
                    setCalibMeta(resp.data);
                }
            } catch { /* backend down */ }
        })();
        return () => { cancelled = true; };
    }, []);

    // ── Poll camera list ─────────────────────────────────────────
    useEffect(() => {
        let cancelled = false;
        const poll = async () => {
            try {
                const { data } = await axios.get('/api/cameras/list');
                if (!cancelled) setCameras(data.cameras || []);
            } catch { /* backend down */ }
        };
        poll();
        const iv = setInterval(poll, 5000);
        return () => { cancelled = true; clearInterval(iv); };
    }, []);

    // ── Build SSE URL based on camera selection ──────────────────
    const sseUrl = selectedCamera
        ? `/api/cameras/heatmap/${selectedCamera}`
        : '/api/heatmap/stream';

    // ── Receive live data from HeatmapCanvas ─────────────────────
    const handleData = useCallback((data) => {
        setLiveData(data);
    }, []);

    // ── Derived analytics ────────────────────────────────────────
    const personCount = liveData.person_count || 0;
    const peakValue = liveData.peak_value || 0;
    const peakLoc = liveData.peak_location || [0.5, 0.5];
    const avgDensity = liveData.avg_density || 0;
    const stability = liveData.stability ?? 1.0;
    const bpWidth = calibMeta.bp_width || liveData.width || 0;
    const bpHeight = calibMeta.bp_height || liveData.height || 0;

    // Peak zone as percentage
    const peakX = Math.round(peakLoc[0] * 100);
    const peakY = Math.round(peakLoc[1] * 100);

    // Occupancy score (0-100)
    const occupancyScore = Math.round(avgDensity * 100);

    // Stability label
    const stabilityLabel = stability > 0.8 ? 'Stable' : stability > 0.5 ? 'Moderate' : 'Volatile';
    const stabilityColor = stability > 0.8 ? '#22c55e' : stability > 0.5 ? '#eab308' : '#ef4444';

    // ── Not calibrated → show empty state ────────────────────────
    if (!calibrated) {
        return (
            <div className="si-empty-state">
                <MapPin size={48} style={{ opacity: 0.18 }} />
                <p className="si-empty-title">Calibration Required</p>
                <p className="si-empty-sub">
                    Navigate to Floor Plan to set up the camera-to-floor
                    homography before the density engine can activate.
                </p>
            </div>
        );
    }

    return (
        <div className="si-layout">
            {/* ── Left: Density Canvas ───────────────────────────── */}
            <div className="si-canvas-container">
                <div className="si-heatmap-fill">
                    <HeatmapCanvas
                        sseUrl={sseUrl}
                        showStatus={true}
                        onData={handleData}
                    />
                </div>
            </div>

            {/* ── Right: Analytics Panel ─────────────────────────── */}
            <div className="si-control-panel">

                {/* Active Detections */}
                <div className="si-panel-section">
                    <div className="si-panel-label">
                        <Activity size={12} />
                        Active Detections
                    </div>
                    <div className="si-panel-value si-panel-value--large">
                        {personCount}
                        <span className="si-panel-unit">people</span>
                    </div>
                </div>

                <div className="si-panel-divider" />

                {/* Peak Density Zone */}
                <div className="si-panel-section">
                    <div className="si-panel-label">
                        <MapPin size={12} />
                        Peak Density Zone
                    </div>
                    <div className="si-panel-value">
                        <span style={{ fontFamily: "'SF Mono', 'Fira Code', monospace", fontSize: '13px' }}>
                            {peakX}%, {peakY}%
                        </span>
                    </div>
                    {peakValue > 0 && (
                        <div style={{
                            marginTop: '6px',
                            width: '100%',
                            height: '4px',
                            borderRadius: '2px',
                            background: 'rgba(255,255,255,0.04)',
                            overflow: 'hidden',
                        }}>
                            <div style={{
                                width: `${Math.min(100, peakValue * 20)}%`,
                                height: '100%',
                                borderRadius: '2px',
                                background: 'linear-gradient(90deg, #22c55e, #eab308, #ef4444)',
                                transition: 'width 300ms ease',
                            }} />
                        </div>
                    )}
                </div>

                <div className="si-panel-divider" />

                {/* Occupancy Score */}
                <div className="si-panel-section">
                    <div className="si-panel-label">
                        <Gauge size={12} />
                        Occupancy Score
                    </div>
                    <div className="si-panel-value si-panel-value--large">
                        {occupancyScore}
                        <span className="si-panel-unit">%</span>
                    </div>
                </div>

                <div className="si-panel-divider" />

                {/* Temporal Stability */}
                <div className="si-panel-section">
                    <div className="si-panel-label">
                        <TrendingUp size={12} />
                        Temporal Stability
                    </div>
                    <div className="si-panel-value" style={{ gap: '8px' }}>
                        <div className="si-alert-dot" style={{
                            background: stabilityColor,
                            boxShadow: `0 0 6px ${stabilityColor}60`,
                        }} />
                        <span style={{ fontSize: '13px', fontWeight: 600 }}>
                            {stabilityLabel}
                        </span>
                        <span style={{
                            fontSize: '11px',
                            fontFamily: "'SF Mono', 'Fira Code', monospace",
                            opacity: 0.5,
                            marginLeft: 'auto',
                        }}>
                            {Math.round(stability * 100)}%
                        </span>
                    </div>
                </div>

                <div className="si-panel-divider" />

                {/* Blueprint Info */}
                <div className="si-panel-section">
                    <div className="si-panel-label">
                        <BarChart3 size={12} />
                        Blueprint
                    </div>
                    <div className="si-panel-value">
                        {bpWidth} × {bpHeight}
                        <span className="si-panel-unit">px</span>
                    </div>
                </div>

                <div className="si-panel-divider" />

                {/* Camera Selector */}
                <div className="si-panel-section">
                    <div className="si-panel-label">
                        <Camera size={12} />
                        Camera Source
                    </div>
                    {cameras.length > 0 ? (
                        <select
                            className="si-camera-select"
                            value={selectedCamera}
                            onChange={(e) => setSelectedCamera(e.target.value)}
                        >
                            <option value="">Default (auto)</option>
                            {cameras.map((c) => (
                                <option key={c.camera_id} value={c.camera_id}>
                                    {c.label || c.camera_id.slice(0, 8)}
                                    {c.running ? ' ●' : ''}
                                </option>
                            ))}
                        </select>
                    ) : (
                        <div className="si-panel-value" style={{ opacity: 0.4, fontSize: '12px' }}>
                            <Radio size={14} />
                            No cameras registered
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default AdminHeatmapView;
