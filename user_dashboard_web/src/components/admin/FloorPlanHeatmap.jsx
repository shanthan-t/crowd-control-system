import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, Upload, RotateCcw, Save, CheckCircle, AlertCircle, Pause, Play } from 'lucide-react';
import axios from 'axios';
import AppleSelect from '../AppleSelect';

const API = '';

const AREA_TYPES = [
    'Open Ground',
    'Closed Room',
    'Passage / Corridor',
    'Entry / Exit Gate',
    'Waiting Zone / Stands',
];

const POINT_LABELS = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left'];

const FloorPlanHeatmap = () => {
    // Camera calibration
    const [cameraPoints, setCameraPoints] = useState([]);
    const [streamActive, setStreamActive] = useState(false);
    const [frozen, setFrozen] = useState(false);
    const [frozenSrc, setFrozenSrc] = useState(null);
    const cameraCanvasRef = useRef(null);

    // Floor plan calibration
    const [floorPlanImage, setFloorPlanImage] = useState(null);
    const [floorPlanFile, setFloorPlanFile] = useState(null);
    const [floorPoints, setFloorPoints] = useState([]);
    const [dragActive, setDragActive] = useState(false);
    const floorCanvasRef = useRef(null);
    const fileInputRef = useRef(null);

    // Area type
    const [areaType, setAreaType] = useState('Open Ground');

    // Calibration save state
    const [saving, setSaving] = useState(false);
    const [saveStatus, setSaveStatus] = useState(null); // 'success' | 'error' | null
    const [saveMessage, setSaveMessage] = useState('');

    const canSave = cameraPoints.length === 4 && floorPoints.length === 4 && floorPlanFile;

    // ── Load existing calibration on mount ─────────────────────────────
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const resp = await axios.get(`${API}/api/calibration/current`);
                if (cancelled) return;
                if (resp.data.calibrated) {
                    // Restore blueprint image from backend
                    setBlueprintUrl(`${API}${resp.data.blueprint_url}`);
                    setFloorPlanImage(`${API}${resp.data.blueprint_url}`);
                    setAreaType(resp.data.area_type || 'Open Ground');
                    setSaveStatus('success');
                    setSaveMessage('Calibration loaded from previous session.');
                }
            } catch { /* backend not up or no calibration */ }
        })();
        return () => { cancelled = true; };
    }, []);

    // Blueprint URL from backend (persisted)
    const [blueprintUrl, setBlueprintUrl] = useState(null);

    // ── Camera stream controls ────────────────────────────────────────
    const startStream = () => {
        setStreamActive(true);
        setFrozen(false);
        setFrozenSrc(null);
        setCameraPoints([]);
    };

    const handleFreezeToggle = async () => {
        if (!frozen) {
            // Freeze: grab snapshot
            try {
                const resp = await axios.get(`${API}/api/monitor/snapshot`, {
                    responseType: 'blob',
                    timeout: 5000,
                });
                const url = URL.createObjectURL(resp.data);
                setFrozenSrc(url);
                setFrozen(true);
            } catch {
                setSaveStatus('error');
                setSaveMessage('Could not freeze frame.');
            }
        } else {
            // Resume live
            if (frozenSrc) URL.revokeObjectURL(frozenSrc);
            setFrozenSrc(null);
            setFrozen(false);
        }
    };

    // Cleanup frozen URL on unmount
    useEffect(() => {
        return () => {
            if (frozenSrc) URL.revokeObjectURL(frozenSrc);
        };
    }, [frozenSrc]);

    // ── Camera click handler ──────────────────────────────────────────
    const handleCameraClick = useCallback((e) => {
        if (cameraPoints.length >= 4) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        setCameraPoints(prev => [...prev, { x, y }]);
    }, [cameraPoints.length]);

    // ── Floor plan upload ─────────────────────────────────────────────
    const processFile = (file) => {
        if (!file || !file.type.startsWith('image/')) return;
        setFloorPlanFile(file);
        const reader = new FileReader();
        reader.onload = (e) => {
            setFloorPlanImage(e.target.result);
            setFloorPoints([]);
        };
        reader.readAsDataURL(file);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragActive(false);
        const file = e.dataTransfer?.files?.[0];
        processFile(file);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setDragActive(true);
    };

    const handleDragLeave = () => setDragActive(false);

    // ── Floor plan click handler ──────────────────────────────────────
    const handleFloorClick = useCallback((e) => {
        if (floorPoints.length >= 4) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        setFloorPoints(prev => [...prev, { x, y }]);
    }, [floorPoints.length]);

    // ── Point marker component ────────────────────────────────────────
    const PointMarker = ({ point, index, color }) => (
        <div
            className="fp-point-marker"
            style={{
                left: `${point.x}%`,
                top: `${point.y}%`,
                '--marker-color': color,
            }}
        >
            <span className="fp-point-label">{index + 1}</span>
        </div>
    );

    return (
        <div className="admin-content-area">

            {/* ── Page Header ───────────────────────────────────────── */}
            <div className="fp-page-header">
                <h2 className="fp-page-title">Floor Plan Crowd Heatmap</h2>
                <p className="fp-page-subtitle">
                    Calibrate camera to floor map, then visualize live crowd density over the floor plan.
                </p>
            </div>

            {/* ── Two-Column Calibration Grid ─────────────────────── */}
            <div className="fp-calibration-grid">

                {/* LEFT: Camera Calibration */}
                <div className="fp-card">
                    <div className="fp-card-header">
                        <h3 className="fp-card-title">
                            <span className="fp-card-step">1.</span>
                            Camera 4-Point Selection
                            <span className="fp-card-count">({cameraPoints.length}/4)</span>
                        </h3>
                    </div>

                    <p className="fp-card-instruction">
                        Start camera preview, freeze the frame, then click 4 matching floor corners:{' '}
                        <span className="text-white/70">{POINT_LABELS.join(', ')}.</span>
                    </p>

                    {!streamActive ? (
                        <button
                            className="fp-btn fp-btn-primary"
                            onClick={startStream}
                        >
                            <Camera size={15} />
                            Start Camera Preview
                        </button>
                    ) : (
                        <div className="fp-camera-controls">
                            <button
                                className={`fp-btn ${frozen ? 'fp-btn-primary' : 'fp-btn-outline'}`}
                                onClick={handleFreezeToggle}
                            >
                                {frozen ? <><Play size={14} /> Resume Live</> : <><Pause size={14} /> Freeze Frame</>}
                            </button>
                        </div>
                    )}

                    {/* Camera feed area */}
                    {streamActive ? (
                        <div
                            className="fp-image-area"
                            onClick={handleCameraClick}
                            ref={cameraCanvasRef}
                        >
                            <img
                                src={frozen && frozenSrc ? frozenSrc : `${API}/api/monitor/stream`}
                                alt="Camera feed"
                                className="fp-image"
                                draggable={false}
                            />
                            {cameraPoints.map((pt, i) => (
                                <PointMarker key={i} point={pt} index={i} color="#facc15" />
                            ))}
                            {cameraPoints.length < 4 && (
                                <div className="fp-image-hint">
                                    {frozen
                                        ? `Click to place point ${cameraPoints.length + 1} — ${POINT_LABELS[cameraPoints.length]}`
                                        : 'Freeze the frame first for precise point placement'
                                    }
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="fp-status-box">
                            Click the button above to start live camera preview.
                        </div>
                    )}

                    <button
                        className="fp-btn fp-btn-ghost"
                        onClick={() => { setCameraPoints([]); setStreamActive(false); setFrozen(false); if (frozenSrc) URL.revokeObjectURL(frozenSrc); setFrozenSrc(null); }}
                    >
                        <RotateCcw size={14} />
                        Reset Camera Points
                    </button>
                </div>

                {/* RIGHT: Floor Plan Calibration */}
                <div className="fp-card">
                    <div className="fp-card-header">
                        <h3 className="fp-card-title">
                            <span className="fp-card-step">2.</span>
                            Floor Plan 4-Point Selection
                            <span className="fp-card-count">({floorPoints.length}/4)</span>
                        </h3>
                    </div>

                    <p className="fp-card-instruction">
                        Upload your floor plan and click the same 4 corners:{' '}
                        <span className="text-white/70">{POINT_LABELS.join(', ')}.</span>
                    </p>

                    {/* Upload / Image area */}
                    {floorPlanImage ? (
                        <div
                            className="fp-image-area"
                            onClick={handleFloorClick}
                            ref={floorCanvasRef}
                        >
                            <img src={floorPlanImage} alt="Floor plan" className="fp-image" />
                            {floorPoints.map((pt, i) => (
                                <PointMarker key={i} point={pt} index={i} color="#60a5fa" />
                            ))}
                            {floorPoints.length < 4 && (
                                <div className="fp-image-hint">
                                    Click to place point {floorPoints.length + 1} — {POINT_LABELS[floorPoints.length]}
                                </div>
                            )}
                        </div>
                    ) : (
                        <div
                            className={`fp-drop-zone ${dragActive ? 'fp-drop-zone-active' : ''}`}
                            onDrop={handleDrop}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                        >
                            <Upload size={28} className="text-[var(--text-muted)] opacity-40" />
                            <p className="fp-drop-text">
                                Drag & drop floor plan image here
                            </p>
                            <button
                                className="fp-btn fp-btn-outline"
                                onClick={() => fileInputRef.current?.click()}
                            >
                                Browse Files
                            </button>
                            <p className="fp-drop-formats">PNG, JPG, SVG — max 10 MB</p>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*"
                                className="hidden"
                                onChange={(e) => processFile(e.target.files?.[0])}
                            />
                        </div>
                    )}

                    {!floorPlanImage && (
                        <div className="fp-status-box">
                            Upload a floor plan image to start calibration.
                        </div>
                    )}

                    <button
                        className="fp-btn fp-btn-ghost"
                        onClick={async () => {
                            // Reset backend calibration
                            try { await axios.delete(`${API}/api/calibration/reset`); } catch { }
                            setFloorPoints([]);
                            setFloorPlanImage(null);
                            setFloorPlanFile(null);
                            setBlueprintUrl(null);
                            setCameraFrame(null);
                            setCameraPoints([]);
                            setSaveStatus(null);
                            setSaveMessage('');
                        }}
                    >
                        <RotateCcw size={14} />
                        Reset Floor Plan Points
                    </button>
                </div>

            </div>

            {/* ── Area Type Selector ──────────────────────────────── */}
            <div className="fp-area-type-section">
                <label className="fp-area-label">Area Type</label>
                <AppleSelect
                    id="area-type"
                    options={AREA_TYPES}
                    value={areaType}
                    onChange={setAreaType}
                />
            </div>

            {/* ── Save Calibration ─────────────────────────────────── */}
            <div className="fp-save-section">
                <button
                    className="fp-btn fp-btn-primary"
                    disabled={!canSave || saving}
                    onClick={async () => {
                        if (!canSave) return;
                        setSaving(true);
                        setSaveStatus(null);
                        try {
                            // Points are stored as % (0-100) of the rendered container.
                            // Send as raw percentages [0-100] — backend converts to
                            // actual pixel coords using known frame/blueprint dimensions.
                            const camPts = cameraPoints.map(p => [p.x, p.y]);
                            const flrPts = floorPoints.map(p => [p.x, p.y]);

                            const formData = new FormData();
                            formData.append('blueprint', floorPlanFile);
                            formData.append('camera_pts', JSON.stringify(camPts));
                            formData.append('floor_pts', JSON.stringify(flrPts));
                            formData.append('area_type', areaType);

                            await axios.post(`${API}/api/calibration/save`, formData, {
                                headers: { 'Content-Type': 'multipart/form-data' },
                            });
                            setSaveStatus('success');
                            setSaveMessage('Calibration saved and activated.');
                        } catch (err) {
                            setSaveStatus('error');
                            setSaveMessage(err?.response?.data?.detail || 'Failed to save calibration.');
                        } finally {
                            setSaving(false);
                        }
                    }}
                >
                    <Save size={15} />
                    {saving ? 'Saving…' : 'Save Calibration'}
                </button>

                {saveStatus === 'success' && (
                    <div className="fp-save-feedback fp-save-feedback--success">
                        <CheckCircle size={14} />
                        {saveMessage}
                    </div>
                )}
                {saveStatus === 'error' && (
                    <div className="fp-save-feedback fp-save-feedback--error">
                        <AlertCircle size={14} />
                        {saveMessage}
                    </div>
                )}

                {!canSave && (
                    <p className="fp-save-hint">
                        Select 4 camera points, upload a floor plan, and select 4 floor points to enable.
                    </p>
                )}
            </div>

        </div>
    );
};

export default FloorPlanHeatmap;
