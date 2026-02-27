/**
 * SpatialIntelligence.jsx — Blueprint-Based Spatial Intelligence
 *
 * Data:
 *   SSE /api/spatial/stream or /api/cameras/spatial/{id}
 *   Payload: { grid, points, width, height, ready }
 *
 * Architecture:
 *   3-Layer strict canvas architecture:
 *   1) Static Blueprint Layer (Rendered once, intrinsic resolution, no blur)
 *   2) Heatmap Layer (Redrawn per frame, Gaussian blur applied, alpha blending)
 *   3) Projected Points Layer (Redrawn per frame, dot markers)
 */

import React, { useEffect, useRef, useCallback, useState } from 'react';
import {
    Activity, Radar, Camera, SlidersHorizontal,
    Layers, Crosshair, MapPin, Upload, Maximize
} from 'lucide-react';

const TURBO_LUT = buildTurboLUT();

function buildTurboLUT() {
    const lut = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
        const t = i / 255;
        const r = Math.max(0, Math.min(255, Math.round(255 * (
            0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (176.78208357 + t * (-354.22945805 + t * (276.23743207)))))
        ))));
        const g = Math.max(0, Math.min(255, Math.round(255 * (
            0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * (2.82956604)))))
        ))));
        const b = Math.max(0, Math.min(255, Math.round(255 * (
            0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90379497 + t * (26.62320709)))))
        ))));
        const idx = i * 4;
        lut[idx] = r;
        lut[idx + 1] = g;
        lut[idx + 2] = b;
        lut[idx + 3] = 255;
    }
    return lut;
}

function flattenGrid(grid) {
    if (!grid || grid.length === 0) return { flat: new Float32Array(0), w: 0, h: 0 };
    if (Array.isArray(grid[0])) {
        const h = grid.length;
        const w = grid[0].length || 0;
        const flat = new Float32Array(w * h);
        let idx = 0;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) flat[idx++] = grid[y][x];
        }
        return { flat, w, h };
    }
    return { flat: new Float32Array(grid), w: 0, h: 0 };
}

function resampleGrid(src, srcW, srcH, dstW, dstH) {
    if (srcW === 0 || srcH === 0 || dstW === 0 || dstH === 0) return new Float32Array(0);
    const dst = new Float32Array(dstW * dstH);
    for (let y = 0; y < dstH; y++) {
        const gy = (dstH === 1) ? 0 : (y / (dstH - 1)) * (srcH - 1);
        const y0 = Math.floor(gy);
        const y1 = Math.min(y0 + 1, srcH - 1);
        const ty = gy - y0;
        for (let x = 0; x < dstW; x++) {
            const gx = (dstW === 1) ? 0 : (x / (dstW - 1)) * (srcW - 1);
            const x0 = Math.floor(gx);
            const x1 = Math.min(x0 + 1, srcW - 1);
            const tx = gx - x0;
            const i00 = y0 * srcW + x0;
            const i10 = y0 * srcW + x1;
            const i01 = y1 * srcW + x0;
            const i11 = y1 * srcW + x1;
            const v0 = src[i00] * (1 - tx) + src[i10] * tx;
            const v1 = src[i01] * (1 - tx) + src[i11] * tx;
            dst[y * dstW + x] = v0 * (1 - ty) + v1 * ty;
        }
    }
    return dst;
}

// ── Blueprint Upload Component ─────────────────────────────────────
const BlueprintUploader = ({ cameraId, onUploadSuccess }) => {
    const fileInputRef = useRef(null);
    const [isUploading, setIsUploading] = useState(false);

    const handleUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(`/api/cameras/spatial/blueprint/${cameraId}`, {
                method: 'POST',
                body: formData
            });
            if (res.ok) {
                onUploadSuccess();
            } else {
                alert("Failed to upload blueprint.");
            }
        } catch (err) {
            console.error(err);
            alert("Error uploading blueprint.");
        }
        setIsUploading(false);
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', width: '100%', gap: '1rem', color: '#94a3b8' }}>
            <MapPin size={48} opacity={0.5} />
            <h2 style={{ fontSize: '1.25rem', color: '#f8fafc' }}>No Blueprint Uploaded</h2>
            <p style={{ maxWidth: '400px', textAlign: 'center', lineHeight: 1.5 }}>
                To enable 8-point spatial homography, please upload a top-down floor plan or blueprint of the monitored area.
            </p>
            <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                onChange={handleUpload}
                style={{ display: 'none' }}
            />
            <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                style={{
                    display: 'flex', alignItems: 'center', gap: '0.5rem',
                    padding: '0.75rem 1.5rem',
                    background: '#3b82f6', color: 'white', borderRadius: '0.5rem',
                    border: 'none', cursor: 'pointer', fontWeight: 'bold'
                }}
            >
                <Upload size={18} />
                {isUploading ? 'Uploading...' : 'Upload Blueprint Image'}
            </button>
        </div>
    );
};


// ── Dual-View Calibration Component ───────────────────────────────
const DualViewCalibration = ({ cameraId, blueprintUrl, onComplete }) => {
    const [camPoints, setCamPoints] = useState([]);
    const [bpPoints, setBpPoints] = useState([]);
    const camImgRef = useRef(null);
    const bpImgRef = useRef(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleImgClick = (e, imgRef, points, setPoints) => {
        if (points.length >= 4 || !imgRef.current) return;
        const rect = imgRef.current.getBoundingClientRect();
        const scaleX = imgRef.current.naturalWidth / rect.width;
        const scaleY = imgRef.current.naturalHeight / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        setPoints([...points, [x, y]]);
    };

    const handleClear = () => {
        setCamPoints([]);
        setBpPoints([]);
    };

    const handleSubmit = async () => {
        if (camPoints.length !== 4 || bpPoints.length !== 4) return;
        setIsSubmitting(true);

        try {
            const bpW = bpImgRef.current.naturalWidth;
            const bpH = bpImgRef.current.naturalHeight;

            const resp = await fetch(`/api/cameras/spatial/calibrate/${cameraId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    camera_pts: camPoints,
                    blueprint_pts: bpPoints,
                    bp_w: bpW,
                    bp_h: bpH
                })
            });
            if (resp.ok) {
                onComplete();
            } else {
                alert("Failed to confirm calibration.");
            }
        } catch (e) {
            console.error(e);
            alert("Error submitting calibration.");
        }
        setIsSubmitting(false);
    };

    const renderPoints = (points, imgRef, color) => {
        return points.map((p, idx) => {
            if (!imgRef.current) return null;
            const rect = imgRef.current.getBoundingClientRect();
            const scaleX = rect.width / imgRef.current.naturalWidth;
            const scaleY = rect.height / imgRef.current.naturalHeight;
            const displayX = p[0] * scaleX;
            const displayY = p[1] * scaleY;

            return (
                <div key={idx} style={{
                    position: 'absolute',
                    left: `${displayX}px`, top: `${displayY}px`,
                    width: '24px', height: '24px',
                    marginLeft: '-12px', marginTop: '-12px',
                    backgroundColor: color,
                    border: '2px solid white',
                    borderRadius: '50%',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '12px', fontWeight: 'bold', color: 'white',
                    pointerEvents: 'none', boxShadow: '0 2px 4px rgba(0,0,0,0.5)'
                }}>
                    {idx + 1}
                </div>
            );
        });
    };

    return (
        <div style={{
            position: 'absolute', inset: 0,
            backgroundColor: 'rgba(2, 6, 23, 0.95)', zIndex: 100,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start',
            color: 'white', padding: '20px', overflowY: 'auto'
        }}>
            <h2 style={{ marginBottom: '8px', fontSize: '24px', fontWeight: 'bold' }}>Dual-View Homography Calibration</h2>
            <p style={{ marginBottom: '24px', opacity: 0.8, textAlign: 'center', maxWidth: '800px' }}>
                Click 4 points forming a bounding area on the <strong style={{ color: '#ef4444' }}>Camera View</strong>.
                Then click the exact corresponding 4 points on the <strong style={{ color: '#3b82f6' }}>Blueprint View</strong> in the same order.
            </p>

            <div style={{ display: 'flex', gap: '20px', width: '100%', maxWidth: '1400px', height: '50vh', minHeight: '400px' }}>

                {/* ── Left: Camera View ── */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#0f172a', borderRadius: '8px', overflow: 'hidden', border: '1px solid #334155' }}>
                    <div style={{ padding: '10px 16px', background: '#1e293b', borderBottom: '1px solid #334155', display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '6px' }}><Camera size={16} color="#ef4444" /> Live Camera Area</span>
                        <span style={{ fontSize: '12px', opacity: 0.7 }}>{camPoints.length}/4 Points</span>
                    </div>
                    <div style={{ flex: 1, position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', background: '#000' }}>
                        <img
                            ref={camImgRef}
                            src={`/api/cameras/stream/${cameraId}`}
                            alt="Live Camera"
                            onClick={(e) => handleImgClick(e, camImgRef, camPoints, setCamPoints)}
                            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', cursor: camPoints.length < 4 ? 'crosshair' : 'default' }}
                        />
                        {renderPoints(camPoints, camImgRef, 'rgba(239, 68, 68, 0.9)')}
                    </div>
                </div>

                {/* ── Right: Blueprint View ── */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#0f172a', borderRadius: '8px', overflow: 'hidden', border: '1px solid #334155' }}>
                    <div style={{ padding: '10px 16px', background: '#1e293b', borderBottom: '1px solid #334155', display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '6px' }}><MapPin size={16} color="#3b82f6" /> Blueprint Mapping</span>
                        <span style={{ fontSize: '12px', opacity: 0.7 }}>{bpPoints.length}/4 Points</span>
                    </div>
                    <div style={{ flex: 1, position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', background: '#000' }}>
                        {blueprintUrl ? (
                            <img
                                ref={bpImgRef}
                                src={blueprintUrl}
                                alt="Blueprint"
                                onClick={(e) => handleImgClick(e, bpImgRef, bpPoints, setBpPoints)}
                                style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', cursor: bpPoints.length < 4 ? 'crosshair' : 'default' }}
                            />
                        ) : (
                            <span style={{ opacity: 0.5 }}>Blueprint not loaded</span>
                        )}
                        {renderPoints(bpPoints, bpImgRef, 'rgba(59, 130, 246, 0.9)')}
                    </div>
                </div>

            </div>

            <div style={{ marginTop: '30px', display: 'flex', gap: '15px' }}>
                <button
                    onClick={handleClear}
                    style={{ padding: '10px 24px', background: '#1e293b', color: 'white', border: '1px solid #334155', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' }}
                >
                    Clear Points
                </button>
                <button
                    onClick={handleSubmit}
                    disabled={camPoints.length !== 4 || bpPoints.length !== 4 || isSubmitting}
                    style={{
                        padding: '10px 32px', border: 'none', borderRadius: '6px',
                        background: (camPoints.length === 4 && bpPoints.length === 4) ? '#10b981' : '#475569',
                        color: 'white', fontWeight: 'bold',
                        cursor: (camPoints.length === 4 && bpPoints.length === 4) ? 'pointer' : 'not-allowed',
                        opacity: isSubmitting ? 0.7 : 1, transition: 'all 0.2s'
                    }}
                >
                    {isSubmitting ? 'Processing...' : 'Save Calibration Mapping'}
                </button>
            </div>
        </div>
    );
};


// ── Main Component ─────────────────────────────────────────────────
const SpatialIntelligence = () => {
    const containerRef = useRef(null);
    const bgCanvasRef = useRef(null);
    const heatCanvasRef = useRef(null);
    const pointsCanvasRef = useRef(null);
    const offscreenRef = useRef(null);
    const bgImgRef = useRef(null);

    const latestRef = useRef(null);
    const needsRedrawRef = useRef(true);
    const rafRef = useRef(null);

    const [connected, setConnected] = useState(false);
    const [cameras, setCameras] = useState([]);
    const [selectedCamera, setSelectedCamera] = useState('');
    const [viewMode, setViewMode] = useState('Room 1'); // 'Room 1' or 'Camera'

    const [gridResolution, setGridResolution] = useState(40);
    const [intensity, setIntensity] = useState(1.4);
    const [showPoints, setShowPoints] = useState(true);
    const [peopleCount, setPeopleCount] = useState(0);

    // States driving the UI workflow
    const [blueprintExists, setBlueprintExists] = useState(true); // Assume true, verify on fetch
    const [blueprintUrl, setBlueprintUrl] = useState('');
    const [isReady, setIsReady] = useState(true);
    const [forceCalibration, setForceCalibration] = useState(false);

    // Try finding an active camera on mount if none selected
    useEffect(() => {
        let cancelled = false;
        const poll = async () => {
            try {
                const resp = await fetch('/api/cameras/list');
                const data = await resp.json();
                if (!cancelled) {
                    const sorted = data.cameras || [];
                    setCameras(sorted);
                    if (!selectedCamera && sorted.length > 0) {
                        const active = sorted.find(c => c.running);
                        setSelectedCamera(active ? active.camera_id : sorted[0].camera_id);
                    }
                }
            } catch { /* ignored */ }
        };
        poll();
        const iv = setInterval(poll, 5000);
        return () => { cancelled = true; clearInterval(iv); };
    }, [selectedCamera]);

    // Check if blueprint exists for the selected camera and map its natural resolution
    const checkBlueprint = useCallback(async () => {
        if (viewMode !== 'Camera' || !selectedCamera) return;
        try {
            const url = `/api/cameras/spatial/blueprint/${selectedCamera}?t=${Date.now()}`;
            const res = await fetch(url);
            if (res.ok) {
                setBlueprintExists(true);
                setBlueprintUrl(url);

                const img = new Image();
                img.onload = () => {
                    bgImgRef.current = img;

                    // Critical: Set accurate fixed intrinsic resolution on all 3 layers.
                    const nw = img.naturalWidth;
                    const nh = img.naturalHeight;

                    if (bgCanvasRef.current && heatCanvasRef.current && pointsCanvasRef.current) {
                        bgCanvasRef.current.width = nw;
                        bgCanvasRef.current.height = nh;
                        heatCanvasRef.current.width = nw;
                        heatCanvasRef.current.height = nh;
                        pointsCanvasRef.current.width = nw;
                        pointsCanvasRef.current.height = nh;

                        drawBackground();
                    }
                };
                img.src = url;
            } else {
                setBlueprintExists(false);
                setBlueprintUrl('');
                bgImgRef.current = null;
            }
        } catch {
            setBlueprintExists(false);
        }
    }, [selectedCamera, viewMode]);

    useEffect(() => {
        checkBlueprint();
    }, [checkBlueprint]);

    const sseUrl = viewMode === 'Room 1'
        ? `/api/rooms/spatial/Room 1`
        : (selectedCamera ? `/api/cameras/spatial/${selectedCamera}` : '/api/spatial/stream');

    // ── SSE connection ───────────────────────────────────────────
    useEffect(() => {
        if (viewMode === 'Camera' && !selectedCamera) return;

        let es;
        try {
            es = new EventSource(sseUrl);
        } catch { return; }

        es.onmessage = (ev) => {
            try {
                const data = JSON.parse(ev.data);
                const { flat, w, h } = flattenGrid(data.grid);
                latestRef.current = {
                    grid: flat,
                    gridW: w,
                    gridH: h,
                    points: Array.isArray(data.points) ? data.points : [],
                    topW: data.width || bgCanvasRef.current?.width || 800,
                    topH: data.height || bgCanvasRef.current?.height || 600,
                };

                setIsReady(data.ready ?? true);
                needsRedrawRef.current = true;
                setPeopleCount(Array.isArray(data.points) ? data.points.length : 0);
                setConnected(true);
            } catch { /* parse error */ }
        };
        es.onerror = () => setConnected(false);

        return () => {
            es.close();
            setConnected(false);
        };
    }, [sseUrl, selectedCamera, viewMode]);

    // ── Draw Static Blueprint Layer ──────────────────────────────
    // Drawn exactly ONE time when the image loads.
    const drawBackground = () => {
        const canvas = bgCanvasRef.current;
        const img = bgImgRef.current;
        if (!canvas || !img) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        // Disable smoothing to guarantee untouched original pixels
        ctx.imageSmoothingEnabled = false;

        ctx.drawImage(img, 0, 0, w, h);
    };

    // ── Render Area Layers ──────────────────────────────────────
    const renderHeatmap = useCallback((data) => {
        if (!data || !heatCanvasRef.current) return;
        const { grid, gridW, gridH } = data;
        const canvas = heatCanvasRef.current;
        const ctx = canvas.getContext('2d');
        const cw = canvas.width;
        const ch = canvas.height;

        ctx.clearRect(0, 0, cw, ch);

        if (!grid || grid.length === 0 || gridW === 0 || gridH === 0) return;

        const targetW = Math.max(10, Math.round(gridResolution));
        const aspect = gridH / Math.max(gridW, 1);
        const targetH = Math.max(6, Math.round(targetW * aspect));
        const resampled = resampleGrid(grid, gridW, gridH, targetW, targetH);

        if (!offscreenRef.current) {
            offscreenRef.current = document.createElement('canvas');
        }
        const off = offscreenRef.current;
        off.width = targetW;
        off.height = targetH;
        const octx = off.getContext('2d', { willReadFrequently: false });
        const imageData = octx.createImageData(targetW, targetH);
        const pixels = imageData.data;

        for (let i = 0; i < resampled.length; i++) {
            let v = resampled[i] * intensity;
            if (v < 0.003) {
                pixels[i * 4 + 3] = 0;
                continue;
            }
            if (v > 1) v = 1;
            const lutIdx = Math.min(255, Math.round(v * 255)) * 4;
            const pidx = i * 4;
            pixels[pidx] = TURBO_LUT[lutIdx];
            pixels[pidx + 1] = TURBO_LUT[lutIdx + 1];
            pixels[pidx + 2] = TURBO_LUT[lutIdx + 2];
            // Density alpha mapping
            pixels[pidx + 3] = Math.min(220, Math.round(v * v * 255));
        }

        octx.putImageData(imageData, 0, 0);

        // Map heatmap over the exact same canvas bounds 
        // using soft gaussian filtering purely on this separate layer
        ctx.save();
        ctx.filter = 'blur(16px)';
        ctx.globalAlpha = 0.8;
        ctx.drawImage(off, 0, 0, cw, ch);
        ctx.restore();
    }, [gridResolution, intensity]);

    const renderPoints = useCallback((data) => {
        if (!data || !pointsCanvasRef.current) return;
        const { points, topW, topH } = data;
        const canvas = pointsCanvasRef.current;
        const ctx = canvas.getContext('2d');
        const cw = canvas.width;
        const ch = canvas.height;

        ctx.clearRect(0, 0, cw, ch);

        if (!showPoints || !points || points.length === 0) return;

        const sx = cw / Math.max(topW, 1);
        const sy = ch / Math.max(topH, 1);

        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = '#000000';
        ctx.shadowBlur = 4;

        for (let i = 0; i < points.length; i++) {
            const pt = points[i];
            const x = pt[0] * sx;
            const y = pt[1] * sy;

            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fill();

            ctx.beginPath();
            ctx.fillStyle = '#eab308'; // Amber inner dot
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#ffffff';
        }
        ctx.shadowBlur = 0;
    }, [showPoints]);

    // ── Animation Loop ──────────────────────────────────────────
    useEffect(() => {
        const loop = () => {
            // Only redraw the overlay layers. Background remains completely untouched.
            if (needsRedrawRef.current && latestRef.current && bgImgRef.current) {
                renderHeatmap(latestRef.current);
                renderPoints(latestRef.current);
                needsRedrawRef.current = false;
            }
            rafRef.current = requestAnimationFrame(loop);
        };
        rafRef.current = requestAnimationFrame(loop);
        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
        };
    }, [renderHeatmap, renderPoints]);

    // ── Redraw on control changes ───────────────────────────────
    useEffect(() => {
        needsRedrawRef.current = true;
    }, [gridResolution, intensity, showPoints]);


    // Determine UI State
    const showUploader = connected && !blueprintExists;
    const showCalibration = connected && blueprintExists && (!isReady || forceCalibration);
    const showVisualizer = connected && blueprintExists && isReady && !forceCalibration;

    // Strict css rules for stacked canvases ensuring perfect aspect scaling through CSS 
    // without altering their exact internal pixel counts.
    const strictCanvasStyle = {
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        objectFit: 'contain',
        pointerEvents: 'none',
        imageRendering: 'pixelated' /* Ensures base fidelity is uncompromised during browser-scaling if desired */
    };

    return (
        <div className="sd-layout" style={{ position: 'relative' }}>

            {showCalibration && selectedCamera && (
                <DualViewCalibration
                    cameraId={selectedCamera}
                    blueprintUrl={blueprintUrl}
                    onComplete={() => {
                        setForceCalibration(false);
                        setIsReady(true);
                        checkBlueprint(); // Remount dimensions if needed
                    }}
                />
            )}

            {/* ── Left: Layered canvas stack ───────────────────── */}
            <div className="sd-canvas-wrap" ref={containerRef} style={{ position: 'relative', overflow: 'hidden', background: '#000' }}>

                {showVisualizer && (
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {/* All 3 canvases perfectly layered and CSS styled identically */}
                        <canvas ref={bgCanvasRef} style={{ ...strictCanvasStyle, zIndex: 1 }} />
                        <canvas ref={heatCanvasRef} style={{ ...strictCanvasStyle, zIndex: 2 }} />
                        <canvas ref={pointsCanvasRef} style={{ ...strictCanvasStyle, zIndex: 3 }} />
                    </div>
                )}

                {showUploader && (
                    <BlueprintUploader cameraId={selectedCamera} onUploadSuccess={checkBlueprint} />
                )}

                {showVisualizer && (
                    <div className="sd-status-bar" style={{ zIndex: 10 }}>
                        <Radar size={14} />
                        <span className="sd-status-label">Spatial Intelligence</span>
                        <span className={`sd-status-dot ${connected ? 'sd-status-dot--live' : ''}`} />
                        {connected && showVisualizer && <span className="sd-status-live">LIVE</span>}
                        {showVisualizer && (
                            <span className="sd-status-count">
                                {peopleCount} {peopleCount === 1 ? 'person' : 'people'}
                            </span>
                        )}
                    </div>
                )}

                {showVisualizer && (
                    <div className="sd-legend" style={{ zIndex: 10 }}>
                        <span className="sd-legend-label">Low</span>
                        <div className="sd-legend-bar" />
                        <span className="sd-legend-label">High</span>
                    </div>
                )}

                {(!connected) && (
                    <div className="sd-empty">
                        <div className="sd-empty-title">Waiting for connection</div>
                        <div className="sd-empty-sub">
                            Ensure a camera is active and running
                        </div>
                    </div>
                )}
            </div>

            {/* ── Right: Controls + stats ──────────────────────── */}
            <div className="sd-panel">
                <div className="sd-panel-section">
                    <div className="sd-panel-label">
                        <Layers size={12} />
                        Model Status
                    </div>
                    <div className="sd-panel-value">
                        {!connected ? 'Disconnected' :
                            !blueprintExists ? 'Needs Blueprint' :
                                !isReady ? 'Pending Calibration' : 'Active (Bird\'s-Eye)'}
                    </div>
                </div>

                <div className="sd-panel-divider" />

                <div className="sd-panel-section">
                    <div className="sd-panel-label">
                        <Activity size={12} />
                        People Count
                    </div>
                    <div className="sd-panel-value sd-panel-value--large">
                        {showVisualizer ? peopleCount : '--'}
                    </div>
                </div>

                <div className="sd-panel-divider" />

                <div className="sd-panel-section">
                    <div className="sd-panel-label">
                        <SlidersHorizontal size={12} />
                        Grid Resolution
                    </div>
                    <input
                        className="sd-slider"
                        type="range"
                        min="20"
                        max="80"
                        step="2"
                        value={gridResolution}
                        onChange={(e) => setGridResolution(Number(e.target.value))}
                        disabled={!showVisualizer}
                    />
                    <div className="sd-panel-value">
                        <span>{gridResolution}</span>
                        <span className="sd-panel-unit">cells</span>
                    </div>
                </div>

                <div className="sd-panel-divider" />

                <div className="sd-panel-section">
                    <div className="sd-panel-label">
                        <SlidersHorizontal size={12} />
                        Intensity Multiplier
                    </div>
                    <input
                        className="sd-slider"
                        type="range"
                        min="0.6"
                        max="3.0"
                        step="0.1"
                        value={intensity}
                        onChange={(e) => setIntensity(Number(e.target.value))}
                        disabled={!showVisualizer}
                    />
                    <div className="sd-panel-value">
                        <span>{intensity.toFixed(1)}</span>
                        <span className="sd-panel-unit">x</span>
                    </div>
                </div>

                <div className="sd-panel-divider" />

                <div className="sd-panel-section">
                    <div className="sd-panel-label">
                        <Maximize size={12} />
                        Calibration
                    </div>
                    <button
                        onClick={() => setForceCalibration(true)}
                        disabled={!connected || !selectedCamera || !blueprintExists}
                        style={{
                            width: '100%', padding: '8px',
                            background: (connected && blueprintExists) ? '#3b82f6' : '#1e293b',
                            color: '#fff', border: 'none',
                            borderRadius: '4px', cursor: (connected && blueprintExists) ? 'pointer' : 'not-allowed',
                            fontSize: '12px', marginTop: '4px', fontWeight: 'bold'
                        }}
                    >
                        Recalibrate Layout
                    </button>

                    {blueprintExists && (
                        <button
                            onClick={() => {
                                if (window.confirm("Delete this blueprint? You will need to upload a new one.")) {
                                    setBlueprintExists(false);
                                    setBlueprintUrl('');
                                    bgImgRef.current = null;
                                }
                            }}
                            style={{
                                width: '100%', padding: '8px',
                                background: 'transparent',
                                color: '#ef4444', border: '1px solid #7f1d1d',
                                borderRadius: '4px', cursor: 'pointer',
                                fontSize: '12px', marginTop: '8px'
                            }}
                        >
                            Remove Selected Blueprint
                        </button>
                    )}
                </div>

                <div className="sd-panel-divider" />

                <div className="sd-panel-section">
                    <div className="sd-panel-label">
                        <Crosshair size={12} />
                        Projected Points
                    </div>
                    <label className="sd-toggle">
                        <input
                            type="checkbox"
                            checked={showPoints}
                            onChange={(e) => setShowPoints(e.target.checked)}
                            disabled={!showVisualizer}
                        />
                        <span className="sd-toggle-slider" />
                        <span className="sd-toggle-label">
                            {showPoints ? 'Visible' : 'Hidden'}
                        </span>
                    </label>
                </div>

                <div className="sd-panel-divider" />

                <div className="sd-panel-section">
                    <div className="sd-panel-label">
                        <Camera size={12} />
                        View Source
                    </div>
                    {cameras.length > 0 ? (
                        <>
                            <select
                                className="sd-camera-select"
                                value={viewMode === 'Room 1' ? 'Room 1' : 'Camera'}
                                onChange={(e) => {
                                    setViewMode(e.target.value);
                                    setForceCalibration(false);
                                }}
                                style={{ marginBottom: '8px' }}
                            >
                                <option value="Room 1">Room 1 (All Cameras)</option>
                                <option value="Camera">Specific Camera</option>
                            </select>

                            {viewMode === 'Camera' && (
                                <select
                                    className="sd-camera-select"
                                    value={selectedCamera}
                                    onChange={(e) => {
                                        setSelectedCamera(e.target.value);
                                        setForceCalibration(false);
                                    }}
                                >
                                    {cameras.map((c) => (
                                        <option key={c.camera_id} value={c.camera_id}>
                                            {c.label || c.camera_id.slice(0, 8)}
                                            {c.running ? ' ●' : ''}
                                        </option>
                                    ))}
                                </select>
                            )}
                        </>
                    ) : (
                        <div className="sd-panel-value" style={{ opacity: 0.4, fontSize: '12px' }}>
                            No cameras registered
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default SpatialIntelligence;
