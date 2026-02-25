import React, { useEffect, useState, useRef, useCallback } from 'react';

const API = 'http://localhost:8000';

/**
 * HeatmapVisualizer — Real-time blueprint + heatmap overlay
 *
 * Architecture:
 *   Layer 0: <img>    — blueprint (rendered once)
 *   Layer 1: <canvas> — heatmap density (Gaussian-blurred grid)
 *   Layer 2: <canvas> — projected people dots
 *
 * Data: SSE from /api/heatmap/stream at ~3 Hz
 */
const HeatmapVisualizer = () => {
    const [calibrated, setCalibrated] = useState(false);
    const [blueprintSrc, setBlueprintSrc] = useState(null);
    const [bpDims, setBpDims] = useState({ w: 0, h: 0 });
    const [connected, setConnected] = useState(false);
    const [peopleCount, setPeopleCount] = useState(0);

    const containerRef = useRef(null);
    const heatCanvasRef = useRef(null);
    const dotCanvasRef = useRef(null);
    const eventSourceRef = useRef(null);

    // ── Check calibration status on mount ────────────────────────────
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const resp = await fetch(`${API}/api/calibration/status`);
                const data = await resp.json();
                if (!cancelled && data.calibrated) {
                    setCalibrated(true);
                    setBpDims({ w: data.bp_width, h: data.bp_height });
                    setBlueprintSrc(`${API}/api/calibration/blueprint`);
                }
            } catch { /* backend not up */ }
        })();
        return () => { cancelled = true; };
    }, []);

    // ── SSE connection ───────────────────────────────────────────────
    useEffect(() => {
        if (!calibrated) return;

        const es = new EventSource(`${API}/api/heatmap/stream`);
        eventSourceRef.current = es;

        es.onopen = () => setConnected(true);
        es.onerror = () => setConnected(false);

        es.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setPeopleCount(data.people || 0);
                drawHeatmap(data.grid);
                drawDots(data.positions);
            } catch { /* bad frame, skip */ }
        };

        return () => {
            es.close();
            eventSourceRef.current = null;
            setConnected(false);
        };
    }, [calibrated, bpDims]);

    // ── Heatmap rendering (canvas layer 1) ───────────────────────────
    const drawHeatmap = useCallback((grid) => {
        const canvas = heatCanvasRef.current;
        if (!canvas || !grid || grid.length === 0) return;

        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        const rows = grid.length;
        const cols = grid[0]?.length || 1;
        const cellW = w / cols;
        const cellH = h / rows;

        // Find max for normalization
        let maxVal = 1;
        for (const row of grid) {
            for (const v of row) {
                if (v > maxVal) maxVal = v;
            }
        }

        // Draw cells with color ramp
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const val = grid[r][c] / maxVal;
                if (val < 0.01) continue;

                const x = c * cellW;
                const y = r * cellH;

                // Color ramp: blue → yellow → red (Increased opacity scale)
                let color;
                if (val < 0.4) {
                    const t = val / 0.4;
                    color = `rgba(59, 130, 246, ${t * 0.8})`;
                } else if (val < 0.7) {
                    const t = (val - 0.4) / 0.3;
                    color = `rgba(250, 204, 21, ${0.4 + t * 0.5})`;
                } else {
                    const t = (val - 0.7) / 0.3;
                    color = `rgba(239, 68, 68, ${0.6 + t * 0.4})`;
                }

                ctx.fillStyle = color;
                // Slight padding for visual separation
                const pad = 1;
                ctx.beginPath();
                ctx.roundRect(x + pad, y + pad, cellW - pad * 2, cellH - pad * 2, 4);
                ctx.fill();
            }
        }

        // Apply blur for Gaussian-like effect
        ctx.filter = 'blur(10px) contrast(1.2)';
        ctx.globalAlpha = 0.85;
        ctx.drawImage(canvas, 0, 0);
        ctx.filter = 'none';
        ctx.globalAlpha = 1;
    }, []);

    // ── People dots rendering (canvas layer 2) ───────────────────────
    const drawDots = useCallback((positions) => {
        const canvas = dotCanvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        if (!positions || positions.length === 0 || bpDims.w === 0) return;

        const scaleX = w / bpDims.w;
        const scaleY = h / bpDims.h;

        for (const [px, py] of positions) {
            const x = px * scaleX;
            const y = py * scaleY;

            // Outer glow
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, 10);
            gradient.addColorStop(0, 'rgba(59, 130, 246, 0.6)');
            gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(x, y, 10, 0, Math.PI * 2);
            ctx.fill();

            // Inner dot
            ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
            ctx.beginPath();
            ctx.arc(x, y, 3.5, 0, Math.PI * 2);
            ctx.fill();

            // White center
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.beginPath();
            ctx.arc(x, y, 1.5, 0, Math.PI * 2);
            ctx.fill();
        }
    }, [bpDims]);

    // ── Resize canvases to match container ───────────────────────────
    useEffect(() => {
        const resize = () => {
            const container = containerRef.current;
            if (!container) return;
            const rect = container.getBoundingClientRect();
            [heatCanvasRef, dotCanvasRef].forEach(ref => {
                if (ref.current) {
                    ref.current.width = rect.width;
                    ref.current.height = rect.height;
                }
            });
        };

        resize();
        window.addEventListener('resize', resize);
        return () => window.removeEventListener('resize', resize);
    }, [calibrated]);

    // ── Not calibrated state ─────────────────────────────────────────
    if (!calibrated) {
        return (
            <div className="hv-container hv-container--empty">
                <div className="hv-empty">
                    <span className="hv-empty-icon">📐</span>
                    <p className="hv-empty-title">No Calibration</p>
                    <p className="hv-empty-desc">
                        Go to Floor Plan and complete camera-to-blueprint calibration to enable live heatmap.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="hv-container" ref={containerRef}>
            {/* Status bar */}
            <div className="hv-status-bar">
                <span className="hv-status-label">Spatial Density Map</span>
                <span className={`hv-status-dot ${connected ? 'hv-status-dot--live' : ''}`} />
                <span className="hv-status-text">
                    {connected ? 'LIVE' : 'CONNECTING…'}
                </span>
                {connected && (
                    <span className="hv-status-count">{peopleCount} detected</span>
                )}
            </div>

            {/* Layer 0: Blueprint image */}
            {blueprintSrc && (
                <img
                    src={blueprintSrc}
                    alt="Floor plan blueprint"
                    className="hv-blueprint"
                    draggable={false}
                    style={{ filter: 'brightness(1.1) contrast(1.05)' }}
                />
            )}

            {/* Layer 1: Heatmap overlay */}
            <canvas ref={heatCanvasRef} className="hv-canvas" />

            {/* Layer 2: People dots */}
            <canvas ref={dotCanvasRef} className="hv-canvas" />
        </div>
    );
};

export default HeatmapVisualizer;
