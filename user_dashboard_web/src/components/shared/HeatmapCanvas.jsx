/**
 * HeatmapCanvas.jsx — Continuous Density Field Renderer
 * =====================================================
 *
 * Renders a smooth, professional heatmap overlay on top of a blueprint image.
 *
 * Architecture:
 *   SSE (/api/heatmap/stream)
 *     → base64-encoded float32 density buffer
 *     → decode to Float32Array
 *     → map through Turbo colormap (256-entry LUT)
 *     → write to ImageData
 *     → draw to offscreen canvas
 *     → drawImage with bilinear upscaling over blueprint
 *
 * Color mapping: Google Turbo (perceptually uniform, no rainbow artifacts).
 * Alpha: 0 at density=0, ramps up smoothly for occupied areas.
 * Rendering: requestAnimationFrame — no flicker, no full re-render of blueprint.
 */

import React, { useEffect, useRef, useCallback, useState } from 'react';

/* ── Turbo Colormap LUT (256 entries) ─────────────────────────────────────
 *
 *  Google Research's Turbo colormap, designed to be perceptually uniform
 *  with no luminance reversals. Blue→Cyan→Green→Yellow→Red progression.
 *  Each entry: [R, G, B] in 0-255 range.
 *  Reference: https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
 */
const TURBO_SRGB_BYTES = buildTurboLUT();

function buildTurboLUT() {
    // Polynomial approximation of the Turbo colormap
    const lut = new Uint8Array(256 * 4);  // RGBA

    for (let i = 0; i < 256; i++) {
        const t = i / 255;
        // Red channel
        const r = Math.max(0, Math.min(255, Math.round(255 * (
            0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (176.78208357 +
                t * (-354.22945805 + t * (276.23743207)))))
        ))));
        // Green channel
        const g = Math.max(0, Math.min(255, Math.round(255 * (
            0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 +
                t * (4.27729857 + t * (2.82956604)))))
        ))));
        // Blue channel
        const b = Math.max(0, Math.min(255, Math.round(255 * (
            0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 +
                t * (-89.90379497 + t * (26.62320709)))))
        ))));

        const idx = i * 4;
        lut[idx] = r;
        lut[idx + 1] = g;
        lut[idx + 2] = b;
        lut[idx + 3] = 255;  // alpha set per-pixel during render
    }
    return lut;
}


/* ── HeatmapCanvas Component ──────────────────────────────────────────── */

const HeatmapCanvas = ({
    sseUrl = '/api/heatmap/stream',
    showStatus = true,
    onData = null,
}) => {
    const containerRef = useRef(null);
    const blueprintRef = useRef(null);
    const canvasRef = useRef(null);
    const offscreenRef = useRef(null);  // offscreen canvas for density ImageData

    const [calibrated, setCalibrated] = useState(false);
    const [blueprintSrc, setBlueprintSrc] = useState(null);
    const [connected, setConnected] = useState(false);
    const [personCount, setPersonCount] = useState(0);

    // Refs for animation loop (avoid re-renders)
    const latestDataRef = useRef(null);
    const needsRedrawRef = useRef(false);
    const rafRef = useRef(null);

    // ── Check calibration status ─────────────────────────────────
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const resp = await fetch('/api/calibration/status');
                const data = await resp.json();
                if (!cancelled && data.calibrated) {
                    setCalibrated(true);
                    setBlueprintSrc('/api/calibration/blueprint');
                }
            } catch { /* backend not up */ }
        })();
        return () => { cancelled = true; };
    }, []);

    // ── SSE connection ───────────────────────────────────────────
    useEffect(() => {
        if (!calibrated) return;
        let es;
        try {
            es = new EventSource(sseUrl);
        } catch { return; }

        es.onmessage = (ev) => {
            try {
                const data = JSON.parse(ev.data);
                latestDataRef.current = data;
                needsRedrawRef.current = true;
                setPersonCount(data.person_count || 0);
                setConnected(true);
                if (onData) onData(data);
            } catch { /* parse error */ }
        };
        es.onerror = () => setConnected(false);

        return () => {
            es.close();
            setConnected(false);
        };
    }, [calibrated, sseUrl, onData]);

    // ── Decode base64 float32 density buffer ─────────────────────
    const decodeDensity = useCallback((b64, width, height) => {
        if (!b64 || !width || !height) return null;
        const binary = atob(b64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return new Float32Array(bytes.buffer);
    }, []);

    // ── Render density to canvas using Turbo colormap ────────────
    const renderDensity = useCallback((density, width, height) => {
        if (!density || !canvasRef.current) return;

        // Ensure offscreen canvas exists at density resolution
        if (!offscreenRef.current) {
            offscreenRef.current = document.createElement('canvas');
        }
        const offscreen = offscreenRef.current;
        offscreen.width = width;
        offscreen.height = height;

        const octx = offscreen.getContext('2d', { willReadFrequently: false });
        const imageData = octx.createImageData(width, height);
        const pixels = imageData.data;

        // Map each density value to Turbo RGBA
        for (let i = 0; i < density.length; i++) {
            const v = density[i];
            const pidx = i * 4;

            if (v < 0.005) {
                // Below noise floor → fully transparent
                pixels[pidx] = 0;
                pixels[pidx + 1] = 0;
                pixels[pidx + 2] = 0;
                pixels[pidx + 3] = 0;
                continue;
            }

            // Map density [0, 1] → LUT index [0, 255]
            const lutIdx = Math.min(255, Math.round(v * 255)) * 4;
            pixels[pidx] = TURBO_SRGB_BYTES[lutIdx];
            pixels[pidx + 1] = TURBO_SRGB_BYTES[lutIdx + 1];
            pixels[pidx + 2] = TURBO_SRGB_BYTES[lutIdx + 2];

            // Alpha ramp: 0 at low density, up to ~200 at full density
            // Smooth cubic ramp avoids harsh edges
            const alpha = Math.min(200, Math.round(v * v * 250));
            pixels[pidx + 3] = alpha;
        }

        octx.putImageData(imageData, 0, 0);

        // Draw upscaled to visible canvas with bilinear interpolation
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.globalCompositeOperation = 'screen';
        ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
    }, []);

    // ── Animation loop ───────────────────────────────────────────
    useEffect(() => {
        const loop = () => {
            if (needsRedrawRef.current && latestDataRef.current) {
                const data = latestDataRef.current;
                const density = decodeDensity(data.density, data.width, data.height);
                if (density) {
                    renderDensity(density, data.width, data.height);
                }
                needsRedrawRef.current = false;
            }
            rafRef.current = requestAnimationFrame(loop);
        };
        rafRef.current = requestAnimationFrame(loop);
        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
        };
    }, [decodeDensity, renderDensity]);

    // ── Handle container resize ──────────────────────────────────
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;
        const observer = new ResizeObserver(() => {
            const canvas = canvasRef.current;
            if (canvas) {
                canvas.width = container.clientWidth;
                canvas.height = container.clientHeight;
                needsRedrawRef.current = true;
            }
        });
        observer.observe(container);
        return () => observer.disconnect();
    }, []);

    // ── Not calibrated → show empty state ────────────────────────
    if (!calibrated) {
        return (
            <div className="hv-container hv-container--empty">
                <div className="hv-empty">
                    <div className="hv-empty-icon">📐</div>
                    <p className="hv-empty-title">Calibration Required</p>
                    <p className="hv-empty-desc">
                        Navigate to Floor Plan to set calibration points before
                        the density field can render.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="hv-container" ref={containerRef}>
            {/* Layer 0: Blueprint image */}
            {blueprintSrc && (
                <img
                    ref={blueprintRef}
                    src={blueprintSrc}
                    alt="Floor plan"
                    className="hv-blueprint"
                    draggable={false}
                />
            )}

            {/* Layer 1: Density heatmap overlay */}
            <canvas
                ref={canvasRef}
                className="hv-canvas"
            />

            {/* Status bar */}
            {showStatus && (
                <div className="hv-status-bar">
                    <span className="hv-status-label">Density Engine</span>
                    <span className={`hv-status-dot ${connected ? 'hv-status-dot--live' : ''}`} />
                    {connected && <span className="hv-status-text">LIVE</span>}
                    <span className="hv-status-count">
                        {personCount} {personCount === 1 ? 'person' : 'people'} detected
                    </span>
                </div>
            )}

            {/* Color legend */}
            <div className="hv-legend">
                <span className="hv-legend-label">Low</span>
                <div className="hv-legend-bar hv-legend-bar--turbo" />
                <span className="hv-legend-label">High</span>
            </div>
        </div>
    );
};

export default HeatmapCanvas;
