import React from 'react';
import HeatmapCanvas from './shared/HeatmapCanvas';

/**
 * HeatmapVisualizer — User Dashboard wrapper for shared HeatmapCanvas.
 * Delegates all rendering to the shared component.
 */
const HeatmapVisualizer = ({ cameraId, roomName }) => {
    // Determine the SSE URL based on selected camera or general floorplan
    let sseUrl = '/api/heatmap/stream'; // Default generic

    // Prioritize roomName (public dashboard integration over multiple cameras)
    if (roomName) {
        sseUrl = `/api/public/heatmap/stream?room=${encodeURIComponent(roomName)}`;
    } else if (cameraId) {
        sseUrl = `/api/cameras/spatial/${cameraId}`;
    }

    // Always attach the dominant camera's blueprint if available
    const blueprintUrl = cameraId ? `/api/cameras/spatial/blueprint/${cameraId}` : null;

    return (
        <HeatmapCanvas
            sseUrl={sseUrl}
            blueprintUrl={blueprintUrl}
            showDots={true}
            showStatus={true}
        />
    );
};

export default HeatmapVisualizer;
