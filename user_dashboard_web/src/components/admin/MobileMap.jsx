import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix leaflet marker icon URL issue with bundlers
import iconMarker from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
    iconUrl: iconMarker,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

// Make L available globally for leaflet.heat
window.L = L;
import 'leaflet.heat';

const HeatmapLayer = ({ points }) => {
    const map = useMap();

    useEffect(() => {
        if (!points || !points.length) return;
        // Map points to [lat, lng, intensity]
        const heatPoints = points.map(p => [p.lat, p.lng, 1.0]);
        const heat = L.heatLayer(heatPoints, { radius: 25, blur: 15, maxZoom: 17 }).addTo(map);

        return () => {
            map.removeLayer(heat);
        };
    }, [points, map]);

    return null;
};

const AutoCenter = ({ points }) => {
    const map = useMap();
    useEffect(() => {
        if (points && points.length > 0) {
            const bounds = L.latLngBounds(points.map(p => [p.lat, p.lng]));
            map.fitBounds(bounds, { padding: [50, 50], maxZoom: 18 });
        }
    }, [points, map]);
    return null;
};

const MobileMap = ({ positions }) => {
    // Default center to a generalized location if no points
    const defaultCenter = [12.6423, 77.4403];
    const center = positions.length > 0 ? [positions[0].lat, positions[0].lng] : defaultCenter;

    return (
        <MapContainer center={center} zoom={15} style={{ width: '100%', height: '100%', background: '#0f172a' }}>
            <TileLayer
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
            />
            {positions.map((p, i) => (
                <Marker key={i} position={[p.lat, p.lng]} />
            ))}
            <HeatmapLayer points={positions} />
            <AutoCenter points={positions} />
        </MapContainer>
    );
};

export default MobileMap;
