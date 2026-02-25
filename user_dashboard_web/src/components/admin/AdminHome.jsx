import React, { useState, useEffect } from 'react';

const AdminHome = () => {
    const [currentTime, setCurrentTime] = useState(new Date());

    // Update clock every second
    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    const formattedTime = currentTime.toLocaleTimeString([], {
        hour: '2-digit', minute: '2-digit', second: '2-digit'
    });

    return (
        <div className="admin-content-area home-hero-wrapper">

            {/* Ambient Backgrounds */}
            <div className="hero-scanlines" />
            <div className="hero-ambient-glow" />

            <div className="home-hero-content">

                {/* Main Typography */}
                <div className="hero-text-container">
                    <h1 className="hero-title">
                        Welcome to Sentinel Live
                    </h1>

                    {/* Minimal animated underline */}
                    <div className="hero-line-draw" />

                    <p className="hero-subtitle">
                        Real-Time Crowd Intelligence & Spatial Risk Monitoring
                    </p>
                </div>

                {/* Live Status Strip */}
                <div className="hero-status-strip">
                    <div className="status-item">
                        <span className="status-label">System Status</span>
                        <div className="status-value-group">
                            <span className="status-dot status-live" />
                            <span className="status-value text-green-400">CONNECTED</span>
                        </div>
                    </div>

                    <div className="status-divider" />

                    <div className="status-item">
                        <span className="status-label">Detection Engine</span>
                        <span className="status-value text-blue-400">ACTIVE</span>
                    </div>

                    <div className="status-divider" />

                    <div className="status-item">
                        <span className="status-label">Heatmap Engine</span>
                        <span className="status-value text-[var(--accent-yellow)]">READY</span>
                    </div>

                    <div className="status-divider" />

                    <div className="status-item">
                        <span className="status-label">Local Sync Time</span>
                        <span className="status-value text-white font-mono">{formattedTime}</span>
                    </div>
                </div>

            </div>
        </div>
    );
};

export default AdminHome;
