import React from 'react';

function getRiskLevel(score) {
    if (score <= 30) return { level: "SAFE", color: "green" };
    if (score <= 60) return { level: "MODERATE", color: "yellow" };
    if (score <= 80) return { level: "HIGH RISK", color: "orange" };
    return { level: "CRITICAL", color: "red" };
}

const CrowdSafetyIndex = ({ csi }) => {
    const score = csi?.crowd_safety_index;

    if (score === undefined || score === null || typeof score !== "number") {
        return (
            <div className="db-card flex flex-col justify-center items-center h-full text-[var(--text-muted)] animate-pulse">
                INITIALIZING CSI SYSTEM...
            </div>
        );
    }

    const { level, color } = getRiskLevel(score);

    // Convert named colors into actual Tailwind/CSS classes
    const colorMap = {
        green: { text: "text-green-400", bg: "bg-green-500/20", border: "border-green-500/50", shadow: "shadow-[0_0_15px_rgba(34,197,94,0.3)]" },
        yellow: { text: "text-yellow-400", bg: "bg-yellow-500/20", border: "border-yellow-500/50", shadow: "shadow-[0_0_15px_rgba(234,179,8,0.3)]" },
        orange: { text: "text-orange-400", bg: "bg-orange-500/20", border: "border-orange-500/50", shadow: "shadow-[0_0_15px_rgba(249,115,22,0.3)]" },
        red: { text: "text-red-500", bg: "bg-red-500/20", border: "border-red-500/80", shadow: "shadow-[0_0_20px_rgba(239,68,68,0.6)]" }
    };

    const style = colorMap[color] || colorMap.green;

    // Critical state animations
    const isCritical = score > 80;
    const isWarning = score > 60;

    return (
        <div className={`db-card relative flex flex-col justify-between h-full transition-all duration-500 ${isCritical ? `animate-pulse ${style.border} ${style.shadow}` : ''}`}>

            {/* Header / Tooltip info */}
            <div className="flex justify-between items-start mb-2 relative group cursor-help">
                <div>
                    <p className="db-card-label flex items-center gap-2">
                        Crowd Safety Index
                        <span className="text-[10px] w-4 h-4 rounded-full border border-gray-500 flex items-center justify-center text-gray-400">i</span>
                    </p>
                    <p className="db-card-sub mb-0 text-xs">Aggregated spatial risk metric.</p>
                </div>

                {/* Custom Tooltip */}
                <div className="absolute top-8 left-0 w-64 p-3 bg-gray-900 border border-gray-700 text-[10px] text-gray-300 rounded shadow-xl opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none">
                    The score dynamically combines real-time density and current crowd count relative to venue capacity.
                    <div className="mt-2 text-gray-500 border-t border-gray-800 pt-1">
                        Capacity: {csi?.capacity_limit ?? 100} | Current Count: {csi?.current_count ?? 0}
                    </div>
                </div>
            </div>

            {/* Score Display */}
            <div className="flex flex-col items-center justify-center my-4">
                <div className="flex items-baseline gap-2">
                    <span className={`text-6xl font-bold font-mono tracking-tighter ${style.text}`}>
                        {score}
                    </span>
                    <span className="text-xl font-mono text-gray-500">/ 100</span>
                </div>

                <div className={`mt-2 px-4 py-1 rounded-full text-xs font-bold uppercase tracking-widest border ${style.bg} ${style.border} ${style.text}`}>
                    {level}
                </div>
            </div>

            {/* Alert / Warning Banners */}
            <div className="mt-auto">
                {score > 90 ? (
                    <div className="bg-red-500/20 border border-red-500/50 p-2 rounded text-xs text-red-200 text-center animate-pulse">
                        ⚠️ <strong>CRITICAL LOAD:</strong> Recommended exit routing active.
                    </div>
                ) : isWarning ? (
                    <div className="bg-orange-500/20 border border-orange-500/50 p-2 rounded text-xs text-orange-200 text-center">
                        ⚠️ High Density Warning
                    </div>
                ) : (
                    <div className="h-8" /> // placeholder filler
                )}
            </div>

            {/* Passive Audio trigger (render only, hidden) */}
            {isCritical && (
                <audio autoPlay loop style={{ display: 'none' }}>
                    {/* Fallback to simple generic beep if specific audio file missing */}
                    <source src="data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA" type="audio/wav" />
                </audio>
            )}
        </div>
    );
};

export default CrowdSafetyIndex;
