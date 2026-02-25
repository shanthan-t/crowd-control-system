import React from 'react';

const RadarSweep = () => {
    return (
        <div className="relative w-16 h-16 rounded-full border border-green-500/30 overflow-hidden flex items-center justify-center bg-green-900/10 shadow-[0_0_15px_rgba(34,197,94,0.15)]">
            {/* Grid lines */}
            <div className="absolute inset-0 border-[0.5px] border-green-500/20 rounded-full"></div>
            <div className="absolute w-full h-[0.5px] bg-green-500/20 top-1/2 -translate-y-1/2"></div>
            <div className="absolute h-full w-[0.5px] bg-green-500/20 left-1/2 -translate-x-1/2"></div>

            {/* Inner rings */}
            <div className="absolute w-10 h-10 border-[0.5px] border-green-500/20 rounded-full"></div>
            <div className="absolute w-4 h-4 border-[0.5px] border-green-500/20 rounded-full block bg-green-500/10"></div>

            {/* Sweep */}
            <div className="absolute origin-bottom-right bottom-1/2 right-1/2 w-8 h-8 bg-gradient-to-tl from-green-500/40 to-transparent animate-[spin_2s_linear_infinite] rounded-tl-full shadow-[0_0_10px_rgba(34,197,94,0.4)] z-10" style={{ transformOrigin: 'bottom right' }}></div>

            {/* Blips */}
            <div className="absolute top-3 left-4 w-1.5 h-1.5 bg-green-400 rounded-full animate-ping" style={{ animationDuration: '2s', animationDelay: '0.2s' }}></div>
            <div className="absolute bottom-4 right-3 w-1 h-1 bg-green-300 rounded-full animate-ping" style={{ animationDuration: '2s', animationDelay: '1.4s' }}></div>
        </div>
    );
};

export default RadarSweep;
