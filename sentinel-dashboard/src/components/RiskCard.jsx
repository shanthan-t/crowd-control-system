import React from 'react';
import { AlertTriangle, ShieldCheck, ShieldAlert } from 'lucide-react';
import clsx from 'clsx';

const RiskCard = ({ risk, peopleCount, audioStatus }) => {
    const isHigh = risk === 'HIGH';
    const isMed = risk === 'MEDIUM';
    const isLow = risk === 'LOW';

    return (
        <div className={clsx(
            "card flex flex-col justify-between h-full transition-all duration-500",
            isHigh ? "border-red-500 bg-red-50/50" : isMed ? "border-yellow-500 bg-yellow-50/50" : "border-green-500 bg-green-50/50"
        )}>
            <div className="flex justify-between items-start">
                <div>
                    <h2 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Current Status</h2>
                    <div className="flex items-center gap-2 mt-1">
                        <h1 className={clsx("text-4xl font-bold tracking-tight",
                            isHigh ? "text-red-600" : isMed ? "text-yellow-600" : "text-green-600"
                        )}>
                            {risk || "INITIALIZING..."}
                        </h1>
                    </div>
                </div>
                <div className={clsx("p-3 rounded-full",
                    isHigh ? "bg-red-100 text-red-600" : isMed ? "bg-yellow-100 text-yellow-600" : "bg-green-100 text-green-600"
                )}>
                    {isHigh ? <ShieldAlert size={32} /> : isMed ? <AlertTriangle size={32} /> : <ShieldCheck size={32} />}
                </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mt-6">
                <div className="p-3 bg-white/60 rounded-lg">
                    <p className="text-xs text-gray-500">People Count</p>
                    <p className="text-2xl font-semibold text-gray-800">{peopleCount}</p>
                </div>
                <div className="p-3 bg-white/60 rounded-lg">
                    <p className="text-xs text-gray-500">Audio Status</p>
                    <p className={clsx("text-xl font-semibold", audioStatus === 'PANIC' ? "text-red-500 animate-pulse" : "text-gray-800")}>
                        {audioStatus}
                    </p>
                </div>
            </div>
        </div>
    );
};

export default RiskCard;
