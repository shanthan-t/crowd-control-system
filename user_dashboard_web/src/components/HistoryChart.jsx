import React from 'react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';

const HistoryChart = ({ data, delay }) => {
    // Format data
    const chartData = data.map(d => {
        const date = new Date(d.timestamp);
        const timeString = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        return {
            time: timeString,
            density: d.people_count
        };
    }).reverse(); // API returns oldest first usually or newest first. We reverse to show oldest to newest left-to-right.

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="glass-panel p-3 border border-[var(--border-color)]">
                    <p className="premium-label m-0">{label}</p>
                    <p className="text-lg font-bold text-yellow m-0">
                        Density: <span className="text-white">{payload[0].value}</span>
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className={`w-full h-full ${delay || ''}`}>
            <div className="h-full w-full">
                {data.length === 0 ? (
                    <div className="flex-center h-full text-[var(--text-muted)] italic">
                        Waiting for live data...
                    </div>
                ) : (
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                            <defs>
                                <linearGradient id="colorDensity" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="var(--accent-yellow)" stopOpacity={0.25} />
                                    <stop offset="95%" stopColor="var(--accent-yellow)" stopOpacity={0.05} />
                                </linearGradient>
                                <filter id="glow">
                                    <feGaussianBlur stdDeviation="3.5" result="coloredBlur" />
                                    <feMerge>
                                        <feMergeNode in="coloredBlur" />
                                        <feMergeNode in="SourceGraphic" />
                                    </feMerge>
                                </filter>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" vertical={false} />
                            <XAxis
                                dataKey="time"
                                stroke="rgba(255,255,255,0.45)"
                                fontSize={12}
                                tickMargin={10}
                                tickLine={false}
                                axisLine={false}
                            />
                            <YAxis
                                stroke="rgba(255,255,255,0.45)"
                                fontSize={12}
                                tickLine={false}
                                axisLine={false}
                                tickFormatter={(value) => `${value}`}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Area
                                type="monotone"
                                dataKey="density"
                                stroke="var(--accent-yellow)"
                                strokeWidth={3.5}
                                fillOpacity={1}
                                fill="url(#colorDensity)"
                                activeDot={{ r: 6, strokeWidth: 0, fill: "var(--accent-yellow)" }}
                                style={{ filter: 'url(#glow)' }}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                )}
            </div>
        </div>
    );
};

export default HistoryChart;
