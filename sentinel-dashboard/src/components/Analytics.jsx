import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Analytics = ({ data }) => {
    // Data expected to be array of { time: '10:00', count: 20 }
    return (
        <div className="card h-full flex flex-col">
            <h2 className="text-lg font-semibold mb-4 text-gray-800">Crowd Analytics (Live)</h2>
            <div className="flex-1 w-full min-h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                        <XAxis dataKey="time" tick={{ fontSize: 12 }} stroke="#9CA3AF" />
                        <YAxis tick={{ fontSize: 12 }} stroke="#9CA3AF" />
                        <Tooltip
                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                        />
                        <Area type="monotone" dataKey="count" stroke="#007AFF" fill="#007AFF" fillOpacity={0.1} strokeWidth={2} />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default Analytics;
