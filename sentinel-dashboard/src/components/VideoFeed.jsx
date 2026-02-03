import React from 'react';
import { CameraOff } from 'lucide-react';

const VideoFeed = ({ frameData }) => {
    return (
        <div className="card p-0 overflow-hidden relative bg-black aspect-video flex items-center justify-center">
            {frameData ? (
                <img
                    src={`data:image/jpeg;base64,${frameData}`}
                    alt="Live Feed"
                    className="w-full h-full object-cover"
                />
            ) : (
                <div className="flex flex-col items-center text-gray-500">
                    <CameraOff size={48} className="mb-2" />
                    <p>Waiting for Live Feed...</p>
                </div>
            )}

            <div className="absolute top-4 left-4 bg-red-600 text-white text-xs px-2 py-1 rounded-sm flex items-center gap-2">
                <span className="w-2 h-2 bg-white rounded-full animate-pulse"></span>
                LIVE SENTINEL VISION
            </div>
        </div>
    );
};

export default VideoFeed;
