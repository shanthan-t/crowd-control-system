import React, { useState } from 'react';
import { Camera, Loader } from 'lucide-react';

/**
 * LiveCameraFeed.jsx — Privacy-Focused Video Stream
 * Fetches and renders the blurred public stream exclusively.
 */
const LiveCameraFeed = ({ cameraId }) => {
    const [streamReady, setStreamReady] = useState(false);

    // Reset stream state if camera changes
    React.useEffect(() => {
        setStreamReady(false);
    }, [cameraId]);

    return (
        <div className="w-full flex-1 relative overflow-hidden rounded-lg border border-white/10 bg-black/40 shadow-[inset_0_0_40px_rgba(255,255,255,0.02)] min-h-[200px] flex items-center justify-center">
            {cameraId ? (
                <>
                    {!streamReady && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-black/50">
                            <Loader size={22} className="animate-spin text-white/40 mb-2" />
                            <span className="text-xs text-white/60 tracking-wider">Connecting...</span>
                        </div>
                    )}
                    <img
                        src={`/api/cameras/stream/public/${cameraId}`}
                        alt="Live Camera Feed"
                        className="absolute inset-0 w-full h-full object-contain"
                        onLoad={() => setStreamReady(true)}
                        onError={() => setStreamReady(false)}
                    />
                </>
            ) : (
                <div className="flex flex-col items-center justify-center opacity-40">
                    <Camera size={32} className="mb-3" />
                    <p className="text-sm font-medium tracking-wide">No Camera Selected</p>
                </div>
            )}
        </div>
    );
};

export default LiveCameraFeed;
