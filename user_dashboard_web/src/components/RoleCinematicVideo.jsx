import React, { useEffect, useRef, useState } from 'react';

const RoleCinematicVideo = ({ role, onComplete }) => {
    const containerRef = useRef(null);
    const videoRef = useRef(null);
    const rafRef = useRef(null);
    const targetTimeRef = useRef(0);
    const [isVideoReady, setIsVideoReady] = useState(false);
    const isCompleteRef = useRef(false);

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const handleLoadedMetadata = () => {
            video.pause();
            setIsVideoReady(true);
        };

        video.addEventListener("loadedmetadata", handleLoadedMetadata);

        if (video.readyState >= 1) {
            handleLoadedMetadata();
        }

        return () => {
            video.removeEventListener("loadedmetadata", handleLoadedMetadata);
        };
    }, []);

    useEffect(() => {
        if (!isVideoReady) return;

        const updateVideoProgress = () => {
            if (!containerRef.current || !videoRef.current || isCompleteRef.current) return;

            const rect = containerRef.current.getBoundingClientRect();
            // Calculate total scrollable distance within the container
            const total = containerRef.current.offsetHeight - window.innerHeight;
            const scrolled = -rect.top;

            // Ensure progress is strictly between 0 and 1
            const progress = Math.max(0, Math.min(1, scrolled / total));
            targetTimeRef.current = progress * videoRef.current.duration;

            if (progress >= 0.99 && !isCompleteRef.current) {
                isCompleteRef.current = true;
                // Reset scroll position before transitioning to prevent dashboard from jumping
                window.scrollTo(0, 0);
                onComplete();
            }
        };

        const renderLoop = () => {
            if (!isCompleteRef.current && videoRef.current) {
                // Smooth easing towards target time
                const diff = targetTimeRef.current - videoRef.current.currentTime;
                if (Math.abs(diff) > 0.01) {
                    videoRef.current.currentTime += diff * 0.1;
                } else {
                    videoRef.current.currentTime = targetTimeRef.current;
                }
            }
            rafRef.current = requestAnimationFrame(renderLoop);
        };

        window.addEventListener("scroll", updateVideoProgress, { passive: true });
        updateVideoProgress(); // Initial check in case they are already scrolled

        rafRef.current = requestAnimationFrame(renderLoop);

        return () => {
            window.removeEventListener("scroll", updateVideoProgress);
            if (rafRef.current) {
                cancelAnimationFrame(rafRef.current);
            }
        };
    }, [isVideoReady, onComplete]);

    // Determine video based on role
    const videoSrc = role === 'admin' ? '/videos/admin_intro.mp4' : '/videos/user_intro.mp4';

    return (
        <div ref={containerRef} className="relative w-full bg-black" style={{ height: '200vh' }}>
            <div className="sticky top-0 w-full h-screen overflow-hidden bg-black">
                <video
                    ref={videoRef}
                    src={videoSrc}
                    className={`w-full h-full object-cover transition-opacity duration-300 ${isVideoReady ? 'opacity-100' : 'opacity-0'}`}
                    muted
                    playsInline
                    preload="auto"
                />
            </div>
        </div>
    );
};

export default RoleCinematicVideo;
