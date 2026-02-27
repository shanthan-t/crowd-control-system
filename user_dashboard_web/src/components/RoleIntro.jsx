import React, { useEffect, useRef, useState } from 'react';

const TOTAL_FRAMES_ADMIN = 212;
const TOTAL_FRAMES_PUBLIC = 240;

const RoleIntro = ({ role, onComplete }) => {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const [images, setImages] = useState([]);
    const [loaded, setLoaded] = useState(false);
    const [opacity, setOpacity] = useState(1);

    // Determine configuration based on role
    const isPublic = role === 'public';
    const folder = isPublic ? 'public' : 'admin';
    const totalFrames = isPublic ? TOTAL_FRAMES_PUBLIC : TOTAL_FRAMES_ADMIN;

    // --- Preload Strategy ---
    useEffect(() => {
        let loadedCount = 0;
        const imgArray = [];

        const preloadChunk = (start, end, callback) => {
            for (let i = start; i <= end; i++) {
                if (i > totalFrames) break;
                const img = new Image();
                const paddedNum = i.toString().padStart(4, '0');
                img.src = `/intro/${folder}/frame_${paddedNum}.jpg`;

                img.onload = () => {
                    loadedCount++;
                    imgArray[i - 1] = img;

                    if (loadedCount === Math.min(end, totalFrames)) {
                        if (callback) callback();
                    }
                };
            }
        };

        // Load first 20 frames immediately to allow fast start
        preloadChunk(1, 20, () => {
            setImages(imgArray);
            setLoaded(true);

            // Lazy load the rest in background
            preloadChunk(21, totalFrames, () => {
                setImages([...imgArray]);
            });
        });

        return () => { setImages([]); setLoaded(false); };
    }, [folder, totalFrames]);

    // --- Scroll Rendering Logic (60FPS Safe) ---
    useEffect(() => {
        if (!loaded || images.length === 0) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Initial Draw
        if (images[0]) {
            ctx.drawImage(images[0], 0, 0, canvas.width, canvas.height);
        }

        let ticking = false;

        const handleScroll = () => {
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    updateFrame();
                    ticking = false;
                });
                ticking = true;
            }
        };

        const updateFrame = () => {
            if (!containerRef.current) return;

            // Calculate scroll progress within this container
            const startNode = containerRef.current.offsetTop;
            const height = containerRef.current.scrollHeight - window.innerHeight;
            const scrollPos = window.scrollY - startNode;

            let fraction = scrollPos / height;

            // Clamp
            fraction = Math.max(0, Math.min(1, fraction));

            // Map to Frame Index
            const frameIndex = Math.min(
                totalFrames - 1,
                Math.floor(fraction * totalFrames)
            );

            // Draw current frame efficiently
            if (images[frameIndex]) {
                ctx.drawImage(images[frameIndex], 0, 0, canvas.width, canvas.height);
            }

            // End of Intro Condition (95% scroll)
            if (fraction > 0.95 && opacity === 1) {
                setOpacity(0); // Trigger fade out
                setTimeout(() => {
                    onComplete(); // Navigate to specific dashboard
                }, 800); // Wait for CSS transition
            }
        };

        window.addEventListener('scroll', handleScroll, { passive: true });
        return () => window.removeEventListener('scroll', handleScroll);
    }, [loaded, images, totalFrames, opacity, onComplete]);

    // Resize canvas neatly on window resize
    useEffect(() => {
        const resize = () => {
            const canvas = canvasRef.current;
            if (canvas) {
                // To keep ratio sharp and cover the screen like object-fit: cover
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;

                // Redraw frame 0 on resize to avoid flash
                if (images.length > 0 && images[0]) {
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(images[0], 0, 0, canvas.width, canvas.height);
                }
            }
        };
        resize();
        window.addEventListener('resize', resize);
        return () => window.removeEventListener('resize', resize);
    }, [images]);

    return (
        <div
            ref={containerRef}
            className="relative bg-black w-full"
            style={{ height: '400vh' }} // Spacer forcing scroll
        >
            <div
                className="sticky top-0 w-full h-screen overflow-hidden transition-opacity duration-700 ease-in-out"
                style={{ opacity: opacity }}
            >
                <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full object-cover"
                />

                {!loaded && (
                    <div className="absolute inset-0 flex items-center justify-center text-white/50 tracking-widest uppercase text-sm font-semibold">
                        Awaiting Sequence...
                    </div>
                )}

                {/* Subtle Overlay Graphic */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/20 pointer-events-none" />

                <div className="absolute bottom-12 left-0 w-full flex justify-center pointer-events-none animate-pulse opacity-60">
                    <p className="text-white text-xs tracking-[0.2em] font-light">SCROLL TO INITIALIZE</p>
                </div>
            </div>
        </div>
    );
};

export default RoleIntro;
