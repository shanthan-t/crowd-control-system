import React, { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const RoleCinematicGSAP = ({ role, onComplete }) => {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const imagesRef = useRef([]);
    const [imagesLoaded, setImagesLoaded] = useState(false);

    const totalFrames = role === 'admin' ? 212 : 240;
    const folder = role === 'admin' ? '/control_room_frames' : '/user_dashboard_frames';

    useEffect(() => {
        window.scrollTo(0, 0);

        let loadedCount = 0;
        const frames = [];
        let isCancelled = false;
        const pad = (num) => String(num).padStart(3, "0");

        const onLoad = () => {
            if (isCancelled) return;
            loadedCount++;
            if (loadedCount === totalFrames) {
                imagesRef.current = frames;
                setImagesLoaded(true);
            }
        };

        const onError = (e, path) => {
            console.error(`Failed to load frame: ${path}`);
        };

        for (let i = 1; i <= totalFrames; i++) {
            const img = new Image();
            const framePath = `${folder}/ezgif-frame-${pad(i)}.jpg`;
            img.src = framePath;
            img.onload = onLoad;
            img.onerror = (e) => onError(e, framePath);
            frames.push(img);
        }

        return () => {
            isCancelled = true;
        };
    }, [role, folder, totalFrames]);

    useEffect(() => {
        if (!imagesLoaded) return;

        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');

        const handleResize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            if (imagesRef.current.length > 0 && imagesRef.current[0]) {
                context.clearRect(0, 0, canvas.width, canvas.height);
                context.drawImage(imagesRef.current[0], 0, 0, canvas.width, canvas.height);
            }
        };

        handleResize();
        window.addEventListener('resize', handleResize);

        const imageSeq = { frame: 0 };

        const trigger = ScrollTrigger.create({
            trigger: containerRef.current,
            start: "top top",
            end: "200% top",
            pin: true,
            scrub: true,
            onUpdate: (self) => {
                if (imagesRef.current.length > 0) {
                    imageSeq.frame = Math.floor(self.progress * (totalFrames - 1));
                    const img = imagesRef.current[imageSeq.frame];
                    if (img) {
                        context.clearRect(0, 0, canvas.width, canvas.height);
                        context.drawImage(img, 0, 0, canvas.width, canvas.height);
                    }
                }
                if (self.progress >= 0.99) {
                    onComplete();
                }
            }
        });

        return () => {
            window.removeEventListener('resize', handleResize);
            if (trigger) trigger.kill();
        };

    }, [imagesLoaded, totalFrames, onComplete]);

    return (
        <div ref={containerRef} style={{ height: "300vh", backgroundColor: "black" }}>
            <div style={{ position: "sticky", top: 0, height: "100vh", overflow: "hidden" }}>
                {!imagesLoaded && (
                    <div className="absolute inset-0 flex items-center justify-center text-white font-mono text-sm tracking-widest z-50 bg-black">
                        INITIALIZING CINEMATIC SEQUENCE...
                    </div>
                )}
                <canvas
                    ref={canvasRef}
                    style={{
                        display: imagesLoaded ? 'block' : 'none',
                        width: '100vw',
                        height: '100vh',
                        objectFit: 'cover'
                    }}
                />
            </div>
        </div>
    );
};

export default RoleCinematicGSAP;
