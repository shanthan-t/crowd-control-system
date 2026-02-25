import React, { useRef, useEffect } from 'react';

// Custom Canvas Hook for High-Performance Constellation Network
export const useConstellation = (canvasRef, isFocusedRef, isTypingRef, bootCompleteState, loginTransitionState, navClickPulseState) => {
    const mouseRef = useRef({ x: -1000, y: -1000 });
    const particlesRef = useRef([]);
    const animationRef = useRef(null);
    const speedMultiplierRef = useRef(1.0);

    // Boot & Login Sequence States
    const bootPhaseRef = useRef(0); // 0: Pre-init, 1: Activate, 2: Orbit, 3: Stabilize, 4: Done
    // We add a sweep trigger for the login transition
    const sweepPhaseRef = useRef(0); // 0: Idle, 1: Sweeping, 2: Finished

    const orbitCenterRef = useRef({ x: 0, y: 0 });

    // Track mouse global position for magnetic distortion
    useEffect(() => {
        const handleMouseMove = (e) => {
            mouseRef.current = { x: e.clientX, y: e.clientY };
        };
        const handleMouseLeave = () => {
            mouseRef.current = { x: -1000, y: -1000 };
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseout', handleMouseLeave);
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseout', handleMouseLeave);
        };
    }, []);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let bootStartTime = performance.now();
        bootPhaseRef.current = 1; // Start boot

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            orbitCenterRef.current = { x: canvas.width / 2, y: canvas.height / 2 };
            initParticles();
        };

        const initParticles = () => {
            const particleCount = Math.min(Math.floor((window.innerWidth * window.innerHeight) / 15000), 80);
            particlesRef.current = [];
            for (let i = 0; i < particleCount; i++) {

                const angle = Math.random() * Math.PI * 2;
                const radius = Math.random() * (Math.max(canvas.width, canvas.height) / 1.5);

                particlesRef.current.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    vx: (Math.random() - 0.5) * 0.42,
                    vy: (Math.random() - 0.5) * 0.42,
                    radius: Math.random() * 1.5 + 0.5,
                    baseAlpha: Math.random() * 0.5 + 0.2,
                    angle: angle,
                    orbitRadius: radius,
                    orbitSpeed: (Math.random() * 0.02 + 0.01) * (Math.random() > 0.5 ? 1 : -1)
                });
            }
        };

        resize();
        window.addEventListener('resize', resize);

        const render = (currentTime) => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const particles = particlesRef.current;
            const mouse = mouseRef.current;
            const center = orbitCenterRef.current;

            // --- Boot Sequence Phasing Logic ---
            const bootElapsed = (currentTime - bootStartTime) / 1000; // seconds
            let currentBootPhase = bootPhaseRef.current;

            let globalGlowIntensity = 1.0;
            let globalSizeMultiplier = 1.0;
            let targetSpeed = 1.0;

            if (bootElapsed < 1.2) {
                currentBootPhase = 1;
                globalGlowIntensity = 1 + (bootElapsed / 1.2) * 2;
                globalSizeMultiplier = 1 + (bootElapsed / 1.2) * 1.5;
                targetSpeed = 1 + (bootElapsed / 1.2) * 5;
            } else if (bootElapsed >= 1.2 && bootElapsed < 3.5) {
                currentBootPhase = 2;
                globalGlowIntensity = 3;
                globalSizeMultiplier = 2.5;
                targetSpeed = 15;
            } else if (bootElapsed >= 3.5 && bootElapsed < 4.5) {
                currentBootPhase = 3;
                const progress = (bootElapsed - 3.5) / 1.0;
                globalGlowIntensity = 3 - (progress * 2);
                globalSizeMultiplier = 2.5 - (progress * 1.5);
                targetSpeed = 15 - (progress * 14);
            } else if (bootElapsed >= 4.5) {
                if (currentBootPhase !== 4) {
                    currentBootPhase = 4;
                    if (bootCompleteState.setBootComplete) {
                        bootCompleteState.setBootComplete(true);
                    }
                }
            }

            bootPhaseRef.current = currentBootPhase;

            // --- Post-Boot Interactions ---
            let isLoginSweeping = false;

            // Handle Login Transition
            if (loginTransitionState.isTransitioning) {
                // Determine login transition elapsed time
                if (sweepPhaseRef.current === 0) {
                    sweepPhaseRef.current = currentTime; // mark sweep start time
                }
                const sweepElapsed = (currentTime - sweepPhaseRef.current) / 1000;

                if (sweepElapsed < 0.9) {
                    // Activate sweep orbital wrap for ~900ms
                    isLoginSweeping = true;
                    targetSpeed = 18;
                    globalGlowIntensity = 3.5;
                    globalSizeMultiplier = 2.0;
                } else if (sweepElapsed >= 0.9 && sweepElapsed < 1.3) {
                    // Decelerate
                    isLoginSweeping = true;
                    const decelProgress = (sweepElapsed - 0.9) / 0.4;
                    targetSpeed = 18 - (decelProgress * 17);
                    globalGlowIntensity = 3.5 - (decelProgress * 2.5);
                    globalSizeMultiplier = 2.0 - (decelProgress * 1.0);
                } else {
                    targetSpeed = 1.0;
                }
            } else {
                sweepPhaseRef.current = 0; // reset sweep phase when not transitioning
            }

            // Normal typing / nav pulse
            if (currentBootPhase === 4 && !isLoginSweeping) {
                if (navClickPulseState.isPulsing) {
                    targetSpeed = 3.0; // Quick temporary boost
                    globalGlowIntensity = 1.5;
                } else {
                    targetSpeed = isTypingRef.current ? 1.35 : 1.0;
                }
            }

            // Ease towards target speed
            if (isTypingRef.current || currentBootPhase === 2 || isLoginSweeping || navClickPulseState.isPulsing) {
                speedMultiplierRef.current += (targetSpeed - speedMultiplierRef.current) * 0.1;
            } else {
                speedMultiplierRef.current += (targetSpeed - speedMultiplierRef.current) * 0.02;
            }

            // --- Update & Draw Particles ---
            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];

                if (currentBootPhase === 2 || currentBootPhase === 3 || isLoginSweeping) {
                    // ORBITAL OVERRIDE
                    p.angle += p.orbitSpeed * (speedMultiplierRef.current * 0.05);

                    const targetX = center.x + Math.cos(p.angle) * p.orbitRadius;
                    const targetY = center.y + Math.sin(p.angle) * p.orbitRadius * 0.6;

                    p.x += (targetX - p.x) * 0.1;
                    p.y += (targetY - p.y) * 0.1;

                    p.x += p.vx;
                    p.y += p.vy;
                } else {
                    // STANDARD DRIFT
                    p.x += p.vx * speedMultiplierRef.current;
                    p.y += p.vy * speedMultiplierRef.current;
                }

                if (currentBootPhase === 1 || currentBootPhase === 4) {
                    if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                    if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
                }

                // Force magnetism off during forced orbits
                let drawX = p.x;
                let drawY = p.y;
                const magnetismRadius = 150;

                if (currentBootPhase === 4 && !isLoginSweeping) {
                    const dx = mouse.x - p.x;
                    const dy = mouse.y - p.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < magnetismRadius) {
                        const force = (magnetismRadius - dist) / magnetismRadius;
                        drawX -= (dx / dist) * force * 15;
                        drawY -= (dy / dist) * force * 15;
                    }
                }

                const focusMultiplier = isFocusedRef.current ? 1.5 : 1;
                const currentAlpha = Math.min(p.baseAlpha * focusMultiplier * globalGlowIntensity, 1);

                ctx.beginPath();
                ctx.arc(drawX, drawY, p.radius * globalSizeMultiplier, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(250, 204, 21, ${currentAlpha})`;
                ctx.fill();

                // Draw Constellation Lines
                for (let j = i + 1; j < particles.length; j++) {
                    const p2 = particles[j];

                    let drawX2 = p2.x;
                    let drawY2 = p2.y;

                    if (currentBootPhase === 4 && !isLoginSweeping) {
                        const dx2 = mouse.x - p2.x;
                        const dy2 = mouse.y - p2.y;
                        const dist2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);

                        if (dist2 < magnetismRadius) {
                            const force2 = (magnetismRadius - dist2) / magnetismRadius;
                            drawX2 -= (dx2 / dist2) * force2 * 15;
                            drawY2 -= (dy2 / dist2) * force2 * 15;
                        }
                    }

                    const lineDx = drawX - drawX2;
                    const lineDy = drawY - drawY2;
                    const lineDist = Math.sqrt(lineDx * lineDx + lineDy * lineDy);

                    const connectDistance = 180;
                    if (lineDist < connectDistance) {
                        const lineOpacity = Math.min((1 - lineDist / connectDistance) * 0.15 * focusMultiplier * globalGlowIntensity, 1);
                        ctx.beginPath();
                        ctx.moveTo(drawX, drawY);
                        ctx.lineTo(drawX2, drawY2);
                        ctx.strokeStyle = `rgba(250, 204, 21, ${lineOpacity})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }
            animationRef.current = requestAnimationFrame(render);
        };

        animationRef.current = requestAnimationFrame(render);

        return () => {
            window.removeEventListener('resize', resize);
            cancelAnimationFrame(animationRef.current);
        };
    }, [bootCompleteState, loginTransitionState, navClickPulseState]);
};

export const BackgroundCanvas = ({ bootCompleteState, loginTransitionState, isFocusedRef, isTypingRef, navClickPulseState }) => {
    const canvasRef = useRef(null);
    useConstellation(canvasRef, isFocusedRef, isTypingRef, bootCompleteState, loginTransitionState, navClickPulseState);

    return (
        <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
            {/* Slow Animated Radial Gradient Flow */}
            <div className={`absolute inset-0 animate-super-slow-gradient transition-colors duration-1000 ${loginTransitionState.isTransitioning ? 'bg-[radial-gradient(circle_at_50%_50%,rgba(20,28,45,1)_0%,rgba(5,8,15,1)_100%)]' : 'bg-[radial-gradient(circle_at_50%_50%,rgba(15,21,35,1)_0%,rgba(5,8,15,1)_100%)]'}`}></div>

            {/* High Performance Canvas Particle Constellation */}
            <canvas
                ref={canvasRef}
                className="absolute inset-0 z-0 pointer-events-auto"
            />
        </div>
    );
};
