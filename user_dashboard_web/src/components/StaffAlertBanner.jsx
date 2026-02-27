import React, { useState, useEffect, useRef } from 'react';
import { AlertTriangle, CheckCircle, Radio } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const API = '';

/**
 * StaffAlertBanner — SSE-connected dispatch alert overlay for staff/public dashboard.
 * Connects to /api/dispatch/staff-stream and shows alert when dispatch is active.
 */
const StaffAlertBanner = () => {
    const [dispatch, setDispatch] = useState(null);
    const [accepting, setAccepting] = useState(false);
    const [assigned, setAssigned] = useState(null);
    const audioRef = useRef(null);
    const sseRef = useRef(null);

    // SSE connection to staff dispatch stream
    useEffect(() => {
        const connect = () => {
            const es = new EventSource(`${API}/api/dispatch/staff-stream`);
            sseRef.current = es;

            es.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data && data.status === 'active') {
                        setDispatch(data);
                        setAssigned(null);
                        // Play alert sound
                        if (audioRef.current) {
                            audioRef.current.play().catch(() => { });
                        }
                    } else if (data && data.status === 'assigned') {
                        setAssigned(data.assigned_to);
                        setDispatch(prev => prev ? { ...prev, status: 'assigned' } : null);
                    } else {
                        // No active dispatch — clear
                        if (!assigned) {
                            setDispatch(null);
                        }
                    }
                } catch { /* ignore parse errors */ }
            };

            es.onerror = () => {
                es.close();
                // Reconnect after 3s
                setTimeout(connect, 3000);
            };
        };

        connect();
        return () => {
            if (sseRef.current) sseRef.current.close();
        };
    }, []);

    const handleAccept = async () => {
        if (!dispatch) return;
        setAccepting(true);
        try {
            // Use S1 as the staff ID for demo (first registered staff)
            await axios.post(`${API}/api/dispatch/accept`, {
                dispatch_id: dispatch.dispatch_id,
                staff_id: 'S1',
            });
            setAssigned({ id: 'S1', name: 'You' });
        } catch (err) {
            // Another staff may have accepted already
            console.error('Accept failed:', err);
        }
        setAccepting(false);
    };

    // Don't render anything if no dispatch
    if (!dispatch) return null;

    return (
        <>
            {/* Hidden alert audio */}
            <audio
                ref={audioRef}
                src="data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA"
                preload="auto"
            />

            <AnimatePresence>
                {dispatch.status === 'active' && !assigned && (
                    <motion.div
                        className="staff-alert-overlay"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <motion.div
                            className="staff-alert-modal"
                            initial={{ scale: 0.9, y: 30 }}
                            animate={{ scale: 1, y: 0 }}
                            exit={{ scale: 0.9, y: -20 }}
                            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                        >
                            <div className="staff-alert-icon">
                                <AlertTriangle size={36} />
                            </div>

                            <h2 className="staff-alert-title">Crowd Surge Alert</h2>

                            <div className="staff-alert-details">
                                <div className="staff-alert-detail-row">
                                    <span className="staff-alert-label">Room</span>
                                    <span className="staff-alert-value">{dispatch.room}</span>
                                </div>
                                <div className="staff-alert-detail-row">
                                    <span className="staff-alert-label">CSI</span>
                                    <span className="staff-alert-value staff-alert-value--danger">{dispatch.csi}</span>
                                </div>
                                <div className="staff-alert-detail-row">
                                    <span className="staff-alert-label">Status</span>
                                    <span className="staff-alert-value">
                                        <Radio size={10} className="animate-pulse" style={{ color: '#ef4444' }} />
                                        Immediate presence required
                                    </span>
                                </div>
                            </div>

                            <button
                                className="staff-alert-accept-btn"
                                onClick={handleAccept}
                                disabled={accepting}
                            >
                                {accepting ? 'Accepting…' : 'ACCEPT'}
                            </button>
                        </motion.div>
                    </motion.div>
                )}

                {assigned && (
                    <motion.div
                        className="staff-alert-assigned-banner"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                    >
                        <CheckCircle size={18} style={{ color: '#22c55e' }} />
                        <span>
                            <strong>{assigned.name === 'You' ? 'You are' : `${assigned.name} is`}</strong> assigned to <strong>{dispatch?.room || 'Room 1'}</strong>
                        </span>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
};

export default StaffAlertBanner;
