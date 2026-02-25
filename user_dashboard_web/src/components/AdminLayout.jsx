import React, { useState } from 'react';
import { Shield, User, LogOut, Home, Monitor, Map, Users } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import AdminHome from './admin/AdminHome';
import AdminLiveMonitor from './admin/AdminLiveMonitor';
import FloorPlanHeatmap from './admin/FloorPlanHeatmap';
import ManageStaff from './admin/ManageStaff';

const NAV_ITEMS = [
    { key: 'home', label: 'Home', icon: Home },
    { key: 'live', label: 'Live Monitor', icon: Monitor },
    { key: 'heatmap', label: 'Floor Plan', icon: Map },
    { key: 'staff', label: 'Manage Staff', icon: Users },
];

const SECTIONS = {
    home: AdminHome,
    live: AdminLiveMonitor,
    heatmap: FloorPlanHeatmap,
    staff: ManageStaff,
};

/* Page transition variants */
const pageVariants = {
    initial: { opacity: 0, y: 12 },
    enter: {
        opacity: 1,
        y: 0,
        transition: { duration: 0.28, ease: [0.4, 0, 0.2, 1] },
    },
    exit: {
        opacity: 0,
        y: -8,
        transition: { duration: 0.22, ease: [0.4, 0, 0.2, 1] },
    },
};

/* Pill sliding transition — weighted spring, no bounce */
const pillTransition = {
    type: 'spring',
    stiffness: 280,
    damping: 32,
    mass: 0.9,
};

/* Micro-zoom keyframes for active label */
const labelZoom = {
    scale: [1, 1.08, 1],
    transition: {
        duration: 0.24,
        ease: [0.4, 0, 0.2, 1],
        times: [0, 0.4, 1],
    },
};

/* Dropdown transition variants */
const dropdownVariants = {
    initial: { opacity: 0, scale: 0.96, y: -6 },
    animate: {
        opacity: 1,
        scale: 1,
        y: 0,
        transition: { duration: 0.24, ease: [0.4, 0, 0.2, 1] },
    },
    exit: {
        opacity: 0,
        scale: 0.96,
        y: -6,
        transition: { duration: 0.24, ease: [0.4, 0, 0.2, 1] },
    },
};

const AdminLayout = ({ username, onLogout }) => {
    const [activeSection, setActiveSection] = useState('home');
    const [profileOpen, setProfileOpen] = useState(false);

    const ActiveComponent = SECTIONS[activeSection] || AdminHome;

    return (
        <div className="db-root">

            {/* ── HEADER ──────────────────────────────────────────────── */}
            <header className="admin-header">

                {/* Left: Logo */}
                <div className="flex items-center gap-3">
                    <Shield size={22} className="text-[var(--accent-yellow)]" strokeWidth={1.5} />
                    <h1 className="text-base font-semibold tracking-widest text-white uppercase">
                        SENTINEL <span className="font-light text-[var(--text-muted)]">LIVE</span>
                    </h1>
                </div>

                {/* Center: Segmented control */}
                <nav className="seg-nav">
                    {NAV_ITEMS.map(({ key, label }) => (
                        <button
                            key={key}
                            className={`seg-nav-item ${activeSection === key ? 'seg-nav-active' : ''}`}
                            onClick={() => setActiveSection(key)}
                        >
                            {activeSection === key && (
                                <motion.div
                                    className="seg-nav-pill"
                                    layoutId="nav-pill"
                                    transition={pillTransition}
                                />
                            )}
                            {activeSection === key ? (
                                <motion.span
                                    className="seg-nav-label"
                                    key={`label-${key}`}
                                    animate={labelZoom}
                                >
                                    {label}
                                </motion.span>
                            ) : (
                                <span className="seg-nav-label">{label}</span>
                            )}
                        </button>
                    ))}
                </nav>

                {/* Right: Admin Profile */}
                <div className="flex items-center justify-end">
                    <div className="relative" style={{ zIndex: 60 }}>
                        <button
                            onClick={() => setProfileOpen(!profileOpen)}
                            className="rounded-full transition-all duration-200 outline-none focus:outline-none"
                        >
                            <div className="w-9 h-9 rounded-full bg-[rgba(255,255,255,0.04)] border border-[rgba(255,255,255,0.06)] hover:bg-[rgba(255,255,255,0.08)] ring-1 ring-[rgba(255,255,255,0.08)] ring-offset-0 flex items-center justify-center transition-colors duration-200">
                                <User size={17} className="text-[rgba(255,255,255,0.92)]" />
                            </div>
                        </button>

                        <AnimatePresence>
                            {profileOpen && (
                                <motion.div
                                    className="profile-dropdown"
                                    variants={dropdownVariants}
                                    initial="initial"
                                    animate="animate"
                                    exit="exit"
                                >
                                    <div className="px-[22px] pt-[20px] flex flex-col gap-[4px] relative z-10">
                                        <p className="text-[17px] text-[rgba(255,255,255,0.92)] font-[600] leading-none">
                                            Admin
                                        </p>
                                        <p className="text-[13px] text-[rgba(255,255,255,0.55)] font-normal leading-none capitalize">
                                            Administrator
                                        </p>
                                    </div>
                                    <div className="px-[22px] pt-[18px] pb-[20px] relative z-10">
                                        <button
                                            onClick={onLogout}
                                            className="profile-logout-btn"
                                        >
                                            <LogOut size={14} className="stroke-[2.5]" /> Terminate Session
                                        </button>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>
            </header>

            {/* ── BODY ────────────────────────────────────────────────── */}
            <div className="admin-body">
                <AnimatePresence mode="wait">
                    <motion.div
                        key={activeSection}
                        variants={pageVariants}
                        initial="initial"
                        animate="enter"
                        exit="exit"
                    >
                        {activeSection === 'home'
                            ? <ActiveComponent onNavigate={setActiveSection} />
                            : <ActiveComponent />
                        }
                    </motion.div>
                </AnimatePresence>
            </div>

        </div>
    );
};

export default AdminLayout;
