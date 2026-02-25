import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Check } from 'lucide-react';

/* ── Animation Config ────────────────────────────────────────── */
const popoverVariants = {
    hidden: { opacity: 0, scale: 0.96, y: -6 },
    visible: {
        opacity: 1,
        scale: 1,
        y: 0,
        transition: { duration: 0.22, ease: [0.4, 0, 0.2, 1] },
    },
    exit: {
        opacity: 0,
        scale: 0.96,
        y: -6,
        transition: { duration: 0.16, ease: [0.4, 0, 0.2, 1] },
    },
};

const pillTransition = {
    type: 'spring',
    stiffness: 400,
    damping: 34,
    mass: 0.8,
};

const textZoom = {
    scale: [1, 1.06, 1],
    transition: { duration: 0.2, ease: 'easeOut', times: [0, 0.45, 1] },
};

/* ── AppleSelect Component ───────────────────────────────────── */
const AppleSelect = ({
    options = [],       // [{ value, label }] or ['string', …]
    value,
    onChange,
    placeholder = 'Select…',
    disabled = false,
    id,
}) => {
    const [open, setOpen] = useState(false);
    const containerRef = useRef(null);
    const listRef = useRef(null);

    // Normalize options to { value, label }
    const normalized = options.map(o =>
        typeof o === 'string' ? { value: o, label: o } : o
    );

    const selected = normalized.find(o => o.value === value);

    // Click outside close
    useEffect(() => {
        if (!open) return;
        const handler = (e) => {
            if (containerRef.current && !containerRef.current.contains(e.target)) {
                setOpen(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [open]);

    // Keyboard support
    const handleKeyDown = (e) => {
        if (disabled) return;
        if (e.key === 'Escape') { setOpen(false); return; }
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            setOpen(o => !o);
            return;
        }
        if (!open) return;
        const idx = normalized.findIndex(o => o.value === value);
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            const next = Math.min(idx + 1, normalized.length - 1);
            onChange(normalized[next].value);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            const prev = Math.max(idx - 1, 0);
            onChange(normalized[prev].value);
        }
    };

    const handleSelect = (val) => {
        onChange(val);
        setOpen(false);
    };

    return (
        <div
            className="apple-select"
            ref={containerRef}
            tabIndex={disabled ? -1 : 0}
            onKeyDown={handleKeyDown}
            id={id}
        >
            {/* ── Trigger ─────────────────────────────────────────── */}
            <button
                type="button"
                className={`apple-select-trigger ${open ? 'apple-select-trigger--open' : ''}`}
                onClick={() => !disabled && setOpen(o => !o)}
                disabled={disabled}
            >
                <span className="apple-select-value">
                    {selected ? selected.label : placeholder}
                </span>
                <motion.span
                    className="apple-select-chevron"
                    animate={{ rotate: open ? 180 : 0 }}
                    transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
                >
                    <ChevronDown size={15} />
                </motion.span>
            </button>

            {/* ── Popover ─────────────────────────────────────────── */}
            <AnimatePresence>
                {open && (
                    <motion.div
                        className="apple-select-popover"
                        variants={popoverVariants}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                    >
                        <div className="apple-select-list" ref={listRef}>
                            {normalized.map(opt => {
                                const isActive = opt.value === value;
                                return (
                                    <button
                                        key={opt.value}
                                        type="button"
                                        className={`apple-select-option ${isActive ? 'apple-select-option--active' : ''}`}
                                        onClick={() => handleSelect(opt.value)}
                                    >
                                        {isActive && (
                                            <motion.div
                                                className="apple-select-option-pill"
                                                layoutId={`apple-pill-${id || 'default'}`}
                                                transition={pillTransition}
                                            />
                                        )}
                                        {isActive ? (
                                            <motion.span
                                                className="apple-select-option-label apple-select-option-label--active"
                                                key={`active-${opt.value}`}
                                                animate={textZoom}
                                            >
                                                {opt.label}
                                            </motion.span>
                                        ) : (
                                            <span className="apple-select-option-label">
                                                {opt.label}
                                            </span>
                                        )}
                                        {isActive && (
                                            <Check size={14} className="apple-select-check" />
                                        )}
                                    </button>
                                );
                            })}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default AppleSelect;
