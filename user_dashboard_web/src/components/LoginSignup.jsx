import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Shield } from 'lucide-react';

const LoginSignup = ({ onLogin, bootCompleteState, isFocusedRef, isTypingRef }) => {
    const [isRegister, setIsRegister] = useState(false);
    const [form, setForm] = useState({ username: '', password: '', confirmPassword: '' });
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [loading, setLoading] = useState(false);
    const [isFocused, setIsFocused] = useState(false);

    const { bootComplete } = bootCompleteState;
    const typingTimeoutRef = React.useRef(null);

    // Sync focus ref for background particle reactivity
    useEffect(() => {
        if (isFocusedRef) isFocusedRef.current = isFocused;
    }, [isFocused, isFocusedRef]);

    const handleInputTyping = (value, field) => {
        setForm(prev => ({ ...prev, [field]: value }));
        if (isTypingRef) isTypingRef.current = true;
        if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
        typingTimeoutRef.current = setTimeout(() => {
            if (isTypingRef) isTypingRef.current = false;
        }, 600);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');
        setLoading(true);

        try {
            if (isRegister) {
                // ── Register ────────────────────────────────────
                if (form.password !== form.confirmPassword) {
                    setError('Passwords do not match.');
                    setLoading(false);
                    return;
                }
                await axios.post('http://localhost:8000/api/auth/register', {
                    username: form.username,
                    password: form.password,
                });
                setSuccess('Account created! You can now sign in.');
                setIsRegister(false);
                setForm(prev => ({ ...prev, password: '', confirmPassword: '' }));
            } else {
                // ── Login ───────────────────────────────────────
                const res = await axios.post('http://localhost:8000/api/auth/login', {
                    username: form.username,
                    password: form.password,
                });
                // Pass token + role + username to parent
                onLogin(res.data.username, res.data.role, res.data.token);
            }
        } catch (err) {
            setError(
                err.response?.data?.detail ||
                'Connection error. Please try again.'
            );
        } finally {
            setLoading(false);
        }
    };

    const toggleMode = () => {
        setIsRegister(!isRegister);
        setError('');
        setSuccess('');
        setForm({ username: '', password: '', confirmPassword: '' });
    };

    return (
        <div className="min-h-screen w-full flex flex-col lg:flex-row relative bg-transparent overflow-hidden">

            {/* Left — Branding */}
            <div className="hidden lg:flex w-1/2 relative flex-col justify-center items-center z-10">
                <div className={`z-10 flex flex-col items-center p-12 text-center gap-10 opacity-0 ${bootComplete ? 'animate-boot-reveal' : ''}`}>
                    <div className="relative glow-transition">
                        <div className={`absolute -inset-4 bg-yellow-500 blur-2xl rounded-[4rem] z-[-1] glow-transition ${isFocused ? 'animate-shield-breathe' : 'opacity-15'}`} />
                        <Shield
                            size={120}
                            className={`glow-transition ${isFocused ? 'text-yellow-400' : 'text-[var(--accent-yellow)]'}`}
                            strokeWidth={1}
                        />
                    </div>
                    <div className={`flex flex-col opacity-0 ${bootComplete ? 'animate-boot-reveal delay-100' : ''}`}>
                        <h1 className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-br from-white to-gray-500 tracking-tighter">
                            SENTINEL LIVE
                        </h1>
                    </div>
                </div>
            </div>

            {/* Right — Form Panel */}
            <div className="w-full lg:w-1/2 min-h-screen flex flex-col justify-center px-6 sm:px-12 md:px-24 xl:px-32 py-12 z-10 relative">

                {/* Mobile Logo */}
                <div className={`lg:hidden flex flex-col items-center gap-6 mb-8 opacity-0 ${bootComplete ? 'animate-boot-reveal' : ''}`}>
                    <Shield size={48} className="text-[var(--accent-yellow)]" />
                    <h1 className="text-3xl font-bold tracking-tighter">SENTINEL LIVE</h1>
                </div>

                <div className="w-full max-w-xl mx-auto flex flex-col gap-8">

                    {/* Heading */}
                    <div className={`opacity-0 ${bootComplete ? 'animate-boot-reveal' : ''}`}>
                        <h2 className="text-4xl lg:text-5xl font-bold text-white tracking-tight mb-2">
                            {isRegister ? 'Create Account' : 'Sign In'}
                        </h2>
                        <p className="text-[var(--text-secondary)] text-lg font-light">
                            {isRegister
                                ? 'Set up your credentials to get started.'
                                : 'Enter your credentials to access your account.'}
                        </p>
                    </div>

                    {/* Error — animates in */}
                    {error && (
                        <div className="login-error-alert">
                            <Shield size={17} className="flex-shrink-0" />
                            <span>{error}</span>
                        </div>
                    )}

                    {/* Success message */}
                    {success && (
                        <div className="login-success-alert">
                            <Shield size={17} className="flex-shrink-0" />
                            <span>{success}</span>
                        </div>
                    )}

                    {/* Form */}
                    <form onSubmit={handleSubmit} className="flex flex-col gap-6">

                        {/* Username */}
                        <div className={`opacity-0 ${bootComplete ? 'animate-boot-reveal delay-200' : ''}`}>
                            <label className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-widest ml-1 mb-2 block">
                                Username
                            </label>
                            <input
                                type="text"
                                className="w-full bg-[var(--bg-secondary)] border-b-2 border-transparent border-b-[var(--border-color)] text-white px-2 py-4 text-xl focus:outline-none focus:border-b-[var(--accent-yellow)] transition-colors placeholder:text-[var(--bg-tertiary)]"
                                placeholder="..."
                                value={form.username}
                                onChange={(e) => handleInputTyping(e.target.value, 'username')}
                                onFocus={() => setIsFocused(true)}
                                onBlur={() => setIsFocused(false)}
                                required
                            />
                        </div>

                        {/* Password */}
                        <div className={`opacity-0 ${bootComplete ? 'animate-boot-reveal delay-300' : ''}`}>
                            <label className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-widest ml-1 mb-2 block">
                                Password
                            </label>
                            <input
                                type="password"
                                className="w-full bg-[var(--bg-secondary)] border-b-2 border-transparent border-b-[var(--border-color)] text-white px-2 py-4 text-xl focus:outline-none focus:border-b-[var(--accent-yellow)] transition-colors placeholder:text-[var(--bg-tertiary)]"
                                placeholder="••••••••••••"
                                value={form.password}
                                onChange={(e) => handleInputTyping(e.target.value, 'password')}
                                onFocus={() => setIsFocused(true)}
                                onBlur={() => setIsFocused(false)}
                                required
                            />
                        </div>

                        {/* Confirm Password (register only) */}
                        {isRegister && (
                            <div className={`opacity-0 ${bootComplete ? 'animate-boot-reveal delay-400' : ''}`}>
                                <label className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-widest ml-1 mb-2 block">
                                    Confirm Password
                                </label>
                                <input
                                    type="password"
                                    className="w-full bg-[var(--bg-secondary)] border-b-2 border-transparent border-b-[var(--border-color)] text-white px-2 py-4 text-xl focus:outline-none focus:border-b-[var(--accent-yellow)] transition-colors placeholder:text-[var(--bg-tertiary)]"
                                    placeholder="••••••••••••"
                                    value={form.confirmPassword}
                                    onChange={(e) => handleInputTyping(e.target.value, 'confirmPassword')}
                                    onFocus={() => setIsFocused(true)}
                                    onBlur={() => setIsFocused(false)}
                                    required
                                />
                            </div>
                        )}

                        {/* Submit */}
                        <div className={`pt-2 opacity-0 ${bootComplete ? 'animate-boot-reveal delay-400' : ''}`}>
                            <button
                                type="submit"
                                className="btn-primary w-full"
                                disabled={loading}
                            >
                                {loading
                                    ? (isRegister ? 'Creating Account...' : 'Authenticating...')
                                    : (isRegister ? 'Create Account' : 'Sign In')}
                            </button>
                        </div>

                        {/* Toggle link */}
                        <div className={`text-center opacity-0 ${bootComplete ? 'animate-boot-reveal delay-400' : ''}`}>
                            <button
                                type="button"
                                className="auth-toggle-link"
                                onClick={toggleMode}
                            >
                                {isRegister
                                    ? 'Already have an account? Sign In'
                                    : "Don't have an account? Sign Up"}
                            </button>
                        </div>

                    </form>
                </div>
            </div>
        </div>
    );
};

export default LoginSignup;
