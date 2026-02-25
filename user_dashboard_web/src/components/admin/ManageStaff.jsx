import React, { useState } from 'react';
import { Trash2, Eye, EyeOff, Plus, Minus, Rocket } from 'lucide-react';
import AppleSelect from '../AppleSelect';

const ZONES = ['Zone A — Main Gate', 'Zone B — East Wing', 'Zone C — West Wing', 'Zone D — Food Court', 'Zone E — Parking'];

const ManageStaff = () => {
    // ── Staff list state ──────────────────────────────────────────────
    const [staff, setStaff] = useState([
        { id: 1, name: 'Rajan Sharma', age: 32, phone: '+91 98765 43210', zone: 'Zone A — Main Gate' },
        { id: 2, name: 'Priya Nair', age: 28, phone: '+91 87654 32109', zone: 'Zone B — East Wing' },
        { id: 3, name: 'Vikram Singh', age: 35, phone: '+91 76543 21098', zone: 'Zone C — West Wing' },
    ]);

    // ── Add staff form ────────────────────────────────────────────────
    const [newName, setNewName] = useState('');
    const [newAge, setNewAge] = useState(25);
    const [newPhone, setNewPhone] = useState('');
    const [newZone, setNewZone] = useState(ZONES[0]);

    const handleAddStaff = () => {
        if (!newName.trim() || !newPhone.trim()) return;
        setStaff(prev => [
            ...prev,
            {
                id: Date.now(),
                name: newName.trim(),
                age: newAge,
                phone: newPhone.startsWith('+91') ? newPhone : `+91 ${newPhone}`,
                zone: newZone,
            },
        ]);
        setNewName('');
        setNewAge(25);
        setNewPhone('');
    };

    const handleDelete = (id) => setStaff(prev => prev.filter(s => s.id !== id));

    // ── Deploy state ──────────────────────────────────────────────────
    const [deployStaff, setDeployStaff] = useState('');
    const [deployZone, setDeployZone] = useState(ZONES[0]);
    const [twilioSid, setTwilioSid] = useState('');
    const [twilioToken, setTwilioToken] = useState('');
    const [twilioPhone, setTwilioPhone] = useState('');
    const [showSid, setShowSid] = useState(false);
    const [showToken, setShowToken] = useState(false);

    const handleDeploy = () => {
        if (!deployStaff) return;
        alert(`Deployed ${deployStaff} to ${deployZone}. SMS notification sent.`);
    };

    return (
        <div className="admin-content-area">

            {/* ── Page Header ──────────────────────────────────────── */}
            <div className="ms-page-header">
                <h2 className="ms-page-title">Staff Management</h2>
                <p className="ms-page-subtitle">
                    Manage ground staff details and deployment.
                </p>
            </div>

            {/* ── Two-Column Grid ──────────────────────────────────── */}
            <div className="ms-grid">

                {/* LEFT — Staff Management */}
                <div className="ms-card">

                    {/* Inner split: Add Form + Table */}
                    <div className="ms-inner-grid">

                        {/* Add New Staff Form */}
                        <div className="ms-form-section">
                            <h3 className="ms-section-title">Add New Staff</h3>

                            <div className="ms-field">
                                <label className="ms-label">Name</label>
                                <input
                                    className="ms-input"
                                    type="text"
                                    placeholder="Full name"
                                    value={newName}
                                    onChange={e => setNewName(e.target.value)}
                                />
                            </div>

                            <div className="ms-field">
                                <label className="ms-label">Age</label>
                                <div className="ms-age-control">
                                    <button
                                        className="ms-age-btn"
                                        onClick={() => setNewAge(a => Math.max(18, a - 1))}
                                    >
                                        <Minus size={14} />
                                    </button>
                                    <span className="ms-age-value">{newAge}</span>
                                    <button
                                        className="ms-age-btn"
                                        onClick={() => setNewAge(a => Math.min(65, a + 1))}
                                    >
                                        <Plus size={14} />
                                    </button>
                                </div>
                            </div>

                            <div className="ms-field">
                                <label className="ms-label">Phone Number</label>
                                <input
                                    className="ms-input"
                                    type="tel"
                                    placeholder="+91 XXXXX XXXXX"
                                    value={newPhone}
                                    onChange={e => setNewPhone(e.target.value)}
                                />
                            </div>

                            <div className="ms-field">
                                <label className="ms-label">Assign Initial Zone</label>
                                <AppleSelect
                                    id="add-staff-zone"
                                    options={ZONES}
                                    value={newZone}
                                    onChange={setNewZone}
                                />
                            </div>

                            <button className="ms-btn ms-btn-primary" onClick={handleAddStaff}>
                                Register Staff
                            </button>
                        </div>

                        {/* Current Ground Staff Table */}
                        <div className="ms-table-section">
                            <h3 className="ms-section-title">Current Ground Staff</h3>

                            <div className="ms-table-wrap">
                                <table className="ms-table">
                                    <thead>
                                        <tr>
                                            {['Name', 'Age', 'Phone', 'Zone', ''].map(h => (
                                                <th key={h}>{h}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {staff.length === 0 ? (
                                            <tr>
                                                <td colSpan={5} className="ms-table-empty">
                                                    No staff registered yet.
                                                </td>
                                            </tr>
                                        ) : (
                                            staff.map(s => (
                                                <tr key={s.id}>
                                                    <td className="text-white font-medium">{s.name}</td>
                                                    <td>{s.age}</td>
                                                    <td className="font-mono text-xs">{s.phone}</td>
                                                    <td className="text-xs">{s.zone.split(' — ')[0]}</td>
                                                    <td>
                                                        <button
                                                            className="ms-delete-btn"
                                                            onClick={() => handleDelete(s.id)}
                                                            title="Remove staff"
                                                        >
                                                            <Trash2 size={14} />
                                                        </button>
                                                    </td>
                                                </tr>
                                            ))
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                    </div>
                </div>

                {/* RIGHT — Deploy Staff */}
                <div className="ms-card">
                    <h3 className="ms-section-title">
                        <Rocket size={16} className="text-[#0a84ff]" />
                        Deploy Staff
                    </h3>

                    <div className="ms-field">
                        <label className="ms-label">Select Staff</label>
                        <AppleSelect
                            id="deploy-staff"
                            options={[
                                { value: '', label: '— Select —' },
                                ...staff.map(s => ({ value: s.name, label: s.name }))
                            ]}
                            value={deployStaff}
                            onChange={setDeployStaff}
                            placeholder="— Select —"
                        />
                    </div>

                    <div className="ms-field">
                        <label className="ms-label">Deploy To</label>
                        <AppleSelect
                            id="deploy-zone"
                            options={ZONES}
                            value={deployZone}
                            onChange={setDeployZone}
                        />
                    </div>

                    {/* Twilio Credentials */}
                    <div className="ms-divider" />

                    <p className="ms-cred-label">Twilio Credentials for SMS</p>

                    <div className="ms-field">
                        <label className="ms-label">Account SID</label>
                        <div className="ms-password-wrap">
                            <input
                                className="ms-input ms-input-secret"
                                type={showSid ? 'text' : 'password'}
                                placeholder="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                                value={twilioSid}
                                onChange={e => setTwilioSid(e.target.value)}
                            />
                            <button
                                className="ms-eye-btn"
                                onClick={() => setShowSid(v => !v)}
                                type="button"
                            >
                                {showSid ? <EyeOff size={14} /> : <Eye size={14} />}
                            </button>
                        </div>
                    </div>

                    <div className="ms-field">
                        <label className="ms-label">Auth Token</label>
                        <div className="ms-password-wrap">
                            <input
                                className="ms-input ms-input-secret"
                                type={showToken ? 'text' : 'password'}
                                placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                                value={twilioToken}
                                onChange={e => setTwilioToken(e.target.value)}
                            />
                            <button
                                className="ms-eye-btn"
                                onClick={() => setShowToken(v => !v)}
                                type="button"
                            >
                                {showToken ? <EyeOff size={14} /> : <Eye size={14} />}
                            </button>
                        </div>
                    </div>

                    <div className="ms-field">
                        <label className="ms-label">Twilio Phone Number</label>
                        <input
                            className="ms-input"
                            type="tel"
                            placeholder="+1 XXX XXX XXXX"
                            value={twilioPhone}
                            onChange={e => setTwilioPhone(e.target.value)}
                        />
                    </div>

                    <button className="ms-btn ms-btn-deploy" onClick={handleDeploy}>
                        <Rocket size={15} />
                        Deploy &amp; Notify
                    </button>
                </div>

            </div>
        </div>
    );
};

export default ManageStaff;
