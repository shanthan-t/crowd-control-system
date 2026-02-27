import React, { useState, useEffect, useRef } from 'react';
import LoginSignup from './components/LoginSignup';
import AdminLayout from './components/AdminLayout';
import Dashboard from './components/Dashboard';
import RoleCinematicGSAP from './components/RoleCinematicGSAP';
import { BackgroundCanvas } from './components/BackgroundCanvas';

function App() {
  const [username, setUsername] = useState(null);
  const [role, setRole] = useState(null);
  const [loginSequencePhase, setLoginSequencePhase] = useState(0); // 0=login, 1=transition, 3=intro, 2=dashboard
  const [bootComplete, setBootComplete] = useState(false);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [isPulsing, setIsPulsing] = useState(false);

  const isFocusedRef = useRef(false);
  const isTypingRef = useRef(false);

  // Restore session on mount
  useEffect(() => {
    const storedUser = localStorage.getItem('sentinel_user');
    const storedRole = localStorage.getItem('sentinel_role');
    const storedToken = localStorage.getItem('sentinel_token');
    if (storedUser && storedRole && storedToken) {
      setUsername(storedUser);
      setRole(storedRole);
      setLoginSequencePhase(2);
      setBootComplete(true);
    } else {
      // Clear any partial/invalid session
      localStorage.removeItem('sentinel_user');
      localStorage.removeItem('sentinel_role');
      localStorage.removeItem('sentinel_token');
    }
  }, []);

  const handleLoginStart = (user, userRole, token) => {
    setLoginSequencePhase(1);
    setIsTransitioning(true);

    setTimeout(() => {
      setUsername(user);
      setRole(userRole);
      localStorage.setItem('sentinel_user', user);
      localStorage.setItem('sentinel_role', userRole);
      localStorage.setItem('sentinel_token', token);
      setIsTransitioning(false);
      setLoginSequencePhase(3); // Start Intro
    }, 1100);
  };

  const handleIntroComplete = () => {
    setLoginSequencePhase(2); // Start Dashboard
  };

  const handleLogout = () => {
    setUsername(null);
    setRole(null);
    setLoginSequencePhase(0);
    localStorage.removeItem('sentinel_user');
    localStorage.removeItem('sentinel_role');
    localStorage.removeItem('sentinel_token');
  };

  return (
    <div className="App min-h-screen relative overflow-hidden text-white font-sans">

      <BackgroundCanvas
        bootCompleteState={{ bootComplete, setBootComplete }}
        loginTransitionState={{ isTransitioning, setIsTransitioning }}
        navClickPulseState={{ isPulsing, setIsPulsing }}
        isFocusedRef={isFocusedRef}
        isTypingRef={isTypingRef}
      />

      {/* Login phase */}
      <div className={`relative z-10 transition-opacity duration-500 ${loginSequencePhase === 1 ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
        {loginSequencePhase === 0 && (
          <LoginSignup
            onLogin={(user, userRole, token) => handleLoginStart(user, userRole, token)}
            bootCompleteState={{ bootComplete }}
            isFocusedRef={isFocusedRef}
            isTypingRef={isTypingRef}
          />
        )}
      </div>

      {/* Intro Phase */}
      {loginSequencePhase === 3 && (
        <div className="relative z-[200]">
          <RoleCinematicGSAP
            role={role}
            onComplete={handleIntroComplete}
          />
        </div>
      )}

      {/* Admin dashboard */}
      {loginSequencePhase === 2 && role === 'admin' && (
        <div className="relative z-10 animate-fade-in delay-200">
          <AdminLayout
            username={username}
            onLogout={handleLogout}
          />
        </div>
      )}

      {/* Public user dashboard */}
      {loginSequencePhase === 2 && role === 'user' && (
        <div className="relative z-10 animate-fade-in delay-200">
          <Dashboard
            username={username}
            onLogout={handleLogout}
          />
        </div>
      )}

    </div>
  );
}

export default App;
