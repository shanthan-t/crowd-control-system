import React, { useState, useEffect, useRef } from 'react';
import LoginSignup from './components/LoginSignup';
import AdminLayout from './components/AdminLayout';
import Dashboard from './components/Dashboard';
import { BackgroundCanvas } from './components/BackgroundCanvas';

function App() {
  const [username, setUsername] = useState(null);
  const [role, setRole] = useState(null);
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
      setBootComplete(true);
    } else {
      localStorage.removeItem('sentinel_user');
      localStorage.removeItem('sentinel_role');
      localStorage.removeItem('sentinel_token');
    }
  }, []);

  const handleLogin = (user, userRole, token) => {
    setUsername(user);
    setRole(userRole);
    localStorage.setItem('sentinel_user', user);
    localStorage.setItem('sentinel_role', userRole);
    localStorage.setItem('sentinel_token', token);
  };

  const handleLogout = () => {
    setUsername(null);
    setRole(null);
    localStorage.removeItem('sentinel_user');
    localStorage.removeItem('sentinel_role');
    localStorage.removeItem('sentinel_token');
  };

  const isLoggedIn = !!(username && role);

  return (
    <div className="App min-h-screen relative overflow-hidden text-white font-sans">

      <BackgroundCanvas
        bootCompleteState={{ bootComplete, setBootComplete }}
        loginTransitionState={{ isTransitioning, setIsTransitioning }}
        navClickPulseState={{ isPulsing, setIsPulsing }}
        isFocusedRef={isFocusedRef}
        isTypingRef={isTypingRef}
      />

      {/* Login */}
      {!isLoggedIn && (
        <div className="relative z-10">
          <LoginSignup
            onLogin={(user, userRole, token) => handleLogin(user, userRole, token)}
            bootCompleteState={{ bootComplete }}
            isFocusedRef={isFocusedRef}
            isTypingRef={isTypingRef}
          />
        </div>
      )}

      {/* Admin → Control Room */}
      {isLoggedIn && role === 'admin' && (
        <div className="relative z-10">
          <AdminLayout
            username={username}
            onLogout={handleLogout}
          />
        </div>
      )}

      {/* Public → User Dashboard */}
      {isLoggedIn && role === 'user' && (
        <div className="relative z-10">
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
