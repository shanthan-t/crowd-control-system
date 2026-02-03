import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';
import RiskCard from './components/RiskCard';
import VideoFeed from './components/VideoFeed';
import Analytics from './components/Analytics';
import { LayoutDashboard, Video, Activity } from 'lucide-react';

// Connect to Python Backend
const socket = io('http://localhost:5000');

function App() {
  const [data, setData] = useState({
    risk_level: 'LOW',
    people_count: 0,
    audio_status: 'NORMAL'
  });

  const [frame, setFrame] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    socket.on('connect', () => {
      console.log("Connected to Sentinel Backend");
    });

    socket.on('sentinel_update', (newData) => {
      setData(newData);

      // Update history for charts
      setHistory(prev => {
        const newEntry = { time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }), count: newData.people_count };
        const newHist = [...prev, newEntry];
        if (newHist.length > 20) newHist.shift(); // Keep last 20 points
        return newHist;
      });
    });

    socket.on('video_frame', (base64Frame) => {
      setFrame(base64Frame);
    });

    return () => {
      socket.off('connect');
      socket.off('sentinel_update');
      socket.off('video_frame');
    };
  }, []);

  return (
    <div className="min-h-screen bg-sentinel-bg p-8">
      {/* Header */}
      <header className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 tracking-tight flex items-center gap-2">
            <LayoutDashboard className="text-sentinel-accent" />
            Sentinel Dashboard
          </h1>
          <p className="text-gray-500 mt-1">Integrated Crowd Safety & Risk Monitoring</p>
        </div>
        <div className="flex gap-4">
          {/* Connection Status Indicator */}
          <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-full border border-gray-200 shadow-sm">
            <span className={`w-2 h-2 rounded-full ${socket.connected ? 'bg-green-500' : 'bg-red-500 animate-pulse'}`}></span>
            <span className="text-sm font-medium text-gray-600">{socket.connected ? 'System Online' : 'Connecting...'}</span>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-160px)]">

        {/* Left Col: Video & Quick Stats */}
        <div className="lg:col-span-2 flex flex-col gap-6 h-full">
          {/* Video Section */}
          <div className="flex-1 min-h-[400px]">
            <VideoFeed frameData={frame} />
          </div>

          {/* Analytics Section */}
          <div className="h-1/3">
            <Analytics data={history} />
          </div>
        </div>

        {/* Right Col: Risk Status */}
        <div className="lg:col-span-1 h-full">
          <RiskCard
            risk={data.risk_level}
            peopleCount={data.people_count}
            audioStatus={data.audio_status}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
