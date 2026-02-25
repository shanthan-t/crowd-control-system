import React from 'react';
import { Shield, Users, Clock, AlertTriangle } from 'lucide-react';

const MetricsCard = ({ title, value, type, actionIcon }) => {

  const getStatusColor = () => {
    if (type !== 'status' && type !== 'alert') return 'var(--text-primary)';
    if (value === 'SAFE' || value === 'LOW') return 'var(--status-safe)';
    if (value === 'CAUTION' || value === 'MEDIUM') return 'var(--status-caution)';
    if (value === 'DANGEROUS' || value === 'HIGH') return 'var(--status-danger)';
    return 'var(--text-primary)';
  };

  return (
    <div className="metric-block">
      <p className="metric-label">{title}</p>
      <div className="flex items-end gap-5">
        <h3
          className="metric-value"
          style={{
            color: getStatusColor(),
            textShadow: type === 'status' ? `0 0 24px ${getStatusColor()}40, 0 0 60px ${getStatusColor()}15` : 'none'
          }}
        >
          {value}
        </h3>
        {actionIcon && (
          <div className="mb-2 opacity-70">
            {actionIcon}
          </div>
        )}
      </div>
    </div>
  );
};

export default MetricsCard;
