import React from 'react';
import { Chip, Tooltip } from '@mui/material';

const PHASE_CONFIG = {
  regular: { label: 'Market Open', color: 'success' },
  pre: { label: 'Pre-Market', color: 'warning' },
  after: { label: 'After Hours', color: 'warning' },
  closed: { label: 'Closed', color: 'default' },
  unknown: { label: 'Unknown', color: 'default' },
};

/**
 * Badge showing current market session status.
 */
const MarketStatusBadge = ({ marketPhase = 'unknown', isOpen = false }) => {
  const config = PHASE_CONFIG[marketPhase] || PHASE_CONFIG.unknown;

  const tooltip = isOpen
    ? 'US equity market is open for regular trading'
    : marketPhase === 'pre'
    ? 'Pre-market session (4:00 AM - 9:30 AM ET)'
    : marketPhase === 'after'
    ? 'After-hours session (4:00 PM - 8:00 PM ET)'
    : 'Market closed';

  return (
    <Tooltip title={tooltip}>
      <Chip
        label={config.label}
        color={config.color}
        size="small"
        variant={isOpen ? 'filled' : 'outlined'}
        sx={{ fontWeight: 500 }}
      />
    </Tooltip>
  );
};

export default MarketStatusBadge;
