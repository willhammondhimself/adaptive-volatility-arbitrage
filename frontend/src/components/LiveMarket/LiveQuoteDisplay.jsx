import React from 'react';
import { Box, Typography, Skeleton, Tooltip, IconButton } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

/**
 * Display live quote data with price changes.
 */
const LiveQuoteDisplay = ({
  symbol = 'SPY',
  spotPrice,
  spotChange,
  vix,
  vixChange,
  lastUpdated,
  isLoading,
  error,
  onRefresh,
  isStale = false,
}) => {
  const formatTime = (date) => {
    if (!date) return '--:--:--';
    const d = new Date(date);
    return d.toLocaleTimeString('en-US', { hour12: false });
  };

  const formatChange = (change) => {
    if (change == null) return null;
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)}%`;
  };

  const ChangeIndicator = ({ change }) => {
    if (change == null) return null;
    const isPositive = change >= 0;
    const Icon = isPositive ? TrendingUpIcon : TrendingDownIcon;
    return (
      <Box
        component="span"
        sx={{
          display: 'inline-flex',
          alignItems: 'center',
          color: isPositive ? 'success.main' : 'error.main',
          fontSize: '0.75rem',
          ml: 0.5,
        }}
      >
        <Icon sx={{ fontSize: 14, mr: 0.25 }} />
        {formatChange(change)}
      </Box>
    );
  };

  if (error) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Typography color="error" variant="caption">
          Error loading data
        </Typography>
        <IconButton size="small" onClick={onRefresh}>
          <RefreshIcon sx={{ fontSize: 16 }} />
        </IconButton>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        opacity: isStale ? 0.6 : 1,
      }}
    >
      {/* Spot Price */}
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.5 }}>
        <Typography variant="caption" color="text.secondary">
          {symbol}
        </Typography>
        {isLoading && !spotPrice ? (
          <Skeleton width={60} height={20} />
        ) : (
          <Typography variant="body2" sx={{ fontWeight: 600, fontFamily: 'monospace' }}>
            ${spotPrice?.toFixed(2) ?? '--'}
          </Typography>
        )}
        <ChangeIndicator change={spotChange} />
      </Box>

      {/* VIX */}
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.5 }}>
        <Typography variant="caption" color="text.secondary">
          VIX
        </Typography>
        {isLoading && !vix ? (
          <Skeleton width={40} height={20} />
        ) : (
          <Typography variant="body2" sx={{ fontWeight: 600, fontFamily: 'monospace' }}>
            {vix?.toFixed(2) ?? '--'}
          </Typography>
        )}
        <ChangeIndicator change={vixChange} />
      </Box>

      {/* Last Updated */}
      <Tooltip title={isStale ? 'Data may be stale' : 'Last updated'}>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{
            fontFamily: 'monospace',
            fontSize: '0.7rem',
          }}
        >
          {formatTime(lastUpdated)}
        </Typography>
      </Tooltip>

      {/* Refresh Button */}
      <IconButton
        size="small"
        onClick={onRefresh}
        disabled={isLoading}
        sx={{ ml: -0.5 }}
      >
        <RefreshIcon
          sx={{
            fontSize: 16,
            animation: isLoading ? 'spin 1s linear infinite' : 'none',
            '@keyframes spin': {
              '0%': { transform: 'rotate(0deg)' },
              '100%': { transform: 'rotate(360deg)' },
            },
          }}
        />
      </IconButton>
    </Box>
  );
};

export default LiveQuoteDisplay;
