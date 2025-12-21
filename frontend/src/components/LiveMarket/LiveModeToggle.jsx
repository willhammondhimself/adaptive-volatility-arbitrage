import React from 'react';
import { Box, Switch, Typography, Tooltip } from '@mui/material';
import WifiIcon from '@mui/icons-material/Wifi';
import WifiOffIcon from '@mui/icons-material/WifiOff';

/**
 * Toggle switch for enabling/disabling live market data mode.
 */
const LiveModeToggle = ({ isLiveMode, onToggle, disabled = false }) => {
  return (
    <Tooltip title={isLiveMode ? 'Disable live data' : 'Enable live data (30s refresh)'}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          px: 1.5,
          py: 0.5,
          borderRadius: 1,
          bgcolor: isLiveMode ? 'success.dark' : 'action.hover',
          opacity: disabled ? 0.5 : 1,
        }}
      >
        {isLiveMode ? (
          <WifiIcon sx={{ fontSize: 18, color: 'success.light' }} />
        ) : (
          <WifiOffIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
        )}
        <Typography variant="caption" sx={{ fontWeight: 500 }}>
          LIVE
        </Typography>
        <Switch
          size="small"
          checked={isLiveMode}
          onChange={onToggle}
          disabled={disabled}
          sx={{
            '& .MuiSwitch-switchBase.Mui-checked': {
              color: 'success.main',
            },
            '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
              backgroundColor: 'success.main',
            },
          }}
        />
      </Box>
    </Tooltip>
  );
};

export default LiveModeToggle;
