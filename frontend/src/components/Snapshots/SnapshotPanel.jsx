import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Slider,
  Divider,
  CircularProgress,
  Alert,
  Chip,
  Tooltip,
} from '@mui/material';
import {
  CameraAlt,
  Delete,
  Refresh,
  PlayArrow,
  Stop,
  History,
} from '@mui/icons-material';
import { useSnapshots } from '../../hooks/useSnapshots';

const formatDate = (isoString) => {
  const date = new Date(isoString);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const SnapshotPanel = ({
  symbol,
  expiryCount,
  onSnapshotSelect,
  onLiveMode,
  isLiveMode = true,
}) => {
  const {
    snapshots,
    selectedSnapshotId,
    isLoading,
    isCapturing,
    error,
    capture,
    remove,
    select,
    clearSelection,
    refresh,
  } = useSnapshots();

  const [sliderValue, setSliderValue] = useState(0);

  const handleCapture = async () => {
    try {
      const metadata = await capture(symbol, expiryCount);
      // Optionally switch to viewing the new snapshot
      if (onSnapshotSelect) {
        onSnapshotSelect(metadata.id);
      }
    } catch (err) {
      // Error is already handled in the hook
    }
  };

  const handleDelete = async (e, snapshotId) => {
    e.stopPropagation();
    if (window.confirm('Delete this snapshot?')) {
      try {
        await remove(snapshotId);
        if (selectedSnapshotId === snapshotId && onLiveMode) {
          onLiveMode();
        }
      } catch (err) {
        // Error is already handled in the hook
      }
    }
  };

  const handleSelect = (snapshotId) => {
    select(snapshotId);
    if (onSnapshotSelect) {
      onSnapshotSelect(snapshotId);
    }
  };

  const handleLiveMode = () => {
    clearSelection();
    if (onLiveMode) {
      onLiveMode();
    }
  };

  const handleSliderChange = (event, newValue) => {
    setSliderValue(newValue);
    if (snapshots.length > 0 && newValue < snapshots.length) {
      const snapshot = snapshots[snapshots.length - 1 - newValue];
      handleSelect(snapshot.id);
    }
  };

  // Filter snapshots for current symbol
  const filteredSnapshots = snapshots.filter(
    (s) => !symbol || s.symbol === symbol.toUpperCase()
  );

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <History fontSize="small" />
          Snapshots
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Refresh list">
            <IconButton size="small" onClick={refresh} disabled={isLoading}>
              <Refresh fontSize="small" />
            </IconButton>
          </Tooltip>
          <Button
            variant="contained"
            size="small"
            startIcon={isCapturing ? <CircularProgress size={16} color="inherit" /> : <CameraAlt />}
            onClick={handleCapture}
            disabled={isCapturing || !symbol}
          >
            Capture
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ mb: 2 }}>
        <Button
          variant={isLiveMode ? 'contained' : 'outlined'}
          size="small"
          color={isLiveMode ? 'success' : 'primary'}
          startIcon={isLiveMode ? <PlayArrow /> : <Stop />}
          onClick={handleLiveMode}
          fullWidth
        >
          {isLiveMode ? 'Live Mode' : 'Return to Live'}
        </Button>
      </Box>

      <Divider sx={{ mb: 2 }} />

      {filteredSnapshots.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" color="textSecondary" gutterBottom>
            Time Navigation
          </Typography>
          <Slider
            value={sliderValue}
            onChange={handleSliderChange}
            min={0}
            max={Math.max(0, filteredSnapshots.length - 1)}
            step={1}
            marks
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => {
              if (filteredSnapshots.length > 0 && value < filteredSnapshots.length) {
                return formatDate(filteredSnapshots[filteredSnapshots.length - 1 - value].captured_at);
              }
              return '';
            }}
            disabled={filteredSnapshots.length < 2}
          />
        </Box>
      )}

      <Typography variant="caption" color="textSecondary" sx={{ mb: 1, display: 'block' }}>
        {filteredSnapshots.length} snapshot{filteredSnapshots.length !== 1 ? 's' : ''} saved
      </Typography>

      {isLoading && filteredSnapshots.length === 0 ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress size={24} />
        </Box>
      ) : filteredSnapshots.length === 0 ? (
        <Typography variant="body2" color="textSecondary" sx={{ py: 2, textAlign: 'center' }}>
          No snapshots yet. Capture one to get started.
        </Typography>
      ) : (
        <List dense sx={{ maxHeight: 300, overflow: 'auto' }}>
          {filteredSnapshots.map((snapshot) => (
            <ListItem key={snapshot.id} disablePadding>
              <ListItemButton
                selected={selectedSnapshotId === snapshot.id}
                onClick={() => handleSelect(snapshot.id)}
              >
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2">
                        {formatDate(snapshot.captured_at)}
                      </Typography>
                      <Chip
                        label={snapshot.symbol}
                        size="small"
                        variant="outlined"
                      />
                    </Box>
                  }
                  secondary={
                    <Typography variant="caption" color="textSecondary">
                      ${snapshot.underlying_price.toFixed(2)}
                      {snapshot.vix_level && ` | VIX: ${snapshot.vix_level.toFixed(2)}`}
                    </Typography>
                  }
                />
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    size="small"
                    onClick={(e) => handleDelete(e, snapshot.id)}
                  >
                    <Delete fontSize="small" />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      )}
    </Paper>
  );
};

export default SnapshotPanel;
