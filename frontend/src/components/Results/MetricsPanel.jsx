import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Chip,
  Divider,
  Skeleton,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import SwapVertIcon from '@mui/icons-material/SwapVert';

const MetricCard = ({ label, value, format, icon, color }) => {
  const formattedValue = (() => {
    if (value === null || value === undefined) return '-';
    switch (format) {
      case 'percent':
        return `${(value * 100).toFixed(2)}%`;
      case 'ratio':
        return value.toFixed(2);
      case 'number':
        return value.toLocaleString();
      case 'currency':
        return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      default:
        return value;
    }
  })();

  const isPositive = value > 0;
  const displayColor = color || (format === 'percent' || format === 'ratio' ? (isPositive ? 'success.main' : 'error.main') : 'text.primary');

  return (
    <Paper
      elevation={1}
      sx={{
        p: 2,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
        height: '100%',
      }}
    >
      {icon && (
        <Box sx={{ color: displayColor, mb: 1 }}>
          {icon}
        </Box>
      )}
      <Typography variant="h5" sx={{ color: displayColor, fontWeight: 600 }}>
        {formattedValue}
      </Typography>
      <Typography variant="body2" color="textSecondary">
        {label}
      </Typography>
    </Paper>
  );
};

const MetricsPanel = ({ metrics, phase2Status, computationTime, dataRange, isLoading }) => {
  if (isLoading) {
    return (
      <Box sx={{ p: 2 }}>
        <Grid container spacing={2}>
          {[1, 2, 3, 4].map((i) => (
            <Grid item xs={6} md={3} key={i}>
              <Skeleton variant="rectangular" height={100} />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  if (!metrics) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="body1" color="textSecondary">
          Run a backtest to see performance metrics
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      {/* Main metrics */}
      <Grid container spacing={2}>
        <Grid item xs={6} md={3}>
          <MetricCard
            label="Total Return"
            value={metrics.total_return}
            format="percent"
            icon={metrics.total_return >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
          />
        </Grid>
        <Grid item xs={6} md={3}>
          <MetricCard
            label="Sharpe Ratio"
            value={metrics.sharpe_ratio}
            format="ratio"
            icon={<ShowChartIcon />}
          />
        </Grid>
        <Grid item xs={6} md={3}>
          <MetricCard
            label="Max Drawdown"
            value={metrics.max_drawdown}
            format="percent"
            icon={<TrendingDownIcon />}
            color="error.main"
          />
        </Grid>
        <Grid item xs={6} md={3}>
          <MetricCard
            label="Total Trades"
            value={metrics.total_trades}
            format="number"
            icon={<SwapVertIcon />}
            color="primary.main"
          />
        </Grid>
      </Grid>

      <Divider sx={{ my: 2 }} />

      {/* Phase 2 status chips */}
      {phase2Status && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Phase 2 Features
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip
              label="Bayesian LSTM"
              size="small"
              color={phase2Status.bayesian_lstm_active ? 'success' : 'default'}
              variant={phase2Status.bayesian_lstm_active ? 'filled' : 'outlined'}
            />
            <Chip
              label="Impact Model"
              size="small"
              color={phase2Status.impact_model_active ? 'success' : 'default'}
              variant={phase2Status.impact_model_active ? 'filled' : 'outlined'}
            />
            <Chip
              label="Uncertainty Sizing"
              size="small"
              color={phase2Status.uncertainty_sizer_active ? 'success' : 'default'}
              variant={phase2Status.uncertainty_sizer_active ? 'filled' : 'outlined'}
            />
            <Chip
              label="Leverage"
              size="small"
              color={phase2Status.leverage_active ? 'success' : 'default'}
              variant={phase2Status.leverage_active ? 'filled' : 'outlined'}
            />
          </Box>
        </Box>
      )}

      {/* Metadata */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1 }}>
        {dataRange && (
          <Typography variant="caption" color="textSecondary">
            Data: {dataRange.start} to {dataRange.end}
          </Typography>
        )}
        {computationTime && (
          <Typography variant="caption" color="textSecondary">
            Computed in {(computationTime / 1000).toFixed(1)}s
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default MetricsPanel;
