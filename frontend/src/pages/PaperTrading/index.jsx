import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  TextField,
  Slider,
  Chip,
  Divider,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import RefreshIcon from '@mui/icons-material/Refresh';
import Plot from 'react-plotly.js';

import usePaperTrading from '../../hooks/usePaperTrading';

const MetricCard = ({ label, value, color = 'text.primary', prefix = '', suffix = '' }) => (
  <Paper
    elevation={2}
    sx={{
      p: 2,
      textAlign: 'center',
      minWidth: 120,
    }}
  >
    <Typography variant="caption" color="text.secondary" gutterBottom>
      {label}
    </Typography>
    <Typography variant="h5" color={color} fontWeight="bold">
      {prefix}{value}{suffix}
    </Typography>
  </Paper>
);

const ParameterInput = ({ label, value, onChange, min, max, step, unit, disabled }) => (
  <Box sx={{ mb: 2 }}>
    <Typography variant="body2" gutterBottom>
      {label}: {value}{unit || ''}
    </Typography>
    <Slider
      value={value}
      onChange={(_, v) => onChange(v)}
      min={min}
      max={max}
      step={step}
      valueLabelDisplay="auto"
      size="small"
      disabled={disabled}
    />
  </Box>
);

const PaperTrading = () => {
  const {
    config,
    updateConfig,
    status,
    trades,
    stats,
    isStarting,
    isStopping,
    error,
    isRunning,
    sessionId,
    capital,
    position,
    avgCost,
    cumulativePnl,
    tickCount,
    lastUpdate,
    startTrading,
    stopTrading,
    refresh,
  } = usePaperTrading();

  // Prepare P&L chart data
  const pnlData = trades
    .filter((t) => t.cumulative_pnl !== null)
    .reverse()
    .map((t, i) => ({
      x: i,
      pnl: t.cumulative_pnl,
      time: t.timestamp,
    }));

  // Prepare uncertainty histogram
  const uncertainties = trades.map((t) => t.uncertainty);

  return (
    <Box sx={{ display: 'flex', gap: 3, p: 3, minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* Left Panel - Configuration */}
      <Paper
        elevation={3}
        sx={{
          width: '320px',
          p: 3,
          position: 'sticky',
          top: 24,
          height: 'fit-content',
          maxHeight: 'calc(100vh - 48px)',
          overflow: 'auto',
        }}
      >
        <Typography variant="h6" gutterBottom>
          Paper Trading
        </Typography>

        <Divider sx={{ my: 2 }} />

        {/* Status */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <Chip
            label={isRunning ? 'Running' : 'Stopped'}
            color={isRunning ? 'success' : 'default'}
            size="small"
          />
          {sessionId && (
            <Typography variant="caption" color="text.secondary">
              Session #{sessionId}
            </Typography>
          )}
        </Box>

        {/* Capital */}
        <Box sx={{ mb: 2 }}>
          <TextField
            label="Initial Capital"
            type="number"
            size="small"
            fullWidth
            value={config.initialCapital}
            onChange={(e) => updateConfig('initialCapital', parseFloat(e.target.value) || 100000)}
            disabled={isRunning}
            InputProps={{ startAdornment: '$' }}
          />
        </Box>

        {/* Uncertainty Threshold */}
        <ParameterInput
          label="Uncertainty Threshold"
          value={config.uncertaintyThreshold}
          onChange={(v) => updateConfig('uncertaintyThreshold', v)}
          min={0.005}
          max={0.1}
          step={0.005}
          disabled={isRunning}
        />

        {/* Position Size */}
        <ParameterInput
          label="Position Size"
          value={config.positionPct * 100}
          onChange={(v) => updateConfig('positionPct', v / 100)}
          min={5}
          max={50}
          step={5}
          unit="%"
          disabled={isRunning}
        />

        <Divider sx={{ my: 2 }} />

        {/* Actions */}
        <Box sx={{ display: 'flex', gap: 1 }}>
          {!isRunning ? (
            <Button
              variant="contained"
              color="success"
              onClick={startTrading}
              disabled={isStarting}
              startIcon={isStarting ? <CircularProgress size={20} /> : <PlayArrowIcon />}
              fullWidth
            >
              {isStarting ? 'Starting...' : 'Start Trading'}
            </Button>
          ) : (
            <Button
              variant="contained"
              color="error"
              onClick={stopTrading}
              disabled={isStopping}
              startIcon={isStopping ? <CircularProgress size={20} /> : <StopIcon />}
              fullWidth
            >
              {isStopping ? 'Stopping...' : 'Stop Trading'}
            </Button>
          )}
          <Button variant="outlined" onClick={refresh}>
            <RefreshIcon />
          </Button>
        </Box>

        {isStarting && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Training forecaster (~5 seconds)...
          </Typography>
        )}

        {/* Current Position */}
        {isRunning && (
          <Box sx={{ mt: 3, p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              Current Position
            </Typography>
            <Typography variant="body2">
              {position > 0 ? `${position} SPY @ $${avgCost.toFixed(2)}` : 'No position'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Capital: ${capital.toLocaleString()}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Tick #{tickCount}
            </Typography>
          </Box>
        )}
      </Paper>

      {/* Right Panel - Results */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 3 }}>
        {error && (
          <Alert severity="error" onClose={() => {}}>
            {error}
          </Alert>
        )}

        {/* Metrics Row */}
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <MetricCard
            label="Total P&L"
            value={cumulativePnl.toFixed(2)}
            color={cumulativePnl >= 0 ? 'success.main' : 'error.main'}
            prefix="$"
          />
          <MetricCard label="Trades" value={stats?.total_trades ?? 0} />
          <MetricCard
            label="Win Rate"
            value={stats?.win_rate?.toFixed(1) ?? '0.0'}
            suffix="%"
          />
          <MetricCard
            label="Sharpe"
            value={stats?.sharpe_estimate?.toFixed(2) ?? '0.00'}
          />
          <MetricCard
            label="Max Drawdown"
            value={stats?.max_drawdown?.toFixed(1) ?? '0.0'}
            suffix="%"
            color="warning.main"
          />
          <MetricCard label="Skipped" value={stats?.skipped_ticks ?? 0} color="text.secondary" />
        </Box>

        {/* Charts */}
        <Box sx={{ display: 'flex', gap: 3 }}>
          {/* P&L Curve */}
          <Paper elevation={3} sx={{ flex: 1, p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Cumulative P&L
            </Typography>
            {pnlData.length > 0 ? (
              <Plot
                data={[
                  {
                    x: pnlData.map((d) => d.x),
                    y: pnlData.map((d) => d.pnl),
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#2196f3', width: 2 },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(33, 150, 243, 0.1)',
                  },
                ]}
                layout={{
                  height: 250,
                  margin: { t: 20, r: 30, b: 40, l: 60 },
                  xaxis: { title: 'Trade #' },
                  yaxis: { title: 'P&L ($)', tickprefix: '$' },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#888' },
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            ) : (
              <Box sx={{ height: 250, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography color="text.secondary">No trades yet</Typography>
              </Box>
            )}
          </Paper>

          {/* Uncertainty Distribution */}
          <Paper elevation={3} sx={{ flex: 1, p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Uncertainty Distribution
            </Typography>
            {uncertainties.length > 0 ? (
              <Plot
                data={[
                  {
                    x: uncertainties,
                    type: 'histogram',
                    marker: { color: '#9c27b0' },
                    nbinsx: 20,
                  },
                ]}
                layout={{
                  height: 250,
                  margin: { t: 20, r: 30, b: 40, l: 60 },
                  xaxis: { title: 'Epistemic Uncertainty' },
                  yaxis: { title: 'Count' },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#888' },
                  shapes: [
                    {
                      type: 'line',
                      x0: config.uncertaintyThreshold,
                      x1: config.uncertaintyThreshold,
                      y0: 0,
                      y1: 1,
                      yref: 'paper',
                      line: { color: 'red', width: 2, dash: 'dash' },
                    },
                  ],
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            ) : (
              <Box sx={{ height: 250, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography color="text.secondary">No data yet</Typography>
              </Box>
            )}
          </Paper>
        </Box>

        {/* Recent Trades Table */}
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Recent Trades
          </Typography>
          <TableContainer sx={{ maxHeight: 300 }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Time</TableCell>
                  <TableCell>Side</TableCell>
                  <TableCell align="right">Qty</TableCell>
                  <TableCell align="right">Price</TableCell>
                  <TableCell align="right">Uncertainty</TableCell>
                  <TableCell align="right">Vol Forecast</TableCell>
                  <TableCell align="right">P&L</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {trades.length > 0 ? (
                  trades.map((trade) => (
                    <TableRow key={trade.id}>
                      <TableCell>
                        {new Date(trade.timestamp).toLocaleTimeString()}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={trade.side}
                          size="small"
                          color={trade.side === 'BUY' ? 'success' : 'error'}
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell align="right">{trade.quantity}</TableCell>
                      <TableCell align="right">${trade.price.toFixed(2)}</TableCell>
                      <TableCell align="right">{trade.uncertainty.toFixed(4)}</TableCell>
                      <TableCell align="right">{(trade.forecast_vol * 100).toFixed(1)}%</TableCell>
                      <TableCell
                        align="right"
                        sx={{
                          color: trade.pnl
                            ? trade.pnl >= 0
                              ? 'success.main'
                              : 'error.main'
                            : 'text.secondary',
                        }}
                      >
                        {trade.pnl !== null ? `$${trade.pnl.toFixed(2)}` : '-'}
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={7} align="center">
                      <Typography color="text.secondary">No trades yet</Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      </Box>
    </Box>
  );
};

export default PaperTrading;
