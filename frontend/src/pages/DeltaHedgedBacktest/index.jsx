import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Slider,
  Divider,
  CircularProgress,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import Plot from 'react-plotly.js';
import { runDeltaHedgedBacktest } from '../../api/deltaHedgedApi';
import { useThemeMode } from '../../App';

const MetricCard = ({ label, value, color = 'text.primary', prefix = '', suffix = '' }) => (
  <Paper
    elevation={2}
    sx={{
      p: 2,
      textAlign: 'center',
      minWidth: 140,
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

const ParameterSlider = ({ label, value, onChange, min, max, step, unit, disabled }) => (
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

const DeltaHedgedBacktest = () => {
  const { mode } = useThemeMode();

  // Configuration state
  const [config, setConfig] = useState({
    days: 30,
    initial_spot: 450,
    initial_iv: 0.20,
    vol_of_vol: 0.5,
    mean_reversion: 2.0,
    strike: 450,
    expiry_days: 45,
    option_type: 'call',
    option_position: 10,
    rebalance_frequency: 'daily',
    delta_threshold: 0.10,
  });

  // Results state
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const updateConfig = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const runBacktest = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await runDeltaHedgedBacktest(config);
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to run backtest');
    } finally {
      setIsLoading(false);
    }
  }, [config]);

  // Chart colors
  const colors = {
    total: '#2196f3',       // Blue
    delta: '#4caf50',       // Green
    vegaGamma: '#9c27b0',   // Purple
    theta: '#ff9800',       // Orange
    costs: '#f44336',       // Red
  };

  // Prepare chart data
  const chartData = results?.attribution || [];
  const plotLayout = {
    height: 400,
    margin: { t: 40, r: 30, b: 60, l: 80 },
    xaxis: {
      title: 'Time',
      showgrid: true,
      gridcolor: mode === 'dark' ? '#333' : '#eee',
    },
    yaxis: {
      title: 'Cumulative P&L ($)',
      tickprefix: '$',
      showgrid: true,
      gridcolor: mode === 'dark' ? '#333' : '#eee',
    },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: mode === 'dark' ? '#e0e0e0' : '#212121' },
    legend: {
      orientation: 'h',
      y: -0.2,
      x: 0.5,
      xanchor: 'center',
    },
    hovermode: 'x unified',
  };

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
          Delta-Hedged Backtest
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          P&L Attribution Analysis
        </Typography>

        <Divider sx={{ my: 2 }} />

        {/* Simulation Parameters */}
        <Typography variant="subtitle2" gutterBottom>
          Simulation
        </Typography>

        <ParameterSlider
          label="Days"
          value={config.days}
          onChange={(v) => updateConfig('days', v)}
          min={10}
          max={90}
          step={5}
          disabled={isLoading}
        />

        <Box sx={{ mb: 2 }}>
          <TextField
            label="Initial Spot"
            type="number"
            size="small"
            fullWidth
            value={config.initial_spot}
            onChange={(e) => updateConfig('initial_spot', parseFloat(e.target.value) || 450)}
            disabled={isLoading}
            InputProps={{ startAdornment: '$' }}
          />
        </Box>

        <ParameterSlider
          label="Initial IV"
          value={config.initial_iv * 100}
          onChange={(v) => updateConfig('initial_iv', v / 100)}
          min={10}
          max={60}
          step={1}
          unit="%"
          disabled={isLoading}
        />

        <Divider sx={{ my: 2 }} />

        {/* Option Parameters */}
        <Typography variant="subtitle2" gutterBottom>
          Option
        </Typography>

        <Box sx={{ mb: 2 }}>
          <TextField
            label="Strike"
            type="number"
            size="small"
            fullWidth
            value={config.strike}
            onChange={(e) => updateConfig('strike', parseFloat(e.target.value) || 450)}
            disabled={isLoading}
            InputProps={{ startAdornment: '$' }}
          />
        </Box>

        <ParameterSlider
          label="Expiry Days"
          value={config.expiry_days}
          onChange={(v) => updateConfig('expiry_days', v)}
          min={20}
          max={90}
          step={5}
          disabled={isLoading}
        />

        <FormControl fullWidth size="small" sx={{ mb: 2 }}>
          <InputLabel>Option Type</InputLabel>
          <Select
            value={config.option_type}
            label="Option Type"
            onChange={(e) => updateConfig('option_type', e.target.value)}
            disabled={isLoading}
          >
            <MenuItem value="call">Call</MenuItem>
            <MenuItem value="put">Put</MenuItem>
          </Select>
        </FormControl>

        <Divider sx={{ my: 2 }} />

        {/* Hedging Parameters */}
        <Typography variant="subtitle2" gutterBottom>
          Hedging
        </Typography>

        <FormControl fullWidth size="small" sx={{ mb: 2 }}>
          <InputLabel>Rebalance Frequency</InputLabel>
          <Select
            value={config.rebalance_frequency}
            label="Rebalance Frequency"
            onChange={(e) => updateConfig('rebalance_frequency', e.target.value)}
            disabled={isLoading}
          >
            <MenuItem value="continuous">Continuous</MenuItem>
            <MenuItem value="hourly">Hourly</MenuItem>
            <MenuItem value="four_hour">4-Hour</MenuItem>
            <MenuItem value="daily">Daily</MenuItem>
          </Select>
        </FormControl>

        <ParameterSlider
          label="Delta Threshold"
          value={config.delta_threshold * 100}
          onChange={(v) => updateConfig('delta_threshold', v / 100)}
          min={1}
          max={50}
          step={1}
          unit="%"
          disabled={isLoading}
        />

        <Divider sx={{ my: 2 }} />

        {/* Run Button */}
        <Button
          variant="contained"
          color="primary"
          onClick={runBacktest}
          disabled={isLoading}
          startIcon={isLoading ? <CircularProgress size={20} /> : <PlayArrowIcon />}
          fullWidth
        >
          {isLoading ? 'Running...' : 'Run Backtest'}
        </Button>
      </Paper>

      {/* Right Panel - Results */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 3 }}>
        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Metrics Row */}
        {results && (
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <MetricCard
              label="Total P&L"
              value={results.metrics.total_pnl.toFixed(0)}
              color={results.metrics.total_pnl >= 0 ? 'success.main' : 'error.main'}
              prefix="$"
            />
            <MetricCard
              label="Vega+Gamma P&L"
              value={results.metrics.vega_gamma_pnl.toFixed(0)}
              color="secondary.main"
              prefix="$"
            />
            <MetricCard
              label="Vega+Gamma Sharpe"
              value={results.metrics.vega_gamma_sharpe.toFixed(2)}
              color={results.metrics.vega_gamma_sharpe > 1 ? 'success.main' : 'text.primary'}
            />
            <MetricCard
              label="Hedge Effectiveness"
              value={(results.metrics.hedge_effectiveness * 100).toFixed(1)}
              color={results.metrics.hedge_effectiveness > 0.9 ? 'success.main' : 'warning.main'}
              suffix="%"
            />
            <MetricCard
              label="Transaction Costs"
              value={results.metrics.transaction_costs.toFixed(0)}
              color="error.main"
              prefix="$"
            />
            <MetricCard
              label="Rebalances"
              value={results.metrics.rebalance_count}
            />
          </Box>
        )}

        {/* P&L Attribution Chart */}
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            P&L Attribution
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Delta P&L should stay flat (effective hedge), while Vega+Gamma shows the alpha source
          </Typography>

          {chartData.length > 0 ? (
            <Plot
              data={[
                {
                  x: chartData.map(d => d.timestamp),
                  y: chartData.map(d => d.total_pnl),
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Total P&L',
                  line: { color: colors.total, width: 2 },
                },
                {
                  x: chartData.map(d => d.timestamp),
                  y: chartData.map(d => d.delta_pnl),
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Delta P&L',
                  line: { color: colors.delta, width: 2 },
                },
                {
                  x: chartData.map(d => d.timestamp),
                  y: chartData.map(d => d.gamma_pnl + d.vega_pnl),
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Vega+Gamma P&L',
                  line: { color: colors.vegaGamma, width: 2 },
                },
                {
                  x: chartData.map(d => d.timestamp),
                  y: chartData.map(d => d.theta_pnl),
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Theta P&L',
                  line: { color: colors.theta, width: 2, dash: 'dot' },
                },
                {
                  x: chartData.map(d => d.timestamp),
                  y: chartData.map(d => -d.transaction_costs),
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Transaction Costs',
                  line: { color: colors.costs, width: 2, dash: 'dash' },
                },
              ]}
              layout={plotLayout}
              config={{ displayModeBar: false }}
              style={{ width: '100%' }}
            />
          ) : (
            <Box sx={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Typography color="text.secondary">
                Click "Run Backtest" to see P&L attribution
              </Typography>
            </Box>
          )}
        </Paper>

        {/* Explanation */}
        {results && (
          <Paper elevation={2} sx={{ p: 2, bgcolor: 'action.hover' }}>
            <Typography variant="subtitle2" gutterBottom>
              What This Shows
            </Typography>
            <Typography variant="body2" color="text.secondary">
              A properly delta-hedged options portfolio removes directional risk, leaving only
              volatility exposure as the alpha source. The <strong>Delta P&L</strong> line
              (green) should be flat near zero. The <strong>Vega+Gamma P&L</strong> line (purple)
              shows cumulative gains from volatility moves and gamma convexity.
              <strong> Theta</strong> (orange) shows the cost of carrying the position over time.
              When Vega+Gamma exceeds Theta + Transaction Costs, the strategy is profitable.
            </Typography>
          </Paper>
        )}
      </Box>
    </Box>
  );
};

export default DeltaHedgedBacktest;
