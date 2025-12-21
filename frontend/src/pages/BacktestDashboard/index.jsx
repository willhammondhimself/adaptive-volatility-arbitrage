import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Slider,
  TextField,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  CircularProgress,
  Chip,
  Stack,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import RefreshIcon from '@mui/icons-material/Refresh';

import useBacktestStore from '../../store/backtestStore';
import { runBacktest } from '../../api/backtestApi';
import useLiveMarket from '../../hooks/useLiveMarket';
import EquityCurveChart from '../../components/Charts/EquityCurveChart';
import MetricsPanel from '../../components/Results/MetricsPanel';
import MonteCarloPanel from '../../components/MonteCarlo/MonteCarloPanel';
import {
  LiveModeToggle,
  MarketStatusBadge,
  LiveQuoteDisplay,
} from '../../components/LiveMarket';

const ParameterInput = ({ label, value, onChange, min, max, step, unit }) => (
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
    />
  </Box>
);

const BacktestDashboard = () => {
  const {
    config,
    results,
    isLoading,
    error,
    updateConfig,
    setResults,
    setLoading,
    setError,
    reset,
  } = useBacktestStore();

  // Live market data
  const {
    isLiveMode,
    setLiveMode,
    spotPrice,
    spotChange,
    vix,
    vixChange,
    lastUpdated,
    isLoading: isLiveLoading,
    error: liveError,
    isMarketOpen,
    marketPhase,
    refresh: refreshLive,
  } = useLiveMarket('SPY');

  const [showBuyHold, setShowBuyHold] = useState(true);

  // Coordinate live mode and demo mode - they're mutually exclusive
  const handleLiveModeToggle = () => {
    const newLiveMode = !isLiveMode;
    setLiveMode(newLiveMode);
    if (newLiveMode && config.demoMode) {
      updateConfig('demoMode', false);
    }
  };

  const handleRunBacktest = async () => {
    setLoading(true);
    setError(null);

    try {
      const requestConfig = {
        data_dir: config.dataDir,
        max_days: config.maxDays,
        initial_capital: config.initialCapital,
        entry_threshold_pct: config.entryThresholdPct,
        exit_threshold_pct: config.exitThresholdPct,
        position_size_pct: config.positionSizePct,
        max_positions: config.maxPositions,
        demo_mode: config.demoMode,
        selected_years: config.demoMode ? null : config.selectedYears,
        use_bayesian_lstm: config.useBayesianLstm,
        use_impact_model: config.useImpactModel,
        use_uncertainty_sizing: config.useUncertaintySizing,
        use_leverage: config.useLeverage,
      };

      const data = await runBacktest(requestConfig);
      setResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Backtest failed');
    }
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
          Backtest Configuration
        </Typography>

        <Divider sx={{ my: 2 }} />

        {/* Capital & Days */}
        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
          General
        </Typography>

        <FormControlLabel
          control={
            <Switch
              checked={config.demoMode}
              onChange={(e) => updateConfig('demoMode', e.target.checked)}
              size="small"
              color="secondary"
            />
          }
          label={
            <Typography variant="body2" color={config.demoMode ? 'secondary' : 'textSecondary'}>
              Demo Mode (instant mock data)
            </Typography>
          }
          sx={{ mb: 1 }}
        />

        {/* Year Selector - only visible when not in Demo Mode */}
        {!config.demoMode && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" gutterBottom>
              Data Years
            </Typography>
            <Stack direction="row" spacing={0.5} sx={{ flexWrap: 'wrap', gap: 0.5 }}>
              {[2019, 2020, 2021, 2024].map((year) => (
                <Chip
                  key={year}
                  label={year}
                  size="small"
                  color={config.selectedYears?.includes(year) ? 'primary' : 'default'}
                  onClick={() => {
                    const currentYears = config.selectedYears || [];
                    const newYears = currentYears.includes(year)
                      ? currentYears.filter((y) => y !== year)
                      : [...currentYears, year];
                    if (newYears.length > 0) {
                      updateConfig('selectedYears', newYears.sort());
                    }
                  }}
                  variant={config.selectedYears?.includes(year) ? 'filled' : 'outlined'}
                  sx={{ cursor: 'pointer' }}
                />
              ))}
            </Stack>
            <Typography variant="caption" color="textSecondary" sx={{ mt: 0.5, display: 'block' }}>
              Select year(s) to load. More years = longer load time.
            </Typography>
          </Box>
        )}

        <Box sx={{ mb: 2 }}>
          <TextField
            label="Initial Capital"
            type="number"
            size="small"
            fullWidth
            value={config.initialCapital}
            onChange={(e) => updateConfig('initialCapital', parseFloat(e.target.value) || 100000)}
            InputProps={{ startAdornment: '$' }}
          />
        </Box>

        <ParameterInput
          label="Max Days"
          value={config.maxDays}
          onChange={(v) => updateConfig('maxDays', v)}
          min={10}
          max={500}
          step={10}
        />

        <Divider sx={{ my: 2 }} />

        {/* Strategy Parameters */}
        <Typography variant="subtitle2" gutterBottom>
          Strategy
        </Typography>

        <ParameterInput
          label="Entry Threshold"
          value={config.entryThresholdPct}
          onChange={(v) => updateConfig('entryThresholdPct', v)}
          min={1}
          max={20}
          step={0.5}
          unit="%"
        />

        <ParameterInput
          label="Exit Threshold"
          value={config.exitThresholdPct}
          onChange={(v) => updateConfig('exitThresholdPct', v)}
          min={0.5}
          max={10}
          step={0.5}
          unit="%"
        />

        <ParameterInput
          label="Position Size"
          value={config.positionSizePct}
          onChange={(v) => updateConfig('positionSizePct', v)}
          min={5}
          max={30}
          step={1}
          unit="%"
        />

        <ParameterInput
          label="Max Positions"
          value={config.maxPositions}
          onChange={(v) => updateConfig('maxPositions', v)}
          min={1}
          max={10}
          step={1}
        />

        <Divider sx={{ my: 2 }} />

        {/* Phase 2 Features */}
        <Typography variant="subtitle2" gutterBottom>
          Phase 2 Features
        </Typography>

        <FormControlLabel
          control={
            <Switch
              checked={config.useBayesianLstm}
              onChange={(e) => updateConfig('useBayesianLstm', e.target.checked)}
              size="small"
            />
          }
          label="Bayesian LSTM Forecaster"
        />

        <FormControlLabel
          control={
            <Switch
              checked={config.useImpactModel}
              onChange={(e) => updateConfig('useImpactModel', e.target.checked)}
              size="small"
            />
          }
          label="Impact Cost Model"
        />

        <FormControlLabel
          control={
            <Switch
              checked={config.useUncertaintySizing}
              onChange={(e) => updateConfig('useUncertaintySizing', e.target.checked)}
              size="small"
            />
          }
          label="Uncertainty Sizing"
        />

        <FormControlLabel
          control={
            <Switch
              checked={config.useLeverage}
              onChange={(e) => updateConfig('useLeverage', e.target.checked)}
              size="small"
            />
          }
          label="Leverage"
        />

        <Divider sx={{ my: 2 }} />

        {/* Actions */}
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleRunBacktest}
            disabled={isLoading}
            startIcon={isLoading ? <CircularProgress size={20} /> : <PlayArrowIcon />}
            fullWidth
          >
            {isLoading ? 'Running...' : 'Run Backtest'}
          </Button>
          <Button
            variant="outlined"
            onClick={reset}
            disabled={isLoading}
          >
            <RefreshIcon />
          </Button>
        </Box>

        {isLoading && !config.demoMode && (
          <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
            Loading {config.selectedYears?.length || 1} year(s) of data
            (~{Math.round((config.selectedYears?.length || 1) * 1.5)} min first time, cached after)...
          </Typography>
        )}
      </Paper>

      {/* Right Panel - Results */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 3 }}>
        {/* Live Market Header */}
        <Paper elevation={1} sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="h6">Market Data</Typography>
              {isLiveMode && <MarketStatusBadge marketPhase={marketPhase} isOpen={isMarketOpen} />}
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {isLiveMode && (
                <LiveQuoteDisplay
                  symbol="SPY"
                  spotPrice={spotPrice}
                  spotChange={spotChange}
                  vix={vix}
                  vixChange={vixChange}
                  lastUpdated={lastUpdated}
                  isLoading={isLiveLoading}
                  error={liveError}
                  onRefresh={refreshLive}
                />
              )}
              <LiveModeToggle isLiveMode={isLiveMode} onToggle={handleLiveModeToggle} />
            </Box>
          </Box>
        </Paper>

        {error && (
          <Alert severity="error">
            {error}
          </Alert>
        )}

        {/* Metrics Panel */}
        <Paper elevation={3}>
          <MetricsPanel
            metrics={results?.metrics}
            phase2Status={results?.phase2_status}
            computationTime={results?.computation_time_ms}
            dataRange={results?.data_range}
            isLoading={isLoading}
          />
        </Paper>

        {/* Equity Curve & Drawdown */}
        <Paper elevation={3} sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="h6">
              Equity & Drawdown
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={showBuyHold}
                  onChange={(e) => setShowBuyHold(e.target.checked)}
                  size="small"
                  color="success"
                />
              }
              label={
                <Typography variant="body2">
                  Show Buy & Hold
                </Typography>
              }
            />
          </Box>
          <EquityCurveChart
            equityCurve={results?.equity_curve}
            initialCapital={config.initialCapital}
            isLoading={isLoading}
            showBuyHold={showBuyHold}
          />
        </Paper>

        {/* Monte Carlo Analysis */}
        <Paper elevation={3}>
          <MonteCarloPanel backtestResults={results} />
        </Paper>
      </Box>
    </Box>
  );
};

export default BacktestDashboard;
