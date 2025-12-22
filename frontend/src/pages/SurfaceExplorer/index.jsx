import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Alert,
} from '@mui/material';
import { ViewInAr, GridOn, Refresh, History } from '@mui/icons-material';
import useSurfaceExplorerStore from '../../store/surfaceExplorerStore';
import { useSurfaceExplorer } from '../../hooks/useSurfaceExplorer';
import ParameterSlider from '../../components/Controls/ParameterSlider';
import SurfaceHeatmap from '../../components/Charts/SurfaceHeatmap';
import Surface3D from '../../components/Charts/Surface3D';
import SnapshotPanel from '../../components/Snapshots/SnapshotPanel';
import { getSnapshot } from '../../api/optionsApi';

const SurfaceExplorer = () => {
  const {
    mode,
    valueType,
    hestonParams,
    bsSigma,
    spot,
    r,
    symbol,
    expiryCount,
    surface,
    isLoading,
    viewMode,
    setMode,
    setValueType,
    updateHestonParam,
    setBsSigma,
    setSpot,
    setR,
    setSymbol,
    setExpiryCount,
    toggleViewMode,
    reset,
  } = useSurfaceExplorerStore();

  useSurfaceExplorer();

  // Snapshot state
  const [isLiveMode, setIsLiveMode] = useState(true);
  const [snapshotData, setSnapshotData] = useState(null);
  const [snapshotLoading, setSnapshotLoading] = useState(false);

  // Load snapshot data when selected
  const handleSnapshotSelect = async (snapshotId) => {
    setSnapshotLoading(true);
    try {
      const data = await getSnapshot(snapshotId);
      setSnapshotData(data);
      setIsLiveMode(false);
    } catch (err) {
      console.error('Failed to load snapshot:', err);
    } finally {
      setSnapshotLoading(false);
    }
  };

  const handleLiveMode = () => {
    setIsLiveMode(true);
    setSnapshotData(null);
  };

  // Determine which surface data to display
  const displaySurface = isLiveMode ? surface : snapshotData;
  const displayLoading = isLiveMode ? isLoading : snapshotLoading;

  const handleViewChange = (event, newView) => {
    if (newView !== null) {
      toggleViewMode();
    }
  };

  const handleModeChange = (event) => {
    setMode(event.target.value);
  };

  const handleValueTypeChange = (event) => {
    setValueType(event.target.value);
  };

  // Generate title and subtitle based on mode and value type
  const getChartTitle = () => {
    if (!isLiveMode && snapshotData) {
      return `${snapshotData.symbol} IV Surface (Snapshot)`;
    }

    const valueLabel = surface?.value_type === 'iv' ? 'Implied Volatility' : 'Price';

    if (mode === 'heston') {
      return `Heston ${valueLabel} Surface`;
    } else if (mode === 'black_scholes') {
      return `Black-Scholes ${valueLabel} Surface`;
    } else {
      return `${symbol} Live IV Surface`;
    }
  };

  const getChartSubtitle = () => {
    if (!isLiveMode && snapshotData) {
      const date = new Date(snapshotData.captured_at);
      return `Captured: ${date.toLocaleString()} | Underlying: $${snapshotData.underlying_price.toFixed(2)}${snapshotData.vix_level ? ` | VIX: ${snapshotData.vix_level.toFixed(2)}` : ''}`;
    }

    if (mode === 'heston') {
      const p = hestonParams;
      return `v₀=${p.v0}, θ=${p.theta}, κ=${p.kappa}, σᵥ=${p.sigma_v}, ρ=${p.rho}`;
    } else if (mode === 'black_scholes') {
      return `σ=${bsSigma}, S=${spot}, r=${r}`;
    } else {
      return surface?.underlying_price
        ? `Underlying: $${surface.underlying_price.toFixed(2)}`
        : '';
    }
  };

  const getZLabel = () => {
    if (!isLiveMode && snapshotData) {
      return 'IV';
    }
    return displaySurface?.value_type === 'iv' ? 'IV' : 'Price ($)';
  };

  const getColorscale = () => {
    if (!isLiveMode && snapshotData) {
      return 'RdYlGn';
    }
    return displaySurface?.value_type === 'iv' ? 'RdYlGn' : 'Viridis';
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">
          Option Surface Explorer
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Compare pricing models and visualize price or implied volatility surfaces
        </Typography>
      </Box>

      <Box sx={{ display: 'flex', gap: 3 }}>
        {/* Left Panel - Parameters */}
        <Paper
          sx={{
            width: '350px',
            p: 3,
            height: 'fit-content',
            position: 'sticky',
            top: 20,
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" fontWeight="bold">
              Configuration
            </Typography>
            <Button size="small" startIcon={<Refresh />} onClick={reset}>
              Reset
            </Button>
          </Box>

          <Divider sx={{ mb: 3 }} />

          {/* Mode Selection */}
          <FormControl fullWidth sx={{ mb: 3 }}>
            <InputLabel>Pricing Model</InputLabel>
            <Select
              value={mode}
              label="Pricing Model"
              onChange={handleModeChange}
            >
              <MenuItem value="black_scholes">Black-Scholes</MenuItem>
              <MenuItem value="heston">Heston</MenuItem>
              <MenuItem value="market_iv">Live Market IV</MenuItem>
            </Select>
          </FormControl>

          {/* Value Type Selection (not for market IV) */}
          {mode !== 'market_iv' && (
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Surface Type</InputLabel>
              <Select
                value={valueType}
                label="Surface Type"
                onChange={handleValueTypeChange}
              >
                <MenuItem value="price">Price Surface</MenuItem>
                <MenuItem value="iv">IV Surface</MenuItem>
              </Select>
            </FormControl>
          )}

          <Divider sx={{ mb: 3 }} />

          {/* Market IV Parameters */}
          {mode === 'market_iv' && (
            <>
              <TextField
                fullWidth
                label="Symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                sx={{ mb: 3 }}
              />
              <ParameterSlider
                label="Expiry Count"
                value={expiryCount}
                onChange={setExpiryCount}
                min={1}
                max={10}
                step={1}
                description="Number of expirations to include"
              />
            </>
          )}

          {/* BS Parameters */}
          {mode === 'black_scholes' && (
            <>
              <ParameterSlider
                label="Spot Price (S)"
                value={spot}
                onChange={setSpot}
                min={50}
                max={500}
                step={1}
                description="Current stock price"
              />
              <ParameterSlider
                label="Volatility (σ)"
                value={bsSigma}
                onChange={setBsSigma}
                min={0.05}
                max={1.0}
                step={0.01}
                description="Annualized volatility"
              />
              <ParameterSlider
                label="Risk-free Rate (r)"
                value={r}
                onChange={setR}
                min={0.0}
                max={0.15}
                step={0.01}
                description="Annualized rate"
              />
            </>
          )}

          {/* Heston Parameters */}
          {mode === 'heston' && (
            <>
              <ParameterSlider
                label="Spot Price (S)"
                value={spot}
                onChange={setSpot}
                min={50}
                max={500}
                step={1}
                description="Current stock price"
              />
              <ParameterSlider
                label="Initial Variance (v₀)"
                value={hestonParams.v0}
                onChange={(val) => updateHestonParam('v0', val)}
                min={0.01}
                max={0.2}
                step={0.01}
                description="Starting volatility level"
              />
              <ParameterSlider
                label="Long-run Variance (θ)"
                value={hestonParams.theta}
                onChange={(val) => updateHestonParam('theta', val)}
                min={0.01}
                max={0.2}
                step={0.01}
                description="Equilibrium volatility"
              />
              <ParameterSlider
                label="Mean Reversion (κ)"
                value={hestonParams.kappa}
                onChange={(val) => updateHestonParam('kappa', val)}
                min={0.1}
                max={5.0}
                step={0.1}
                description="Speed of vol mean reversion"
              />
              <ParameterSlider
                label="Vol of Vol (σᵥ)"
                value={hestonParams.sigma_v}
                onChange={(val) => updateHestonParam('sigma_v', val)}
                min={0.1}
                max={1.0}
                step={0.05}
                description="Volatility uncertainty"
              />
              <ParameterSlider
                label="Correlation (ρ)"
                value={hestonParams.rho}
                onChange={(val) => updateHestonParam('rho', val)}
                min={-1.0}
                max={1.0}
                step={0.05}
                description="Stock-vol correlation"
              />
              <ParameterSlider
                label="Risk-free Rate (r)"
                value={r}
                onChange={setR}
                min={0.0}
                max={0.15}
                step={0.01}
                description="Annualized rate"
              />
            </>
          )}

          {/* Performance Info */}
          {surface && (
            <Box sx={{ mt: 3, p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
              <Typography variant="caption" display="block" gutterBottom>
                <strong>Computation Time:</strong> {surface.computation_time_ms.toFixed(2)}ms
              </Typography>
              <Typography variant="caption" display="block" gutterBottom>
                <strong>Mode:</strong> {surface.mode}
              </Typography>
              <Typography variant="caption" display="block" gutterBottom>
                <strong>Value Type:</strong> {surface.value_type}
              </Typography>
              {surface.underlying_price && (
                <Typography variant="caption" display="block" gutterBottom>
                  <strong>Underlying:</strong> ${surface.underlying_price.toFixed(2)}
                </Typography>
              )}
              {surface.cache_hit !== undefined && (
                <Chip
                  label={surface.cache_hit ? 'Cache Hit' : 'Cache Miss'}
                  size="small"
                  color={surface.cache_hit ? 'success' : 'default'}
                />
              )}
            </Box>
          )}
        </Paper>

        {/* Center Panel - Visualization */}
        <Box sx={{ flex: 1 }}>
          {!isLiveMode && (
            <Alert
              severity="info"
              sx={{ mb: 2 }}
              icon={<History />}
              action={
                <Button color="inherit" size="small" onClick={handleLiveMode}>
                  Return to Live
                </Button>
              }
            >
              Viewing snapshot from {snapshotData && new Date(snapshotData.captured_at).toLocaleString()}
            </Alert>
          )}
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Box>
                <Typography variant="h6" fontWeight="bold">
                  {getChartTitle()}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  {getChartSubtitle()}
                </Typography>
              </Box>
              <ToggleButtonGroup
                value={viewMode}
                exclusive
                onChange={handleViewChange}
                size="small"
              >
                <ToggleButton value="2d">
                  <GridOn sx={{ mr: 1 }} />
                  2D Heatmap
                </ToggleButton>
                <ToggleButton value="3d">
                  <ViewInAr sx={{ mr: 1 }} />
                  3D Surface
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>

            {viewMode === '2d' ? (
              <SurfaceHeatmap
                strikes={displaySurface?.strikes}
                maturities={displaySurface?.maturities}
                values={displaySurface?.values}
                title={getChartTitle()}
                subtitle={getChartSubtitle()}
                zLabel={getZLabel()}
                colorscale={getColorscale()}
                isLoading={displayLoading}
              />
            ) : (
              <Surface3D
                strikes={displaySurface?.strikes}
                maturities={displaySurface?.maturities}
                values={displaySurface?.values}
                title={getChartTitle()}
                subtitle={getChartSubtitle()}
                zLabel={getZLabel()}
                colorscale={getColorscale()}
                isLoading={displayLoading}
              />
            )}
          </Paper>
        </Box>

        {/* Right Panel - Snapshots (only for market_iv mode) */}
        {mode === 'market_iv' && (
          <Box sx={{ width: '300px' }}>
            <SnapshotPanel
              symbol={symbol}
              expiryCount={expiryCount}
              onSnapshotSelect={handleSnapshotSelect}
              onLiveMode={handleLiveMode}
              isLiveMode={isLiveMode}
            />
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default SurfaceExplorer;
