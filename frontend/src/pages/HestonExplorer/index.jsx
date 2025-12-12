import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
  Divider,
} from '@mui/material';
import { ViewInAr, GridOn, Refresh } from '@mui/icons-material';
import useHestonStore from '../../store/hestonStore';
import { useHestonPricing } from '../../hooks/useHestonPricing';
import ParameterSlider from '../../components/Controls/ParameterSlider';
import HestonHeatmap from '../../components/Charts/HestonHeatmap';
import HestonSurface3D from '../../components/Charts/HestonSurface3D';

const HestonExplorer = () => {
  const {
    params,
    spot,
    priceSurface,
    isLoading,
    viewMode,
    updateParam,
    setSpot,
    toggleViewMode,
    reset,
  } = useHestonStore();

  // Fetch price surface when parameters change
  useHestonPricing();

  const handleViewChange = (event, newView) => {
    if (newView !== null) {
      toggleViewMode();
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom fontWeight="bold">
          Heston Option Pricing Explorer
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Interactive 2D/3D visualization of option prices using the Heston stochastic volatility
          model with FFT
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
              Parameters
            </Typography>
            <Button size="small" startIcon={<Refresh />} onClick={reset}>
              Reset
            </Button>
          </Box>

          <Divider sx={{ mb: 3 }} />

          {/* Spot Price */}
          <ParameterSlider
            label="Spot Price (S)"
            value={spot}
            onChange={setSpot}
            min={50}
            max={200}
            step={1}
            description="Current stock price"
          />

          {/* Initial Variance (v0) */}
          <ParameterSlider
            label="Initial Variance (v₀)"
            value={params.v0}
            onChange={(val) => updateParam('v0', val)}
            min={0.01}
            max={0.2}
            step={0.01}
            description="Starting volatility level"
          />

          {/* Long-run Variance (theta) */}
          <ParameterSlider
            label="Long-run Variance (θ)"
            value={params.theta}
            onChange={(val) => updateParam('theta', val)}
            min={0.01}
            max={0.2}
            step={0.01}
            description="Equilibrium volatility"
          />

          {/* Mean Reversion Speed (kappa) */}
          <ParameterSlider
            label="Mean Reversion (κ)"
            value={params.kappa}
            onChange={(val) => updateParam('kappa', val)}
            min={0.1}
            max={5.0}
            step={0.1}
            description="Speed of vol mean reversion"
          />

          {/* Vol of Vol (sigma_v) */}
          <ParameterSlider
            label="Vol of Vol (σᵥ)"
            value={params.sigma_v}
            onChange={(val) => updateParam('sigma_v', val)}
            min={0.1}
            max={1.0}
            step={0.05}
            description="Volatility uncertainty"
          />

          {/* Correlation (rho) */}
          <ParameterSlider
            label="Correlation (ρ)"
            value={params.rho}
            onChange={(val) => updateParam('rho', val)}
            min={-1.0}
            max={1.0}
            step={0.05}
            description="Stock-vol correlation"
          />

          {/* Risk-free Rate (r) */}
          <ParameterSlider
            label="Risk-free Rate (r)"
            value={params.r}
            onChange={(val) => updateParam('r', val)}
            min={0.0}
            max={0.15}
            step={0.01}
            description="Annualized rate"
          />

          {/* Performance Info */}
          {priceSurface && (
            <Box sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
              <Typography variant="caption" display="block" gutterBottom>
                <strong>Computation Time:</strong> {priceSurface.computation_time_ms.toFixed(2)}ms
              </Typography>
              <Chip
                label={priceSurface.cache_hit ? 'Cache Hit' : 'Cache Miss'}
                size="small"
                color={priceSurface.cache_hit ? 'success' : 'default'}
              />
            </Box>
          )}
        </Paper>

        {/* Right Panel - Visualization */}
        <Box sx={{ flex: 1 }}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6" fontWeight="bold">
                Option Price Surface
              </Typography>
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
              <HestonHeatmap
                strikes={priceSurface?.strikes}
                maturities={priceSurface?.maturities}
                prices={priceSurface?.prices}
                params={params}
                isLoading={isLoading}
              />
            ) : (
              <HestonSurface3D
                strikes={priceSurface?.strikes}
                maturities={priceSurface?.maturities}
                prices={priceSurface?.prices}
                params={params}
                isLoading={isLoading}
              />
            )}
          </Paper>
        </Box>
      </Box>
    </Box>
  );
};

export default HestonExplorer;
