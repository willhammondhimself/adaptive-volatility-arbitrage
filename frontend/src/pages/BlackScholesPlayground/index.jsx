import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Slider,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
  TextField,
} from '@mui/material';
import useBSPlaygroundStore from '../../store/bsPlaygroundStore';
import { useBSPricing } from '../../hooks/useBSPricing';
import GreeksDisplay from '../../components/Greeks/GreeksDisplay';
import PnLHeatmap from '../../components/Charts/PnLHeatmap';

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

const BlackScholesPlayground = () => {
  const {
    S,
    K,
    T,
    r,
    sigma,
    optionType,
    spotRange,
    volRange,
    pricing,
    isLoading,
    heatmapData,
    isHeatmapLoading,
    setS,
    setK,
    setT,
    setR,
    setSigma,
    setOptionType,
    setSpotRange,
    setVolRange,
  } = useBSPlaygroundStore();

  // Hook to fetch pricing and heatmap
  useBSPricing();

  const handleOptionTypeChange = (_, newType) => {
    if (newType !== null) {
      setOptionType(newType);
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        gap: 3,
        p: 3,
        minHeight: '100vh',
        bgcolor: 'background.default',
      }}
    >
      {/* Left Panel - Parameters */}
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
          Black-Scholes Playground
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Interactive option pricing with Greeks
        </Typography>

        <Divider sx={{ my: 2 }} />

        {/* Option Type Toggle */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Option Type
          </Typography>
          <ToggleButtonGroup
            value={optionType}
            exclusive
            onChange={handleOptionTypeChange}
            size="small"
            fullWidth
          >
            <ToggleButton value="call">Call</ToggleButton>
            <ToggleButton value="put">Put</ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Core Parameters */}
        <Typography variant="subtitle2" gutterBottom>
          Option Parameters
        </Typography>

        <Box sx={{ mb: 2 }}>
          <TextField
            label="Spot Price (S)"
            type="number"
            size="small"
            fullWidth
            value={S}
            onChange={(e) => setS(parseFloat(e.target.value) || 100)}
            InputProps={{ startAdornment: '$' }}
          />
        </Box>

        <Box sx={{ mb: 2 }}>
          <TextField
            label="Strike Price (K)"
            type="number"
            size="small"
            fullWidth
            value={K}
            onChange={(e) => setK(parseFloat(e.target.value) || 100)}
            InputProps={{ startAdornment: '$' }}
          />
        </Box>

        <ParameterSlider
          label="Time to Expiry"
          value={T}
          onChange={setT}
          min={0.01}
          max={3.0}
          step={0.01}
          unit=" years"
        />

        <ParameterSlider
          label="Volatility (σ)"
          value={sigma * 100}
          onChange={(v) => setSigma(v / 100)}
          min={5}
          max={100}
          step={1}
          unit="%"
        />

        <ParameterSlider
          label="Risk-Free Rate"
          value={r * 100}
          onChange={(v) => setR(v / 100)}
          min={0}
          max={15}
          step={0.25}
          unit="%"
        />

        <Divider sx={{ my: 2 }} />

        {/* Heatmap Range */}
        <Typography variant="subtitle2" gutterBottom>
          Heatmap Range
        </Typography>

        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" gutterBottom>
            Spot Range: ${spotRange[0]} - ${spotRange[1]}
          </Typography>
          <Slider
            value={spotRange}
            onChange={(_, v) => setSpotRange(v)}
            min={S * 0.5}
            max={S * 1.5}
            step={1}
            valueLabelDisplay="auto"
            size="small"
          />
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" gutterBottom>
            Vol Range: {(volRange[0] * 100).toFixed(0)}% - {(volRange[1] * 100).toFixed(0)}%
          </Typography>
          <Slider
            value={volRange.map((v) => v * 100)}
            onChange={(_, v) => setVolRange(v.map((x) => x / 100))}
            min={5}
            max={100}
            step={1}
            valueLabelDisplay="auto"
            size="small"
          />
        </Box>
      </Paper>

      {/* Right Panel - Results */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 3 }}>
        {/* Greeks Display */}
        <Paper elevation={3} sx={{ p: 3 }}>
          <GreeksDisplay pricing={pricing} isLoading={isLoading} />
        </Paper>

        {/* P&L Heatmap */}
        <Paper elevation={3} sx={{ p: 3 }}>
          <PnLHeatmap
            heatmapData={heatmapData}
            isLoading={isHeatmapLoading}
            currentSpot={S}
            currentVol={sigma}
          />
        </Paper>

        {/* Explanation */}
        <Paper elevation={2} sx={{ p: 2, bgcolor: 'action.hover' }}>
          <Typography variant="subtitle2" gutterBottom>
            About Black-Scholes Greeks
          </Typography>
          <Typography variant="body2" color="text.secondary">
            <strong>Delta</strong>: Price change per $1 move in underlying.{' '}
            <strong>Gamma</strong>: Rate of delta change (convexity).{' '}
            <strong>Theta</strong>: Time decay per day.{' '}
            <strong>Vega</strong>: Sensitivity to 1% vol change.{' '}
            <strong>Rho</strong>: Sensitivity to 1% rate change.
          </Typography>
        </Paper>
      </Box>
    </Box>
  );
};

export default BlackScholesPlayground;
