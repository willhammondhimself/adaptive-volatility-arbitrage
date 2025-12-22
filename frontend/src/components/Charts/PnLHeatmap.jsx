import React from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import Plot from 'react-plotly.js';
import { useThemeMode } from '../../App';

const PnLHeatmap = ({ heatmapData, isLoading, currentSpot, currentVol }) => {
  const { mode } = useThemeMode();

  if (isLoading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: 400,
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (!heatmapData) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: 400,
        }}
      >
        <Typography color="text.secondary">
          Adjust parameters to generate P&L heatmap
        </Typography>
      </Box>
    );
  }

  const { spots, vols, pnl } = heatmapData;

  // Find min/max for color scale centering at 0
  const allPnl = pnl.flat();
  const maxAbs = Math.max(...allPnl.map(Math.abs));

  // Current position marker
  const annotations = [];
  if (currentSpot !== undefined && currentVol !== undefined) {
    annotations.push({
      x: currentSpot,
      y: currentVol * 100,
      text: 'Current',
      showarrow: true,
      arrowhead: 2,
      arrowcolor: mode === 'dark' ? '#fff' : '#000',
      font: { color: mode === 'dark' ? '#fff' : '#000' },
    });
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        P&L Heatmap (Spot × Volatility)
      </Typography>
      <Plot
        data={[
          {
            x: spots,
            y: vols.map((v) => (v * 100).toFixed(0)),
            z: pnl,
            type: 'heatmap',
            colorscale: [
              [0, '#d32f2f'],
              [0.5, '#ffffff'],
              [1, '#388e3c'],
            ],
            zmid: 0,
            zmin: -maxAbs,
            zmax: maxAbs,
            colorbar: {
              title: 'P&L ($)',
              tickprefix: '$',
              tickformat: '.2f',
            },
            hovertemplate:
              'Spot: $%{x:.2f}<br>Vol: %{y}%<br>P&L: $%{z:.2f}<extra></extra>',
          },
        ]}
        layout={{
          height: 400,
          margin: { t: 20, r: 80, b: 60, l: 80 },
          xaxis: {
            title: 'Spot Price ($)',
            tickprefix: '$',
            showgrid: true,
            gridcolor: mode === 'dark' ? '#333' : '#eee',
          },
          yaxis: {
            title: 'Volatility (%)',
            ticksuffix: '%',
            showgrid: true,
            gridcolor: mode === 'dark' ? '#333' : '#eee',
          },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { color: mode === 'dark' ? '#e0e0e0' : '#212121' },
          annotations,
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
        Shows P&L from entry price across different spot prices and volatility
        levels. Green = profit, Red = loss.
      </Typography>
    </Box>
  );
};

export default PnLHeatmap;
