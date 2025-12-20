import React from 'react';
import Plot from 'react-plotly.js';
import { Box, CircularProgress, Typography, useTheme } from '@mui/material';
import { getPlotlyLayout, getChartColors, getSceneLayout } from '../../utils/plotlyTheme';

const HestonSurface3D = ({ strikes, maturities, prices, params, isLoading }) => {
  const theme = useTheme();
  const mode = theme.palette.mode;
  const plotlyTheme = getPlotlyLayout(mode);
  const colors = getChartColors(mode);
  const sceneTheme = getSceneLayout(mode);

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="700px">
        <CircularProgress />
      </Box>
    );
  }

  if (!strikes || !maturities || !prices) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="700px">
        <Typography variant="h6" color="textSecondary">
          No data available
        </Typography>
      </Box>
    );
  }

  const data = [
    {
      type: 'surface',
      x: strikes,
      y: maturities,
      z: prices,
      colorscale: 'Viridis',
      hovertemplate:
        'Strike: $%{x:.2f}<br>Maturity: %{y:.2f}y<br>Price: $%{z:.2f}<extra></extra>',
      contours: {
        z: {
          show: true,
          project: { z: true },
          color: mode === 'dark' ? '#888' : '#666',
          width: 1,
        },
      },
      colorbar: {
        title: 'Price ($)',
        titleside: 'right',
        tickfont: { color: colors.text },
        titlefont: { color: colors.text },
      },
    },
  ];

  const layout = {
    ...plotlyTheme,
    title: {
      text: `Heston Call Option Price Surface (3D)<br><sub>v₀=${params.v0}, θ=${params.theta}, κ=${params.kappa}, σᵥ=${params.sigma_v}, ρ=${params.rho}</sub>`,
      font: { size: 16, color: colors.text },
    },
    scene: {
      ...sceneTheme,
      xaxis: {
        ...sceneTheme.xaxis,
        title: { text: 'Strike Price ($)', font: { color: colors.text } },
      },
      yaxis: {
        ...sceneTheme.yaxis,
        title: { text: 'Maturity (years)', font: { color: colors.text } },
      },
      zaxis: {
        ...sceneTheme.zaxis,
        title: { text: 'Call Price ($)', font: { color: colors.text } },
      },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.3 },
      },
    },
    autosize: true,
    margin: { l: 0, r: 0, t: 100, b: 0 },
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    toImageButtonOptions: {
      format: 'png',
      filename: 'heston_surface_3d',
      width: 1920,
      height: 1080,
      scale: 2,
    },
  };

  return (
    <Box sx={{ width: '100%', height: '700px' }}>
      <Plot
        data={data}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </Box>
  );
};

export default HestonSurface3D;
