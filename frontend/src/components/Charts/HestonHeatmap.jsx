import React from 'react';
import Plot from 'react-plotly.js';
import { Box, CircularProgress, Typography } from '@mui/material';

const HestonHeatmap = ({ strikes, maturities, prices, params, isLoading }) => {
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="600px">
        <CircularProgress />
      </Box>
    );
  }

  if (!strikes || !maturities || !prices) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="600px">
        <Typography variant="h6" color="textSecondary">
          No data available
        </Typography>
      </Box>
    );
  }

  const data = [
    {
      type: 'heatmap',
      x: strikes,
      y: maturities,
      z: prices,
      colorscale: 'Viridis',
      hovertemplate:
        'Strike: $%{x:.2f}<br>Maturity: %{y:.2f}y<br>Price: $%{z:.2f}<extra></extra>',
      colorbar: {
        title: 'Price ($)',
        titleside: 'right',
      },
    },
  ];

  const layout = {
    title: {
      text: `Heston Call Option Price Heatmap<br><sub>v₀=${params.v0}, θ=${params.theta}, κ=${params.kappa}, σᵥ=${params.sigma_v}, ρ=${params.rho}</sub>`,
      font: { size: 16 },
    },
    xaxis: {
      title: 'Strike Price ($)',
      showgrid: true,
      gridcolor: '#e0e0e0',
    },
    yaxis: {
      title: 'Time to Maturity (years)',
      showgrid: true,
      gridcolor: '#e0e0e0',
    },
    autosize: true,
    margin: { l: 60, r: 60, t: 100, b: 60 },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    toImageButtonOptions: {
      format: 'png',
      filename: 'heston_heatmap',
      width: 1920,
      height: 1080,
      scale: 2,
    },
  };

  return (
    <Box sx={{ width: '100%', height: '600px' }}>
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

export default HestonHeatmap;
