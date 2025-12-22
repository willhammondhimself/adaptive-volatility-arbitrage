import React from 'react';
import Plot from 'react-plotly.js';
import { Box, CircularProgress, Typography, useTheme } from '@mui/material';
import { getPlotlyLayout, getChartColors, getSceneLayout } from '../../utils/plotlyTheme';

const Surface3D = ({
  strikes,
  maturities,
  values,
  title,
  subtitle,
  zLabel = 'Value',
  colorscale = 'Viridis',
  isLoading,
}) => {
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

  if (!strikes || !maturities || !values) {
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
      z: values,
      colorscale,
      hovertemplate:
        `Strike: $%{x:.2f}<br>Maturity: %{y:.2f}y<br>${zLabel}: %{z:.4f}<extra></extra>`,
      contours: {
        z: {
          show: true,
          project: { z: true },
          color: mode === 'dark' ? '#888' : '#666',
          width: 1,
        },
      },
      colorbar: {
        title: zLabel,
        titleside: 'right',
        tickfont: { color: colors.text },
        titlefont: { color: colors.text },
      },
    },
  ];

  const layout = {
    ...plotlyTheme,
    title: {
      text: subtitle ? `${title}<br><sub>${subtitle}</sub>` : title,
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
        title: { text: zLabel, font: { color: colors.text } },
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
      filename: 'surface_3d',
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

export default Surface3D;
