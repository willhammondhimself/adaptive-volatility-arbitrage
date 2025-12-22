import React from 'react';
import Plot from 'react-plotly.js';
import { Box, CircularProgress, Typography, useTheme } from '@mui/material';
import { getPlotlyLayout, getChartColors } from '../../utils/plotlyTheme';

const SurfaceHeatmap = ({
  strikes,
  maturities,
  values,
  title,
  subtitle,
  zLabel = 'Value',
  colorscale = 'Viridis',
  isLoading,
  formatZ = (z) => z.toFixed(4),
}) => {
  const theme = useTheme();
  const mode = theme.palette.mode;
  const plotlyTheme = getPlotlyLayout(mode);
  const colors = getChartColors(mode);

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="600px">
        <CircularProgress />
      </Box>
    );
  }

  if (!strikes || !maturities || !values) {
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
      z: values,
      colorscale,
      hovertemplate:
        `Strike: $%{x:.2f}<br>Maturity: %{y:.2f}y<br>${zLabel}: %{z:.4f}<extra></extra>`,
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
    xaxis: {
      ...plotlyTheme.xaxis,
      title: { text: 'Strike Price ($)', font: { color: colors.text } },
      showgrid: true,
    },
    yaxis: {
      ...plotlyTheme.yaxis,
      title: { text: 'Time to Maturity (years)', font: { color: colors.text } },
      showgrid: true,
    },
    autosize: true,
    margin: { l: 60, r: 60, t: 100, b: 60 },
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    toImageButtonOptions: {
      format: 'png',
      filename: 'surface_heatmap',
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

export default SurfaceHeatmap;
