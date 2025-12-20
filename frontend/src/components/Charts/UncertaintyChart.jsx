import React from 'react';
import Plot from 'react-plotly.js';
import { Box, CircularProgress, Typography, useTheme } from '@mui/material';
import { getPlotlyLayout, getChartColors } from '../../utils/plotlyTheme';

const UncertaintyChart = ({ forecast, isLoading }) => {
  const theme = useTheme();
  const mode = theme.palette.mode;
  const plotlyTheme = getPlotlyLayout(mode);
  const colors = getChartColors(mode);

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="300px">
        <CircularProgress />
      </Box>
    );
  }

  if (!forecast) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="300px">
        <Typography variant="body1" color="textSecondary">
          Run a forecast to see uncertainty visualization
        </Typography>
      </Box>
    );
  }

  const { mean_vol, epistemic_uncertainty, lower_bound, upper_bound, confidence_scalar } = forecast;

  // Create a simple bar chart showing forecast with uncertainty bands
  const data = [
    {
      type: 'bar',
      x: ['Volatility Forecast'],
      y: [mean_vol * 100], // Convert to percentage
      name: 'Mean Vol',
      marker: { color: colors.primary },
      error_y: {
        type: 'data',
        array: [(upper_bound - mean_vol) * 100],
        arrayminus: [(mean_vol - lower_bound) * 100],
        visible: true,
        color: colors.textSecondary,
        thickness: 2,
        width: 8,
      },
      hovertemplate:
        'Mean: %{y:.2f}%<br>' +
        `Upper: ${(upper_bound * 100).toFixed(2)}%<br>` +
        `Lower: ${(lower_bound * 100).toFixed(2)}%<extra></extra>`,
    },
  ];

  const layout = {
    ...plotlyTheme,
    title: {
      text: 'Volatility Forecast with Uncertainty',
      font: { size: 14, color: colors.text },
    },
    yaxis: {
      ...plotlyTheme.yaxis,
      title: { text: 'Annualized Volatility (%)', font: { color: colors.text } },
      showgrid: true,
      range: [0, Math.max(upper_bound * 100 * 1.2, 50)],
    },
    xaxis: {
      ...plotlyTheme.xaxis,
      showticklabels: false,
    },
    autosize: true,
    height: 280,
    margin: { l: 60, r: 30, t: 50, b: 30 },
    annotations: [
      {
        x: 'Volatility Forecast',
        y: mean_vol * 100,
        text: `${(mean_vol * 100).toFixed(1)}%`,
        showarrow: false,
        yshift: 20,
        font: { size: 12, color: colors.primary },
      },
      {
        x: 0.5,
        y: -0.15,
        xref: 'paper',
        yref: 'paper',
        text: `Uncertainty: ±${(epistemic_uncertainty * 100).toFixed(2)}% | Confidence: ${(confidence_scalar * 100).toFixed(0)}%`,
        showarrow: false,
        font: { size: 11, color: colors.textSecondary },
      },
    ],
  };

  const config = {
    responsive: true,
    displayModeBar: false,
  };

  return (
    <Box sx={{ width: '100%' }}>
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

export default UncertaintyChart;
