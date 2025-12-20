import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { Box, ToggleButton, ToggleButtonGroup, useTheme } from '@mui/material';
import { getPlotlyLayout, getChartColors } from '../../utils/plotlyTheme';

const DistributionHistogram = ({ results, observedMetrics }) => {
  const theme = useTheme();
  const mode = theme.palette.mode;
  const plotlyTheme = getPlotlyLayout(mode);
  const colors = getChartColors(mode);

  const [selectedMetric, setSelectedMetric] = useState('sharpe');

  if (!results) return null;

  const metricConfig = {
    sharpe: {
      label: 'Sharpe Ratio',
      distribution: results.sharpe_distribution,
      stats: results.sharpe_ratio,
      observed: observedMetrics?.sharpe_ratio,
      format: (v) => v.toFixed(2),
      xLabel: 'Sharpe Ratio',
    },
    return: {
      label: 'Total Return',
      distribution: results.return_distribution,
      stats: results.total_return,
      observed: observedMetrics?.total_return != null ? observedMetrics.total_return * 100 : null,
      format: (v) => `${v.toFixed(1)}%`,
      xLabel: 'Total Return (%)',
    },
    drawdown: {
      label: 'Max Drawdown',
      distribution: results.drawdown_distribution,
      stats: results.max_drawdown,
      observed: observedMetrics?.max_drawdown != null ? observedMetrics.max_drawdown * 100 : null,
      format: (v) => `${v.toFixed(1)}%`,
      xLabel: 'Max Drawdown (%)',
    },
  };

  const config = metricConfig[selectedMetric];
  const { distribution, stats, observed, format, xLabel } = config;

  const data = [
    {
      type: 'histogram',
      x: distribution,
      nbinsx: 50,
      marker: {
        color: colors.primary,
        opacity: 0.7,
        line: { color: colors.primary, width: 1 },
      },
      hovertemplate: `${xLabel}: %{x:.2f}<br>Count: %{y}<extra></extra>`,
    },
  ];

  // CI shaded region and observed value line as shapes
  const shapes = [
    // 95% CI shaded region
    {
      type: 'rect',
      x0: stats.ci_lower,
      x1: stats.ci_upper,
      y0: 0,
      y1: 1,
      yref: 'paper',
      fillcolor: mode === 'dark' ? 'rgba(102, 187, 106, 0.2)' : 'rgba(76, 175, 80, 0.15)',
      line: { width: 0 },
    },
    // Mean line
    {
      type: 'line',
      x0: stats.mean,
      x1: stats.mean,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: colors.primary, width: 2 },
    },
  ];

  // Observed value line (if available)
  if (observed != null) {
    shapes.push({
      type: 'line',
      x0: observed,
      x1: observed,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: colors.drawdown, width: 2, dash: 'dash' },
    });
  }

  const annotations = [
    // CI bounds labels
    {
      x: stats.ci_lower,
      y: 1.02,
      yref: 'paper',
      text: format(stats.ci_lower),
      showarrow: false,
      font: { size: 10, color: colors.buyHold },
    },
    {
      x: stats.ci_upper,
      y: 1.02,
      yref: 'paper',
      text: format(stats.ci_upper),
      showarrow: false,
      font: { size: 10, color: colors.buyHold },
    },
  ];

  // Observed value annotation
  if (observed != null) {
    annotations.push({
      x: observed,
      y: 0.95,
      yref: 'paper',
      text: `Observed: ${format(observed)}`,
      showarrow: true,
      arrowhead: 2,
      arrowcolor: colors.drawdown,
      ax: 40,
      ay: -20,
      font: { size: 11, color: colors.drawdown },
      bgcolor: colors.annotationBg,
      borderpad: 3,
    });
  }

  const layout = {
    ...plotlyTheme,
    title: {
      text: `${config.label} Distribution (${results.n_simulations.toLocaleString()} simulations)`,
      font: { size: 14, color: colors.text },
    },
    xaxis: {
      ...plotlyTheme.xaxis,
      title: { text: xLabel, font: { color: colors.text } },
    },
    yaxis: {
      ...plotlyTheme.yaxis,
      title: { text: 'Frequency', font: { color: colors.text } },
    },
    shapes,
    annotations,
    autosize: true,
    height: 350,
    margin: { l: 60, r: 30, t: 60, b: 50 },
    bargap: 0.02,
  };

  const plotConfig = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
        <ToggleButtonGroup
          value={selectedMetric}
          exclusive
          onChange={(e, value) => value && setSelectedMetric(value)}
          size="small"
        >
          <ToggleButton value="sharpe">Sharpe</ToggleButton>
          <ToggleButton value="return">Return</ToggleButton>
          <ToggleButton value="drawdown">Drawdown</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <Plot
        data={data}
        layout={layout}
        config={plotConfig}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </Box>
  );
};

export default DistributionHistogram;
