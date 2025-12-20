import React from 'react';
import Plot from 'react-plotly.js';
import { Box, CircularProgress, Typography, useTheme } from '@mui/material';
import { getPlotlyLayout, getChartColors } from '../../utils/plotlyTheme';

const EquityCurveChart = ({ equityCurve, initialCapital, isLoading, showBuyHold = false }) => {
  const theme = useTheme();
  const mode = theme.palette.mode;
  const plotlyTheme = getPlotlyLayout(mode);
  const colors = getChartColors(mode);

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="500px">
        <CircularProgress />
      </Box>
    );
  }

  if (!equityCurve || equityCurve.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="500px">
        <Typography variant="body1" color="textSecondary">
          Run a backtest to see the equity curve
        </Typography>
      </Box>
    );
  }

  const dates = equityCurve.map((p) => p.date);
  const equity = equityCurve.map((p) => p.equity);
  const buyHoldEquity = equityCurve.map((p) => p.buy_hold_equity);
  const drawdowns = equityCurve.map((p) => -p.drawdown * 100); // Negative percentage

  // Find max drawdown point for annotation
  const maxDDValue = Math.min(...drawdowns);
  const maxDDIndex = drawdowns.indexOf(maxDDValue);
  const maxDDDate = dates[maxDDIndex];

  // Calculate y-axis ranges
  const allEquityValues = showBuyHold && buyHoldEquity[0] != null
    ? [...equity, ...buyHoldEquity]
    : equity;
  const minEquity = Math.min(...allEquityValues);
  const maxEquity = Math.max(...allEquityValues);
  const equityPadding = (maxEquity - minEquity) * 0.1;

  const data = [
    // Top panel: Strategy equity
    {
      type: 'scatter',
      mode: 'lines',
      x: dates,
      y: equity,
      name: 'Strategy',
      line: { color: colors.strategy, width: 2 },
      xaxis: 'x',
      yaxis: 'y',
      hovertemplate: '%{x}<br>Strategy: $%{y:,.0f}<extra></extra>',
    },
    // Top panel: Buy & Hold (conditional)
    {
      type: 'scatter',
      mode: 'lines',
      x: dates,
      y: buyHoldEquity,
      name: 'Buy & Hold',
      line: { color: colors.buyHold, width: 2, dash: 'dot' },
      xaxis: 'x',
      yaxis: 'y',
      visible: showBuyHold && buyHoldEquity[0] != null ? true : 'legendonly',
      hovertemplate: '%{x}<br>Buy & Hold: $%{y:,.0f}<extra></extra>',
    },
    // Top panel: Initial capital reference line
    {
      type: 'scatter',
      mode: 'lines',
      x: [dates[0], dates[dates.length - 1]],
      y: [initialCapital || 100000, initialCapital || 100000],
      name: 'Initial Capital',
      line: { color: colors.reference, width: 1, dash: 'dash' },
      xaxis: 'x',
      yaxis: 'y',
      showlegend: false,
      hoverinfo: 'skip',
    },
    // Bottom panel: Drawdown
    {
      type: 'scatter',
      mode: 'lines',
      x: dates,
      y: drawdowns,
      name: 'Drawdown',
      line: { color: colors.drawdown, width: 1.5 },
      fill: 'tozeroy',
      fillcolor: colors.drawdownFill,
      xaxis: 'x',
      yaxis: 'y2',
      hovertemplate: '%{x}<br>Drawdown: %{y:.1f}%<extra></extra>',
    },
  ];

  const layout = {
    ...plotlyTheme,
    grid: {
      rows: 2,
      columns: 1,
      pattern: 'independent',
      roworder: 'top to bottom',
    },
    // Shared x-axis (only shows on bottom panel)
    xaxis: {
      ...plotlyTheme.xaxis,
      showgrid: true,
      type: 'date',
      domain: [0, 1],
      anchor: 'y2',
    },
    // Top panel: Equity
    yaxis: {
      ...plotlyTheme.yaxis,
      title: { text: 'Equity ($)', standoff: 10, font: { color: colors.text } },
      showgrid: true,
      tickformat: '$,.0f',
      domain: [0.35, 1.0], // Top 65%
      anchor: 'x',
      range: [minEquity - equityPadding, maxEquity + equityPadding],
    },
    // Bottom panel: Drawdown
    yaxis2: {
      ...plotlyTheme.yaxis,
      title: { text: 'Drawdown (%)', standoff: 10, font: { color: colors.text } },
      showgrid: true,
      tickformat: '.0f',
      ticksuffix: '%',
      domain: [0.0, 0.28], // Bottom 28%
      anchor: 'x',
      range: [maxDDValue * 1.15, 2], // 0 at top, max DD at bottom with padding
    },
    // Legend
    legend: {
      ...plotlyTheme.legend,
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.02,
      xanchor: 'center',
      x: 0.5,
    },
    // Max drawdown annotation
    annotations: [
      {
        x: maxDDDate,
        y: maxDDValue,
        xref: 'x',
        yref: 'y2',
        text: `Max DD: ${maxDDValue.toFixed(1)}%`,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 1,
        arrowcolor: colors.drawdown,
        ax: 40,
        ay: -25,
        font: { size: 11, color: colors.drawdown },
        bgcolor: colors.annotationBg,
        borderpad: 3,
      },
    ],
    // Max drawdown horizontal line
    shapes: [
      {
        type: 'line',
        x0: dates[0],
        x1: dates[dates.length - 1],
        y0: maxDDValue,
        y1: maxDDValue,
        xref: 'x',
        yref: 'y2',
        line: { color: colors.drawdown, width: 1, dash: 'dot' },
      },
    ],
    // General layout
    autosize: true,
    height: 500,
    margin: { l: 70, r: 30, t: 50, b: 50 },
    hovermode: 'x unified',
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: {
      format: 'png',
      filename: 'equity_drawdown',
      width: 1920,
      height: 1080,
      scale: 2,
    },
  };

  return (
    <Box sx={{ width: '100%', height: '500px' }}>
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

export default EquityCurveChart;
