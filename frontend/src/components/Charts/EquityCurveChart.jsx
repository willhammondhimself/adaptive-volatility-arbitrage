import React from 'react';
import Plot from 'react-plotly.js';
import { Box, CircularProgress, Typography } from '@mui/material';

const EquityCurveChart = ({ equityCurve, initialCapital, isLoading, showBuyHold = false }) => {
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (!equityCurve || equityCurve.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <Typography variant="body1" color="textSecondary">
          Run a backtest to see the equity curve
        </Typography>
      </Box>
    );
  }

  const dates = equityCurve.map((p) => p.date);
  const equity = equityCurve.map((p) => p.equity);
  const drawdowns = equityCurve.map((p) => p.drawdown * -100);
  const buyHoldEquity = equityCurve.map((p) => p.buy_hold_equity);

  const data = [
    {
      type: 'scatter',
      mode: 'lines',
      x: dates,
      y: equity,
      name: 'Strategy',
      line: { color: '#1976d2', width: 2 },
      yaxis: 'y',
      hovertemplate: 'Date: %{x}<br>Strategy: $%{y:,.0f}<extra></extra>',
    },
    {
      type: 'scatter',
      mode: 'lines',
      x: dates,
      y: drawdowns,
      name: 'Drawdown',
      line: { color: '#dc004e', width: 1 },
      fill: 'tozeroy',
      fillcolor: 'rgba(220, 0, 78, 0.1)',
      yaxis: 'y2',
      hovertemplate: 'Date: %{x}<br>Drawdown: %{y:.1f}%<extra></extra>',
    },
  ];

  // Add buy-and-hold line if toggle is on and data exists
  if (showBuyHold && buyHoldEquity[0] != null) {
    data.push({
      type: 'scatter',
      mode: 'lines',
      x: dates,
      y: buyHoldEquity,
      name: 'Buy & Hold',
      line: { color: '#4caf50', width: 2, dash: 'dot' },
      yaxis: 'y',
      hovertemplate: 'Date: %{x}<br>Buy & Hold: $%{y:,.0f}<extra></extra>',
    });
  }

  // Calculate y-axis ranges including buy-and-hold if shown
  const allEquityValues = showBuyHold && buyHoldEquity[0] != null
    ? [...equity, ...buyHoldEquity]
    : equity;
  const minEquity = Math.min(...allEquityValues);
  const maxEquity = Math.max(...allEquityValues);
  const equityRange = maxEquity - minEquity;
  const minDrawdown = Math.min(...drawdowns);

  const layout = {
    title: {
      text: 'Equity Curve & Drawdown',
      font: { size: 16 },
    },
    xaxis: {
      title: 'Date',
      showgrid: true,
      gridcolor: '#e0e0e0',
      type: 'date',
    },
    yaxis: {
      title: 'Equity ($)',
      showgrid: true,
      gridcolor: '#e0e0e0',
      side: 'left',
      range: [minEquity - equityRange * 0.1, maxEquity + equityRange * 0.1],
      tickformat: '$,.0f',
    },
    yaxis2: {
      title: 'Drawdown (%)',
      showgrid: false,
      side: 'right',
      overlaying: 'y',
      range: [minDrawdown * 1.2, 5],
      tickformat: '.0f',
      ticksuffix: '%',
    },
    legend: {
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.02,
      xanchor: 'center',
      x: 0.5,
    },
    autosize: true,
    height: 400,
    margin: { l: 80, r: 80, t: 80, b: 60 },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
    hovermode: 'x unified',
    shapes: [
      {
        type: 'line',
        x0: dates[0],
        x1: dates[dates.length - 1],
        y0: initialCapital || 100000,
        y1: initialCapital || 100000,
        line: { color: '#999', width: 1, dash: 'dash' },
        yref: 'y',
      },
    ],
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    toImageButtonOptions: {
      format: 'png',
      filename: 'equity_curve',
      width: 1920,
      height: 1080,
      scale: 2,
    },
  };

  return (
    <Box sx={{ width: '100%', height: '400px' }}>
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
