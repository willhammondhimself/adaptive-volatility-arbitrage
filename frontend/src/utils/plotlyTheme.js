/**
 * Plotly theme utilities for light/dark mode support.
 */

/**
 * Get base Plotly layout properties for current theme mode.
 * @param {string} mode - 'light' or 'dark'
 * @returns {object} Plotly layout properties
 */
export const getPlotlyLayout = (mode) => ({
  plot_bgcolor: mode === 'dark' ? '#1e1e1e' : '#fafafa',
  paper_bgcolor: mode === 'dark' ? '#121212' : '#ffffff',
  font: {
    color: mode === 'dark' ? '#e0e0e0' : '#212121',
  },
  xaxis: {
    gridcolor: mode === 'dark' ? '#333333' : '#e8e8e8',
    linecolor: mode === 'dark' ? '#444444' : '#cccccc',
    tickfont: { color: mode === 'dark' ? '#a0a0a0' : '#666666' },
  },
  yaxis: {
    gridcolor: mode === 'dark' ? '#333333' : '#e8e8e8',
    linecolor: mode === 'dark' ? '#444444' : '#cccccc',
    tickfont: { color: mode === 'dark' ? '#a0a0a0' : '#666666' },
  },
  legend: {
    font: { color: mode === 'dark' ? '#e0e0e0' : '#212121' },
  },
});

/**
 * Get chart-specific colors for current theme mode.
 * @param {string} mode - 'light' or 'dark'
 * @returns {object} Color definitions
 */
export const getChartColors = (mode) => ({
  strategy: mode === 'dark' ? '#42a5f5' : '#1976d2',
  buyHold: mode === 'dark' ? '#66bb6a' : '#4caf50',
  drawdown: mode === 'dark' ? '#f48fb1' : '#dc004e',
  drawdownFill: mode === 'dark' ? 'rgba(244, 143, 177, 0.25)' : 'rgba(220, 0, 78, 0.2)',
  reference: mode === 'dark' ? '#666666' : '#999999',
  annotationBg: mode === 'dark' ? 'rgba(30, 30, 30, 0.9)' : 'rgba(255, 255, 255, 0.8)',
  primary: mode === 'dark' ? '#90caf9' : '#1976d2',
  secondary: mode === 'dark' ? '#f48fb1' : '#dc004e',
  text: mode === 'dark' ? '#e0e0e0' : '#212121',
  textSecondary: mode === 'dark' ? '#a0a0a0' : '#666666',
  grid: mode === 'dark' ? '#333333' : '#e0e0e0',
});

/**
 * Get 3D scene properties for Plotly surface charts.
 * @param {string} mode - 'light' or 'dark'
 * @returns {object} Scene layout properties
 */
export const getSceneLayout = (mode) => ({
  bgcolor: mode === 'dark' ? '#1e1e1e' : '#fafafa',
  xaxis: {
    gridcolor: mode === 'dark' ? '#444444' : '#d0d0d0',
    showgrid: true,
  },
  yaxis: {
    gridcolor: mode === 'dark' ? '#444444' : '#d0d0d0',
    showgrid: true,
  },
  zaxis: {
    gridcolor: mode === 'dark' ? '#444444' : '#d0d0d0',
    showgrid: true,
  },
});
