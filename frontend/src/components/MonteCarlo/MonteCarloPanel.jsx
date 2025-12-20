import React from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Box,
  Button,
  Slider,
  LinearProgress,
  Alert,
  Tooltip,
  Paper,
  Grid,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CasinoIcon from '@mui/icons-material/Casino';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import useBacktestStore from '../../store/backtestStore';
import { runMonteCarloSimulation } from '../../api/backtestApi';
import DistributionHistogram from './DistributionHistogram';
import RiskProbabilityTable from './RiskProbabilityTable';

const simulationMarks = [
  { value: 1000, label: '1K' },
  { value: 5000, label: '5K' },
  { value: 10000, label: '10K' },
  { value: 25000, label: '25K' },
  { value: 50000, label: '50K' },
];

const MetricWithCI = ({ label, value, format, ci }) => (
  <Paper elevation={1} sx={{ p: 1.5, textAlign: 'center' }}>
    <Typography variant="caption" color="textSecondary" display="block">
      {label}
    </Typography>
    <Typography variant="h6" fontWeight="medium">
      {format(value)}
    </Typography>
    {ci && (
      <Typography variant="caption" color="textSecondary">
        95% CI: {format(ci.ci_lower)} - {format(ci.ci_upper)}
      </Typography>
    )}
  </Paper>
);

const MonteCarloPanel = ({ backtestResults }) => {
  const {
    monteCarloResults,
    monteCarloLoading,
    monteCarloError,
    numSimulations,
    setMonteCarloResults,
    setMonteCarloLoading,
    setMonteCarloError,
    setNumSimulations,
  } = useBacktestStore();

  const tradeReturns = backtestResults?.trade_returns;
  const hasEnoughTrades = tradeReturns && tradeReturns.length >= 5;

  const handleRunSimulation = async () => {
    if (!hasEnoughTrades) return;

    setMonteCarloLoading(true);
    try {
      const results = await runMonteCarloSimulation(tradeReturns, numSimulations);
      setMonteCarloResults(results);
    } catch (error) {
      setMonteCarloError(error.message || 'Failed to run Monte Carlo simulation');
    }
  };

  return (
    <Accordion defaultExpanded={false}>
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        sx={{ '& .MuiAccordionSummary-content': { alignItems: 'center', gap: 1 } }}
      >
        <CasinoIcon color="primary" />
        <Typography variant="h6">Monte Carlo Analysis</Typography>
        {monteCarloLoading && (
          <LinearProgress sx={{ width: 100, ml: 2, flexShrink: 0 }} />
        )}
        <Tooltip
          title="Runs your strategy thousands of times with randomized trade sequences to estimate the range of possible outcomes."
          arrow
        >
          <InfoOutlinedIcon fontSize="small" color="action" sx={{ ml: 1 }} />
        </Tooltip>
      </AccordionSummary>
      <AccordionDetails>
        {/* Controls */}
        <Box sx={{ display: 'flex', gap: 3, alignItems: 'center', mb: 3, flexWrap: 'wrap' }}>
          <Box sx={{ flex: 1, minWidth: 200 }}>
            <Typography variant="caption" color="textSecondary" gutterBottom display="block">
              Number of Simulations
            </Typography>
            <Slider
              value={numSimulations}
              onChange={(e, value) => setNumSimulations(value)}
              min={1000}
              max={50000}
              step={1000}
              marks={simulationMarks}
              valueLabelDisplay="auto"
              valueLabelFormat={(v) => `${(v / 1000).toFixed(0)}K`}
              disabled={monteCarloLoading}
            />
          </Box>
          <Button
            variant="contained"
            onClick={handleRunSimulation}
            disabled={monteCarloLoading || !hasEnoughTrades}
            startIcon={<CasinoIcon />}
          >
            {monteCarloLoading ? 'Running...' : 'Run Simulation'}
          </Button>
        </Box>

        {/* Insufficient trades warning */}
        {!hasEnoughTrades && (
          <Alert severity="info" sx={{ mb: 2 }}>
            {tradeReturns
              ? `Need at least 5 trades for Monte Carlo analysis (have ${tradeReturns.length}).`
              : 'Run a backtest first to enable Monte Carlo analysis.'}
          </Alert>
        )}

        {/* Error */}
        {monteCarloError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {monteCarloError}
          </Alert>
        )}

        {/* Results */}
        {monteCarloResults && (
          <Box>
            {/* Summary metrics with CIs */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={6} sm={3}>
                <MetricWithCI
                  label="Total Return"
                  value={monteCarloResults.total_return.mean}
                  format={(v) => `${v.toFixed(1)}%`}
                  ci={monteCarloResults.total_return}
                />
              </Grid>
              <Grid item xs={6} sm={3}>
                <MetricWithCI
                  label="Sharpe Ratio"
                  value={monteCarloResults.sharpe_ratio.mean}
                  format={(v) => v.toFixed(2)}
                  ci={monteCarloResults.sharpe_ratio}
                />
              </Grid>
              <Grid item xs={6} sm={3}>
                <MetricWithCI
                  label="Max Drawdown"
                  value={monteCarloResults.max_drawdown.mean}
                  format={(v) => `${v.toFixed(1)}%`}
                  ci={monteCarloResults.max_drawdown}
                />
              </Grid>
              <Grid item xs={6} sm={3}>
                <MetricWithCI
                  label="Win Rate"
                  value={monteCarloResults.win_rate.mean}
                  format={(v) => `${v.toFixed(1)}%`}
                  ci={monteCarloResults.win_rate}
                />
              </Grid>
            </Grid>

            {/* Histogram */}
            <DistributionHistogram
              results={monteCarloResults}
              observedMetrics={backtestResults?.metrics}
            />

            {/* Risk table */}
            <RiskProbabilityTable
              riskAssessment={{
                ...monteCarloResults.risk_assessment,
                n_simulations: monteCarloResults.n_simulations,
              }}
            />

            {/* Computation time */}
            <Typography variant="caption" color="textSecondary" sx={{ mt: 2, display: 'block' }}>
              Completed in {monteCarloResults.computation_time_ms.toFixed(0)}ms
              ({monteCarloResults.n_trades} trades, {monteCarloResults.n_simulations.toLocaleString()} simulations)
            </Typography>
          </Box>
        )}
      </AccordionDetails>
    </Accordion>
  );
};

export default MonteCarloPanel;
