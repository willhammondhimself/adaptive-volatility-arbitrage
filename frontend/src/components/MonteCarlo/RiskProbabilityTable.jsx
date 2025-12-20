import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tooltip,
  Typography,
  Box,
} from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';

const riskMetrics = [
  {
    key: 'prob_loss',
    label: 'P(Loss)',
    tooltip: 'Probability of negative total return. Based on bootstrap simulations of your trade sequence.',
    format: (v) => `${v.toFixed(1)}%`,
    threshold: 10, // Warning if above this
    getColor: (v) => (v < 5 ? 'success.main' : v < 15 ? 'warning.main' : 'error.main'),
  },
  {
    key: 'prob_low_sharpe',
    label: 'P(Sharpe < 0.5)',
    tooltip: 'Probability of Sharpe ratio below 0.5. A Sharpe below 0.5 indicates weak risk-adjusted returns.',
    format: (v) => `${v.toFixed(1)}%`,
    threshold: 20,
    getColor: (v) => (v < 10 ? 'success.main' : v < 25 ? 'warning.main' : 'error.main'),
  },
  {
    key: 'prob_severe_drawdown',
    label: 'P(DD > 20%)',
    tooltip: 'Probability of max drawdown exceeding 20%. A 20%+ drawdown means significant portfolio decline.',
    format: (v) => `${v.toFixed(1)}%`,
    threshold: 15,
    getColor: (v) => (v < 10 ? 'success.main' : v < 20 ? 'warning.main' : 'error.main'),
  },
];

const RiskProbabilityTable = ({ riskAssessment }) => {
  if (!riskAssessment) return null;

  return (
    <TableContainer component={Paper} elevation={1} sx={{ mt: 2 }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Risk Metric</TableCell>
            <TableCell align="right">Probability</TableCell>
            <TableCell align="center" sx={{ width: 40 }}></TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {riskMetrics.map((metric) => {
            const value = riskAssessment[metric.key];
            return (
              <TableRow key={metric.key}>
                <TableCell>
                  <Typography variant="body2">{metric.label}</Typography>
                </TableCell>
                <TableCell align="right">
                  <Typography
                    variant="body2"
                    fontWeight="medium"
                    color={metric.getColor(value)}
                  >
                    {metric.format(value)}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Tooltip
                    title={
                      <Box sx={{ p: 0.5 }}>
                        <Typography variant="body2">{metric.tooltip}</Typography>
                      </Box>
                    }
                    arrow
                    placement="left"
                  >
                    <InfoOutlinedIcon fontSize="small" color="action" sx={{ cursor: 'help' }} />
                  </Tooltip>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      <Box sx={{ p: 1.5, borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="caption" color="textSecondary">
          Probabilities estimated from {riskAssessment.n_simulations?.toLocaleString() || '10,000'} bootstrap simulations.
          Lower values indicate more robust strategies.
        </Typography>
      </Box>
    </TableContainer>
  );
};

export default RiskProbabilityTable;
