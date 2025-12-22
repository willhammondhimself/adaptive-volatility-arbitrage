import React from 'react';
import { Box, Paper, Typography, Grid } from '@mui/material';

const GreekCard = ({ label, value, description, color = 'text.primary' }) => (
  <Paper
    elevation={2}
    sx={{
      p: 2,
      textAlign: 'center',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
    }}
  >
    <Typography variant="caption" color="text.secondary" gutterBottom>
      {label}
    </Typography>
    <Typography variant="h4" color={color} fontWeight="bold">
      {typeof value === 'number' ? value.toFixed(4) : value}
    </Typography>
    {description && (
      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
        {description}
      </Typography>
    )}
  </Paper>
);

const GreeksDisplay = ({ pricing, isLoading }) => {
  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography color="text.secondary">Calculating...</Typography>
      </Box>
    );
  }

  if (!pricing) {
    return null;
  }

  const greeks = [
    {
      label: 'Price',
      value: pricing.price,
      description: 'Theoretical value',
      color: 'primary.main',
    },
    {
      label: 'Delta',
      value: pricing.delta,
      description: 'Price sensitivity',
      color: pricing.delta >= 0 ? 'success.main' : 'error.main',
    },
    {
      label: 'Gamma',
      value: pricing.gamma,
      description: 'Delta sensitivity',
      color: 'secondary.main',
    },
    {
      label: 'Theta',
      value: pricing.theta,
      description: 'Time decay / day',
      color: pricing.theta >= 0 ? 'success.main' : 'error.main',
    },
    {
      label: 'Vega',
      value: pricing.vega,
      description: 'Vol sensitivity / 1%',
      color: 'info.main',
    },
    {
      label: 'Rho',
      value: pricing.rho,
      description: 'Rate sensitivity / 1%',
      color: 'warning.main',
    },
  ];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Option Greeks
      </Typography>
      <Grid container spacing={2}>
        {greeks.map((greek) => (
          <Grid item xs={6} sm={4} md={2} key={greek.label}>
            <GreekCard {...greek} />
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default GreeksDisplay;
