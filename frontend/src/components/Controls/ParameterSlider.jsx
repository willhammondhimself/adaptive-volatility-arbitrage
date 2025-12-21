import React from 'react';
import { Box, Slider, TextField, Typography } from '@mui/material';

const ParameterSlider = ({ label, value, onChange, min, max, step, description, disabled = false }) => {
  const handleSliderChange = (event, newValue) => {
    onChange(newValue);
  };

  const handleInputChange = (event) => {
    const newValue = event.target.value === '' ? 0 : Number(event.target.value);
    onChange(newValue);
  };

  const handleBlur = () => {
    if (value < min) {
      onChange(min);
    } else if (value > max) {
      onChange(max);
    }
  };

  return (
    <Box sx={{ mb: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="subtitle2" fontWeight="bold">
          {label}
        </Typography>
        <TextField
          value={value}
          onChange={handleInputChange}
          onBlur={handleBlur}
          size="small"
          type="number"
          disabled={disabled}
          inputProps={{
            step,
            min,
            max,
            style: { textAlign: 'right', width: '80px' },
          }}
          sx={{ width: '100px' }}
        />
      </Box>
      {description && (
        <Typography variant="caption" color="textSecondary" sx={{ mb: 1, display: 'block' }}>
          {description}
        </Typography>
      )}
      <Slider
        value={typeof value === 'number' ? value : 0}
        onChange={handleSliderChange}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        valueLabelDisplay="auto"
        sx={{
          '& .MuiSlider-thumb': {
            transition: 'left 0.1s ease-out',
          },
        }}
      />
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="caption" color="textSecondary">
          {min}
        </Typography>
        <Typography variant="caption" color="textSecondary">
          {max}
        </Typography>
      </Box>
    </Box>
  );
};

export default ParameterSlider;
