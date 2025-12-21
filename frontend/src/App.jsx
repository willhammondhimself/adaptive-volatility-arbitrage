import React, { createContext, useContext, useState, useMemo, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
} from '@mui/material';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import HestonExplorer from './pages/HestonExplorer';
import BacktestDashboard from './pages/BacktestDashboard';
import PaperTrading from './pages/PaperTrading';

// Theme mode context
const ThemeModeContext = createContext({
  mode: 'light',
  toggleMode: () => {},
});

export const useThemeMode = () => useContext(ThemeModeContext);

// Get initial mode from localStorage or system preference
const getInitialMode = () => {
  const stored = localStorage.getItem('theme-mode');
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

// Theme definitions
const getTheme = (mode) =>
  createTheme({
    palette: {
      mode,
      primary: {
        main: mode === 'dark' ? '#90caf9' : '#1976d2',
      },
      secondary: {
        main: mode === 'dark' ? '#f48fb1' : '#dc004e',
      },
      background: {
        default: mode === 'dark' ? '#121212' : '#f5f5f5',
        paper: mode === 'dark' ? '#1e1e1e' : '#ffffff',
      },
      text: {
        primary: mode === 'dark' ? '#e0e0e0' : '#212121',
        secondary: mode === 'dark' ? '#a0a0a0' : '#666666',
      },
    },
    typography: {
      fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
      h4: {
        fontWeight: 600,
      },
      h6: {
        fontWeight: 600,
      },
    },
  });

function Navigation() {
  const location = useLocation();
  const { mode, toggleMode } = useThemeMode();

  return (
    <AppBar position="static" elevation={1}>
      <Toolbar>
        <Typography variant="h6" sx={{ flexGrow: 0, mr: 4 }}>
          Volatility Arbitrage
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexGrow: 1 }}>
          <Button
            component={Link}
            to="/"
            color="inherit"
            startIcon={<ShowChartIcon />}
            sx={{
              backgroundColor: location.pathname === '/' ? 'rgba(255,255,255,0.15)' : 'transparent',
            }}
          >
            Heston Explorer
          </Button>
          <Button
            component={Link}
            to="/backtest"
            color="inherit"
            startIcon={<TimelineIcon />}
            sx={{
              backgroundColor: location.pathname === '/backtest' ? 'rgba(255,255,255,0.15)' : 'transparent',
            }}
          >
            Backtest Dashboard
          </Button>
          <Button
            component={Link}
            to="/paper-trading"
            color="inherit"
            startIcon={<TrendingUpIcon />}
            sx={{
              backgroundColor: location.pathname === '/paper-trading' ? 'rgba(255,255,255,0.15)' : 'transparent',
            }}
          >
            Paper Trading
          </Button>
        </Box>
        <IconButton onClick={toggleMode} color="inherit" title={`Switch to ${mode === 'dark' ? 'light' : 'dark'} mode`}>
          {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
        </IconButton>
      </Toolbar>
    </AppBar>
  );
}

function App() {
  const [mode, setMode] = useState(getInitialMode);

  const toggleMode = () => {
    setMode((prev) => {
      const newMode = prev === 'light' ? 'dark' : 'light';
      localStorage.setItem('theme-mode', newMode);
      return newMode;
    });
  };

  const theme = useMemo(() => getTheme(mode), [mode]);

  const contextValue = useMemo(() => ({ mode, toggleMode }), [mode]);

  return (
    <ThemeModeContext.Provider value={contextValue}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <BrowserRouter>
          <Navigation />
          <Routes>
            <Route path="/" element={<HestonExplorer />} />
            <Route path="/backtest" element={<BacktestDashboard />} />
            <Route path="/paper-trading" element={<PaperTrading />} />
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </ThemeModeContext.Provider>
  );
}

export default App;
