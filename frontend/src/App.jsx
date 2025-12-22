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
import AccountBalanceIcon from '@mui/icons-material/AccountBalance';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import FunctionsIcon from '@mui/icons-material/Functions';
import LayersIcon from '@mui/icons-material/Layers';
import HestonExplorer from './pages/HestonExplorer';
import BacktestDashboard from './pages/BacktestDashboard';
import PaperTrading from './pages/PaperTrading';
import DeltaHedgedBacktest from './pages/DeltaHedgedBacktest';
import BlackScholesPlayground from './pages/BlackScholesPlayground';
import SurfaceExplorer from './pages/SurfaceExplorer';

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
          <Button
            component={Link}
            to="/delta-hedged"
            color="inherit"
            startIcon={<AccountBalanceIcon />}
            sx={{
              backgroundColor: location.pathname === '/delta-hedged' ? 'rgba(255,255,255,0.15)' : 'transparent',
            }}
          >
            Delta Hedged
          </Button>
          <Button
            component={Link}
            to="/bs-playground"
            color="inherit"
            startIcon={<FunctionsIcon />}
            sx={{
              backgroundColor: location.pathname === '/bs-playground' ? 'rgba(255,255,255,0.15)' : 'transparent',
            }}
          >
            BS Playground
          </Button>
          <Button
            component={Link}
            to="/surface-explorer"
            color="inherit"
            startIcon={<LayersIcon />}
            sx={{
              backgroundColor: location.pathname === '/surface-explorer' ? 'rgba(255,255,255,0.15)' : 'transparent',
            }}
          >
            Surface Explorer
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
            <Route path="/delta-hedged" element={<DeltaHedgedBacktest />} />
            <Route path="/bs-playground" element={<BlackScholesPlayground />} />
            <Route path="/surface-explorer" element={<SurfaceExplorer />} />
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </ThemeModeContext.Provider>
  );
}

export default App;
