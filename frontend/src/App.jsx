import React, { createContext, useContext, useState, useMemo } from 'react';
import { BrowserRouter, Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
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
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import LayersIcon from '@mui/icons-material/Layers';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import AccountBalanceIcon from '@mui/icons-material/AccountBalance';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import SurfaceExplorer from './pages/SurfaceExplorer';
import HestonExplorer from './pages/HestonExplorer';
import BlackScholesPlayground from './pages/BlackScholesPlayground';
import BacktestDashboard from './pages/BacktestDashboard';
import PaperTrading from './pages/PaperTrading';
import DeltaHedgedBacktest from './pages/DeltaHedgedBacktest';

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

  const isOptionsExplorer = location.pathname === '/options' || location.pathname === '/options/bs';
  const isTrading = location.pathname === '/trading' || location.pathname === '/trading/paper';

  return (
    <AppBar position="static" elevation={1}>
      <Toolbar>
        <Typography variant="h6" sx={{ flexGrow: 0, mr: 4 }}>
          Volatility Arbitrage
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexGrow: 1, alignItems: 'center' }}>
          {/* Surface Explorer - First */}
          <Button
            component={Link}
            to="/"
            color="inherit"
            startIcon={<LayersIcon />}
            sx={{
              backgroundColor: location.pathname === '/' ? 'rgba(255,255,255,0.15)' : 'transparent',
            }}
          >
            Surface Explorer
          </Button>

          {/* Options Explorer with Heston/BS toggle */}
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Button
              component={Link}
              to="/options"
              color="inherit"
              startIcon={<ShowChartIcon />}
              sx={{
                backgroundColor: isOptionsExplorer ? 'rgba(255,255,255,0.15)' : 'transparent',
                borderTopRightRadius: 0,
                borderBottomRightRadius: 0,
              }}
            >
              Options Explorer
            </Button>
            {isOptionsExplorer && (
              <ToggleButtonGroup
                size="small"
                value={location.pathname === '/options/bs' ? 'bs' : 'heston'}
                exclusive
                sx={{ ml: 0.5 }}
              >
                <ToggleButton
                  component={Link}
                  to="/options"
                  value="heston"
                  sx={{ color: 'inherit', borderColor: 'rgba(255,255,255,0.3)', py: 0.5, px: 1 }}
                >
                  Heston
                </ToggleButton>
                <ToggleButton
                  component={Link}
                  to="/options/bs"
                  value="bs"
                  sx={{ color: 'inherit', borderColor: 'rgba(255,255,255,0.3)', py: 0.5, px: 1 }}
                >
                  BS
                </ToggleButton>
              </ToggleButtonGroup>
            )}
          </Box>

          {/* Trading with Backtest/Paper toggle */}
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Button
              component={Link}
              to="/trading"
              color="inherit"
              startIcon={<TimelineIcon />}
              sx={{
                backgroundColor: isTrading ? 'rgba(255,255,255,0.15)' : 'transparent',
                borderTopRightRadius: 0,
                borderBottomRightRadius: 0,
              }}
            >
              Trading
            </Button>
            {isTrading && (
              <ToggleButtonGroup
                size="small"
                value={location.pathname === '/trading/paper' ? 'paper' : 'backtest'}
                exclusive
                sx={{ ml: 0.5 }}
              >
                <ToggleButton
                  component={Link}
                  to="/trading"
                  value="backtest"
                  sx={{ color: 'inherit', borderColor: 'rgba(255,255,255,0.3)', py: 0.5, px: 1 }}
                >
                  Backtest
                </ToggleButton>
                <ToggleButton
                  component={Link}
                  to="/trading/paper"
                  value="paper"
                  sx={{ color: 'inherit', borderColor: 'rgba(255,255,255,0.3)', py: 0.5, px: 1 }}
                >
                  Paper
                </ToggleButton>
              </ToggleButtonGroup>
            )}
          </Box>

          {/* Delta Hedged */}
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
            <Route path="/" element={<SurfaceExplorer />} />
            <Route path="/options" element={<HestonExplorer />} />
            <Route path="/options/bs" element={<BlackScholesPlayground />} />
            <Route path="/trading" element={<BacktestDashboard />} />
            <Route path="/trading/paper" element={<PaperTrading />} />
            <Route path="/delta-hedged" element={<DeltaHedgedBacktest />} />
            <Route path="/backtest" element={<Navigate to="/trading" replace />} />
            <Route path="/paper-trading" element={<Navigate to="/trading/paper" replace />} />
            <Route path="/bs-playground" element={<Navigate to="/options/bs" replace />} />
            <Route path="/surface-explorer" element={<Navigate to="/" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </ThemeModeContext.Provider>
  );
}

export default App;
