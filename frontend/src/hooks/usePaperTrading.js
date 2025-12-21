import { useEffect, useRef, useCallback } from 'react';
import usePaperTradingStore from '../store/paperTradingStore';
import * as api from '../api/paperTradingApi';

/**
 * Hook for managing paper trading state and polling.
 *
 * Handles:
 * - Auto-polling when trading is running (5s interval)
 * - Pause when browser tab hidden
 * - Start/stop trading actions
 * - Cleanup on unmount
 */
const usePaperTrading = () => {
  const {
    config,
    status,
    trades,
    stats,
    isLoading,
    isStarting,
    isStopping,
    error,
    updateConfig,
    setStatus,
    setTrades,
    setStats,
    setLoading,
    setStarting,
    setStopping,
    setError,
    reset,
  } = usePaperTradingStore();

  const intervalRef = useRef(null);
  const isVisibleRef = useRef(true);

  // Fetch all status data
  const fetchData = useCallback(async () => {
    if (!isVisibleRef.current) return;

    try {
      const [statusRes, tradesRes, statsRes] = await Promise.all([
        api.getStatus(),
        api.getTrades(50),
        api.getStats(),
      ]);

      setStatus(statusRes.data);
      setTrades(tradesRes.data.trades);
      setStats(statsRes.data);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch data');
    }
  }, [setStatus, setTrades, setStats, setError]);

  // Handle visibility changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      isVisibleRef.current = document.visibilityState === 'visible';

      // Immediate fetch when tab becomes visible
      if (isVisibleRef.current && status?.is_running) {
        fetchData();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [status?.is_running, fetchData]);

  // Polling when trading is running
  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Always fetch once on mount
    fetchData();

    if (status?.is_running) {
      // Poll every 5 seconds when trading
      intervalRef.current = setInterval(fetchData, 5000);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [status?.is_running, fetchData]);

  // Start trading
  const startTrading = useCallback(async () => {
    setStarting(true);
    setError(null);

    try {
      const requestConfig = {
        initial_capital: config.initialCapital,
        uncertainty_threshold: config.uncertaintyThreshold,
        position_pct: config.positionPct,
      };

      await api.startTrading(requestConfig);
      await fetchData();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to start trading');
    } finally {
      setStarting(false);
    }
  }, [config, fetchData, setStarting, setError]);

  // Stop trading
  const stopTrading = useCallback(async () => {
    setStopping(true);
    setError(null);

    try {
      const response = await api.stopTrading();
      setStats(response.data.stats);
      await fetchData();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to stop trading');
    } finally {
      setStopping(false);
    }
  }, [fetchData, setStopping, setStats, setError]);

  return {
    // Config
    config,
    updateConfig,

    // State
    status,
    trades,
    stats,
    isLoading,
    isStarting,
    isStopping,
    error,

    // Derived
    isRunning: status?.is_running ?? false,
    sessionId: status?.session_id ?? null,
    capital: status?.capital ?? config.initialCapital,
    position: status?.position ?? 0,
    avgCost: status?.avg_cost ?? 0,
    cumulativePnl: status?.cumulative_pnl ?? 0,
    tickCount: status?.tick_count ?? 0,
    lastUpdate: status?.last_update ?? null,

    // Actions
    startTrading,
    stopTrading,
    refresh: fetchData,
    reset,
  };
};

export default usePaperTrading;
