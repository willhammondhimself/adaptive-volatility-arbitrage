import { useEffect, useRef, useCallback } from 'react';
import useLiveMarketStore from '../store/liveMarketStore';

/**
 * Hook for managing live market data polling.
 *
 * Handles:
 * - Auto-polling when live mode enabled
 * - Pause when browser tab hidden
 * - Cleanup on unmount
 *
 * @param {string} symbol - Ticker symbol (default: 'SPY')
 */
const useLiveMarket = (symbol = 'SPY') => {
  const {
    isLiveMode,
    pollingInterval,
    liveQuote,
    vixLevel,
    marketStatus,
    isLoading,
    error,
    lastUpdated,
    toggleLiveMode,
    setLiveMode,
    fetchAll,
    clearError,
  } = useLiveMarketStore();

  const intervalRef = useRef(null);
  const isVisibleRef = useRef(true);

  // Fetch data (only when tab visible)
  const poll = useCallback(async () => {
    if (!isVisibleRef.current) return;
    try {
      await fetchAll(symbol);
    } catch (err) {
      // Error already set in store
    }
  }, [fetchAll, symbol]);

  // Handle visibility changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      isVisibleRef.current = document.visibilityState === 'visible';

      // Immediate fetch when tab becomes visible again
      if (isVisibleRef.current && isLiveMode) {
        poll();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isLiveMode, poll]);

  // Polling interval management
  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (isLiveMode) {
      // Initial fetch
      poll();

      // Set up polling
      intervalRef.current = setInterval(poll, pollingInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isLiveMode, pollingInterval, poll]);

  return {
    // State
    isLiveMode,
    liveQuote,
    vixLevel,
    marketStatus,
    isLoading,
    error,
    lastUpdated,

    // Actions
    toggleLiveMode,
    setLiveMode,
    refresh: poll,
    clearError,

    // Derived
    spotPrice: liveQuote?.price ?? null,
    spotChange: liveQuote?.change_percent ?? null,
    vix: vixLevel?.level ?? null,
    vixChange: vixLevel?.change_percent ?? null,
    isMarketOpen: marketStatus?.is_open ?? false,
    marketPhase: marketStatus?.market_phase ?? 'unknown',
  };
};

export default useLiveMarket;
