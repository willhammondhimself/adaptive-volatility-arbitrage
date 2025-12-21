import { create } from 'zustand';
import { fetchQuote, fetchVix, fetchMarketStatus } from '../api/marketApi';

const useLiveMarketStore = create((set, get) => ({
  // Live mode state
  isLiveMode: false,
  pollingInterval: 30000, // 30 seconds

  // Market data
  liveQuote: null,
  vixLevel: null,
  marketStatus: null,

  // Loading/error states
  isLoading: false,
  error: null,
  lastUpdated: null,

  // Actions
  toggleLiveMode: () =>
    set((state) => ({
      isLiveMode: !state.isLiveMode,
      error: null,
    })),

  setLiveMode: (enabled) =>
    set({
      isLiveMode: enabled,
      error: null,
    }),

  setPollingInterval: (interval) =>
    set({ pollingInterval: interval }),

  // Fetch actions
  fetchQuote: async (symbol = 'SPY') => {
    try {
      const quote = await fetchQuote(symbol);
      set({
        liveQuote: quote,
        lastUpdated: new Date(),
        error: null,
      });
      return quote;
    } catch (err) {
      set({ error: err.message || 'Failed to fetch quote' });
      throw err;
    }
  },

  fetchVix: async () => {
    try {
      const vix = await fetchVix();
      set({
        vixLevel: vix,
        lastUpdated: new Date(),
        error: null,
      });
      return vix;
    } catch (err) {
      set({ error: err.message || 'Failed to fetch VIX' });
      throw err;
    }
  },

  fetchMarketStatus: async () => {
    try {
      const status = await fetchMarketStatus();
      set({ marketStatus: status });
      return status;
    } catch (err) {
      // Non-critical, don't set error
      console.warn('Failed to fetch market status:', err);
      return null;
    }
  },

  // Fetch all market data
  fetchAll: async (symbol = 'SPY') => {
    set({ isLoading: true });
    try {
      const [quote, vix, status] = await Promise.all([
        get().fetchQuote(symbol),
        get().fetchVix(),
        get().fetchMarketStatus(),
      ]);
      set({ isLoading: false });
      return { quote, vix, status };
    } catch (err) {
      set({ isLoading: false });
      throw err;
    }
  },

  clearError: () => set({ error: null }),

  reset: () =>
    set({
      isLiveMode: false,
      liveQuote: null,
      vixLevel: null,
      marketStatus: null,
      isLoading: false,
      error: null,
      lastUpdated: null,
    }),
}));

export default useLiveMarketStore;
