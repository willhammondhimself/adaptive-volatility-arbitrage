import { create } from 'zustand';

const usePaperTradingStore = create((set) => ({
  // Configuration
  config: {
    initialCapital: 100000,
    uncertaintyThreshold: 0.02,
    positionPct: 0.10,
  },

  // Status
  status: null,
  trades: [],
  stats: null,

  // Loading states
  isLoading: false,
  isStarting: false,
  isStopping: false,
  error: null,

  // Actions
  updateConfig: (key, value) =>
    set((state) => ({
      config: { ...state.config, [key]: value },
    })),

  setStatus: (status) => set({ status }),

  setTrades: (trades) => set({ trades }),

  setStats: (stats) => set({ stats }),

  setLoading: (isLoading) => set({ isLoading }),

  setStarting: (isStarting) => set({ isStarting }),

  setStopping: (isStopping) => set({ isStopping }),

  setError: (error) => set({ error }),

  reset: () =>
    set({
      status: null,
      trades: [],
      stats: null,
      error: null,
    }),
}));

export default usePaperTradingStore;
