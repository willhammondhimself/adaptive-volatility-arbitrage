import { create } from 'zustand';

const useBacktestStore = create((set) => ({
  // Backtest configuration
  config: {
    dataDir: 'src/volatility_arbitrage/data/SPY_Options_2019_24',
    maxDays: 100,
    initialCapital: 100000.0,
    entryThresholdPct: 5.0,
    exitThresholdPct: 2.0,
    positionSizePct: 15.0,
    maxPositions: 5,
    // Demo mode for instant response
    demoMode: true,
    // Phase 2 toggles
    useBayesianLstm: false,
    useImpactModel: false,
    useUncertaintySizing: false,
    useLeverage: false,
  },

  // Results
  results: null,
  isLoading: false,
  error: null,

  // Actions
  updateConfig: (key, value) =>
    set((state) => ({
      config: { ...state.config, [key]: value },
    })),

  setResults: (data) =>
    set({ results: data, isLoading: false, error: null }),

  setLoading: (isLoading) => set({ isLoading }),

  setError: (error) => set({ error, isLoading: false }),

  reset: () =>
    set({
      config: {
        dataDir: 'src/volatility_arbitrage/data/SPY_Options_2019_24',
        maxDays: 100,
        initialCapital: 100000.0,
        entryThresholdPct: 5.0,
        exitThresholdPct: 2.0,
        positionSizePct: 15.0,
        maxPositions: 5,
        demoMode: true,
        useBayesianLstm: false,
        useImpactModel: false,
        useUncertaintySizing: false,
        useLeverage: false,
      },
      results: null,
      error: null,
    }),
}));

export default useBacktestStore;
