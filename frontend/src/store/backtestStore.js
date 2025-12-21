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
    // Year selection for historical data (default to single year for speed)
    selectedYears: [2019],
    // Phase 2 toggles
    useBayesianLstm: false,
    useImpactModel: false,
    useUncertaintySizing: false,
    useLeverage: false,
  },

  // Backtest results
  results: null,
  isLoading: false,
  error: null,

  // Monte Carlo state
  monteCarloResults: null,
  monteCarloLoading: false,
  monteCarloError: null,
  numSimulations: 10000,

  // Backtest actions
  updateConfig: (key, value) =>
    set((state) => ({
      config: { ...state.config, [key]: value },
    })),

  setResults: (data) =>
    set({ results: data, isLoading: false, error: null }),

  setLoading: (isLoading) => set({ isLoading }),

  setError: (error) => set({ error, isLoading: false }),

  // Monte Carlo actions
  setMonteCarloResults: (results) =>
    set({ monteCarloResults: results, monteCarloLoading: false, monteCarloError: null }),

  setMonteCarloLoading: (loading) => set({ monteCarloLoading: loading }),

  setMonteCarloError: (error) =>
    set({ monteCarloError: error, monteCarloLoading: false }),

  setNumSimulations: (num) => set({ numSimulations: num }),

  clearMonteCarloResults: () =>
    set({ monteCarloResults: null, monteCarloError: null }),

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
        selectedYears: [2019],
        useBayesianLstm: false,
        useImpactModel: false,
        useUncertaintySizing: false,
        useLeverage: false,
      },
      results: null,
      error: null,
      monteCarloResults: null,
      monteCarloError: null,
    }),
}));

export default useBacktestStore;
