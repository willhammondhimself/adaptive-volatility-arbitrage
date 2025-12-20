import { create } from 'zustand';

const useForecastStore = create((set) => ({
  // Forecast configuration
  config: {
    horizon: 1,
    nSamples: 50,
    hiddenSize: 64,
    dropoutP: 0.2,
    uncertaintyPenalty: 2.0,
  },

  // Input returns (for demo, generate from volatility)
  returns: [],

  // Results
  forecast: null,
  isLoading: false,
  error: null,

  // Actions
  updateConfig: (key, value) =>
    set((state) => ({
      config: { ...state.config, [key]: value },
    })),

  setReturns: (returns) => set({ returns }),

  setForecast: (data) =>
    set({ forecast: data, isLoading: false, error: null }),

  setLoading: (isLoading) => set({ isLoading }),

  setError: (error) => set({ error, isLoading: false }),

  reset: () =>
    set({
      config: {
        horizon: 1,
        nSamples: 50,
        hiddenSize: 64,
        dropoutP: 0.2,
        uncertaintyPenalty: 2.0,
      },
      returns: [],
      forecast: null,
      error: null,
    }),
}));

export default useForecastStore;
