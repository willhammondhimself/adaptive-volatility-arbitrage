import { create } from 'zustand';

const useSurfaceExplorerStore = create((set) => ({
  // Mode settings
  mode: 'black_scholes', // 'heston' | 'black_scholes' | 'market_iv'
  valueType: 'price', // 'price' | 'iv'

  // Heston parameters
  hestonParams: {
    v0: 0.04,
    theta: 0.05,
    kappa: 2.0,
    sigma_v: 0.3,
    rho: -0.7,
  },

  // BS parameters
  bsSigma: 0.20,

  // Common parameters
  spot: 100.0,
  r: 0.05,
  strikeRange: [80, 120],
  maturityRange: [0.25, 2.0],
  numStrikes: 40,
  numMaturities: 20,

  // Market IV settings
  symbol: 'SPY',
  expiryCount: 5,

  // Results
  surface: null,
  isLoading: false,
  error: null,

  // View settings
  viewMode: '2d', // '2d' | '3d'

  // Actions
  setMode: (mode) => set({ mode }),

  setValueType: (valueType) => set({ valueType }),

  updateHestonParam: (key, value) =>
    set((state) => ({
      hestonParams: { ...state.hestonParams, [key]: value },
    })),

  setBsSigma: (bsSigma) => set({ bsSigma }),

  setSpot: (spot) => set({ spot }),

  setR: (r) => set({ r }),

  setStrikeRange: (strikeRange) => set({ strikeRange }),

  setMaturityRange: (maturityRange) => set({ maturityRange }),

  setSymbol: (symbol) => set({ symbol }),

  setExpiryCount: (expiryCount) => set({ expiryCount }),

  setSurface: (data) =>
    set({ surface: data, isLoading: false, error: null }),

  setLoading: (isLoading) => set({ isLoading }),

  setError: (error) => set({ error, isLoading: false }),

  toggleViewMode: () =>
    set((state) => ({
      viewMode: state.viewMode === '2d' ? '3d' : '2d',
    })),

  reset: () =>
    set({
      mode: 'black_scholes',
      valueType: 'price',
      hestonParams: {
        v0: 0.04,
        theta: 0.05,
        kappa: 2.0,
        sigma_v: 0.3,
        rho: -0.7,
      },
      bsSigma: 0.20,
      spot: 100.0,
      r: 0.05,
      strikeRange: [80, 120],
      maturityRange: [0.25, 2.0],
      symbol: 'SPY',
      expiryCount: 5,
      viewMode: '2d',
    }),
}));

export default useSurfaceExplorerStore;
