import { create } from 'zustand';

const useHestonStore = create((set) => ({
  // Heston parameters
  params: {
    v0: 0.04,
    theta: 0.05,
    kappa: 2.0,
    sigma_v: 0.3,
    rho: -0.7,
    r: 0.05,
  },

  // Grid configuration
  spot: 100.0,
  strikeRange: [80, 120],
  maturityRange: [0.25, 2.0],
  numStrikes: 40,
  numMaturities: 20,

  // Results
  priceSurface: null,
  isLoading: false,
  error: null,

  // View settings
  viewMode: '2d', // '2d' | '3d'

  // Actions
  updateParam: (key, value) =>
    set((state) => ({
      params: { ...state.params, [key]: value },
    })),

  setSpot: (spot) => set({ spot }),

  setStrikeRange: (strikeRange) => set({ strikeRange }),

  setMaturityRange: (maturityRange) => set({ maturityRange }),

  setPriceSurface: (data) =>
    set({ priceSurface: data, isLoading: false, error: null }),

  setLoading: (isLoading) => set({ isLoading }),

  setError: (error) => set({ error, isLoading: false }),

  toggleViewMode: () =>
    set((state) => ({
      viewMode: state.viewMode === '2d' ? '3d' : '2d',
    })),

  reset: () =>
    set({
      params: {
        v0: 0.04,
        theta: 0.05,
        kappa: 2.0,
        sigma_v: 0.3,
        rho: -0.7,
        r: 0.05,
      },
      spot: 100.0,
      strikeRange: [80, 120],
      maturityRange: [0.25, 2.0],
      viewMode: '2d',
    }),
}));

export default useHestonStore;
