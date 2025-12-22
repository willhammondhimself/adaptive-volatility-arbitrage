import { create } from 'zustand';

const useBSPlaygroundStore = create((set) => ({
  // Option parameters
  S: 100.0,
  K: 100.0,
  T: 1.0,
  r: 0.05,
  sigma: 0.20,
  optionType: 'call',

  // Pricing results
  pricing: null,
  isLoading: false,
  error: null,

  // Heatmap configuration
  spotRange: [80, 120],
  volRange: [0.10, 0.40],
  numSpots: 50,
  numVols: 30,

  // Heatmap results
  heatmapData: null,
  isHeatmapLoading: false,
  heatmapError: null,

  // Actions
  setS: (S) => set({ S }),
  setK: (K) => set({ K }),
  setT: (T) => set({ T }),
  setR: (r) => set({ r }),
  setSigma: (sigma) => set({ sigma }),
  setOptionType: (optionType) => set({ optionType }),

  setSpotRange: (spotRange) => set({ spotRange }),
  setVolRange: (volRange) => set({ volRange }),

  setPricing: (data) => set({ pricing: data, isLoading: false, error: null }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error, isLoading: false }),

  setHeatmapData: (data) =>
    set({ heatmapData: data, isHeatmapLoading: false, heatmapError: null }),
  setHeatmapLoading: (isHeatmapLoading) => set({ isHeatmapLoading }),
  setHeatmapError: (heatmapError) =>
    set({ heatmapError, isHeatmapLoading: false }),

  reset: () =>
    set({
      S: 100.0,
      K: 100.0,
      T: 1.0,
      r: 0.05,
      sigma: 0.20,
      optionType: 'call',
      spotRange: [80, 120],
      volRange: [0.10, 0.40],
      pricing: null,
      heatmapData: null,
      error: null,
      heatmapError: null,
    }),
}));

export default useBSPlaygroundStore;
