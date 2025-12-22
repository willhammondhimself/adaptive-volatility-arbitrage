import { useEffect, useCallback } from 'react';
import { computeBSPrice, computePnLHeatmap } from '../api/optionsApi';
import useBSPlaygroundStore from '../store/bsPlaygroundStore';
import { useDebounce } from './useDebounce';

export const useBSPricing = () => {
  const {
    S,
    K,
    T,
    r,
    sigma,
    optionType,
    spotRange,
    volRange,
    numSpots,
    numVols,
    pricing,
    setLoading,
    setPricing,
    setError,
    setHeatmapLoading,
    setHeatmapData,
    setHeatmapError,
  } = useBSPlaygroundStore();

  // Debounce parameters
  const debouncedS = useDebounce(S, 300);
  const debouncedK = useDebounce(K, 300);
  const debouncedT = useDebounce(T, 300);
  const debouncedR = useDebounce(r, 300);
  const debouncedSigma = useDebounce(sigma, 300);
  const debouncedOptionType = useDebounce(optionType, 300);

  // Fetch pricing
  useEffect(() => {
    const fetchPricing = async () => {
      setLoading(true);
      try {
        const data = await computeBSPrice({
          S: debouncedS,
          K: debouncedK,
          T: debouncedT,
          r: debouncedR,
          sigma: debouncedSigma,
          option_type: debouncedOptionType,
        });
        setPricing(data);
      } catch (error) {
        console.error('Error fetching BS price:', error);
        setError(error.message);
      }
    };

    fetchPricing();
  }, [debouncedS, debouncedK, debouncedT, debouncedR, debouncedSigma, debouncedOptionType]);

  // Fetch heatmap when pricing changes
  const fetchHeatmap = useCallback(async () => {
    if (!pricing) return;

    setHeatmapLoading(true);
    try {
      const data = await computePnLHeatmap({
        K: debouncedK,
        T: debouncedT,
        r: debouncedR,
        sigma: debouncedSigma,
        option_type: debouncedOptionType,
        entry_price: pricing.price,
        spot_range: spotRange,
        vol_range: volRange,
        num_spots: numSpots,
        num_vols: numVols,
      });
      setHeatmapData(data);
    } catch (error) {
      console.error('Error fetching heatmap:', error);
      setHeatmapError(error.message);
    }
  }, [
    pricing,
    debouncedK,
    debouncedT,
    debouncedR,
    debouncedSigma,
    debouncedOptionType,
    spotRange,
    volRange,
    numSpots,
    numVols,
  ]);

  // Auto-fetch heatmap when pricing changes
  useEffect(() => {
    fetchHeatmap();
  }, [fetchHeatmap]);

  return { fetchHeatmap };
};
