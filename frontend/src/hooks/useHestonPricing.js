import { useEffect } from 'react';
import { computePriceSurface } from '../api/hestonApi';
import useHestonStore from '../store/hestonStore';
import { useDebounce } from './useDebounce';

export const useHestonPricing = () => {
  const {
    params,
    spot,
    strikeRange,
    maturityRange,
    numStrikes,
    numMaturities,
    setLoading,
    setPriceSurface,
    setError,
  } = useHestonStore();

  // Debounce parameters to avoid excessive API calls
  const debouncedParams = useDebounce(params, 500);

  useEffect(() => {
    const fetchPriceSurface = async () => {
      setLoading(true);
      try {
        const data = await computePriceSurface({
          params: debouncedParams,
          spot,
          strike_range: strikeRange,
          maturity_range: maturityRange,
          num_strikes: numStrikes,
          num_maturities: numMaturities,
        });
        setPriceSurface(data);
      } catch (error) {
        console.error('Error fetching price surface:', error);
        setError(error.message);
      }
    };

    fetchPriceSurface();
  }, [
    debouncedParams,
    spot,
    strikeRange,
    maturityRange,
    numStrikes,
    numMaturities,
  ]);
};
