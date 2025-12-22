import { useEffect, useRef, useCallback } from 'react';
import useSurfaceExplorerStore from '../store/surfaceExplorerStore';
import { computeUnifiedSurface, getIVSurface } from '../api/optionsApi';

const DEBOUNCE_MS = 300;

export const useSurfaceExplorer = () => {
  const {
    mode,
    valueType,
    hestonParams,
    bsSigma,
    spot,
    r,
    strikeRange,
    maturityRange,
    numStrikes,
    numMaturities,
    symbol,
    expiryCount,
    setSurface,
    setLoading,
    setError,
  } = useSurfaceExplorerStore();

  const debounceRef = useRef(null);
  const abortRef = useRef(null);

  const fetchSurface = useCallback(async () => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
    abortRef.current = new AbortController();

    setLoading(true);

    try {
      let response;

      if (mode === 'market_iv') {
        response = await getIVSurface(symbol, expiryCount);
        // Transform to unified format
        setSurface({
          mode: 'market_iv',
          strikes: response.strikes,
          maturities: response.maturities,
          values: response.ivs,
          value_type: 'iv',
          computation_time_ms: response.computation_time_ms,
          cache_hit: false,
          symbol: response.symbol,
          underlying_price: response.underlying_price,
          expiry_dates: response.expiry_dates,
        });
      } else {
        const params = {
          mode,
          value_type: valueType,
          spot,
          r,
          strike_range: strikeRange,
          maturity_range: maturityRange,
          num_strikes: numStrikes,
          num_maturities: numMaturities,
        };

        if (mode === 'heston') {
          params.heston_params = hestonParams;
        } else if (mode === 'black_scholes') {
          params.bs_sigma = bsSigma;
        }

        response = await computeUnifiedSurface(params);
        setSurface(response);
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message || 'Failed to fetch surface');
      }
    }
  }, [
    mode,
    valueType,
    hestonParams,
    bsSigma,
    spot,
    r,
    strikeRange,
    maturityRange,
    numStrikes,
    numMaturities,
    symbol,
    expiryCount,
    setSurface,
    setLoading,
    setError,
  ]);

  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(() => {
      fetchSurface();
    }, DEBOUNCE_MS);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [fetchSurface]);

  return { refetch: fetchSurface };
};

export default useSurfaceExplorer;
