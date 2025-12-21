import client from './client';

/**
 * Fetch stock quote.
 * @param {string} symbol - Ticker symbol (e.g., "SPY")
 */
export const fetchQuote = async (symbol) => {
  const response = await client.get(`/api/v1/market/quote/${symbol}`);
  return response.data;
};

/**
 * Fetch option chain.
 * @param {string} symbol - Underlying symbol
 * @param {string} [expiry] - Expiration date (YYYY-MM-DD), uses nearest if not specified
 */
export const fetchOptionChain = async (symbol, expiry = null) => {
  const params = expiry ? { expiry } : {};
  const response = await client.get(`/api/v1/market/option-chain/${symbol}`, { params });
  return response.data;
};

/**
 * Fetch VIX quote.
 */
export const fetchVix = async () => {
  const response = await client.get('/api/v1/market/vix');
  return response.data;
};

/**
 * Fetch market status.
 */
export const fetchMarketStatus = async () => {
  const response = await client.get('/api/v1/market/status');
  return response.data;
};

/**
 * Clear market data cache.
 */
export const clearMarketCache = async () => {
  const response = await client.delete('/api/v1/market/cache');
  return response.data;
};
