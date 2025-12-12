import client from './client';

export const computePriceSurface = async (config) => {
  const response = await client.post('/api/v1/heston/price-surface', config);
  return response.data;
};

export const clearCache = async () => {
  const response = await client.delete('/api/v1/heston/cache');
  return response.data;
};

export const getCacheStats = async () => {
  const response = await client.get('/api/v1/heston/cache/stats');
  return response.data;
};
