import client from './client';

export const estimateCosts = async (config) => {
  const response = await client.post('/api/v1/costs/estimate', config);
  return response.data;
};
