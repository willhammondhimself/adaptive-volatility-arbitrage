import client from './client';

export const predictVolatility = async (config) => {
  const response = await client.post('/api/v1/forecast/predict', config);
  return response.data;
};
