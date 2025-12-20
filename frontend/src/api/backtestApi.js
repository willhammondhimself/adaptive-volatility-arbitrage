import client from './client';

export const runBacktest = async (config) => {
  const response = await client.post('/api/v1/backtest/run', config);
  return response.data;
};
