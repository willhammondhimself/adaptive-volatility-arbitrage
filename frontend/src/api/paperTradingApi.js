import client from './client';

export const getStatus = () => client.get('/api/v1/paper-trading/status');

export const getTrades = (limit = 100) =>
  client.get(`/api/v1/paper-trading/trades?limit=${limit}`);

export const getStats = () => client.get('/api/v1/paper-trading/stats');

export const startTrading = (config) =>
  client.post('/api/v1/paper-trading/start', config);

export const stopTrading = () => client.post('/api/v1/paper-trading/stop');
