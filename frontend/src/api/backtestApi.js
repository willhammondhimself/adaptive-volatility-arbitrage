import client from './client';

export const runBacktest = async (config) => {
  const response = await client.post('/api/v1/backtest/run', config);
  return response.data;
};

export const runMonteCarloSimulation = async (tradeReturns, numSimulations = 10000) => {
  const response = await client.post('/api/v1/backtest/monte-carlo', {
    trade_returns: tradeReturns,
    n_simulations: numSimulations,
  });
  return response.data;
};
