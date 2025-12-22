import axios from 'axios';

const API_BASE = 'http://localhost:8000/api/v1';

export const runDeltaHedgedBacktest = async (params) => {
  const response = await axios.post(`${API_BASE}/backtest/delta-hedged`, params);
  return response;
};
