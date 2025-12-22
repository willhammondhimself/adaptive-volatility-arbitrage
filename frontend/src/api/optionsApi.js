import axios from 'axios';

const API_BASE = 'http://localhost:8000/api/v1/options';

export const computeBSPrice = async (params) => {
  const response = await axios.post(`${API_BASE}/bs/price`, params);
  return response.data;
};

export const computePnLHeatmap = async (params) => {
  const response = await axios.post(`${API_BASE}/bs/pnl-heatmap`, params);
  return response.data;
};

export const computeUnifiedSurface = async (params) => {
  const response = await axios.post(`${API_BASE}/surface`, params);
  return response.data;
};

export const getIVSurface = async (symbol, expiryCount = 5) => {
  const response = await axios.get(`${API_BASE}/iv-surface/${symbol}`, {
    params: { expiry_count: expiryCount },
  });
  return response.data;
};

// Snapshot API
const SNAPSHOTS_BASE = 'http://localhost:8000/api/v1/snapshots';

export const captureSnapshot = async (symbol, expiryCount = 5) => {
  const response = await axios.post(`${SNAPSHOTS_BASE}/capture/${symbol}`, null, {
    params: { expiry_count: expiryCount },
  });
  return response.data;
};

export const listSnapshots = async (symbol = null, limit = 100, offset = 0) => {
  const params = { limit, offset };
  if (symbol) params.symbol = symbol;
  const response = await axios.get(SNAPSHOTS_BASE, { params });
  return response.data;
};

export const getSnapshot = async (snapshotId) => {
  const response = await axios.get(`${SNAPSHOTS_BASE}/${snapshotId}`);
  return response.data;
};

export const deleteSnapshot = async (snapshotId) => {
  const response = await axios.delete(`${SNAPSHOTS_BASE}/${snapshotId}`);
  return response.data;
};

export const getSnapshotSymbols = async () => {
  const response = await axios.get(`${SNAPSHOTS_BASE}/symbols`);
  return response.data;
};
