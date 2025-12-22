import { useEffect, useCallback } from 'react';
import useSnapshotStore from '../store/snapshotStore';
import {
  listSnapshots,
  getSnapshot,
  captureSnapshot,
  deleteSnapshot,
} from '../api/optionsApi';

export const useSnapshots = (symbol = null) => {
  const {
    snapshots,
    selectedSnapshotId,
    selectedSnapshot,
    isLoading,
    isCapturing,
    error,
    setSnapshots,
    setSelectedSnapshotId,
    setSelectedSnapshot,
    setIsLoading,
    setIsCapturing,
    setError,
    addSnapshot,
    removeSnapshot,
    clearSelectedSnapshot,
  } = useSnapshotStore();

  // Fetch snapshots list on mount
  useEffect(() => {
    const fetchSnapshots = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await listSnapshots(symbol);
        setSnapshots(data);
      } catch (err) {
        setError(err.message || 'Failed to fetch snapshots');
      } finally {
        setIsLoading(false);
      }
    };

    fetchSnapshots();
  }, [symbol, setSnapshots, setIsLoading, setError]);

  // Fetch snapshot details when selected
  useEffect(() => {
    if (!selectedSnapshotId) {
      setSelectedSnapshot(null);
      return;
    }

    const fetchSnapshotDetails = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await getSnapshot(selectedSnapshotId);
        setSelectedSnapshot(data);
      } catch (err) {
        setError(err.message || 'Failed to fetch snapshot');
        setSelectedSnapshotId(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSnapshotDetails();
  }, [selectedSnapshotId, setSelectedSnapshot, setSelectedSnapshotId, setIsLoading, setError]);

  // Capture new snapshot
  const capture = useCallback(async (captureSymbol, expiryCount = 5) => {
    setIsCapturing(true);
    setError(null);
    try {
      const metadata = await captureSnapshot(captureSymbol, expiryCount);
      addSnapshot(metadata);
      return metadata;
    } catch (err) {
      setError(err.message || 'Failed to capture snapshot');
      throw err;
    } finally {
      setIsCapturing(false);
    }
  }, [setIsCapturing, setError, addSnapshot]);

  // Delete snapshot
  const remove = useCallback(async (snapshotId) => {
    setError(null);
    try {
      await deleteSnapshot(snapshotId);
      removeSnapshot(snapshotId);
    } catch (err) {
      setError(err.message || 'Failed to delete snapshot');
      throw err;
    }
  }, [setError, removeSnapshot]);

  // Select snapshot by ID
  const select = useCallback((snapshotId) => {
    setSelectedSnapshotId(snapshotId);
  }, [setSelectedSnapshotId]);

  // Clear selection
  const clearSelection = useCallback(() => {
    clearSelectedSnapshot();
  }, [clearSelectedSnapshot]);

  // Refresh snapshots list
  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await listSnapshots(symbol);
      setSnapshots(data);
    } catch (err) {
      setError(err.message || 'Failed to refresh snapshots');
    } finally {
      setIsLoading(false);
    }
  }, [symbol, setSnapshots, setIsLoading, setError]);

  return {
    snapshots,
    selectedSnapshotId,
    selectedSnapshot,
    isLoading,
    isCapturing,
    error,
    capture,
    remove,
    select,
    clearSelection,
    refresh,
  };
};
