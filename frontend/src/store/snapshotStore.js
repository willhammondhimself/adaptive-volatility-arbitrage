import { create } from 'zustand';

const useSnapshotStore = create((set, get) => ({
  // State
  snapshots: [],
  selectedSnapshotId: null,
  selectedSnapshot: null,
  isLoading: false,
  isCapturing: false,
  error: null,

  // Actions
  setSnapshots: (snapshots) => set({ snapshots }),

  setSelectedSnapshotId: (id) => set({ selectedSnapshotId: id }),

  setSelectedSnapshot: (snapshot) => set({ selectedSnapshot: snapshot }),

  setIsLoading: (isLoading) => set({ isLoading }),

  setIsCapturing: (isCapturing) => set({ isCapturing }),

  setError: (error) => set({ error }),

  addSnapshot: (snapshot) => set((state) => ({
    snapshots: [snapshot, ...state.snapshots],
  })),

  removeSnapshot: (id) => set((state) => ({
    snapshots: state.snapshots.filter((s) => s.id !== id),
    selectedSnapshotId: state.selectedSnapshotId === id ? null : state.selectedSnapshotId,
    selectedSnapshot: state.selectedSnapshot?.id === id ? null : state.selectedSnapshot,
  })),

  clearSelectedSnapshot: () => set({
    selectedSnapshotId: null,
    selectedSnapshot: null,
  }),

  reset: () => set({
    snapshots: [],
    selectedSnapshotId: null,
    selectedSnapshot: null,
    isLoading: false,
    isCapturing: false,
    error: null,
  }),
}));

export default useSnapshotStore;
