import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  Dataset, JoinRule, ChainStep, ProjectConfig,
  ModelMetrics, StatBucket, FeatureImportance,
} from '../lib/types';
import * as api from '../lib/api';

export interface AppState {
  datasets: Dataset[];
  joins: JoinRule[];
  steps: ChainStep[];
  config: ProjectConfig | null;
  metrics: ModelMetrics;
  stats: StatBucket[];
  importance: FeatureImportance[];
  status: string;
  statusType: 'info' | 'error' | 'success';
}

export function useAppState() {
  const [state, setState] = useState<AppState>({
    datasets: [],
    joins: [],
    steps: [],
    config: null,
    metrics: { trained: false },
    stats: [],
    importance: [],
    status: '',
    statusType: 'info',
  });

  const setStatus = useCallback((msg: string, type: AppState['statusType'] = 'info') => {
    setState(s => ({ ...s, status: msg, statusType: type }));
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [datasets, joins, steps, config, metrics, stats, importance] = await Promise.all([
        api.listDatasets().catch(() => []),
        api.listJoins().catch(() => []),
        api.listChain().catch(() => []),
        api.getConfig().catch(() => null),
        api.getMetrics().catch(() => ({ trained: false })),
        api.getStats().catch(() => []),
        api.getImportance().catch(() => []),
      ]);
      setState(s => ({ ...s, datasets, joins, steps, config, metrics, stats, importance }));
    } catch (_) {}
  }, []);

  // Initial load + polling
  useEffect(() => { refresh(); }, [refresh]);
  useInterval(refresh, 4000);

  // ── Dataset actions ──────────────────────────────────────────────────────

  const uploadDataset = useCallback(async (
    csvText: string,
    filename: string,
    isPrimary: boolean
  ) => {
    setStatus(`Uploading ${filename}…`, 'info');
    try {
      const result = await api.uploadDataset(csvText, filename, { isPrimary });
      setStatus(`${filename} loaded — ${result.row_count} rows`, 'success');
      await refresh();
    } catch (e: any) {
      setStatus(`Upload failed: ${e.message}`, 'error');
    }
  }, [refresh, setStatus]);

  const removeDataset = useCallback(async (id: number) => {
    await api.deleteDataset(id);
    await refresh();
  }, [refresh]);

  const updateCols = useCallback(async (id: number, cols: string[]) => {
    await api.updateDatasetCols(id, cols);
    await refresh();
  }, [refresh]);

  // ── Join actions ─────────────────────────────────────────────────────────

  const addJoin = useCallback(async (rule: Omit<JoinRule, 'rule_id'>) => {
    await api.addJoin(rule);
    await refresh();
  }, [refresh]);

  const removeJoin = useCallback(async (ruleId: number) => {
    await api.deleteJoin(ruleId);
    await refresh();
  }, [refresh]);

  // ── Config actions ───────────────────────────────────────────────────────

  const saveConfig = useCallback(async (patch: Partial<ProjectConfig>) => {
    await api.setConfig(patch);
    setStatus('Config saved, retraining…', 'info');
    await refresh();
  }, [refresh, setStatus]);

  // ── Chain actions ────────────────────────────────────────────────────────

  const toggleStep = useCallback(async (step: ChainStep) => {
    await api.updateChainStep(step.step_id, { enabled: !step.enabled });
    await refresh();
  }, [refresh]);

  const addStep = useCallback(async (step: Omit<ChainStep, 'step_id'>) => {
    await api.addChainStep(step);
    await refresh();
  }, [refresh]);

  const removeStep = useCallback(async (id: number) => {
    await api.deleteChainStep(id);
    await refresh();
  }, [refresh]);

  return {
    state,
    refresh,
    setStatus,
    uploadDataset,
    removeDataset,
    updateCols,
    addJoin,
    removeJoin,
    saveConfig,
    toggleStep,
    addStep,
    removeStep,
  };
}

// ── Interval hook ─────────────────────────────────────────────────────────────

function useInterval(fn: () => void, delay: number) {
  const ref = useRef(fn);
  useEffect(() => { ref.current = fn; }, [fn]);
  useEffect(() => {
    const id = setInterval(() => ref.current(), delay);
    return () => clearInterval(id);
  }, [delay]);
}
