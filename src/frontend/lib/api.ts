import type {
  Dataset, JoinRule, ChainStep, ProjectConfig,
  ModelMetrics, StatBucket, FeatureImportance, ColumnMeta,
} from './types';

const BASE = '/api';

async function request<T>(url: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(BASE + url, init);
  const json = await res.json();
  if (!res.ok) throw new Error(json.error ?? `HTTP ${res.status}`);
  return json as T;
}

// ── Datasets ──────────────────────────────────────────────────────────────────

export const listDatasets = (): Promise<Dataset[]> =>
  request('/datasets');

export const uploadDataset = (
  csvText: string,
  label: string,
  options: { isPrimary?: boolean; outcomeCol?: string; responseCol?: string } = {}
): Promise<Dataset & { schema: any[] }> =>
  request('/datasets/upload', {
    method: 'POST',
    headers: {
      'Content-Type': 'text/csv',
      'X-Dataset-Label': label,
      'X-Is-Primary': options.isPrimary ? 'true' : 'false',
      ...(options.outcomeCol ? { 'X-Outcome-Col': options.outcomeCol } : {}),
      ...(options.responseCol ? { 'X-Response-Col': options.responseCol } : {}),
    },
    body: csvText,
  });

export const deleteDataset = (id: number): Promise<{ success: boolean }> =>
  request(`/datasets/${id}`, { method: 'DELETE' });

export const updateDatasetCols = (id: number, cols: string[]): Promise<{ success: boolean }> =>
  request(`/datasets/${id}/cols`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cols }),
  });

// ── Joins ─────────────────────────────────────────────────────────────────────

export const listJoins = (): Promise<JoinRule[]> =>
  request('/joins');

export const addJoin = (rule: Omit<JoinRule, 'rule_id'>): Promise<{ rule_id: number }> =>
  request('/joins', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(rule),
  });

export const deleteJoin = (ruleId: number): Promise<{ success: boolean }> =>
  request(`/joins/${ruleId}`, { method: 'DELETE' });

// ── Config ────────────────────────────────────────────────────────────────────

export const getConfig = (): Promise<ProjectConfig> =>
  request('/config');

export const setConfig = (config: Partial<ProjectConfig>): Promise<{ success: boolean }> =>
  request('/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });

// ── Chain ─────────────────────────────────────────────────────────────────────

export const listChain = (): Promise<ChainStep[]> =>
  request('/chain');

export const addChainStep = (step: Omit<ChainStep, 'step_id'>): Promise<{ step_id: number }> =>
  request('/chain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(step),
  });

export const updateChainStep = (id: number, patch: Partial<ChainStep>): Promise<{ success: boolean }> =>
  request(`/chain/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });

export const deleteChainStep = (id: number): Promise<{ success: boolean }> =>
  request(`/chain/${id}`, { method: 'DELETE' });

// ── Model ─────────────────────────────────────────────────────────────────────

export const getMetrics = (): Promise<ModelMetrics> =>
  request('/metrics');

export const getImportance = (): Promise<FeatureImportance[]> =>
  request('/importance');

export const getStats = (): Promise<StatBucket[]> =>
  request('/stats');

export const getMergedColumns = (): Promise<ColumnMeta[]> =>
  request('/merged-columns');

export const getPredictInputs = (): Promise<import('./types').ColumnMeta[]> =>
  request('/predict-inputs');

export const predict = (row: Record<string, any>): Promise<{ probability: number; percentage: number }> =>
  request('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(row),
  });