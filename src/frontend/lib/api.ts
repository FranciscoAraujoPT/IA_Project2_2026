import type {
  Dataset, JoinRule, ChainStep, ProjectConfig,
  ModelMetrics, StatBucket, FeatureImportance, ColumnMeta, ModelInfo,
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

export const listModels = (): Promise<ModelInfo[]> =>
  request('/models');

export const getMetrics = (modelId?: string): Promise<ModelMetrics> =>
  request(`/metrics${modelId ? `?model=${modelId}` : ''}`);

export const getImportance = (modelId?: string): Promise<FeatureImportance[]> =>
  request(`/importance${modelId ? `?model=${modelId}` : ''}`);

export const getStats = (): Promise<StatBucket[]> =>
  request('/stats');

export const getMergedColumns = (): Promise<ColumnMeta[]> =>
  request('/merged-columns');

export const getPredictInputs = (): Promise<import('./types').ColumnMeta[]> =>
  request('/predict-inputs');

export const predict = (row: Record<string, any>, modelId?: string): Promise<{ probability: number; percentage: number; model_id?: string }> =>
  request(`/predict${modelId ? `?model=${modelId}` : ''}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(row),
  });

export const getKFoldMetrics = (): Promise<import('./types').KFoldMetrics | Record<string, never>> =>
  request('/kfold-metrics');

export const getKFoldSample = (pct = 0.30, threshold = 0.5): Promise<import('./types').KFoldSampleResponse> =>
  request(`/kfold-sample?pct=${pct}&threshold=${threshold}`);

export const getSampleRows = (n?: number): Promise<import('./types').SampleRowsResponse> =>
  request(`/sample-rows${n !== undefined ? `?n=${n}` : ''}`);

export const predictCompare = (
  row: Record<string, any>,
  realOutcome: number,
  threshold?: number,
): Promise<import('./types').PredictCompareResult> =>
  request('/predict-compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ row, real_outcome: realOutcome, threshold }),
  });