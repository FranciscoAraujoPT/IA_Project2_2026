export interface ColumnMeta {
  name: string;
  type: 'numeric' | 'categorical';
  min: number | null;
  max: number | null;
  mean?: number | null;
  uniqueValues: string[] | null;
}

export interface Dataset {
  dataset_id: number;
  label: string;
  schema: ColumnMeta[];
  row_count: number;
  active_cols: string[];
  uploaded_at: string;
}

export interface JoinRule {
  rule_id: number;
  left_id: number;
  right_id: number;
  left_col: string;
  right_col: string;
  join_type: 'left' | 'inner' | 'outer';
}

export interface ChainStep {
  step_id: number;
  step_order: number;
  name: string;
  type: 'interaction' | 'log' | 'ratio' | 'binary_threshold' | 'encode' | 'passthrough';
  config: Record<string, any>;
  enabled: boolean;
}

export interface ProjectConfig {
  outcome_col: string;
  response_col: string;
  primary_id: number;
  k_folds: number;
}

export interface ModelMetrics {
  trained: boolean;
  samples?: number;
  accuracy?: number;
  roc_auc?: number;
  f1?: number;
  precision?: number;
  recall?: number;
  positive_rate?: number;
  features?: number;
}

export interface StatBucket {
  range: string;
  purchaseRate: number;
  total: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export type Tab = 'upload' | 'join' | 'chain' | 'analyze' | 'predict';

export interface KFoldMetrics {
  k: number;
  accuracy: { mean: number; std: number };
  roc_auc: { mean: number; std: number };
  f1: { mean: number; std: number };
  precision: { mean: number; std: number };
  recall: { mean: number; std: number };
  fold_aucs: number[];
}

export interface SampleRowsResponse {
  outcome_col: string;
  rows: Record<string, any>[];
}

export interface PredictCompareResult {
  probability: number;
  percentage: number;
  predicted: number;
  real: number | null;
  correct: boolean | null;
}

export interface KFoldSampleRow {
  _fold: number;
  _real: number;
  _prob: number;
  _predicted: number;
  _correct: boolean;
  [key: string]: any;
}

export interface KFoldSampleResponse {
  outcome_col: string;
  rows: KFoldSampleRow[];
  total: number;
  k: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  status: 'online' | 'waiting_for_data';
  supports_prediction: boolean;
  metrics: Partial<ModelMetrics> & { model_id?: string };
}