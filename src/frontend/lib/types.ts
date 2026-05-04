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