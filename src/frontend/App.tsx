import React, { useState, useEffect, useRef } from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import {
  Activity, Target, BrainCircuit, Database, Plus, Upload,
  Trash2, RefreshCw, CheckCircle2, AlertCircle, Settings,
  Layers, Link2, GitBranch, ToggleLeft, ToggleRight, X, ChevronDown, ChevronUp,
  SlidersHorizontal, Lock, Unlock, TrendingUp, Zap,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

import { useAppState } from './hooks/useAppState';
import * as api from './lib/api';
import type {
  Dataset, JoinRule, ChainStep, ProjectConfig,
  ModelMetrics, StatBucket, FeatureImportance, ColumnMeta,
} from './lib/types';

function cn(...inputs: ClassValue[]) { return twMerge(clsx(inputs)); }

type Page = 'analytics' | 'predictions' | 'datasets' | 'joins' | 'chain';

// ── Shared UI ─────────────────────────────────────────────────────────────────

function Badge({ children, color = 'slate' }: { children: React.ReactNode; color?: string }) {
  const colors: Record<string, string> = {
    slate: 'bg-slate-100 text-slate-600', green: 'bg-emerald-100 text-emerald-700',
    red: 'bg-rose-100 text-rose-600', indigo: 'bg-indigo-100 text-indigo-700',
    amber: 'bg-amber-100 text-amber-700', purple: 'bg-purple-100 text-purple-700',
  };
  return (
    <span className={cn('px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wide', colors[color] ?? colors.slate)}>
      {children}
    </span>
  );
}

function Card({ title, icon, children, className }: { title: string; icon: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn('bg-white border border-slate-200 p-4 rounded-2xl shadow-sm hover:shadow-md transition-all', className)}>
      <div className="flex items-center gap-2 mb-3">
        <div className="p-2 bg-slate-50 rounded-lg text-slate-400">{icon}</div>
        <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">{title}</h3>
      </div>
      {children}
    </div>
  );
}

function Toast({ msg, onDone }: { msg: { text: string; ok: boolean } | null; onDone: () => void }) {
  useEffect(() => { if (!msg) return; const t = setTimeout(onDone, 3500); return () => clearTimeout(t); }, [msg]);
  return (
    <AnimatePresence>
      {msg && (
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
          className={cn('flex items-center gap-2 px-4 py-3 rounded-xl text-sm font-medium',
            msg.ok ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-rose-50 text-rose-700 border border-rose-200')}>
          {msg.ok ? <CheckCircle2 className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
          {msg.text}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function Sidebar({ page, setPage, datasets, metrics }: {
  page: Page; setPage: (p: Page) => void;
  datasets: Dataset[]; metrics: ModelMetrics;
}) {
  const nav: { id: Page; label: string; icon: React.ReactNode }[] = [
    { id: 'analytics', label: 'Analytics', icon: <Activity className="w-5 h-5" /> },
    { id: 'predictions', label: 'Predictions', icon: <Target className="w-5 h-5" /> },
    { id: 'datasets', label: 'Datasets', icon: <Database className="w-5 h-5" /> },
    { id: 'joins', label: 'Joins', icon: <Link2 className="w-5 h-5" /> },
    { id: 'chain', label: 'Chain', icon: <GitBranch className="w-5 h-5" /> },
  ];
  return (
    <aside className="w-64 h-full bg-white border-r border-slate-200 hidden lg:flex flex-col shrink-0">
      <div className="p-6 border-b border-slate-100">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold text-sm">P</div>
          <span className="font-bold text-lg tracking-tight">Predictron</span>
        </div>
        <p className="text-[10px] uppercase tracking-widest text-slate-400 font-semibold">Predictive ML Suite</p>
      </div>

      <nav className="p-4 flex-1 space-y-1">
        {nav.map(item => (
          <button key={item.id} onClick={() => setPage(item.id)}
            className={cn('w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-colors text-left',
              page === item.id ? 'bg-indigo-50 text-indigo-700' : 'text-slate-500 hover:bg-slate-50')}>
            {item.icon}{item.label}
          </button>
        ))}
      </nav>

      <div className="p-6 border-t border-slate-100 space-y-3">
        {datasets.length > 0 && (
          <div className="bg-slate-50 rounded-xl p-3 border border-slate-200">
            <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400 mb-1">Loaded Datasets</p>
            {datasets.slice(0, 3).map(d => (
              <p key={d.dataset_id} className="text-xs font-semibold text-slate-700 truncate">{d.label}</p>
            ))}
            {datasets.length > 3 && <p className="text-[10px] text-slate-400">+{datasets.length - 3} more</p>}
          </div>
        )}
        <div className="bg-slate-900 rounded-xl p-4 text-white">
          <p className="text-xs text-slate-400 mb-1">Model Status</p>
          <div className="flex items-center gap-2">
            <div className={cn('w-2 h-2 rounded-full', metrics.trained ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500')} />
            <p className="text-sm font-medium">{metrics.trained ? `AUC ${metrics.roc_auc?.toFixed(3)}` : 'Awaiting data'}</p>
          </div>
          {metrics.trained && <p className="text-[10px] text-slate-400 mt-1">{metrics.features} features · {metrics.samples?.toLocaleString()} rows</p>}
        </div>
      </div>
    </aside>
  );
}

// ── Upload Modal ──────────────────────────────────────────────────────────────

function UploadModal({ datasets, onClose, onDone }: {
  datasets: Dataset[]; onClose: () => void; onDone: () => void;
}) {
  const [step, setStep] = useState<'pick' | 'map' | 'uploading' | 'done'>('pick');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<{ headers: string[]; rows: Record<string, any>[]; inferred: ColumnMeta[] } | null>(null);
  const [outcomeCol, setOutcomeCol] = useState('');
  const [responseCol, setResponseCol] = useState('');
  const [label, setLabel] = useState('');
  const [isPrimary, setIsPrimary] = useState(datasets.length === 0);
  const [error, setError] = useState('');
  const [progress, setProgress] = useState('');

  const handleFile = async (f: File) => {
    setFile(f);
    setLabel(f.name.replace(/\.csv$/i, ''));
    const text = await f.text();
    const lines = text.trim().split(/\r?\n/);
    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
    const parseRow = (line: string) => {
      const vals = line.split(',');
      const obj: Record<string, any> = {};
      headers.forEach((h, i) => {
        const v = (vals[i] ?? '').trim().replace(/^"|"$/g, '');
        obj[h] = v === '' ? '' : isNaN(Number(v)) ? v : Number(v);
      });
      return obj;
    };
    const sampleRows = lines.slice(1, 201).filter(l => l.trim()).map(parseRow);
    const inferred: ColumnMeta[] = headers.map(col => {
      const vals = sampleRows.map(r => r[col]).filter(v => v !== '' && v !== null);
      const nums = vals.map(Number).filter(n => !isNaN(n));
      if (nums.length > vals.length * 0.8) {
        return { name: col, type: 'numeric' as const, min: Math.min(...nums), max: Math.max(...nums), uniqueValues: null };
      }
      return { name: col, type: 'categorical' as const, min: null, max: null, uniqueValues: [...new Set(vals.map(String))].slice(0, 20) };
    });
    const outcomeGuess = inferred.find(c => c.type === 'numeric' && /buy|bought|purchase|convert|sale|outcome/i.test(c.name))?.name || inferred[inferred.length - 1]?.name || '';
    const responseGuess = inferred.find(c => c.type === 'numeric' && /time|wait|response|delay/i.test(c.name) && c.name !== outcomeGuess)?.name || inferred.find(c => c.type === 'numeric' && c.name !== outcomeGuess)?.name || '';
    setOutcomeCol(outcomeGuess); setResponseCol(responseGuess);
    setPreview({ headers, rows: sampleRows.slice(0, 5), inferred });
    setStep('map');
  };

  const handleUpload = async () => {
    if (!file) return;
    setStep('uploading'); setProgress('Uploading…');
    try {
      const text = await file.text();
      const result = await api.uploadDataset(text, label || file.name, {
        isPrimary,
        outcomeCol: outcomeCol || undefined,
        responseCol: responseCol || undefined,
      });
      setProgress(`Imported ${result.row_count.toLocaleString()} rows!`);
      setStep('done');
      setTimeout(() => { onDone(); onClose(); }, 1200);
    } catch (e: any) { setError(e.message); setStep('map'); }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={onClose}>
      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
        className="bg-white rounded-3xl shadow-2xl w-full max-w-2xl mx-4 overflow-hidden"
        onClick={e => e.stopPropagation()}>
        <div className="bg-indigo-600 px-8 py-6 text-white">
          <h2 className="text-xl font-bold">Import Dataset</h2>
          <p className="text-indigo-200 text-sm mt-1">Upload any CSV — columns are detected automatically</p>
        </div>
        <div className="p-8">
          {step === 'pick' && (
            <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-300 rounded-2xl p-12 cursor-pointer hover:border-indigo-400 hover:bg-indigo-50 transition-all">
              <Upload className="w-10 h-10 text-slate-300 mb-4" />
              <p className="font-semibold text-slate-600">Drop a CSV file or click to browse</p>
              <input type="file" accept=".csv" className="hidden" onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
            </label>
          )}

          {step === 'map' && preview && (
            <div className="space-y-5">
              <div className="flex items-center gap-3 px-4 py-3 bg-slate-50 rounded-xl border border-slate-200">
                <Database className="w-4 h-4 text-indigo-500 shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-slate-700 truncate">{file?.name}</p>
                  <p className="text-[10px] text-slate-400">{preview.headers.length} columns</p>
                </div>
                <button onClick={() => setStep('pick')} className="text-xs text-indigo-500 font-bold hover:underline shrink-0">Change</button>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Dataset name</label>
                  <input className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={label} onChange={e => setLabel(e.target.value)} />
                </div>
                <div className="space-y-1.5 flex flex-col justify-end">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" checked={isPrimary} onChange={e => setIsPrimary(e.target.checked)} className="accent-indigo-600" />
                    <span className="text-sm text-slate-600 font-medium">Set as primary dataset</span>
                  </label>
                  <p className="text-[10px] text-slate-400">Primary = base for model training & joins</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-green-500 inline-block" />Outcome column
                  </label>
                  <select className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={outcomeCol} onChange={e => setOutcomeCol(e.target.value)}>
                    <option value="">— select —</option>
                    {preview.headers.map(h => <option key={h} value={h}>{h}</option>)}
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-indigo-500 inline-block" />X-axis column
                  </label>
                  <select className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={responseCol} onChange={e => setResponseCol(e.target.value)}>
                    <option value="">— select —</option>
                    {preview.headers.map(h => <option key={h} value={h}>{h}</option>)}
                  </select>
                </div>
              </div>

              <div>
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2">Detected columns</p>
                <div className="flex flex-wrap gap-2">
                  {preview.inferred.map(col => (
                    <span key={col.name} className={cn('px-2.5 py-1 rounded-lg text-xs font-medium border',
                      col.name === outcomeCol ? 'bg-green-50 border-green-300 text-green-700' :
                        col.name === responseCol ? 'bg-indigo-50 border-indigo-300 text-indigo-700' :
                          col.type === 'numeric' ? 'bg-slate-50 border-slate-200 text-slate-600' : 'bg-amber-50 border-amber-200 text-amber-700')}>
                      {col.name}<span className="ml-1 opacity-50 text-[9px]">{col.type === 'numeric' ? '#' : 'Aa'}</span>
                    </span>
                  ))}
                </div>
              </div>

              {error && <p className="text-rose-500 text-sm">{error}</p>}
              <div className="flex gap-3">
                <button onClick={handleUpload}
                  className="flex-1 py-3 bg-indigo-600 text-white rounded-2xl font-bold text-sm hover:bg-indigo-700 transition-colors">
                  Upload Dataset
                </button>
                <button onClick={onClose} className="px-6 py-3 bg-slate-100 text-slate-600 rounded-2xl font-bold text-sm hover:bg-slate-200">Cancel</button>
              </div>
            </div>
          )}

          {step === 'uploading' && (
            <div className="flex flex-col items-center py-12 gap-4">
              <RefreshCw className="w-10 h-10 text-indigo-500 animate-spin" />
              <p className="font-semibold text-slate-600">{progress}</p>
            </div>
          )}
          {step === 'done' && (
            <div className="flex flex-col items-center py-12 gap-4">
              <CheckCircle2 className="w-12 h-12 text-emerald-500" />
              <p className="font-bold text-slate-700">{progress}</p>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}

// ── Analytics Page ────────────────────────────────────────────────────────────

function AnalyticsPage({ config, metrics, stats, importance, onUpload }: {
  config: ProjectConfig | null; metrics: ModelMetrics;
  stats: StatBucket[]; importance: FeatureImportance[];
  onUpload: () => void;
}) {
  const avgRate = stats.length ? stats.reduce((a, s) => a + s.purchaseRate, 0) / stats.filter(s => s.total > 0).length : 0;

  return (
    <>
      <header className="h-20 bg-white border-b border-slate-200 px-8 flex items-center justify-between shadow-sm sticky top-0 z-10">
        <div>
          <h1 className="text-xl font-bold text-slate-800">Analytics</h1>
          {config?.outcome_col && <p className="text-xs text-slate-400 mt-0.5">Outcome: <span className="font-semibold text-indigo-600">{config.outcome_col}</span> · X-axis: <span className="font-semibold text-indigo-600">{config.response_col}</span></p>}
        </div>
        <button onClick={onUpload} className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-xl text-xs font-bold hover:bg-indigo-700">
          <Upload className="w-3.5 h-3.5" /> Add Dataset
        </button>
      </header>

      {!config?.outcome_col ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-4 text-slate-300">
          <Layers className="w-16 h-16" />
          <p className="text-lg font-bold">No dataset loaded</p>
          <button onClick={onUpload} className="px-6 py-3 bg-indigo-600 text-white rounded-2xl font-bold text-sm hover:bg-indigo-700">Upload a CSV to get started</button>
        </div>
      ) : (
        <div className="p-8 flex-1 grid grid-cols-1 md:grid-cols-12 gap-6 content-start">
          <div className="md:col-span-12 grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'AUC-ROC', value: metrics.roc_auc?.toFixed(4) ?? '—', sub: 'model performance' },
              { label: 'Accuracy', value: metrics.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : '—', sub: 'training set' },
              { label: 'Avg Outcome Rate', value: `${(avgRate * 100).toFixed(1)}%`, sub: `"${config.outcome_col}" = 1` },
              { label: 'Features', value: metrics.features ?? '—', sub: 'after chain' },
            ].map(k => (
              <Card key={k.label} title={k.label} icon={<Activity className="w-4 h-4" />}>
                <div className="text-2xl font-mono font-bold truncate">{k.value}</div>
                <p className="text-[10px] opacity-50 uppercase mt-2 font-semibold truncate">{k.sub}</p>
              </Card>
            ))}
          </div>

          <div className="md:col-span-8 bg-white border border-slate-200 rounded-3xl p-6 shadow-sm flex flex-col min-h-[400px]">
            <div className="flex justify-between items-start mb-6">
              <div>
                <h2 className="text-lg font-bold">Outcome Rate by {config.response_col}</h2>
                <p className="text-sm text-slate-500">How <span className="font-semibold">{config.outcome_col}</span> varies with <span className="font-semibold">{config.response_col}</span></p>
              </div>
              <Badge color="indigo">Live</Badge>
            </div>
            {stats.length === 0 ? (
              <div className="flex-1 flex items-center justify-center text-slate-300 font-mono">NO DATA</div>
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={stats}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis dataKey="range" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 9, fontWeight: 600 }} interval={0} angle={-20} textAnchor="end" height={40} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip contentStyle={{ border: 'none', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,.08)', fontSize: '12px' }}
                    formatter={(v: number, _n, p) => [`${(v * 100).toFixed(1)}% (n=${p.payload.total})`, config.outcome_col]} />
                  <Bar dataKey="purchaseRate" fill="#6366f1" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="md:col-span-4 bg-white border border-slate-200 rounded-3xl p-6 shadow-sm overflow-y-auto max-h-[400px]">
            <h3 className="text-sm font-bold text-slate-700 mb-4">Feature Importance</h3>
            {importance.slice(0, 12).map((f, i) => (
              <div key={f.feature} className="mb-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-600 truncate max-w-[150px]">{f.feature}</span>
                  <span className="font-mono text-slate-400">{f.importance.toFixed(4)}</span>
                </div>
                <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${(f.importance / (importance[0]?.importance || 1)) * 100}%`, opacity: 1 - i * 0.05 }} />
                </div>
              </div>
            ))}
            {!importance.length && <p className="text-sm text-slate-300 text-center py-8">Train the model to see importance</p>}
          </div>

          {metrics.trained && (
            <div className="md:col-span-12 grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { label: 'F1 Score', value: metrics.f1?.toFixed(4) ?? '—' },
                { label: 'Precision', value: metrics.precision?.toFixed(4) ?? '—' },
                { label: 'Recall', value: metrics.recall?.toFixed(4) ?? '—' },
                { label: 'Positive Rate', value: metrics.positive_rate ? `${(metrics.positive_rate * 100).toFixed(1)}%` : '—' },
              ].map(k => (
                <Card key={k.label} title={k.label} icon={<Activity className="w-4 h-4" />}>
                  <div className="text-xl font-mono font-bold">{k.value}</div>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}
    </>
  );
}

// ── Predictions Page ──────────────────────────────────────────────────────────

type SimMode = 'single' | 'sweep';

function PredictionsPage({ config, datasets }: { config: ProjectConfig | null; datasets: Dataset[] }) {
  const [featureCols, setFeatureCols] = useState<import('./lib/types').ColumnMeta[]>([]);

  // Load chain input columns (raw source cols the chain reads from)
  useEffect(() => {
    if (!config?.outcome_col) return;
    api.getPredictInputs().then(cols => setFeatureCols(cols)).catch(() => { });
  }, [config?.outcome_col]);

  // Mode: single prediction vs sweep
  const [mode, setMode] = useState<SimMode>('single');

  // ── Shared baseline values for all features ────────────────────────────────
  const [baseValues, setBaseValues] = useState<Record<string, any>>({});
  // Features the user has toggled OFF — they use their neutral midpoint so the model
  // still gets a value, but the user is effectively saying "don't vary this"
  const [ignoredFeatures, setIgnoredFeatures] = useState<Set<string>>(new Set());
  const [sweepFeature, setSweepFeature] = useState<string>('');

  // Single mode state
  const [prediction, setPrediction] = useState<number | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<{ input: Record<string, any>; result: number; label: string }[]>([]);

  // Sweep mode state
  const [sweepResults, setSweepResults] = useState<{ x: number | string; prob: number }[]>([]);
  const [sweeping, setSweeping] = useState(false);
  const [sweepError, setSweepError] = useState<string | null>(null);
  const [sweepSteps, setSweepSteps] = useState(20);
  const [currentProb, setCurrentProb] = useState<number | null>(null);

  // Neutral values for ignored features — use actual mean if available, else midpoint
  const neutralValues = React.useMemo(() => {
    const n: Record<string, any> = {};
    featureCols.forEach(col => {
      if (col.type === 'numeric') {
        const mid = col.mean != null
          ? col.mean
          : parseFloat(((col.min ?? 0) + (col.max ?? 100) / 2).toFixed(2));
        n[col.name] = parseFloat(mid.toFixed(2));
      } else {
        n[col.name] = col.uniqueValues?.[0] ?? '';
      }
    });
    return n;
  }, [featureCols.map(c => c.name).join(',')]);

  // Build prediction row: ignored features get neutral value, active ones get baseValues
  const buildRow = (overrides: Record<string, any> = {}) => {
    const row: Record<string, any> = {};
    featureCols.forEach(col => {
      if (ignoredFeatures.has(col.name)) {
        row[col.name] = neutralValues[col.name];
      } else {
        row[col.name] = baseValues[col.name];
      }
    });
    return { ...row, ...overrides };
  };

  const toggleIgnore = (name: string) => {
    setIgnoredFeatures(prev => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
    setPrediction(null);
    setSweepResults([]);
  };

  // Initialise baseline values when featureCols load
  useEffect(() => {
    if (!featureCols.length) return;
    const init: Record<string, any> = {};
    featureCols.forEach(col => {
      if (col.type === 'numeric') {
        const mid = col.mean != null
          ? col.mean
          : parseFloat(((col.min ?? 0) + (col.max ?? 100) / 2).toFixed(2));
        init[col.name] = parseFloat(mid.toFixed(2));
      } else {
        init[col.name] = col.uniqueValues?.[0] ?? '';
      }
    });
    setBaseValues(init);
    setIgnoredFeatures(new Set());
    setPrediction(null);
    setSweepResults([]);
    const firstNum = featureCols.find(c => c.type === 'numeric');
    if (firstNum) setSweepFeature(firstNum.name);
  }, [featureCols.map(c => c.name).join(',')]);

  // ── Single prediction ──────────────────────────────────────────────────────
  const handlePredict = async () => {
    setPredicting(true); setError(null);
    try {
      const row = buildRow();
      const d = await api.predict(row);
      setPrediction(d.probability);
      const activeEntries = Object.entries(row).filter(([k]) => !ignoredFeatures.has(k));
      const label = activeEntries.slice(0, 2).map(([k, v]) => `${k.replace(/_/g, ' ')}=${typeof v === 'number' ? Number(v).toFixed(1) : v}`).join(', ');
      setHistory(prev => [{ input: row, result: d.probability, label }, ...prev].slice(0, 8));
    } catch (err: any) { setError(err.message); }
    finally { setPredicting(false); }
  };

  // ── Sweep simulation ───────────────────────────────────────────────────────
  const handleSweep = async () => {
    const col = featureCols.find(c => c.name === sweepFeature);
    if (!col) return;
    setSweeping(true); setSweepError(null); setSweepResults([]); setCurrentProb(null);

    try {
      let points: { x: number | string; prob: number }[] = [];

      if (col.type === 'numeric') {
        const min = col.min ?? 0;
        const max = col.max ?? 100;
        const step = (max - min) / sweepSteps;
        const xs = Array.from({ length: sweepSteps + 1 }, (_, i) => parseFloat((min + i * step).toFixed(2)));

        for (const x of xs) {
          const row = buildRow({ [sweepFeature]: x });
          const d = await api.predict(row);
          points.push({ x, prob: parseFloat((d.probability * 100).toFixed(1)) });
          setSweepResults([...points]);
        }
      } else {
        const vals = col.uniqueValues ?? [];
        for (const v of vals) {
          const row = buildRow({ [sweepFeature]: v });
          const d = await api.predict(row);
          points.push({ x: v, prob: parseFloat((d.probability * 100).toFixed(1)) });
          setSweepResults([...points]);
        }
      }

      // Baseline probability with current ignored set
      const base = await api.predict(buildRow());
      setCurrentProb(base.probability);
    } catch (err: any) { setSweepError(err.message); }
    finally { setSweeping(false); }
  };

  const sweepCol = featureCols.find(c => c.name === sweepFeature);
  const activeCount = featureCols.length - ignoredFeatures.size;
  const prob = prediction ?? 0;
  const probColor = (p: number) => p > 0.6 ? 'text-emerald-500' : p > 0.35 ? 'text-amber-500' : 'text-rose-500';
  const probBg = (p: number) => p > 0.6 ? 'bg-emerald-500' : p > 0.35 ? 'bg-amber-500' : 'bg-rose-500';

  if (!config?.outcome_col) return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 text-slate-300">
      <BrainCircuit className="w-16 h-16" />
      <p className="text-lg font-bold">Upload a dataset first</p>
    </div>
  );

  return (
    <>
      <header className="h-20 bg-white border-b border-slate-200 px-8 flex items-center justify-between shadow-sm sticky top-0 z-10">
        <div>
          <h1 className="text-xl font-bold text-slate-800">Scenario Simulator</h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Predicting <span className="font-semibold text-indigo-600">{config.outcome_col}</span> · <span className="font-semibold text-slate-600">{activeCount}</span>/{featureCols.length} features active
          </p>
        </div>
        {/* Mode toggle */}
        <div className="flex items-center gap-1 bg-slate-100 rounded-xl p-1">
          <button onClick={() => setMode('single')}
            className={cn('px-4 py-1.5 rounded-lg text-xs font-bold transition-all', mode === 'single' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700')}>
            <span className="flex items-center gap-1.5"><Zap className="w-3.5 h-3.5" />Single</span>
          </button>
          <button onClick={() => setMode('sweep')}
            className={cn('px-4 py-1.5 rounded-lg text-xs font-bold transition-all', mode === 'sweep' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700')}>
            <span className="flex items-center gap-1.5"><TrendingUp className="w-3.5 h-3.5" />Sweep</span>
          </button>
        </div>
      </header>

      <div className="p-8 flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 content-start">

        {/* ── Left panel: feature controls ── */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          <div className="bg-indigo-600 rounded-3xl p-6 text-white shadow-lg flex flex-col gap-4">
            <div>
              <h2 className="text-base font-bold">
                {mode === 'sweep' ? 'Baseline & Context' : 'Feature Inputs'}
              </h2>
              <p className="text-indigo-200 text-[11px] mt-0.5">
                Toggle features off to ignore them — they'll use a neutral midpoint value.
              </p>
            </div>

            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] text-indigo-200 font-semibold">
                {activeCount} of {featureCols.length} active
              </span>
              {ignoredFeatures.size === featureCols.length ? (
                <button onClick={() => setIgnoredFeatures(new Set())}
                  className="text-[9px] text-indigo-300 hover:text-white underline">
                  enable all
                </button>
              ) : ignoredFeatures.size > 0 ? (
                <div className="flex gap-2">
                  <button onClick={() => setIgnoredFeatures(new Set())}
                    className="text-[9px] text-indigo-300 hover:text-white underline">
                    enable all
                  </button>
                  <span className="text-indigo-400">·</span>
                  <button onClick={() => setIgnoredFeatures(new Set(featureCols.map(c => c.name)))}
                    className="text-[9px] text-indigo-300 hover:text-white underline">
                    disable all
                  </button>
                </div>
              ) : (
                <button onClick={() => setIgnoredFeatures(new Set(featureCols.map(c => c.name)))}
                  className="text-[9px] text-indigo-300 hover:text-white underline">
                  disable all
                </button>
              )}
            </div>

            <div className="space-y-2 max-h-[52vh] overflow-y-auto pr-1">
              {featureCols.map(col => {
                const isSweepAxis = mode === 'sweep' && col.name === sweepFeature;
                const isIgnored = ignoredFeatures.has(col.name);
                return (
                  <div key={col.name} className={cn(
                    'rounded-2xl p-3 transition-all',
                    isSweepAxis ? 'bg-white/5 border border-indigo-300/30' :
                      isIgnored ? 'bg-white/5 opacity-50' : 'bg-white/10'
                  )}>
                    {/* Header row: name + badge/toggle */}
                    <div className="flex items-center justify-between mb-1">
                      <span className={cn(
                        'text-[10px] font-bold uppercase tracking-wider truncate max-w-[130px]',
                        isIgnored ? 'text-indigo-300 line-through' : 'text-indigo-100'
                      )}>
                        {col.name.replace(/_/g, ' ')}
                      </span>
                      <div className="flex items-center gap-1.5 shrink-0">
                        {isSweepAxis ? (
                          <span className="text-[9px] bg-white/20 text-white px-2 py-0.5 rounded-full font-bold">AXIS</span>
                        ) : (
                          <>
                            {!isIgnored && col.type === 'numeric' && (
                              <span className="text-white font-mono text-[10px]">
                                {Number(baseValues[col.name] ?? 0).toFixed(1)}
                              </span>
                            )}
                            <button
                              onClick={() => toggleIgnore(col.name)}
                              title={isIgnored ? 'Enable this feature' : 'Ignore this feature'}
                              className={cn(
                                'w-6 h-3.5 rounded-full transition-all relative shrink-0',
                                isIgnored ? 'bg-white/20' : 'bg-white/50'
                              )}>
                              <span className={cn(
                                'absolute top-0.5 w-2.5 h-2.5 rounded-full transition-all',
                                isIgnored ? 'left-0.5 bg-indigo-300' : 'left-3 bg-white'
                              )} />
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                    {/* Control: only shown when active and not sweep axis */}
                    {!isSweepAxis && !isIgnored && (
                      col.type === 'numeric' ? (
                        <input type="range"
                          min={col.min ?? 0} max={col.max ?? 100}
                          step={Math.max(0.01, ((col.max ?? 100) - (col.min ?? 0)) / 100)}
                          className="w-full h-1.5 bg-indigo-400/30 rounded-full appearance-none cursor-pointer mt-1"
                          value={baseValues[col.name] ?? 0}
                          onChange={e => setBaseValues(v => ({ ...v, [col.name]: parseFloat(e.target.value) }))} />
                      ) : (
                        <select
                          className="w-full bg-white/10 border border-white/20 rounded-xl px-2 py-1.5 text-xs focus:outline-none mt-1"
                          value={baseValues[col.name] ?? ''}
                          onChange={e => setBaseValues(v => ({ ...v, [col.name]: e.target.value }))}>
                          {col.uniqueValues?.map(v => <option key={v} value={v} className="text-slate-800">{v}</option>)}
                        </select>
                      )
                    )}
                    {/* Ignored state hint */}
                    {isIgnored && !isSweepAxis && (
                      <p className="text-[9px] text-indigo-300/70 mt-0.5 italic">
                        using neutral value · toggle to set manually
                      </p>
                    )}
                  </div>
                );
              })}
            </div>

            {mode === 'single' && (
              <>
                {error && <p className="text-rose-200 text-xs">{error}</p>}
                <button onClick={handlePredict} disabled={predicting}
                  className="w-full bg-white text-indigo-600 py-3.5 rounded-2xl font-bold text-sm shadow-xl active:scale-[0.98] transition-all disabled:opacity-50">
                  {predicting
                    ? <span className="flex items-center justify-center gap-2"><RefreshCw className="w-4 h-4 animate-spin" />Calculating…</span>
                    : <span className="flex items-center justify-center gap-2"><Zap className="w-4 h-4" />Run Prediction</span>}
                </button>
              </>
            )}
          </div>
        </div>

        {/* ── Right panel ── */}
        <div className="lg:col-span-8 flex flex-col gap-6">

          {/* ── SINGLE MODE ── */}
          {mode === 'single' && (
            <>
              <div className="bg-white border border-slate-200 rounded-3xl p-8 shadow-sm">
                <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-6">
                  Prediction Output — {config.outcome_col}
                </h3>
                {prediction === null ? (
                  <div className="flex flex-col items-center justify-center py-12 text-slate-300">
                    <BrainCircuit className="w-12 h-12 mb-4" />
                    <p className="text-sm font-semibold">Set inputs and run a prediction</p>
                  </div>
                ) : (
                  <AnimatePresence mode="wait">
                    <motion.div key={prediction} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
                      <div className="flex items-end gap-4">
                        <div className={cn('text-7xl font-black tabular-nums', probColor(prob))}>{(prob * 100).toFixed(1)}</div>
                        <div className="pb-2">
                          <p className="text-2xl font-black text-slate-300">%</p>
                          <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">P({config.outcome_col} = 1)</p>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                          <motion.div className={cn('h-full rounded-full', probBg(prob))}
                            initial={{ width: 0 }} animate={{ width: `${prob * 100}%` }} transition={{ duration: 0.6, ease: 'easeOut' }} />
                        </div>
                        <div className="flex justify-between text-[9px] font-mono text-slate-300"><span>0%</span><span>50%</span><span>100%</span></div>
                      </div>
                      <div className={cn('rounded-2xl px-5 py-4 flex items-center gap-4',
                        prob > 0.6 ? 'bg-emerald-50 border border-emerald-200' : prob > 0.35 ? 'bg-amber-50 border border-amber-200' : 'bg-rose-50 border border-rose-200')}>
                        <div className={cn('w-10 h-10 rounded-xl flex items-center justify-center shrink-0', probBg(prob))}>
                          {prob > 0.6 ? <CheckCircle2 className="w-5 h-5 text-white" /> : <AlertCircle className="w-5 h-5 text-white" />}
                        </div>
                        <div>
                          <p className="font-bold text-slate-800 text-sm">
                            {prob > 0.6 ? 'High likelihood' : prob > 0.35 ? 'Moderate likelihood' : 'Low likelihood'}
                          </p>
                          <p className="text-xs text-slate-500 mt-0.5">
                            {prob > 0.6 ? 'High confidence positive prediction.' : prob > 0.35 ? 'Uncertain — consider adjusting inputs.' : 'Negative prediction given these inputs.'}
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  </AnimatePresence>
                )}
              </div>

              {history.length > 0 && (
                <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm">
                  <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-4">Prediction History</h3>
                  <div className="space-y-2">
                    {history.map((h, i) => (
                      <div key={i} className="flex items-center justify-between px-4 py-3 bg-slate-50 rounded-xl gap-3">
                        <span className="text-xs text-slate-500 truncate">{h.label}</span>
                        <div className="flex items-center gap-2 shrink-0">
                          <div className="w-16 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                            <div className={cn('h-full rounded-full', probBg(h.result))} style={{ width: `${h.result * 100}%` }} />
                          </div>
                          <span className={cn('text-sm font-black w-12 text-right', probColor(h.result))}>
                            {(h.result * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {/* ── SWEEP MODE ── */}
          {mode === 'sweep' && (
            <>
              {/* Sweep config */}
              <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm">
                <div className="flex items-center gap-2 mb-5">
                  <TrendingUp className="w-4 h-4 text-indigo-500" />
                  <h3 className="text-sm font-bold text-slate-700">Sweep Configuration</h3>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1.5">Sweep Feature (X axis)</label>
                    <select
                      className="w-full border border-slate-200 rounded-xl px-3 py-2 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-300"
                      value={sweepFeature}
                      onChange={e => { setSweepFeature(e.target.value); setSweepResults([]); setCurrentProb(null); }}>
                      {featureCols.map(c => (
                        <option key={c.name} value={c.name}>{c.name.replace(/_/g, ' ')}</option>
                      ))}
                    </select>
                  </div>
                  {sweepCol?.type === 'numeric' && (
                    <div>
                      <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1.5">
                        Resolution ({sweepSteps} steps)
                      </label>
                      <input type="range" min={5} max={50} step={5}
                        className="w-full h-1.5 bg-slate-200 rounded-full appearance-none cursor-pointer mt-2"
                        value={sweepSteps}
                        onChange={e => setSweepSteps(parseInt(e.target.value))} />
                      <div className="flex justify-between text-[9px] font-mono text-slate-300 mt-1"><span>5</span><span>50</span></div>
                    </div>
                  )}
                </div>

                {sweepCol && (
                  <div className="bg-indigo-50 border border-indigo-100 rounded-2xl px-4 py-3 mb-4 text-xs text-indigo-700">
                    <span className="font-bold">{sweepFeature.replace(/_/g, ' ')}</span>
                    {sweepCol.type === 'numeric'
                      ? ` will sweep from ${sweepCol.min?.toFixed(1)} to ${sweepCol.max?.toFixed(1)} in ${sweepSteps} steps. All other features use the baseline values on the left.`
                      : ` will iterate over all ${sweepCol.uniqueValues?.length} unique values. All other features use the baseline values.`}
                  </div>
                )}

                {sweepError && <p className="text-rose-500 text-xs mb-3">{sweepError}</p>}

                <button onClick={handleSweep} disabled={sweeping || !sweepFeature}
                  className="w-full bg-indigo-600 text-white py-3 rounded-2xl font-bold text-sm shadow-md active:scale-[0.98] transition-all disabled:opacity-50 flex items-center justify-center gap-2">
                  {sweeping
                    ? <><RefreshCw className="w-4 h-4 animate-spin" />Running sweep… {sweepResults.length}/{sweepCol?.type === 'numeric' ? sweepSteps + 1 : sweepCol?.uniqueValues?.length ?? 0}</>
                    : <><TrendingUp className="w-4 h-4" />Run Sweep</>}
                </button>
              </div>

              {/* Sweep chart */}
              {(sweepResults.length > 0 || sweeping) && (
                <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm">
                  <div className="flex items-center justify-between mb-5">
                    <div>
                      <h3 className="text-sm font-bold text-slate-700">
                        P({config.outcome_col} = 1) vs {sweepFeature.replace(/_/g, ' ')}
                      </h3>
                      <p className="text-[10px] text-slate-400 mt-0.5">
                        How outcome probability changes as {sweepFeature.replace(/_/g, ' ')} varies
                      </p>
                    </div>
                    {currentProb !== null && (
                      <div className="text-right">
                        <p className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Baseline prob</p>
                        <p className={cn('text-lg font-black', probColor(currentProb))}>{(currentProb * 100).toFixed(1)}%</p>
                      </div>
                    )}
                  </div>

                  <ResponsiveContainer width="100%" height={260}>
                    {sweepCol?.type === 'numeric' ? (
                      <LineChart data={sweepResults} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="x" tick={{ fontSize: 10 }} tickFormatter={v => Number(v).toFixed(0)}
                          label={{ value: sweepFeature.replace(/_/g, ' '), position: 'insideBottom', offset: -2, fontSize: 10, fill: '#94a3b8' }} />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} tickFormatter={v => `${v}%`} />
                        <Tooltip
                          formatter={(v: any) => [`${Number(v).toFixed(1)}%`, `P(${config.outcome_col}=1)`]}
                          labelFormatter={v => `${sweepFeature.replace(/_/g, ' ')}: ${Number(v).toFixed(1)}`}
                          contentStyle={{ borderRadius: 12, border: '1px solid #e2e8f0', fontSize: 11 }} />
                        {currentProb !== null && (
                          <ReferenceLine y={currentProb * 100} stroke="#6366f1" strokeDasharray="5 3"
                            label={{ value: 'baseline', fill: '#6366f1', fontSize: 9, position: 'insideTopRight' }} />
                        )}
                        <Line type="monotone" dataKey="prob" stroke="#6366f1" strokeWidth={2.5}
                          dot={false} activeDot={{ r: 5, fill: '#6366f1' }} isAnimationActive={false} />
                      </LineChart>
                    ) : (
                      <BarChart data={sweepResults} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="x" tick={{ fontSize: 10 }} />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} tickFormatter={v => `${v}%`} />
                        <Tooltip
                          formatter={(v: any) => [`${Number(v).toFixed(1)}%`, `P(${config.outcome_col}=1)`]}
                          contentStyle={{ borderRadius: 12, border: '1px solid #e2e8f0', fontSize: 11 }} />
                        <Bar dataKey="prob" fill="#6366f1" radius={[6, 6, 0, 0]} isAnimationActive={false} />
                      </BarChart>
                    )}
                  </ResponsiveContainer>

                  {/* Insight callouts */}
                  {sweepResults.length > 1 && !sweeping && (() => {
                    const maxPt = sweepResults.reduce((a, b) => a.prob > b.prob ? a : b);
                    const minPt = sweepResults.reduce((a, b) => a.prob < b.prob ? a : b);
                    const range = maxPt.prob - minPt.prob;
                    return (
                      <div className="mt-5 grid grid-cols-3 gap-3">
                        <div className="bg-emerald-50 border border-emerald-100 rounded-2xl px-4 py-3">
                          <p className="text-[9px] font-bold uppercase tracking-wider text-emerald-600 mb-1">Peak probability</p>
                          <p className="text-lg font-black text-emerald-600">{maxPt.prob.toFixed(1)}%</p>
                          <p className="text-[10px] text-emerald-500">at {sweepFeature.replace(/_/g, ' ')} = {typeof maxPt.x === 'number' ? Number(maxPt.x).toFixed(1) : maxPt.x}</p>
                        </div>
                        <div className="bg-rose-50 border border-rose-100 rounded-2xl px-4 py-3">
                          <p className="text-[9px] font-bold uppercase tracking-wider text-rose-600 mb-1">Trough probability</p>
                          <p className="text-lg font-black text-rose-600">{minPt.prob.toFixed(1)}%</p>
                          <p className="text-[10px] text-rose-500">at {sweepFeature.replace(/_/g, ' ')} = {typeof minPt.x === 'number' ? Number(minPt.x).toFixed(1) : minPt.x}</p>
                        </div>
                        <div className="bg-indigo-50 border border-indigo-100 rounded-2xl px-4 py-3">
                          <p className="text-[9px] font-bold uppercase tracking-wider text-indigo-600 mb-1">Total range</p>
                          <p className="text-lg font-black text-indigo-600">{range.toFixed(1)}pp</p>
                          <p className="text-[10px] text-indigo-500">{range > 20 ? 'High sensitivity' : range > 8 ? 'Moderate sensitivity' : 'Low sensitivity'}</p>
                        </div>
                      </div>
                    );
                  })()}

                  {/* Threshold finder */}
                  {sweepResults.length > 1 && !sweeping && sweepCol?.type === 'numeric' && (() => {
                    const threshold50 = sweepResults.find(r => r.prob >= 50);
                    const threshold70 = sweepResults.find(r => r.prob >= 70);
                    return (threshold50 || threshold70) ? (
                      <div className="mt-4 bg-slate-50 border border-slate-100 rounded-2xl px-5 py-4">
                        <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-3">
                          <SlidersHorizontal className="w-3 h-3 inline mr-1" />Probability Thresholds
                        </p>
                        <div className="space-y-2">
                          {threshold50 && (
                            <div className="flex items-center justify-between">
                              <span className="text-xs text-slate-600">Probability ≥ <span className="font-bold text-amber-600">50%</span></span>
                              <span className="text-xs font-mono font-bold text-slate-700">
                                {sweepFeature.replace(/_/g, ' ')} ≥ {Number(threshold50.x).toFixed(1)}
                              </span>
                            </div>
                          )}
                          {threshold70 && (
                            <div className="flex items-center justify-between">
                              <span className="text-xs text-slate-600">Probability ≥ <span className="font-bold text-emerald-600">70%</span></span>
                              <span className="text-xs font-mono font-bold text-slate-700">
                                {sweepFeature.replace(/_/g, ' ')} ≥ {Number(threshold70.x).toFixed(1)}
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    ) : null;
                  })()}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}

// ── Datasets Page ─────────────────────────────────────────────────────────────

function DatasetsPage({ datasets, config, onUpload, onRefresh, onSaveConfig }: {
  datasets: Dataset[]; config: ProjectConfig | null;
  onUpload: () => void; onRefresh: () => void;
  onSaveConfig: (c: Partial<ProjectConfig>) => void;
}) {
  const [toast, setToast] = useState<{ text: string; ok: boolean } | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [editOutcome, setEditOutcome] = useState(config?.outcome_col ?? '');
  const [editResponse, setEditResponse] = useState(config?.response_col ?? '');
  const [editPrimary, setEditPrimary] = useState(config?.primary_id ?? 0);

  const prevConfig = useRef(config);
  if (config !== prevConfig.current) {
    prevConfig.current = config;
    if (config) { setEditOutcome(config.outcome_col); setEditResponse(config.response_col); setEditPrimary(config.primary_id); }
  }

  const primaryDs = datasets.find(d => d.dataset_id === editPrimary) ?? datasets[0];
  const primaryCols = primaryDs?.schema.map(c => c.name) ?? [];

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this dataset?')) return;
    try {
      await api.deleteDataset(id);
      setToast({ text: 'Dataset deleted', ok: true });
      onRefresh();
    } catch (e: any) { setToast({ text: e.message, ok: false }); }
  };

  const toggleCol = async (ds: Dataset, col: string) => {
    const active = new Set(ds.active_cols);
    active.has(col) ? active.delete(col) : active.add(col);
    await api.updateDatasetCols(ds.dataset_id, [...active]);
    onRefresh();
  };

  return (
    <>
      <header className="h-20 bg-white border-b border-slate-200 px-8 flex items-center justify-between shadow-sm sticky top-0 z-10">
        <div>
          <h1 className="text-xl font-bold text-slate-800">Datasets</h1>
          <p className="text-xs text-slate-400 mt-0.5">{datasets.length} loaded · {datasets.reduce((a, d) => a + d.row_count, 0).toLocaleString()} total rows</p>
        </div>
        <button onClick={onUpload} className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-xl text-xs font-bold hover:bg-indigo-700">
          <Plus className="w-3.5 h-3.5" /> Add Dataset
        </button>
      </header>

      <div className="p-8 flex-1 flex flex-col gap-6">
        <Toast msg={toast} onDone={() => setToast(null)} />

        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
          <h3 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2"><Settings className="w-4 h-4" /> Model Configuration</h3>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="space-y-1.5">
              <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Primary dataset</label>
              <select className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                value={editPrimary} onChange={e => setEditPrimary(+e.target.value)}>
                {datasets.map(d => <option key={d.dataset_id} value={d.dataset_id}>{d.label}</option>)}
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Outcome column</label>
              <select className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                value={editOutcome} onChange={e => setEditOutcome(e.target.value)}>
                <option value="">— select —</option>
                {primaryCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500">X-axis column</label>
              <select className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                value={editResponse} onChange={e => setEditResponse(e.target.value)}>
                <option value="">— select —</option>
                {primaryCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>
          <button onClick={() => onSaveConfig({ outcome_col: editOutcome, response_col: editResponse, primary_id: editPrimary })}
            className="px-5 py-2 bg-indigo-600 text-white rounded-xl text-sm font-bold hover:bg-indigo-700">
            Save & Retrain
          </button>
        </div>

        {datasets.length === 0 && (
          <div className="flex flex-col items-center justify-center py-16 text-slate-300 gap-4">
            <Database className="w-12 h-12" /><p className="font-semibold">No datasets loaded</p>
            <button onClick={onUpload} className="text-sm text-indigo-500 font-bold hover:underline">Upload a CSV</button>
          </div>
        )}

        {datasets.map(ds => (
          <div key={ds.dataset_id} className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
            <div className="px-6 py-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={cn('w-2 h-2 rounded-full', ds.dataset_id === config?.primary_id ? 'bg-indigo-500' : 'bg-slate-300')} />
                <div>
                  <p className="font-semibold text-slate-800">{ds.label}</p>
                  <p className="text-xs text-slate-400">{ds.row_count.toLocaleString()} rows · {ds.schema.length} columns · {ds.active_cols.length} active</p>
                </div>
                {ds.dataset_id === config?.primary_id && <Badge color="indigo">primary</Badge>}
              </div>
              <div className="flex items-center gap-2">
                <button onClick={() => setExpanded(expanded === ds.dataset_id ? null : ds.dataset_id)}
                  className="p-2 text-slate-400 hover:bg-slate-50 rounded-lg transition-colors">
                  {expanded === ds.dataset_id ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                <button onClick={() => handleDelete(ds.dataset_id)} className="p-2 text-slate-300 hover:text-rose-400 hover:bg-rose-50 rounded-lg transition-colors">
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
            <AnimatePresence>
              {expanded === ds.dataset_id && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                  className="border-t border-slate-100 px-6 py-4 overflow-hidden">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-3">Active columns (click to toggle)</p>
                  <div className="flex flex-wrap gap-2">
                    {ds.schema.map(col => {
                      const isActive = ds.active_cols.includes(col.name);
                      return (
                        <button key={col.name} onClick={() => toggleCol(ds, col.name)}
                          className={cn('px-2.5 py-1 rounded-lg text-xs font-medium border transition-all',
                            isActive ? (col.type === 'numeric' ? 'bg-indigo-50 border-indigo-300 text-indigo-700' : 'bg-amber-50 border-amber-200 text-amber-700')
                              : 'bg-slate-50 border-slate-200 text-slate-400 line-through')}>
                          {col.name}<span className="ml-1 opacity-50">{col.type === 'numeric' ? '#' : 'Aa'}</span>
                        </button>
                      );
                    })}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>
    </>
  );
}

// ── Joins Page ────────────────────────────────────────────────────────────────

function JoinsPage({ datasets, joins, onRefresh }: {
  datasets: Dataset[]; joins: JoinRule[]; onRefresh: () => void;
}) {
  const [leftId, setLeftId] = useState(datasets[0]?.dataset_id ?? 0);
  const [rightId, setRightId] = useState(datasets[1]?.dataset_id ?? 0);
  const [leftCol, setLeftCol] = useState('');
  const [rightCol, setRightCol] = useState('');
  const [joinType, setJoinType] = useState<'left' | 'inner' | 'outer'>('left');
  const [toast, setToast] = useState<{ text: string; ok: boolean } | null>(null);

  const leftSchema = datasets.find(d => d.dataset_id === leftId)?.schema ?? [];
  const rightSchema = datasets.find(d => d.dataset_id === rightId)?.schema ?? [];
  const dsName = (id: number) => datasets.find(d => d.dataset_id === id)?.label ?? `Dataset ${id}`;

  const addJoin = async () => {
    try {
      await api.addJoin({ left_id: leftId, right_id: rightId, left_col: leftCol, right_col: rightCol, join_type: joinType });
      setToast({ text: 'Join rule added', ok: true }); setLeftCol(''); setRightCol(''); onRefresh();
    } catch (e: any) { setToast({ text: e.message, ok: false }); }
  };

  if (datasets.length < 2) return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 text-slate-300">
      <Link2 className="w-16 h-16" />
      <p className="text-lg font-bold">Upload at least 2 datasets to define joins</p>
    </div>
  );

  return (
    <>
      <header className="h-20 bg-white border-b border-slate-200 px-8 flex items-center justify-between shadow-sm sticky top-0 z-10">
        <div>
          <h1 className="text-xl font-bold text-slate-800">Join Rules</h1>
          <p className="text-xs text-slate-400 mt-0.5">{joins.length} rules · {datasets.length} datasets</p>
        </div>
        <Badge color="indigo">{joins.length} active</Badge>
      </header>

      <div className="p-8 flex-1 flex flex-col gap-6">
        <Toast msg={toast} onDone={() => setToast(null)} />

        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
          <h3 className="text-sm font-bold text-slate-700 mb-5">New Join Rule</h3>
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 items-end">
            <div className="lg:col-span-2 space-y-3">
              <div>
                <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block mb-1.5">Left dataset</label>
                <select className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  value={leftId} onChange={e => { setLeftId(+e.target.value); setLeftCol(''); }}>
                  {datasets.map(d => <option key={d.dataset_id} value={d.dataset_id}>{d.label}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block mb-1.5">Key column</label>
                <select className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  value={leftCol} onChange={e => setLeftCol(e.target.value)}>
                  <option value="">— select —</option>
                  {leftSchema.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                </select>
              </div>
            </div>

            <div className="flex flex-col items-center gap-2">
              <Link2 className="w-5 h-5 text-slate-400" />
              <div className="flex gap-1">
                {(['left', 'inner', 'outer'] as const).map(jt => (
                  <button key={jt} onClick={() => setJoinType(jt)}
                    className={cn('px-2 py-1 text-[10px] font-bold uppercase rounded-lg border',
                      joinType === jt ? 'bg-indigo-600 text-white border-indigo-600' : 'border-slate-200 text-slate-500 hover:bg-slate-50')}>
                    {jt}
                  </button>
                ))}
              </div>
            </div>

            <div className="lg:col-span-2 space-y-3">
              <div>
                <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block mb-1.5">Right dataset</label>
                <select className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  value={rightId} onChange={e => { setRightId(+e.target.value); setRightCol(''); }}>
                  {datasets.map(d => <option key={d.dataset_id} value={d.dataset_id}>{d.label}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block mb-1.5">Key column</label>
                <select className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  value={rightCol} onChange={e => setRightCol(e.target.value)}>
                  <option value="">— select —</option>
                  {rightSchema.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                </select>
              </div>
            </div>
          </div>
          <button onClick={addJoin} disabled={!leftCol || !rightCol || leftId === rightId}
            className="mt-5 px-6 py-2.5 bg-indigo-600 text-white rounded-xl text-sm font-bold hover:bg-indigo-700 disabled:opacity-40">
            Add Join Rule
          </button>
        </div>

        {joins.length > 0 && (
          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-100">
              <h3 className="text-sm font-bold text-slate-700">Active Join Rules</h3>
            </div>
            {joins.map(rule => (
              <div key={rule.rule_id} className="px-6 py-4 flex items-center gap-3 border-b border-slate-50 hover:bg-slate-50">
                <div className="flex-1 flex items-center gap-3 flex-wrap">
                  <Badge color="slate">{dsName(rule.left_id)}</Badge>
                  <span className="text-xs font-mono text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">{rule.left_col}</span>
                  <span className="text-[10px] text-slate-400 uppercase font-bold">{rule.join_type} join</span>
                  <span className="text-xs font-mono text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">{rule.right_col}</span>
                  <Badge color="slate">{dsName(rule.right_id)}</Badge>
                </div>
                <button onClick={() => { api.deleteJoin(rule.rule_id).then(onRefresh); }}
                  className="p-1.5 text-slate-300 hover:text-rose-400 hover:bg-rose-50 rounded-lg">
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
        {joins.length === 0 && (
          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 text-sm text-slate-500">
            No join rules defined. Datasets with matching column sets will be unioned automatically.
          </div>
        )}
      </div>
    </>
  );
}

// ── Chain Page ────────────────────────────────────────────────────────────────

const STEP_COLORS: Record<string, string> = {
  passthrough: 'slate', log: 'purple', interaction: 'red',
  ratio: 'amber', binary_threshold: 'green', encode: 'indigo',
};

function ChainPage({ steps, columns, joinsCount, onRefresh }: {
  steps: ChainStep[]; columns: ColumnMeta[]; joinsCount: number; onRefresh: () => void;
}) {
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState({ name: '', type: 'interaction', colA: '', colB: '', col: '', threshold: '' });
  // Fetch merged columns (includes joined cols like purchases_purchase_value)
  const [mergedCols, setMergedCols] = useState<ColumnMeta[]>([]);
  useEffect(() => {
    api.getMergedColumns()
      .then(setMergedCols)
      .catch((err) => {
        console.error('getMergedColumns failed:', err);
        setMergedCols([]);
      });
  }, [joinsCount]);

  const effectiveCols = mergedCols.length > 0 ? mergedCols : columns;
  const numericCols = effectiveCols.filter(c => c.type === 'numeric').map(c => c.name);
  const allCols = effectiveCols.map(c => c.name);

  const toggleStep = async (step: ChainStep) => {
    await api.updateChainStep(step.step_id, { enabled: !step.enabled });
    onRefresh();
  };

  const removeStep = async (id: number) => {
    await api.deleteChainStep(id);
    onRefresh();
  };

  const addStep = async () => {
    const config: Record<string, any> = {};
    if (['interaction', 'ratio'].includes(form.type)) { config.colA = form.colA; config.colB = form.colB; }
    else if (['log', 'passthrough', 'encode'].includes(form.type)) { config.col = form.col; }
    else if (form.type === 'binary_threshold') { config.col = form.col; config.threshold = parseFloat(form.threshold) || 0; }
    await api.addChainStep({
      step_order: steps.length + 1,
      name: form.name || `${form.type}_${Date.now()}`,
      type: form.type as ChainStep['type'],
      config,
      enabled: true,
    });
    setAdding(false);
    setForm({ name: '', type: 'interaction', colA: '', colB: '', col: '', threshold: '' });
    onRefresh();
  };

  return (
    <>
      <header className="h-20 bg-white border-b border-slate-200 px-8 flex items-center justify-between shadow-sm sticky top-0 z-10">
        <div>
          <h1 className="text-xl font-bold text-slate-800">Forward Chain</h1>
          <p className="text-xs text-slate-400 mt-0.5">{steps.filter(s => s.enabled).length} of {steps.length} steps enabled</p>
        </div>
        <button onClick={() => setAdding(!adding)} className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-xl text-xs font-bold hover:bg-indigo-700">
          <Plus className="w-3.5 h-3.5" /> Add Step
        </button>
      </header>

      <div className="p-8 flex-1 flex flex-col gap-6">
        <div className="bg-indigo-50 border border-indigo-200 rounded-2xl px-5 py-3 text-sm text-indigo-700">
          Features are computed <strong>in order</strong> — each step can reference columns from previous steps. Toggle steps on/off to control the feature set.
        </div>

        {adding && (
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-sm font-bold text-slate-700 mb-4">New Chain Step</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div>
                <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block mb-1.5">Type</label>
                <select className="w-full border border-slate-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  value={form.type} onChange={e => setForm(f => ({ ...f, type: e.target.value }))}>
                  <option value="passthrough">Passthrough (as-is)</option>
                  <option value="interaction">Interaction A × B</option>
                  <option value="log">Log transform</option>
                  <option value="ratio">Ratio A / B</option>
                  <option value="binary_threshold">Binary threshold</option>
                  <option value="encode">Encode categorical</option>
                </select>
              </div>
              <div>
                <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block mb-1.5">Name</label>
                <input className="w-full border border-slate-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  placeholder="feature_name" value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} />
              </div>
              {['interaction', 'ratio'].includes(form.type) && (<>
                <div>
                  <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block mb-1.5">Column A</label>
                  <select className="w-full border border-slate-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={form.colA} onChange={e => setForm(f => ({ ...f, colA: e.target.value }))}>
                    <option value="">— select —</option>{numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block mb-1.5">Column B</label>
                  <select className="w-full border border-slate-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={form.colB} onChange={e => setForm(f => ({ ...f, colB: e.target.value }))}>
                    <option value="">— select —</option>{numericCols.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
              </>)}
              {['log', 'passthrough', 'encode', 'binary_threshold'].includes(form.type) && (
                <div>
                  <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block mb-1.5">Column</label>
                  <select className="w-full border border-slate-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={form.col} onChange={e => setForm(f => ({ ...f, col: e.target.value }))}>
                    <option value="">— select —</option>
                    {(form.type === 'encode' || form.type === 'passthrough' ? allCols : numericCols).map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
              )}
              {form.type === 'binary_threshold' && (
                <div>
                  <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block mb-1.5">Threshold</label>
                  <input type="number" className="w-full border border-slate-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={form.threshold} onChange={e => setForm(f => ({ ...f, threshold: e.target.value }))} />
                </div>
              )}
            </div>
            <div className="flex gap-3">
              <button onClick={addStep} className="px-5 py-2 bg-indigo-600 text-white rounded-xl text-sm font-bold hover:bg-indigo-700">Add</button>
              <button onClick={() => setAdding(false)} className="px-5 py-2 bg-slate-100 text-slate-600 rounded-xl text-sm font-bold hover:bg-slate-200">Cancel</button>
            </div>
          </div>
        )}

        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
          {steps.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 text-slate-300 gap-3">
              <GitBranch className="w-10 h-10" /><p className="font-semibold">Upload a dataset to auto-generate chain steps</p>
            </div>
          )}
          {steps.map((step, idx) => (
            <div key={step.step_id} className={cn('flex items-center gap-4 px-6 py-4 border-b border-slate-50 hover:bg-slate-50 transition-colors', !step.enabled && 'opacity-40')}>
              <div className="w-6 h-6 rounded-full bg-slate-100 flex items-center justify-center text-[10px] font-bold text-slate-400 shrink-0">{idx + 1}</div>
              <div className="flex-1 flex items-center gap-3 flex-wrap min-w-0">
                <span className="text-sm font-semibold text-slate-700 truncate">{step.name}</span>
                <Badge color={STEP_COLORS[step.type] ?? 'slate'}>{step.type}</Badge>
                <span className="text-xs text-slate-400">
                  {step.config.colA && `${step.config.colA} ↔ ${step.config.colB}`}
                  {step.config.col && step.config.threshold !== undefined && `${step.config.col} > ${step.config.threshold}`}
                  {step.config.col && step.config.threshold === undefined && !step.config.colA && step.config.col}
                </span>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <button onClick={() => toggleStep(step)} className={cn('transition-colors', step.enabled ? 'text-indigo-500' : 'text-slate-300')}>
                  {step.enabled ? <ToggleRight className="w-6 h-6" /> : <ToggleLeft className="w-6 h-6" />}
                </button>
                <button onClick={() => removeStep(step.step_id)} className="p-1.5 text-slate-300 hover:text-rose-400 hover:bg-rose-50 rounded-lg">
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

// ── Root App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [page, setPage] = useState<Page>('analytics');
  const [showUpload, setShowUpload] = useState(false);

  const {
    state: { datasets, joins, steps, config, metrics, stats, importance },
    refresh,
    saveConfig,
  } = useAppState();

  const primaryDs = datasets.find(d => d.dataset_id === config?.primary_id) ?? datasets[0];

  return (
    <div className="w-full h-full min-h-screen bg-[#f8fafc] text-[#1e293b] font-sans flex overflow-hidden">
      <Sidebar page={page} setPage={setPage} datasets={datasets} metrics={metrics} />

      <main className="flex-1 flex flex-col h-full overflow-y-auto">
        {page === 'analytics' && <AnalyticsPage config={config} metrics={metrics} stats={stats} importance={importance} onUpload={() => setShowUpload(true)} />}
        {page === 'predictions' && <PredictionsPage config={config} datasets={datasets} />}
        {page === 'datasets' && <DatasetsPage datasets={datasets} config={config} onUpload={() => setShowUpload(true)} onRefresh={refresh} onSaveConfig={saveConfig} />}
        {page === 'joins' && <JoinsPage datasets={datasets} joins={joins} onRefresh={refresh} />}
        {page === 'chain' && <ChainPage steps={steps} columns={primaryDs?.schema ?? []} joinsCount={joins.length} onRefresh={refresh} />}
      </main>

      {showUpload && <UploadModal datasets={datasets} onClose={() => setShowUpload(false)} onDone={refresh} />}
    </div>
  );
}