import React, { useState, useEffect, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import {
  Activity, Target, BrainCircuit, Database, Plus, Upload,
  Trash2, RefreshCw, CheckCircle2, AlertCircle, ArrowUpDown,
  Search, Download, Settings, ChevronRight, Layers,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) { return twMerge(clsx(inputs)); }

// ── Types ─────────────────────────────────────────────────────────────────────

type Page = 'analytics' | 'predictions' | 'datasets';

interface ColumnMeta {
  name: string;
  type: 'numeric' | 'categorical';
  min: number | null;
  max: number | null;
  uniqueValues: string[] | null;
}

interface Schema {
  columns: ColumnMeta[];
  outcome_col: string;
  response_col: string;
  label: string;
  row_count: number;
  uploaded_at: string;
}

interface Stat {
  range: string;
  purchaseRate: number;
  total: number;
}

// ── Shared UI ─────────────────────────────────────────────────────────────────

function Badge({ children, color = 'slate' }: { children: React.ReactNode; color?: string }) {
  const colors: Record<string, string> = {
    slate: 'bg-slate-100 text-slate-600',
    green: 'bg-emerald-100 text-emerald-700',
    red: 'bg-rose-100 text-rose-600',
    indigo: 'bg-indigo-100 text-indigo-700',
    amber: 'bg-amber-100 text-amber-700',
    purple: 'bg-purple-100 text-purple-700',
  };
  return (
    <span className={cn('px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wide', colors[color] || colors.slate)}>
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
  useEffect(() => {
    if (!msg) return;
    const t = setTimeout(onDone, 3500);
    return () => clearTimeout(t);
  }, [msg]);
  return (
    <AnimatePresence>
      {msg && (
        <motion.div
          initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
          className={cn('flex items-center gap-2 px-4 py-3 rounded-xl text-sm font-medium',
            msg.ok ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-rose-50 text-rose-700 border border-rose-200')}
        >
          {msg.ok ? <CheckCircle2 className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
          {msg.text}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function Sidebar({ page, setPage, schema }: { page: Page; setPage: (p: Page) => void; schema: Schema | null }) {
  const nav: { id: Page; label: string; icon: React.ReactNode }[] = [
    { id: 'analytics', label: 'Analytics', icon: <Activity className="w-5 h-5" /> },
    { id: 'predictions', label: 'Predictions', icon: <Target className="w-5 h-5" /> },
    { id: 'datasets', label: 'Datasets', icon: <Database className="w-5 h-5" /> },
  ];
  return (
    <aside className="w-64 h-full bg-white border-r border-slate-200 hidden lg:flex flex-col shrink-0">
      <div className="p-6 border-b border-slate-100">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold">C</div>
          <span className="font-bold text-lg tracking-tight">ConvAI</span>
        </div>
        <p className="text-[10px] uppercase tracking-widest text-slate-400 font-semibold">Intelligence Suite</p>
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
        {schema && (
          <div className="bg-slate-50 rounded-xl p-3 border border-slate-200">
            <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400 mb-1">Active Dataset</p>
            <p className="text-xs font-semibold text-slate-700 truncate">{schema.label}</p>
            <p className="text-[10px] text-slate-400 mt-0.5">{schema.row_count.toLocaleString()} rows · {schema.columns.length} cols</p>
            <div className="flex gap-1 mt-2 flex-wrap">
              <Badge color="indigo">{schema.response_col}</Badge>
              <Badge color="green">{schema.outcome_col}</Badge>
            </div>
          </div>
        )}
        <div className="bg-slate-900 rounded-xl p-4 text-white">
          <p className="text-xs text-slate-400 mb-1">Model Status</p>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
            <p className="text-sm font-medium">Adaptive ML Ready</p>
          </div>
        </div>
      </div>
    </aside>
  );
}

// ── CSV Upload + Column Mapping Modal ─────────────────────────────────────────

function UploadModal({ onClose, onUploaded }: { onClose: () => void; onUploaded: () => void }) {
  const [step, setStep] = useState<'pick' | 'map' | 'uploading' | 'done'>('pick');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<{ headers: string[]; rows: Record<string, any>[]; inferred: ColumnMeta[] } | null>(null);
  const [outcomeCol, setOutcomeCol] = useState('');
  const [responseCol, setResponseCol] = useState('');
  const [label, setLabel] = useState('');
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

    // Infer types client-side for display
    const inferred: ColumnMeta[] = headers.map(col => {
      const vals = sampleRows.map(r => r[col]).filter(v => v !== '' && v !== null);
      const nums = vals.map(Number).filter(n => !isNaN(n));
      const isNumeric = nums.length > vals.length * 0.8;
      if (isNumeric) {
        return { name: col, type: 'numeric' as const, min: Math.min(...nums), max: Math.max(...nums), uniqueValues: null };
      }
      return { name: col, type: 'categorical' as const, min: null, max: null, uniqueValues: [...new Set(vals.map(String))].slice(0, 20) };
    });

    // Auto-detect columns
    const outcomGuess = inferred.find(c => c.type === 'numeric' && /buy|bought|purchase|convert|sale|outcome/i.test(c.name))?.name
      || inferred[inferred.length - 1]?.name || '';
    const responseGuess = inferred.find(c => c.type === 'numeric' && /time|wait|response|delay|latency|duration/i.test(c.name) && c.name !== outcomGuess)?.name
      || inferred.find(c => c.type === 'numeric' && c.name !== outcomGuess)?.name || '';

    setOutcomeCol(outcomGuess);
    setResponseCol(responseGuess);
    setPreview({ headers, rows: sampleRows.slice(0, 5), inferred });
    setStep('map');
  };

  const handleUpload = async () => {
    if (!file || !outcomeCol || !responseCol) { setError('Please select both columns.'); return; }
    setStep('uploading');
    setProgress('Reading file...');
    try {
      const text = await file.text();
      setProgress('Uploading to server...');
      const res = await fetch('/api/dataset/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'text/csv',
          'x-outcome-col': outcomeCol,
          'x-response-col': responseCol,
          'x-dataset-label': label || file.name,
        },
        body: text,
      });
      const d = await res.json();
      if (!res.ok) throw new Error(d.error);
      setProgress(`Imported ${d.inserted.toLocaleString()} rows!`);
      setStep('done');
      setTimeout(() => { onUploaded(); onClose(); }, 1200);
    } catch (e: any) {
      setError(e.message);
      setStep('map');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={onClose}>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
        className="bg-white rounded-3xl shadow-2xl w-full max-w-2xl mx-4 overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="bg-indigo-600 px-8 py-6 text-white">
          <h2 className="text-xl font-bold">Import Dataset</h2>
          <p className="text-indigo-200 text-sm mt-1">Upload any CSV — the app will adapt to your columns</p>
        </div>

        <div className="p-8">
          {step === 'pick' && (
            <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-300 rounded-2xl p-12 cursor-pointer hover:border-indigo-400 hover:bg-indigo-50 transition-all">
              <Upload className="w-10 h-10 text-slate-300 mb-4" />
              <p className="font-semibold text-slate-600">Drop a CSV file or click to browse</p>
              <p className="text-xs text-slate-400 mt-1">Any columns — the app will detect types automatically</p>
              <input type="file" accept=".csv" className="hidden" onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
            </label>
          )}

          {step === 'map' && preview && (
            <div className="space-y-6">
              {/* File info */}
              <div className="flex items-center gap-3 px-4 py-3 bg-slate-50 rounded-xl border border-slate-200">
                <Database className="w-4 h-4 text-indigo-500 shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-slate-700 truncate">{file?.name}</p>
                  <p className="text-[10px] text-slate-400">{preview.headers.length} columns detected</p>
                </div>
                <button onClick={() => setStep('pick')} className="text-xs text-indigo-500 font-bold hover:underline shrink-0">Change</button>
              </div>

              {/* Dataset label */}
              <div className="space-y-1.5">
                <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Dataset Name</label>
                <input
                  className="w-full border border-slate-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  value={label}
                  onChange={e => setLabel(e.target.value)}
                  placeholder="My Dataset"
                />
              </div>

              {/* Column mapping */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-green-500 inline-block"></span>
                    Outcome / Target column
                  </label>
                  <p className="text-[10px] text-slate-400">The 0/1 column you want to predict (e.g. "bought")</p>
                  <select
                    className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={outcomeCol}
                    onChange={e => setOutcomeCol(e.target.value)}
                  >
                    <option value="">— select —</option>
                    {preview.headers.map(h => <option key={h} value={h}>{h}</option>)}
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-indigo-500 inline-block"></span>
                    X-axis / Analysis column
                  </label>
                  <p className="text-[10px] text-slate-400">The main variable to chart (e.g. "response_time_min")</p>
                  <select
                    className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={responseCol}
                    onChange={e => setResponseCol(e.target.value)}
                  >
                    <option value="">— select —</option>
                    {preview.headers.map(h => <option key={h} value={h}>{h}</option>)}
                  </select>
                </div>
              </div>

              {/* Column type summary */}
              <div>
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2">Detected Columns</p>
                <div className="flex flex-wrap gap-2">
                  {preview.inferred.map(col => (
                    <span key={col.name} className={cn(
                      'px-2.5 py-1 rounded-lg text-xs font-medium border',
                      col.name === outcomeCol ? 'bg-green-50 border-green-300 text-green-700' :
                        col.name === responseCol ? 'bg-indigo-50 border-indigo-300 text-indigo-700' :
                          col.type === 'numeric' ? 'bg-slate-50 border-slate-200 text-slate-600' :
                            'bg-amber-50 border-amber-200 text-amber-700'
                    )}>
                      {col.name}
                      <span className="ml-1 opacity-50 text-[9px]">{col.type === 'numeric' ? '#' : 'Aa'}</span>
                    </span>
                  ))}
                </div>
              </div>

              {/* Data preview */}
              <div>
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2">Preview (first 5 rows)</p>
                <div className="overflow-x-auto rounded-xl border border-slate-200">
                  <table className="w-full text-xs">
                    <thead className="bg-slate-50">
                      <tr>{preview.headers.map(h => <th key={h} className="px-3 py-2 text-left font-bold text-slate-500 whitespace-nowrap">{h}</th>)}</tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {preview.rows.map((row, i) => (
                        <tr key={i}>{preview.headers.map(h => <td key={h} className="px-3 py-2 text-slate-600 whitespace-nowrap">{String(row[h] ?? '')}</td>)}</tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {error && <p className="text-rose-500 text-sm font-medium">{error}</p>}

              <div className="flex gap-3">
                <button onClick={handleUpload} disabled={!outcomeCol || !responseCol}
                  className="flex-1 py-3 bg-indigo-600 text-white rounded-2xl font-bold text-sm hover:bg-indigo-700 transition-colors disabled:opacity-40">
                  Upload & Apply Dataset
                </button>
                <button onClick={onClose} className="px-6 py-3 bg-slate-100 text-slate-600 rounded-2xl font-bold text-sm hover:bg-slate-200 transition-colors">
                  Cancel
                </button>
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

function AnalyticsPage({ schema, onUpload }: { schema: Schema | null; onUpload: () => void }) {
  const [stats, setStats] = useState<Stat[]>([]);
  const [loading, setLoading] = useState(true);
  const [buckets, setBuckets] = useState(10);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/stats?buckets=${buckets}`)
      .then(r => r.json()).then(setStats).catch(console.error).finally(() => setLoading(false));
  }, [schema, buckets]);

  const avgRate = stats.length > 0 ? stats.reduce((a, s) => a + s.purchaseRate, 0) / stats.filter(s => s.total > 0).length : 0;
  const totalRows = stats.reduce((a, s) => a + s.total, 0);

  return (
    <>
      <header className="h-20 bg-white border-b border-slate-200 px-8 flex items-center justify-between shadow-sm sticky top-0 z-10">
        <div>
          <h1 className="text-xl font-bold text-slate-800">{schema?.label ?? 'Analytics'}</h1>
          {schema && <p className="text-xs text-slate-400 mt-0.5">Outcome: <span className="font-semibold text-indigo-600">{schema.outcome_col}</span> · X-axis: <span className="font-semibold text-indigo-600">{schema.response_col}</span></p>}
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right">
            <p className="text-xs text-slate-400 font-medium uppercase tracking-wider">Avg Outcome Rate</p>
            <p className="text-lg font-bold text-slate-900">{loading ? '...' : `${(avgRate * 100).toFixed(1)}%`}</p>
          </div>
          <div className="h-10 w-[1px] bg-slate-200"></div>
          <div className="text-right">
            <p className="text-xs text-slate-400 font-medium uppercase tracking-wider">Rows</p>
            <p className="text-lg font-bold text-slate-900">{loading ? '...' : totalRows.toLocaleString()}</p>
          </div>
          <button onClick={onUpload} className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-xl text-xs font-bold hover:bg-indigo-700 transition-colors">
            <Upload className="w-3.5 h-3.5" /> New Dataset
          </button>
        </div>
      </header>

      {!schema ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-4 text-slate-300">
          <Layers className="w-16 h-16" />
          <p className="text-lg font-bold">No dataset loaded</p>
          <button onClick={onUpload} className="px-6 py-3 bg-indigo-600 text-white rounded-2xl font-bold text-sm hover:bg-indigo-700 transition-colors">
            Upload a CSV to get started
          </button>
        </div>
      ) : (
        <div className="p-8 flex-1 grid grid-cols-1 md:grid-cols-12 gap-6 content-start">
          {/* KPI cards */}
          <div className="md:col-span-12 grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Total Rows', value: totalRows.toLocaleString(), sub: 'in dataset' },
              { label: 'Columns', value: schema.columns.length, sub: `${schema.columns.filter(c => c.type === 'numeric').length} numeric, ${schema.columns.filter(c => c.type === 'categorical').length} categorical` },
              { label: 'Avg Outcome Rate', value: `${(avgRate * 100).toFixed(1)}%`, sub: `"${schema.outcome_col}" = 1` },
              { label: 'X-axis Column', value: schema.response_col, sub: 'analysis variable' },
            ].map(k => (
              <Card key={k.label} title={k.label} icon={<Activity className="w-4 h-4" />}>
                <div className="text-2xl font-mono font-bold truncate">{k.value}</div>
                <p className="text-[10px] opacity-50 uppercase mt-2 font-semibold truncate">{k.sub}</p>
              </Card>
            ))}
          </div>

          {/* Main chart */}
          <div className="md:col-span-8 bg-white border border-slate-200 rounded-3xl p-6 shadow-sm flex flex-col min-h-[460px]">
            <div className="flex justify-between items-start mb-6">
              <div>
                <h2 className="text-lg font-bold">Outcome Rate by {schema.response_col}</h2>
                <p className="text-sm text-slate-500">How <span className="font-semibold">{schema.outcome_col}</span> varies across <span className="font-semibold">{schema.response_col}</span> values</p>
              </div>
              <Badge color="indigo">Live</Badge>
            </div>
            <div className="h-[320px] w-full">
              {loading ? (
                <div className="h-full flex items-center justify-center text-slate-300 font-mono">COMPUTING...</div>
              ) : (
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={stats}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis dataKey="range" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 9, fontWeight: 600 }} interval={0} angle={-20} textAnchor="end" height={40} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip
                      contentStyle={{ border: 'none', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.08)', fontSize: '12px' }}
                      formatter={(v: number, _n, p) => [`${(v * 100).toFixed(1)}% (n=${p.payload.total})`, schema.outcome_col]}
                    />
                    <Bar dataKey="purchaseRate" fill="#6366f1" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
            {/* Bucket slider — only for numeric x-axis */}
            {schema.columns.find(c => c.name === schema.response_col)?.type === 'numeric' && (
              <div className="md:col-span-8 bg-white border border-slate-200 rounded-2xl px-6 py-4 shadow-sm flex items-center gap-4">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 shrink-0">Intervals</span>
                <input
                  type="range" min={2} max={30} step={1}
                  value={buckets}
                  onChange={e => setBuckets(Number(e.target.value))}
                  className="flex-1 accent-indigo-600"
                />
                <span className="text-sm font-bold text-indigo-600 w-6 text-center shrink-0">{buckets}</span>
              </div>
            )}
          </div>

          {/* Bucket breakdown */}
          <div className="md:col-span-4 flex flex-col gap-3 overflow-y-auto max-h-[460px]">
            {stats.filter(s => s.total > 0).map((s, i) => (
              <div key={i} className="bg-white border border-slate-200 rounded-2xl px-5 py-3 flex items-center justify-between shadow-sm">
                <div>
                  <p className="text-xs text-slate-400 font-semibold truncate max-w-[120px]">{s.range}</p>
                  <p className="text-xl font-black mt-0.5">{(s.purchaseRate * 100).toFixed(1)}%</p>
                </div>
                <div className="text-right">
                  <p className="text-[10px] text-slate-400">n = {s.total}</p>
                  <Badge color={s.purchaseRate > 0.5 ? 'green' : s.purchaseRate > 0.25 ? 'amber' : 'red'}>
                    {s.purchaseRate > 0.5 ? 'High' : s.purchaseRate > 0.25 ? 'Mid' : 'Low'}
                  </Badge>
                </div>
              </div>
            ))}
          </div>

          {/* Column breakdown */}
          <div className="md:col-span-12">
            <h3 className="text-sm font-bold text-slate-700 mb-3">Dataset Columns</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {schema.columns.map(col => (
                <div key={col.name} className={cn(
                  'bg-white border rounded-xl px-4 py-3 text-xs shadow-sm',
                  col.name === schema.outcome_col ? 'border-green-300 bg-green-50' :
                    col.name === schema.response_col ? 'border-indigo-300 bg-indigo-50' : 'border-slate-200'
                )}>
                  <p className="font-bold text-slate-700 truncate">{col.name}</p>
                  <p className="text-slate-400 mt-0.5">{col.type === 'numeric' ? `${col.min?.toFixed(1)} – ${col.max?.toFixed(1)}` : col.uniqueValues?.slice(0, 3).join(', ')}</p>
                  <Badge color={col.name === schema.outcome_col ? 'green' : col.name === schema.response_col ? 'indigo' : col.type === 'numeric' ? 'slate' : 'amber'}>
                    {col.name === schema.outcome_col ? 'outcome' : col.name === schema.response_col ? 'x-axis' : col.type}
                  </Badge>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// ── Predictions Page ──────────────────────────────────────────────────────────

function PredictionsPage({ schema }: { schema: Schema | null }) {
  const [formValues, setFormValues] = useState<Record<string, any>>({});
  const [prediction, setPrediction] = useState<number | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<{ input: Record<string, any>; result: number }[]>([]);

  // Re-initialise form when schema changes
  useEffect(() => {
    if (!schema) return;
    const init: Record<string, any> = {};
    schema.columns.filter(c => c.name !== schema.outcome_col).forEach(col => {
      if (col.type === 'numeric') {
        const mid = col.min !== null && col.max !== null ? (col.min + col.max) / 2 : 0;
        init[col.name] = parseFloat(mid.toFixed(2));
      } else {
        init[col.name] = col.uniqueValues?.[0] ?? '';
      }
    });
    setFormValues(init);
    setPrediction(null);
  }, [schema]);

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setPredicting(true);
    setError(null);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formValues),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setPrediction(data.purchase_probability);
      setHistory(prev => [{ input: { ...formValues }, result: data.purchase_probability }, ...prev].slice(0, 6));
    } catch (err: any) {
      setError(err.message);
    } finally {
      setPredicting(false);
    }
  };

  const featureCols = schema?.columns.filter(c => c.name !== schema.outcome_col) ?? [];
  const prob = prediction ?? 0;
  const probColor = prob > 0.6 ? 'text-emerald-500' : prob > 0.35 ? 'text-amber-500' : 'text-rose-500';
  const probBg = prob > 0.6 ? 'bg-emerald-500' : prob > 0.35 ? 'bg-amber-500' : 'bg-rose-500';

  if (!schema) return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 text-slate-300">
      <BrainCircuit className="w-16 h-16" />
      <p className="text-lg font-bold">Upload a dataset first</p>
    </div>
  );

  return (
    <>
      <header className="h-20 bg-white border-b border-slate-200 px-8 flex items-center justify-between shadow-sm sticky top-0 z-10">
        <div>
          <h1 className="text-xl font-bold text-slate-800">Predictions</h1>
          <p className="text-xs text-slate-400 mt-0.5">Predicting <span className="font-semibold text-indigo-600">{schema.outcome_col}</span> · {featureCols.length} features</p>
        </div>
        <Badge color="indigo">Logistic Regression</Badge>
      </header>

      <div className="p-8 flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 content-start">

        {/* Dynamic form */}
        <div className="lg:col-span-5 bg-indigo-600 rounded-3xl p-8 text-white shadow-lg flex flex-col">
          <h2 className="text-xl font-bold mb-1">Configure Scenario</h2>
          <p className="text-indigo-200 text-xs mb-6">Predict: <span className="text-white font-semibold">{schema.outcome_col}</span></p>

          <form onSubmit={handlePredict} className="flex-1 flex flex-col gap-4">
            <div className="flex-1 space-y-4 overflow-y-auto pr-1">
              {featureCols.map(col => (
                <div key={col.name} className="space-y-1.5">
                  {col.type === 'numeric' ? (
                    <>
                      <label className="flex justify-between text-[11px] font-bold uppercase tracking-wider">
                        <span className="text-indigo-100">{col.name.replace(/_/g, ' ')}</span>
                        <span className="text-white font-mono">{Number(formValues[col.name] ?? 0).toFixed(2)}</span>
                      </label>
                      <input
                        type="range"
                        min={col.min ?? 0} max={col.max ?? 100}
                        step={(col.max !== null && col.min !== null) ? Math.max(0.01, (col.max - col.min) / 100) : 0.01}
                        className="w-full h-1.5 bg-indigo-400/30 rounded-full appearance-none cursor-pointer accent-white"
                        value={formValues[col.name] ?? 0}
                        onChange={e => setFormValues(v => ({ ...v, [col.name]: parseFloat(e.target.value) }))}
                      />
                      <div className="flex justify-between text-[9px] text-indigo-300 font-mono">
                        <span>{col.min?.toFixed(1)}</span><span>{col.max?.toFixed(1)}</span>
                      </div>
                    </>
                  ) : (
                    <>
                      <label className="block text-[11px] text-indigo-200 font-bold uppercase">{col.name.replace(/_/g, ' ')}</label>
                      <select
                        className="w-full bg-white/10 border border-white/20 rounded-xl px-3 py-2 text-xs focus:outline-none focus:bg-white/20 appearance-none"
                        value={formValues[col.name] ?? ''}
                        onChange={e => setFormValues(v => ({ ...v, [col.name]: e.target.value }))}
                      >
                        {col.uniqueValues?.map(v => <option key={v} value={v} className="text-slate-800">{v}</option>)}
                      </select>
                    </>
                  )}
                </div>
              ))}
            </div>

            {error && (
              <div className="flex items-center gap-2 bg-rose-500/20 border border-rose-400/40 rounded-xl px-4 py-3">
                <AlertCircle className="w-4 h-4 text-rose-200 shrink-0" />
                <p className="text-xs text-rose-100">{error}</p>
              </div>
            )}

            <button type="submit" disabled={predicting}
              className="w-full bg-white text-indigo-600 py-4 rounded-2xl font-bold text-sm shadow-xl active:scale-[0.98] transition-all disabled:opacity-50">
              {predicting ? <span className="flex items-center justify-center gap-2"><RefreshCw className="w-4 h-4 animate-spin" />Calculating...</span> : 'Run Prediction'}
            </button>
          </form>
        </div>

        {/* Result */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          <div className="bg-white border border-slate-200 rounded-3xl p-8 shadow-sm">
            <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-6">Prediction Output — {schema.outcome_col}</h3>

            {prediction === null ? (
              <div className="flex flex-col items-center justify-center py-12 text-slate-300">
                <BrainCircuit className="w-12 h-12 mb-4" />
                <p className="text-sm font-semibold">Set inputs and run a prediction</p>
              </div>
            ) : (
              <AnimatePresence mode="wait">
                <motion.div key={prediction} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
                  <div className="flex items-end gap-4">
                    <div className={cn('text-7xl font-black tabular-nums', probColor)}>{(prob * 100).toFixed(1)}</div>
                    <div className="pb-2">
                      <p className="text-2xl font-black text-slate-300">%</p>
                      <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">P({schema.outcome_col} = 1)</p>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                      <motion.div className={cn('h-full rounded-full', probBg)} initial={{ width: 0 }} animate={{ width: `${prob * 100}%` }} transition={{ duration: 0.6, ease: 'easeOut' }} />
                    </div>
                    <div className="flex justify-between text-[9px] font-mono text-slate-300"><span>0%</span><span>50%</span><span>100%</span></div>
                  </div>

                  <div className={cn('rounded-2xl px-5 py-4 flex items-center gap-4',
                    prob > 0.6 ? 'bg-emerald-50 border border-emerald-200' :
                      prob > 0.35 ? 'bg-amber-50 border border-amber-200' : 'bg-rose-50 border border-rose-200')}>
                    <div className={cn('w-10 h-10 rounded-xl flex items-center justify-center shrink-0',
                      prob > 0.6 ? 'bg-emerald-500' : prob > 0.35 ? 'bg-amber-500' : 'bg-rose-500')}>
                      {prob > 0.6 ? <CheckCircle2 className="w-5 h-5 text-white" /> : <AlertCircle className="w-5 h-5 text-white" />}
                    </div>
                    <div>
                      <p className="font-bold text-slate-800 text-sm">
                        {prob > 0.6 ? 'High likelihood' : prob > 0.35 ? 'Moderate likelihood' : 'Low likelihood'}
                      </p>
                      <p className="text-xs text-slate-500 mt-0.5">
                        {prob > 0.6 ? `Model predicts ${schema.outcome_col} = 1 with high confidence.`
                          : prob > 0.35 ? 'Outcome is uncertain — consider adjusting inputs.'
                            : `Model predicts ${schema.outcome_col} = 0 given these inputs.`}
                      </p>
                    </div>
                  </div>

                  {/* Input values used */}
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {featureCols.map(col => (
                      <div key={col.name} className="bg-slate-50 rounded-xl px-3 py-2.5">
                        <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400 truncate">{col.name}</p>
                        <p className="text-sm font-bold text-slate-700 mt-0.5 truncate capitalize">{String(formValues[col.name])}</p>
                      </div>
                    ))}
                  </div>
                </motion.div>
              </AnimatePresence>
            )}
          </div>

          {/* History */}
          {history.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm">
              <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-4">Recent Predictions</h3>
              <div className="space-y-2">
                {history.map((h, i) => (
                  <div key={i} className="flex items-center justify-between px-4 py-3 bg-slate-50 rounded-xl">
                    <div className="flex items-center gap-3 min-w-0">
                      <span className="text-[10px] font-mono text-slate-400 shrink-0">#{history.length - i}</span>
                      <span className="text-xs text-slate-600 truncate">
                        {Object.entries(h.input).slice(0, 3).map(([k, v]) => `${k}: ${v}`).join(' · ')}
                      </span>
                    </div>
                    <span className={cn('text-sm font-black shrink-0 ml-2',
                      h.result > 0.6 ? 'text-emerald-500' : h.result > 0.35 ? 'text-amber-500' : 'text-rose-500')}>
                      {(h.result * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// ── Datasets Page ─────────────────────────────────────────────────────────────

const PAGE_SIZE = 20;

function DatasetsPage({ schema, onUpload, onSchemaChange }: { schema: Schema | null; onUpload: () => void; onSchemaChange: () => void }) {
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [showAddForm, setShowAddForm] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [newRow, setNewRow] = useState<Record<string, any>>({});
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState<{ text: string; ok: boolean } | null>(null);
  const [editOutcome, setEditOutcome] = useState('');
  const [editResponse, setEditResponse] = useState('');

  useEffect(() => {
    if (schema) { setEditOutcome(schema.outcome_col); setEditResponse(schema.response_col); }
  }, [schema]);

  const fetchRows = useCallback((page = 1) => {
    setLoading(true);
    fetch(`/api/dataset?page=${page}&limit=${PAGE_SIZE}`)
      .then(r => r.json())
      .then(d => { setRows(d.rows ?? []); setTotal(d.total ?? 0); setError(null); })
      .catch(() => setError('Failed to load dataset'))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { fetchRows(currentPage); }, [currentPage, schema]);

  const showMsg = (text: string, ok: boolean) => setToast({ text, ok });

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      const res = await fetch('/api/dataset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(newRow) });
      const d = await res.json();
      if (!res.ok) throw new Error(d.error);
      showMsg('Row added', true);
      setShowAddForm(false);
      fetchRows(1); setCurrentPage(1);
    } catch (e: any) { showMsg(e.message, false); }
    finally { setSaving(false); }
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this row?')) return;
    try {
      const res = await fetch(`/api/dataset/${id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Delete failed');
      showMsg('Row deleted', true);
      fetchRows(currentPage);
    } catch { showMsg('Delete failed', false); }
  };

  const handleExport = async () => {
    const all = await fetch(`/api/dataset?page=1&limit=100000`).then(r => r.json());
    const allRows: Record<string, any>[] = all.rows ?? [];
    if (!allRows.length) return;
    const headers = Object.keys(allRows[0]).filter(k => k !== 'row_id');
    const csv = [headers.join(','), ...allRows.map(r => headers.map(h => r[h] ?? '').join(','))].join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `${schema?.label ?? 'dataset'}.csv`;
    a.click();
  };

  const handleSaveSettings = async () => {
    const res = await fetch('/api/schema', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ outcome_col: editOutcome, response_col: editResponse }) });
    if (res.ok) { showMsg('Mapping updated', true); setShowSettings(false); onSchemaChange(); }
    else showMsg('Failed to update', false);
  };

  const cols = schema?.columns ?? [];
  const displayCols = cols.length > 0 ? cols : (rows[0] ? Object.keys(rows[0]).filter(k => k !== 'row_id').map(k => ({ name: k, type: 'numeric' as const, min: null, max: null, uniqueValues: null })) : []);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <>
      <header className="h-20 bg-white border-b border-slate-200 px-8 flex items-center justify-between shadow-sm sticky top-0 z-10">
        <div>
          <h1 className="text-xl font-bold text-slate-800">Dataset Manager</h1>
          <p className="text-xs text-slate-400 mt-0.5">{total.toLocaleString()} rows · {schema?.label ?? 'no dataset'}</p>
        </div>
        <div className="flex items-center gap-3">
          {schema && <button onClick={() => setShowSettings(s => !s)} className="flex items-center gap-2 px-4 py-2 bg-slate-100 text-slate-600 rounded-xl text-xs font-bold hover:bg-slate-200 transition-colors"><Settings className="w-3.5 h-3.5" /> Mapping</button>}
          <button onClick={handleExport} className="flex items-center gap-2 px-4 py-2 bg-slate-100 text-slate-600 rounded-xl text-xs font-bold hover:bg-slate-200 transition-colors"><Download className="w-3.5 h-3.5" /> Export</button>
          <button onClick={onUpload} className="flex items-center gap-2 px-4 py-2 bg-slate-600 text-white rounded-xl text-xs font-bold hover:bg-slate-700 transition-colors"><Upload className="w-3.5 h-3.5" /> New Dataset</button>
          {schema && <button onClick={() => { setNewRow(Object.fromEntries(cols.map(c => [c.name, c.type === 'numeric' ? 0 : c.uniqueValues?.[0] ?? '']))); setShowAddForm(s => !s); }}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-xl text-xs font-bold hover:bg-indigo-700 transition-colors"><Plus className="w-3.5 h-3.5" /> Add Row</button>}
        </div>
      </header>

      <div className="p-8 flex-1 flex flex-col gap-6">
        <Toast msg={toast} onDone={() => setToast(null)} />

        {/* Column mapping settings */}
        <AnimatePresence>
          {showSettings && schema && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
              className="bg-white border border-indigo-200 rounded-2xl p-6 shadow-sm overflow-hidden">
              <h3 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2"><Settings className="w-4 h-4" /> Column Mapping</h3>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Outcome Column (target)</label>
                  <select className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={editOutcome} onChange={e => setEditOutcome(e.target.value)}>
                    {schema.columns.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500">X-axis Column (analysis)</label>
                  <select className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={editResponse} onChange={e => setEditResponse(e.target.value)}>
                    {schema.columns.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                  </select>
                </div>
              </div>
              <div className="flex gap-3">
                <button onClick={handleSaveSettings} className="px-5 py-2 bg-indigo-600 text-white rounded-xl text-sm font-bold hover:bg-indigo-700 transition-colors">Save & Retrain</button>
                <button onClick={() => setShowSettings(false)} className="px-5 py-2 bg-slate-100 text-slate-600 rounded-xl text-sm font-bold hover:bg-slate-200 transition-colors">Cancel</button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Add row form */}
        <AnimatePresence>
          {showAddForm && schema && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
              className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm overflow-hidden">
              <h3 className="text-sm font-bold text-slate-700 mb-4">Add New Row</h3>
              <form onSubmit={handleAdd}>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  {cols.map(col => (
                    <div key={col.name} className="space-y-1">
                      <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500 truncate block">{col.name}</label>
                      {col.type === 'numeric' ? (
                        <input type="number" step="any" min={col.min ?? undefined} max={col.max ?? undefined} required
                          className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                          value={newRow[col.name] ?? ''}
                          onChange={e => setNewRow(v => ({ ...v, [col.name]: parseFloat(e.target.value) }))} />
                      ) : (
                        <select className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
                          value={newRow[col.name] ?? ''}
                          onChange={e => setNewRow(v => ({ ...v, [col.name]: e.target.value }))}>
                          {col.uniqueValues?.map(v => <option key={v} value={v}>{v}</option>)}
                        </select>
                      )}
                    </div>
                  ))}
                </div>
                <div className="flex gap-3">
                  <button type="submit" disabled={saving} className="px-6 py-2 bg-indigo-600 text-white rounded-xl text-sm font-bold hover:bg-indigo-700 disabled:opacity-50 transition-colors">{saving ? 'Saving...' : 'Save'}</button>
                  <button type="button" onClick={() => setShowAddForm(false)} className="px-6 py-2 bg-slate-100 text-slate-600 rounded-xl text-sm font-bold hover:bg-slate-200 transition-colors">Cancel</button>
                </div>
              </form>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input type="text" placeholder="Search current page..." className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 bg-white"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>

        {/* Table */}
        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden flex-1">
          {loading ? (
            <div className="flex items-center justify-center h-64 text-slate-300 font-mono">LOADING...</div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-64 gap-3">
              <AlertCircle className="w-8 h-8 text-rose-300" />
              <p className="text-sm text-rose-400 font-semibold">{error}</p>
              <button onClick={() => fetchRows(currentPage)} className="text-xs text-indigo-500 font-bold hover:underline">Retry</button>
            </div>
          ) : !schema ? (
            <div className="flex flex-col items-center justify-center h-64 gap-3 text-slate-300">
              <Database className="w-10 h-10" />
              <p className="font-semibold">No dataset loaded yet</p>
              <button onClick={onUpload} className="text-xs text-indigo-500 font-bold hover:underline">Upload a CSV</button>
            </div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-slate-50 border-b border-slate-200">
                    <tr>
                      <th className="px-4 py-3 text-left text-[10px] font-bold uppercase tracking-wider text-slate-400 whitespace-nowrap">#</th>
                      {displayCols.map(col => (
                        <th key={col.name} className={cn(
                          'px-4 py-3 text-left text-[10px] font-bold uppercase tracking-wider whitespace-nowrap',
                          col.name === schema.outcome_col ? 'text-green-600' :
                            col.name === schema.response_col ? 'text-indigo-600' : 'text-slate-400'
                        )}>
                          {col.name}
                          {col.name === schema.outcome_col && <span className="ml-1 text-[8px] normal-case">(outcome)</span>}
                          {col.name === schema.response_col && <span className="ml-1 text-[8px] normal-case">(x-axis)</span>}
                        </th>
                      ))}
                      <th className="px-4 py-3 text-[10px] font-bold uppercase tracking-wider text-slate-400">Del</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {rows
                      .filter(row => !search || Object.values(row).some(v => String(v).toLowerCase().includes(search.toLowerCase())))
                      .map((row, i) => (
                        <tr key={row.row_id ?? i} className="hover:bg-slate-50 transition-colors">
                          <td className="px-4 py-3 font-mono text-xs text-slate-300">{row.row_id}</td>
                          {displayCols.map(col => {
                            const v = row[col.name];
                            const isOutcome = col.name === schema.outcome_col;
                            return (
                              <td key={col.name} className="px-4 py-3 whitespace-nowrap">
                                {isOutcome ? (
                                  <Badge color={Number(v) > 0.5 ? 'green' : 'red'}>{Number(v) > 0.5 ? 'Yes' : 'No'}</Badge>
                                ) : col.type === 'categorical' ? (
                                  <Badge>{String(v)}</Badge>
                                ) : (
                                  <span className="text-slate-700 font-mono text-xs">{typeof v === 'number' ? v.toFixed(2) : String(v)}</span>
                                )}
                              </td>
                            );
                          })}
                          <td className="px-4 py-3">
                            <button onClick={() => handleDelete(row.row_id)} className="p-1.5 text-slate-300 hover:text-rose-400 transition-colors rounded-lg hover:bg-rose-50">
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div className="px-4 py-3 flex items-center justify-between border-t border-slate-100 bg-slate-50">
                <p className="text-xs text-slate-400">
                  Showing <span className="font-semibold text-slate-600">{(currentPage - 1) * PAGE_SIZE + 1}–{Math.min(currentPage * PAGE_SIZE, total)}</span> of <span className="font-semibold text-slate-600">{total.toLocaleString()}</span>
                </p>
                {totalPages > 1 && (
                  <div className="flex items-center gap-1">
                    <button onClick={() => setCurrentPage(1)} disabled={currentPage === 1} className="px-2 py-1 text-xs rounded-lg text-slate-500 hover:bg-slate-200 disabled:opacity-30 font-bold">«</button>
                    <button onClick={() => setCurrentPage(p => Math.max(1, p - 1))} disabled={currentPage === 1} className="px-2 py-1 text-xs rounded-lg text-slate-500 hover:bg-slate-200 disabled:opacity-30 font-bold">‹</button>
                    {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                      const p = Math.min(Math.max(currentPage - 2, 1) + i, totalPages);
                      return (
                        <button key={p} onClick={() => setCurrentPage(p)}
                          className={cn('min-w-[28px] px-2 py-1 text-xs rounded-lg font-semibold', p === currentPage ? 'bg-indigo-600 text-white' : 'text-slate-500 hover:bg-slate-200')}>
                          {p}
                        </button>
                      );
                    })}
                    <button onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))} disabled={currentPage === totalPages} className="px-2 py-1 text-xs rounded-lg text-slate-500 hover:bg-slate-200 disabled:opacity-30 font-bold">›</button>
                    <button onClick={() => setCurrentPage(totalPages)} disabled={currentPage === totalPages} className="px-2 py-1 text-xs rounded-lg text-slate-500 hover:bg-slate-200 disabled:opacity-30 font-bold">»</button>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}

// ── Root App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [page, setPage] = useState<Page>('analytics');
  const [schema, setSchema] = useState<Schema | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  const loadSchema = useCallback(() => {
    fetch('/api/schema')
      .then(r => r.ok ? r.json() : null)
      .then(s => setSchema(s))
      .catch(() => setSchema(null));
  }, []);

  useEffect(() => { loadSchema(); }, []);

  return (
    <div className="w-full h-full min-h-screen bg-[#f8fafc] text-[#1e293b] font-sans flex overflow-hidden">
      <Sidebar page={page} setPage={setPage} schema={schema} />

      <main className="flex-1 flex flex-col h-full overflow-y-auto">
        {page === 'analytics' && <AnalyticsPage schema={schema} onUpload={() => setShowUpload(true)} />}
        {page === 'predictions' && <PredictionsPage schema={schema} />}
        {page === 'datasets' && <DatasetsPage schema={schema} onUpload={() => setShowUpload(true)} onSchemaChange={loadSchema} />}
      </main>

      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onUploaded={() => { loadSchema(); setPage('analytics'); }}
        />
      )}
    </div>
  );
}
