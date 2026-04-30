import express from 'express';
import { createServer as createViteServer } from 'vite';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import Database from 'better-sqlite3';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Database ──────────────────────────────────────────────────────────────────

const dbDir = path.join(__dirname, 'db');
if (!fs.existsSync(dbDir)) fs.mkdirSync(dbDir, { recursive: true });

const db = new Database(path.join(dbDir, 'data.db'));

function initDb() {
  db.exec(`
    CREATE TABLE IF NOT EXISTS generic_rows (
      row_id  INTEGER PRIMARY KEY AUTOINCREMENT,
      data    TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS dataset_schema (
      id            INTEGER PRIMARY KEY CHECK (id = 1),
      columns       TEXT    NOT NULL,
      outcome_col   TEXT    NOT NULL,
      response_col  TEXT    NOT NULL,
      label         TEXT    NOT NULL,
      row_count     INTEGER NOT NULL DEFAULT 0,
      uploaded_at   TEXT    NOT NULL
    );
  `);
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface ColumnMeta {
  name: string;
  type: 'numeric' | 'categorical';
  min: number | null;
  max: number | null;
  uniqueValues: string[] | null;
}

// ── Schema inference ──────────────────────────────────────────────────────────

function inferSchema(rows: Record<string, any>[]): ColumnMeta[] {
  if (rows.length === 0) return [];

  return Object.keys(rows[0]).map(col => {
    const values = rows.map(r => r[col]).filter(v => v !== null && v !== undefined && v !== '');
    const nums = values.map(Number).filter(n => !isNaN(n));
    const isNumeric = nums.length > values.length * 0.8;

    if (isNumeric) {
      return {
        name: col,
        type: 'numeric' as const,
        min: Math.min(...nums),
        max: Math.max(...nums),
        uniqueValues: null,
      };
    }

    return {
      name: col,
      type: 'categorical' as const,
      min: null,
      max: null,
      uniqueValues: [...new Set(values.map(String))].slice(0, 50),
    };
  });
}

// ── Logistic Regression ───────────────────────────────────────────────────────

class SimpleLogistic {
  private weights: number[] = [];
  private bias = 0;
  private featureCols: string[] = [];
  private encodings: Record<string, Record<string, number>> = {};
  private means: number[] = [];
  private stds: number[] = [];
  trained = false;

  train(rows: Record<string, any>[], outcomeCol: string, schema: ColumnMeta[]) {
    this.featureCols = schema.filter(c => c.name !== outcomeCol).map(c => c.name);
    this.encodings = {};

    schema.forEach(col => {
      if (col.type === 'categorical' && col.uniqueValues) {
        const enc: Record<string, number> = {};
        col.uniqueValues.forEach((v, i) => { enc[v] = i; });
        this.encodings[col.name] = enc;
      }
    });

    const X: number[][] = [];
    const y: number[] = [];

    for (const row of rows) {
      const outcome = Number(row[outcomeCol]);
      if (isNaN(outcome)) continue;
      const features = this.extractFeatures(row);
      if (features.some(isNaN)) continue;
      X.push(features);
      y.push(outcome > 0.5 ? 1 : 0);
    }

    if (X.length < 10) { this.trained = false; return; }

    const n = X[0].length;
    this.means = Array(n).fill(0);
    this.stds = Array(n).fill(1);

    for (let j = 0; j < n; j++) {
      const col = X.map(r => r[j]);
      this.means[j] = col.reduce((a, b) => a + b, 0) / col.length;
      const variance = col.reduce((a, b) => a + (b - this.means[j]) ** 2, 0) / col.length;
      this.stds[j] = Math.sqrt(variance) || 1;
    }

    const Xn = X.map(row => row.map((v, j) => (v - this.means[j]) / this.stds[j]));

    this.weights = Array(n).fill(0);
    this.bias = 0;
    const lr = 0.1;
    const epochs = 200;

    for (let e = 0; e < epochs; e++) {
      const grads = Array(n).fill(0);
      let biasGrad = 0;
      for (let i = 0; i < Xn.length; i++) {
        const z = Xn[i].reduce((s, v, j) => s + v * this.weights[j], this.bias);
        const pred = 1 / (1 + Math.exp(-z));
        const err = pred - y[i];
        for (let j = 0; j < n; j++) grads[j] += err * Xn[i][j];
        biasGrad += err;
      }
      for (let j = 0; j < n; j++) this.weights[j] -= lr * grads[j] / Xn.length;
      this.bias -= lr * biasGrad / Xn.length;
    }

    this.trained = true;
  }

  private extractFeatures(row: Record<string, any>): number[] {
    return this.featureCols.map(col => {
      const v = row[col];
      if (this.encodings[col]) return this.encodings[col][String(v)] ?? 0;
      return Number(v);
    });
  }

  predict(row: Record<string, any>): number {
    if (!this.trained) throw new Error('Model not trained');
    const features = this.extractFeatures(row);
    const norm = features.map((v, j) => (v - this.means[j]) / this.stds[j]);
    const z = norm.reduce((s, v, j) => s + v * this.weights[j], this.bias);
    return 1 / (1 + Math.exp(-z));
  }
}

const model = new SimpleLogistic();

// ── Helpers ───────────────────────────────────────────────────────────────────

function getSchema() {
  return db.prepare('SELECT * FROM dataset_schema WHERE id = 1').get() as any;
}

async function trainModel() {
  try {
    const schema = getSchema();
    if (!schema) return;
    const rows = db.prepare('SELECT data FROM generic_rows').all() as { data: string }[];
    const parsed = rows.map(r => JSON.parse(r.data));
    const cols = JSON.parse(schema.columns) as ColumnMeta[];
    model.train(parsed, schema.outcome_col, cols);
    console.log(`[model] Trained on ${parsed.length} rows, outcome="${schema.outcome_col}"`);
  } catch (e) {
    console.error('[model] Training failed:', e);
  }
}

// ── Server ────────────────────────────────────────────────────────────────────

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '500mb' }));
  app.use(express.urlencoded({ extended: true, limit: '500mb' }));
  app.use(express.text({ type: 'text/csv', limit: '500mb' }));
  app.use(express.text({ type: 'text/plain', limit: '500mb' }));

  initDb();
  await trainModel();

  // ── GET /api/schema ──────────────────────────────────────────────────────────

  app.get('/api/schema', (req, res) => {
    const schema = getSchema();
    if (!schema) return res.status(404).json({ error: 'No dataset loaded' });
    res.json({ ...schema, columns: JSON.parse(schema.columns) });
  });

  // ── POST /api/schema ─────────────────────────────────────────────────────────

  app.post('/api/schema', (req, res) => {
    try {
      const { outcome_col, response_col } = req.body;
      const existing = getSchema();
      if (!existing) return res.status(404).json({ error: 'No dataset loaded' });

      const cols: ColumnMeta[] = JSON.parse(existing.columns);
      const names = cols.map(c => c.name);
      if (!names.includes(outcome_col)) return res.status(400).json({ error: `Column "${outcome_col}" not found` });
      if (!names.includes(response_col)) return res.status(400).json({ error: `Column "${response_col}" not found` });

      db.prepare('UPDATE dataset_schema SET outcome_col = ?, response_col = ? WHERE id = 1').run(outcome_col, response_col);
      trainModel().catch(console.error);
      res.json({ success: true });
    } catch (e) {
      res.status(500).json({ error: (e as Error).message });
    }
  });

  // ── GET /api/stats ───────────────────────────────────────────────────────────

  app.get('/api/stats', (req, res) => {
    try {
      const schema = getSchema();
      if (!schema) return res.json([]);

      const rows = db.prepare('SELECT data FROM generic_rows').all() as { data: string }[];
      const data = rows.map(r => JSON.parse(r.data));
      const cols = JSON.parse(schema.columns) as ColumnMeta[];

      const xCol = schema.response_col;
      const yCol = schema.outcome_col;
      const xMeta = cols.find(c => c.name === xCol);
      if (!xMeta) return res.json([]);

      if (xMeta.type === 'numeric') {
        const buckets = Math.max(2, Math.min(50, parseInt(String(req.query.buckets ?? '10'), 10)));
        const xMin = xMeta.min ?? 0;
        const xMax = xMeta.max ?? 100;
        const step = (xMax - xMin) / buckets;

        const groups: { total: number; pos: number }[] = Array.from({ length: buckets }, () => ({ total: 0, pos: 0 }));

        for (const row of data) {
          const x = Number(row[xCol]);
          const y = Number(row[yCol]);
          if (isNaN(x) || isNaN(y)) continue;
          let idx = Math.floor((x - xMin) / step);
          if (idx >= buckets) idx = buckets - 1;
          if (idx < 0) idx = 0;
          groups[idx].total++;
          if (y > 0.5) groups[idx].pos++;
        }

        return res.json(groups.map((g, i) => ({
          range: `${(xMin + i * step).toFixed(1)}–${(xMin + (i + 1) * step).toFixed(1)}`,
          purchaseRate: g.total > 0 ? g.pos / g.total : 0,
          total: g.total,
        })));
      }

      // Categorical
      const groups: Record<string, { total: number; pos: number }> = {};
      for (const row of data) {
        const x = String(row[xCol]);
        const y = Number(row[yCol]);
        if (isNaN(y)) continue;
        if (!groups[x]) groups[x] = { total: 0, pos: 0 };
        groups[x].total++;
        if (y > 0.5) groups[x].pos++;
      }

      return res.json(Object.entries(groups).map(([k, v]) => ({
        range: k,
        purchaseRate: v.total > 0 ? v.pos / v.total : 0,
        total: v.total,
      })));
    } catch (e) {
      res.status(500).json({ error: (e as Error).message });
    }
  });

  // ── POST /api/predict ────────────────────────────────────────────────────────

  app.post('/api/predict', (req, res) => {
    try {
      if (!model.trained) return res.status(503).json({ error: 'Model not trained yet' });
      const prob = model.predict(req.body);
      res.json({ purchase_probability: prob });
    } catch (e) {
      res.status(500).json({ error: (e as Error).message });
    }
  });

  // ── GET /api/dataset ─────────────────────────────────────────────────────────

  app.get('/api/dataset', (req, res) => {
    try {
      const page = Math.max(1, parseInt(String(req.query.page ?? '1'), 10));
      const limit = Math.max(1, parseInt(String(req.query.limit ?? '20'), 10));
      const offset = (page - 1) * limit;

      const total = (db.prepare('SELECT COUNT(*) as n FROM generic_rows').get() as any).n;
      const rows = db.prepare('SELECT row_id, data FROM generic_rows ORDER BY row_id ASC LIMIT ? OFFSET ?').all(limit, offset) as { row_id: number; data: string }[];

      res.json({
        total,
        rows: rows.map(r => ({ row_id: r.row_id, ...JSON.parse(r.data) })),
      });
    } catch (e) {
      res.status(500).json({ error: (e as Error).message });
    }
  });

  // ── POST /api/dataset (single row) ──────────────────────────────────────────

  app.post('/api/dataset', (req, res) => {
    try {
      const result = db.prepare('INSERT INTO generic_rows (data) VALUES (?)').run(JSON.stringify(req.body));
      db.prepare('UPDATE dataset_schema SET row_count = row_count + 1 WHERE id = 1').run();
      trainModel().catch(console.error);
      res.json({ row_id: result.lastInsertRowid, success: true });
    } catch (e) {
      res.status(500).json({ error: (e as Error).message });
    }
  });

  // ── DELETE /api/dataset/:id ──────────────────────────────────────────────────

  app.delete('/api/dataset/:id', (req, res) => {
    try {
      const result = db.prepare('DELETE FROM generic_rows WHERE row_id = ?').run(parseInt(req.params.id));
      if (result.changes === 0) return res.status(404).json({ error: 'Row not found' });
      db.prepare('UPDATE dataset_schema SET row_count = MAX(0, row_count - 1) WHERE id = 1').run();
      res.json({ success: true });
    } catch (e) {
      res.status(500).json({ error: (e as Error).message });
    }
  });

  // ── POST /api/dataset/upload ─────────────────────────────────────────────────

  app.post('/api/dataset/upload', (req, res) => {
    try {
      const csvText: string = typeof req.body === 'string' ? req.body : '';
      if (!csvText.trim()) return res.status(400).json({ error: 'Empty CSV' });

      const outcomeCol = String(req.headers['x-outcome-col'] ?? '').trim();
      const responseCol = String(req.headers['x-response-col'] ?? '').trim();
      const label = String(req.headers['x-dataset-label'] ?? 'Uploaded Dataset').trim();

      const lines = csvText.trim().split(/\r?\n/);
      if (lines.length < 2) return res.status(400).json({ error: 'CSV needs header + data rows' });

      const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));

      if (outcomeCol && !headers.includes(outcomeCol))
        return res.status(400).json({ error: `Outcome column "${outcomeCol}" not found in CSV` });
      if (responseCol && !headers.includes(responseCol))
        return res.status(400).json({ error: `Response column "${responseCol}" not found in CSV` });

      const parseRow = (line: string): Record<string, any> => {
        const vals = line.split(',');
        const obj: Record<string, any> = {};
        headers.forEach((h, i) => {
          const v = (vals[i] ?? '').trim().replace(/^"|"$/g, '');
          obj[h] = v === '' ? '' : isNaN(Number(v)) ? v : Number(v);
        });
        return obj;
      };

      const rows = lines.slice(1).filter(l => l.trim()).map(parseRow);
      if (rows.length === 0) return res.status(400).json({ error: 'No data rows found' });

      const schema = inferSchema(rows);

      // Use user-selected columns, fall back to heuristics only if not provided
      const resolvedOutcome = outcomeCol
        || schema.find(c => /buy|bought|purchase|convert|outcome/i.test(c.name))?.name
        || schema[schema.length - 1]?.name
        || '';

      const resolvedResponse = responseCol
        || schema.find(c => c.type === 'numeric' && /time|wait|response|delay|duration/i.test(c.name) && c.name !== resolvedOutcome)?.name
        || schema.find(c => c.type === 'numeric' && c.name !== resolvedOutcome)?.name
        || '';

      db.transaction(() => {
        db.prepare('DELETE FROM generic_rows').run();
        const insert = db.prepare('INSERT INTO generic_rows (data) VALUES (?)');
        for (const row of rows) insert.run(JSON.stringify(row));
        db.prepare(`
          INSERT OR REPLACE INTO dataset_schema (id, columns, outcome_col, response_col, label, row_count, uploaded_at)
          VALUES (1, ?, ?, ?, ?, ?, ?)
        `).run(JSON.stringify(schema), resolvedOutcome, resolvedResponse, label, rows.length, new Date().toISOString());
      })();

      trainModel().catch(console.error);

      res.json({ success: true, inserted: rows.length, outcome_col: resolvedOutcome, response_col: resolvedResponse, schema });
    } catch (e) {
      res.status(500).json({ error: (e as Error).message });
    }
  });

  // ── Vite / Static ────────────────────────────────────────────────────────────

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      root: path.resolve(__dirname, '../'),
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (_req, res) => res.sendFile(path.join(__dirname, 'dist', 'index.html')));
  }

  app.listen(PORT, '0.0.0.0', () => console.log(`Server running at http://0.0.0.0:${PORT}`));
}

startServer();