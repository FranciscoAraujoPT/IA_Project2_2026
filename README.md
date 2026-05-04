# Predictron

Generic predictive modelling tool — upload any number of CSVs, define joins, tune the forward chain, and get a trained logistic regression model.

## Architecture

```
src/
├── backend/
│   ├── app.py                        # Flask API entry point
│   ├── requirements.txt
│   ├── server.ts                     # Node Vite dev server + proxy
│   ├── db/                           # SQLite database (auto-created)
│   └── predictive_model/
│       ├── __init__.py
│       ├── chain.py                  # Forward chain engine
│       ├── chain_store.py            # Chain steps persistence
│       ├── datasets.py               # N-dataset manager + merge logic
│       └── model.py                  # Logistic regression + metrics
└── frontend/
    ├── App.tsx                       # Root component
    ├── main.tsx
    ├── styles/globals.css
    ├── lib/
    │   ├── api.ts                    # API client
    │   └── types.ts                  # Shared TypeScript types
    └──hooks/
        └── useAppState.ts            # Central state + actions
```

## Quick Start

### 1. Python backend

```bash
cd src/backend
pip install -r requirements.txt
python app.py
# Runs on http://localhost:5000
```

### 2. Frontend (dev mode)

```bash
npm install
npm run dev
# Opens http://localhost:3000
# Proxies /api/* to Flask on :5000
```

## How to use

1. **Upload** — drag & drop one or more CSV files. Select which columns to include per file. Mark a primary dataset.
2. **Join** — if you have multiple files, define which key columns link them (left / inner / outer join). Files with identical columns are auto-unioned.
3. **Chain** — review auto-generated forward chain steps. Toggle on/off or add custom ones (interactions, log transforms, ratios, etc). Each step can reference columns from earlier steps.
4. **Analyze** — view model metrics (AUC, F1, accuracy…), the outcome-rate chart, and feature importance.
5. **Predict** — enter values for any row and get a live probability estimate.

## Forward Chaining

Features are engineered in step order. Step N can reference a column created by step N-1.

Example pipeline:
```
1. passthrough(response_time_min)          → response_time_min
2. passthrough(purchase_value)             → purchase_value
3. log(response_time_min)                  → log_response_time_min
4. interaction(response_time_min × purchase_value) → response_time_min_x_purchase_value
5. ratio(purchase_value / response_time_min)  → ratio_purchase_value_response_time_min
6. binary_threshold(response_time_min > 30)   → response_time_min_high
7. encode(product_category)                → product_category_enc
```
All enabled features are then scaled and fed to logistic regression.

## Join logic

When multiple datasets are loaded:
- If join rules are defined, they are applied in order starting from the primary dataset
- Right-side columns are prefixed with `ds<id>_` to avoid name collisions  
- If no join rules are defined, datasets with identical column sets are unioned (stacked)
