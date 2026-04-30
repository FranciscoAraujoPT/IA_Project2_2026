# Customer Response Time Analyzer

A full-stack web application that analyzes how response time to customer questions affects the probability of purchase, powered by a **Logistic Regression** predictive model.

## Architecture

```
Frontend (React + Vite)  ←→  Node/Express server  ←→  SQLite DB
                                                   ←→  Python Flask (ML)
```

## Pages

### Analytics
Bar chart showing purchase rate bucketed by response time (0–60 min). Live stats from the database.

### Predictions
Run the Python scikit-learn model with custom inputs:
- Response time, price, patience level, segment, time of day, complexity
- Shows probability with colour-coded verdict + prediction history

### Datasets
Full CRUD interface for the interactions table:
- View all rows with sorting and search
- Add individual rows via form
- Bulk import via CSV upload
- Delete rows
- Export to CSV

## Running

### Node.js server
```bash
pip install -r src/backend/predictive_model/requirements.txt
npm install
npm run dev
# → http://localhost:3000
```
## Python Model

`backend/model.py` — `PurchaseModel` class:
- **Algorithm**: Logistic Regression (scikit-learn)
- **Features**: response_time, price, patience, segment, time_of_day, complexity + engineered interaction features
- **Engineered features**: `rt × (1-patience)`, `price × complexity`, `log(response_time)`
- **Auto-retrain**: model retrains after every dataset insert/bulk import

## CSV Import Format

Required columns:
```
response_time_min,price,patience_level,segment,time_of_day,complexity,bought
5.2,99.99,0.7,high,morning,3,1
45.0,299.0,0.2,low,evening,8,0
```

Optional columns: `customer_id`, `product_id`
