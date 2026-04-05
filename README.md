# Loan Approval Predictor

A web application that predicts loan approval using a trained XGBoost model (93.72% accuracy) on the credit risk dataset.

## Project Structure

```
loan_app/
├── app.py              ← Flask backend (API + serves frontend)
├── model.pkl           ← Trained XGBoost model + encoders
├── requirements.txt    ← Python dependencies
└── static/
    └── index.html      ← Frontend UI
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

## API Endpoint

**POST** `/predict`

Request body (JSON):
```json
{
  "age": "28",
  "income": "60000",
  "home": "RENT",
  "emp_length": "5",
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": "15000",
  "int_rate": "11.5",
  "pct_income": "0.25",
  "default_on_file": "N",
  "cred_hist": "4"
}
```

Response:
```json
{
  "decision": "APPROVED",
  "confidence": 87.3,
  "prob_default": 12.7,
  "prob_safe": 87.3
}
```

## Model Info

- **Algorithm**: XGBoost Classifier
- **Accuracy**: 93.72% on held-out test set
- **Features**: 11 (age, income, home ownership, employment length, loan intent, grade, amount, interest rate, % of income, prior default, credit history length)
- **Target**: `loan_status` — 0 = repaid, 1 = defaulted
