from flask import Flask, request, jsonify, send_from_directory, session, redirect
import pickle
import numpy as np
import sqlite3
import hashlib
import os
import secrets

app = Flask(__name__, static_folder='static')
app.secret_key = secrets.token_hex(32)

DB = 'users.db'

def init_db():
    with sqlite3.connect(DB) as con:
        con.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT    NOT NULL,
                email    TEXT    NOT NULL UNIQUE,
                password TEXT    NOT NULL,
                created  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        con.commit()

init_db()

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

model    = data['model']
encoders = data['encoders']
features = data['features']

@app.route('/')
def index():
    if 'user' not in session:
        return redirect('/login')
    return send_from_directory('static', 'index.html')

@app.route('/login')
def login_page():
    return send_from_directory('static', 'login.html')

@app.route('/signup')
def signup_page():
    return send_from_directory('static', 'signup.html')

@app.route('/api/signup', methods=['POST'])
def signup():
    body  = request.get_json()
    name  = (body.get('name') or '').strip()
    email = (body.get('email') or '').strip().lower()
    pw    = body.get('password') or ''
    if not name or not email or not pw:
        return jsonify({'error': 'All fields are required.'}), 400
    if len(pw) < 6:
        return jsonify({'error': 'Password must be at least 6 characters.'}), 400
    try:
        with sqlite3.connect(DB) as con:
            con.execute('INSERT INTO users (name, email, password) VALUES (?,?,?)',
                        (name, email, hash_pw(pw)))
            con.commit()
        session['user'] = {'name': name, 'email': email}
        return jsonify({'ok': True, 'name': name})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'An account with this email already exists.'}), 409

@app.route('/api/login', methods=['POST'])
def login():
    body  = request.get_json()
    email = (body.get('email') or '').strip().lower()
    pw    = body.get('password') or ''
    with sqlite3.connect(DB) as con:
        row = con.execute(
            'SELECT name, email FROM users WHERE email=? AND password=?',
            (email, hash_pw(pw))
        ).fetchone()
    if not row:
        return jsonify({'error': 'Invalid email or password.'}), 401
    session['user'] = {'name': row[0], 'email': row[1]}
    return jsonify({'ok': True, 'name': row[0]})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'ok': True})

@app.route('/api/me')
def me():
    if 'user' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    return jsonify(session['user'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorised'}), 401
    try:
        body = request.get_json()
        raw = {
            'person_age':                 float(body['age']),
            'person_income':              float(body['income']),
            'person_home_ownership':      body['home'],
            'person_emp_length':          float(body['emp_length']),
            'loan_intent':                body['loan_intent'],
            'loan_grade':                 body['loan_grade'],
            'loan_amnt':                  float(body['loan_amnt']),
            'loan_int_rate':              float(body['int_rate']),
            'loan_percent_income':        float(body['pct_income']),
            'cb_person_default_on_file':  body['default_on_file'],
            'cb_person_cred_hist_length': float(body['cred_hist']),
        }
        for col, mapping in encoders.items():
            if col in raw:
                val = raw[col]
                if val not in mapping:
                    return jsonify({'error': f'Unknown value "{val}" for {col}'}), 400
                raw[col] = mapping[val]
        X     = np.array([[raw[f] for f in features]])
        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        return jsonify({
            'decision':     'REJECTED' if pred == 1 else 'APPROVED',
            'confidence':   round(float(proba[pred]) * 100, 1),
            'prob_default': round(float(proba[1]) * 100, 1),
            'prob_safe':    round(float(proba[0]) * 100, 1),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
