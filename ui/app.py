"""
ArvyaX — Flask Web UI
Run: python app.py
Open: http://localhost:5000
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# Add parent directory to path so pipeline.py is found
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from flask import Flask, render_template, request, jsonify
from pipeline import (text_features, meta_features, build_features,
                      decide, compute_unc)

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# ── Load saved models
MODEL_PATH = os.path.join(BASE, 'models', 'models.pkl')

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

xgb    = bundle['xgb']
rf     = bundle['rf']
lr     = bundle['lr']
reg    = bundle['reg']
tfidf  = bundle['tfidf']
imp    = bundle['imputer']
le     = bundle['le']

print(f"[ArvyaX] Models loaded. Classes: {list(le.classes_)}")


# ── Single prediction function
def predict_single(journal_text, ambience_type, duration_min, sleep_hours,
                   energy_level, stress_level, time_of_day,
                   previous_day_mood, face_emotion_hint, reflection_quality):

    row = {
        'journal_text':      journal_text,
        'ambience_type':     ambience_type,
        'duration_min':      duration_min,
        'sleep_hours':       sleep_hours,
        'energy_level':      energy_level,
        'stress_level':      stress_level,
        'time_of_day':       time_of_day,
        'previous_day_mood': previous_day_mood,
        'face_emotion_hint': face_emotion_hint,
        'reflection_quality':reflection_quality,
    }

    df_single = pd.DataFrame([row])
    X, _ = build_features(df_single, tfidf=tfidf, fit=False)
    X = pd.DataFrame(imp.transform(X), columns=X.columns)

    # Ensemble probabilities
    xp = xgb.predict_proba(X)
    rp = rf.predict_proba(X)
    lp = lr.predict_proba(X)
    proba = (xp * 0.50 + rp * 0.35 + lp * 0.15)[0]

    state_enc = int(np.argmax(proba))
    state     = le.inverse_transform([state_enc])[0]

    # Intensity
    intensity = int(np.clip(round(float(reg.predict(X)[0])), 1, 5))

    # Decision
    what, when, message = decide(state, row)

    # Uncertainty
    confidence, uncertain_flag = compute_unc(journal_text, reflection_quality, proba)

    # Class probabilities
    class_probs = {cls: round(float(p), 3) for cls, p in zip(le.classes_, proba)}

    return {
        'predicted_state':     state,
        'predicted_intensity': intensity,
        'confidence':          round(confidence, 3),
        'uncertain_flag':      uncertain_flag,
        'what_to_do':          what,
        'when_to_do':          when,
        'support_message':     message,
        'class_probabilities': class_probs,
    }


# ── Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.json
        result = predict_single(
            journal_text       = d.get('journal_text', ''),
            ambience_type      = d.get('ambience_type', 'ocean'),
            duration_min       = float(d.get('duration_min', 15)),
            sleep_hours        = float(d.get('sleep_hours', 7)),
            energy_level       = float(d.get('energy_level', 3)),
            stress_level       = float(d.get('stress_level', 3)),
            time_of_day        = d.get('time_of_day', 'morning'),
            previous_day_mood  = d.get('previous_day_mood', 'neutral'),
            face_emotion_hint  = d.get('face_emotion_hint', 'neutral_face'),
            reflection_quality = d.get('reflection_quality', 'clear'),
        )
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/predictions')
def predictions():
    path = os.path.join(BASE, 'outputs', 'predictions.csv')
    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient='records'))


if __name__ == '__main__':
    print("🌿 ArvyaX UI starting at http://localhost:5000")
    app.run(debug=True, port=5000)