"""
ArvyaX — Flask Web UI
Run: python app.py
Then open: http://localhost:5000
"""
import os, sys, pickle, json, re, warnings
import numpy as np
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from pipeline import (text_feats, meta_feats, decide, message,
                      FACE_M, TIME_M, MOOD_M, AMB_M, QUAL_M)

app = Flask(__name__, template_folder='ui/templates', static_folder='ui/static')

# Load models
def load_models():
    mdir = os.path.join(BASE, 'models')
    with open(f'{mdir}/ensemble_state.pkl','rb') as f: ens = pickle.load(f)
    with open(f'{mdir}/xgb_intensity.pkl','rb') as f:  reg = pickle.load(f)
    with open(f'{mdir}/tfidf.pkl','rb') as f:           tfidf = pickle.load(f)
    with open(f'{mdir}/label_encoder.pkl','rb') as f:   le = pickle.load(f)
    with open(f'{mdir}/imputer.pkl','rb') as f:         imp = pickle.load(f)
    with open(f'{mdir}/meta.json') as f:                meta = json.load(f)
    return ens, reg, tfidf, le, imp, meta

ens, reg, tfidf, le, imp, meta_info = load_models()

def predict_single(journal_text, ambience_type, duration_min, sleep_hours,
                   energy_level, stress_level, time_of_day, previous_day_mood,
                   face_emotion_hint, reflection_quality):
    import pandas as pd
    from scipy.stats import entropy as scipy_entropy

    row = dict(journal_text=journal_text, ambience_type=ambience_type,
               duration_min=duration_min, sleep_hours=sleep_hours,
               energy_level=energy_level, stress_level=stress_level,
               time_of_day=time_of_day, previous_day_mood=previous_day_mood,
               face_emotion_hint=face_emotion_hint, reflection_quality=reflection_quality)

    tf_row = pd.Series(text_feats(journal_text)).values.reshape(1,-1).astype(float)
    mf_row = pd.Series(meta_feats(row)).values.reshape(1,-1).astype(float)
    combined = np.hstack([tf_row, mf_row])
    combined = imp.transform(combined)
    tfidf_row = tfidf.transform([journal_text or 'neutral']).toarray()
    X = np.hstack([combined, tfidf_row])

    proba = ens.predict_proba(X)[0]
    state = le.inverse_transform([np.argmax(proba)])[0]
    conf  = float(proba.max())

    ent   = scipy_entropy(proba + 1e-10) / np.log(len(le.classes_))
    wc    = len(str(journal_text or '').split())
    unc   = int(conf < 0.45 or ent > 0.7 or wc <= 4 or reflection_quality == 'conflicted')

    # Intensity
    intensity = int(np.clip(round(float(reg.predict(X)[0])), 1, 5))

    # Decision
    st_f   = float(stress_level or 3)
    en_f   = float(energy_level or 3)
    tod    = str(time_of_day or 'afternoon')
    what, when = decide(state, intensity, st_f, en_f, tod)
    msg    = message(state, intensity, what, when, unc)

    # All class probabilities
    class_probs = {le.classes_[i]: round(float(proba[i]),3)
                   for i in range(len(le.classes_))}

    return dict(
        predicted_state=state,
        predicted_intensity=intensity,
        confidence=round(conf,3),
        uncertain_flag=unc,
        what_to_do=what,
        when_to_do=when,
        supportive_message=msg,
        class_probabilities=class_probs,
    )

@app.route('/')
def index():
    return render_template('index.html', meta=meta_info)

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    result = predict_single(
        journal_text      = d.get('journal_text',''),
        ambience_type     = d.get('ambience_type','ocean'),
        duration_min      = d.get('duration_min', 15),
        sleep_hours       = d.get('sleep_hours', 7),
        energy_level      = d.get('energy_level', 3),
        stress_level      = d.get('stress_level', 3),
        time_of_day       = d.get('time_of_day','morning'),
        previous_day_mood = d.get('previous_day_mood','neutral'),
        face_emotion_hint = d.get('face_emotion_hint','neutral_face'),
        reflection_quality= d.get('reflection_quality','clear'),
    )
    return jsonify(result)

@app.route('/predictions')
def predictions():
    import pandas as pd
    df = pd.read_csv(os.path.join(BASE, 'outputs/predictions.csv'))
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    print("Starting ArvyaX UI at http://localhost:5000")
    app.run(debug=True, port=5000)
