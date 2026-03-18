# ArvyaX — Reflective Intelligence System
### Machine Learning Internship Assignment | Team ArvyaX · RevoltronX
_Theme: From Understanding Humans → To Guiding Them_

---

## Project Summary
Takes post-session reflections + contextual signals and produces:
- **Emotional state** (calm / focused / restless / mixed / neutral / overwhelmed)
- **Intensity** score (1–5, regression)
- **Decision**: What to do + When to do it
- **Uncertainty**: Confidence score + uncertain flag
- **Supportive message** (bonus)

---

## Project Structure
```
arvyax/
├── data/
│   ├── train_weighted.csv      # 987 weighted training samples
│   ├── train_full.csv          # 891 rows from full PDF parse
│   ├── train.csv               # Original 114 clean rows
│   └── test.csv                # 120 test samples
├── models/
│   └── models.pkl              # Saved ensemble (generated on run)
├── outputs/
│   └── predictions.csv         # Final predictions (120 rows)
├── ui/
│   ├── app.py                  # Flask web server
│   └── templates/index.html    # Dark-theme UI
├── pipeline.py                 # Main ML pipeline
├── README.md
├── ERROR_ANALYSIS.md
└── EDGE_PLAN.md
```

---

## Setup & Run

### Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost flask
```

### Run pipeline (trains + predicts)
```bash
cd arvyax
python pipeline.py
# → outputs/predictions.csv
```

### Run web UI
```bash
cd arvyax/ui
python app.py
# Open: http://localhost:5000
```

---

## Approach & Model Design

### Dataset
- **987 training samples** (114 high-quality + 873 additional)
- **120 test samples** (unlabeled)
- 6 emotional state classes: calm, focused, restless, mixed, neutral, overwhelmed
- Text ranges from detailed reflections to ultra-short phrases ("ok", "mind racing")

### Feature Engineering (151 total features)

**Text Features (15 hand-crafted)**
- Positive/Negative/Mixed/Uncertainty word counts
- Sentiment polarity score
- Short-text flag (≤5 words), vague phrase flag
- Contrast phrase, body cue, task cue detection
- Template pattern flag (for rows 115+)
- Word count, char count, avg word length

**TF-IDF Features (120)**
- Bigram TF-IDF on journal text
- Top terms: "ideas", "track", "things felt", "concentrate", "mind quiet"

**Metadata Features (16)**
- Sleep hours + quality index
- Energy, stress, duration
- Time of day, ambience, previous mood, face emotion, reflection quality
- Derived: energy-stress ratio, low_sleep flag, high_stress flag

### Part 1 — Emotional State Classification
- **Model**: Soft-voting ensemble (XGBoost 50% + RandomForest 35% + LogReg 15%)
- **CV Accuracy**: 60.1% ± 4.1% (6-class, random baseline = 16.7%)
- **Sample weighting**: Long/rich texts weighted 3×, medium 2×, short 1×
- _Note: Lower accuracy than baseline (92% on clean 114 rows) reflects genuine label noise in the full dataset — rows 115–1080 contain template-generated text with inconsistent labeling_

### Part 2 — Intensity Prediction
- **Model**: XGBoost Regressor → round to nearest int
- **Why regression**: Intensity is ordinal (1→5 ordering is meaningful). Regression preserves this; classification would ignore that predicting 3 when true=4 is better than predicting 1.
- **CV MAE**: 1.30 (baseline: predicting mean = ~1.5 MAE)

### Part 3 — Decision Engine
Rule-based system: predicted_state + stress + energy + time_of_day → what + when

```
overwhelmed + morning   → box_breathing     → now
overwhelmed + night     → rest              → tonight
focused     + morning   → deep_work         → now
restless    + stress≥4  → grounding         → now
mixed       + any       → journaling
stress = 5              → override: now (any state)
```

### Part 4 — Uncertainty Modeling
```
confidence = max_proba×0.6 + margin×0.3 − uncertainty_words×0.03
             − short_text×0.05 + quality_adjustment
uncertain_flag = 1 if confidence < 0.55 OR text ≤ 5 words
                   OR reflection = "conflicted" OR ≥2 uncertainty signals
```

### Part 5 — Feature Importance
Text (TF-IDF bigrams) dominates: 60% of top-20 features are text-based.
Key metadata: `sentiment`, `energy_stress_ratio`, `stress_level`

### Part 6 — Ablation Study
| Model | Accuracy |
|---|---|
| Text-Only | 60.9% |
| Metadata-Only | 18.8% |
| Text + Metadata (Full) | 59.5% |

### Part 9 — Robustness
| Input Type | Handling |
|---|---|
| Very short ("ok", "fine") | `is_short=1` → confidence −0.05, uncertain_flag=1, defer to metadata |
| Missing values | `SimpleImputer(mean)` for numeric; default encodings for categorical |
| Contradictory | `has_contrast` feature, `mixed_score` captures conflicting signals; uncertainty raised |

---

## Output Format (`predictions.csv`)
| Column | Description |
|---|---|
| `id` | Test ID (10001–10120) |
| `predicted_state` | Emotional state |
| `predicted_intensity` | 1–5 |
| `confidence` | 0–1 |
| `uncertain_flag` | 0 or 1 |
| `what_to_do` | Recommended action |
| `when_to_do` | Timing |
| `support_message` | Human-like guidance |

---

## Constraints Respected
- ✅ No OpenAI / Gemini / Claude APIs
- ✅ No hosted LLMs
- ✅ Fully local: scikit-learn + XGBoost + Flask
- ✅ Reproducible: `random_state=42` everywhere
