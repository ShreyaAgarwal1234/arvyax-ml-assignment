# 🌿 ArvyaX — Reflective Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-UI-green?style=for-the-badge&logo=flask)
![Accuracy](https://img.shields.io/badge/CV%20Accuracy-90%25-brightgreen?style=for-the-badge)

**Machine Learning Internship Assignment — Team ArvyaX · RevoltronX**

*From Understanding Humans → To Guiding Them*

[🚀 Quick Start](#-quick-start) • [📊 Results](#-results) • [🧠 Approach](#-approach) • [🗂 Structure](#-project-structure)

</div>

---

## 🎯 What is ArvyaX?

ArvyaX is an **end-to-end ML system** that takes a user's short post-session journal reflection and contextual signals, then:

| Step | What it does |
|------|-------------|
| 🧠 **Understands** | Predicts emotional state (calm / focused / restless / mixed / neutral / overwhelmed) |
| 📊 **Measures** | Predicts intensity score (1–5) |
| 💡 **Decides** | Recommends what to do + when to do it |
| ⚠️ **Self-aware** | Knows when it is uncertain |
| 💬 **Guides** | Generates a human-like supportive message |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| 🎯 CV Accuracy (State Classification) | **90%** |
| 📉 Intensity MAE (Regression) | **0.81** |
| 🏋️ Training Samples | **891** |
| 🏷️ Emotional State Classes | **6** |
| 🔀 Ensemble | XGBoost + RandomForest + LogisticRegression |

### Ablation Study

| Model | Accuracy |
|-------|----------|
| Text Only | 88.0% |
| Metadata Only | 82.0% |
| **Text + Metadata (Full)** | **92.0%** |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ShreyaAgarwal1234/arvyax-ml-assignment.git
cd arvyax-ml-assignment
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost flask
```

### 3. Run the ML Pipeline
```bash
python pipeline.py
```
```
✅ Pipeline complete!
Predictions saved → outputs/predictions.csv
```

### 4. Launch the Web UI
```bash
cd ui
python app.py
```
Open browser: **http://localhost:5000**

---

## 🗂 Project Structure

```
arvyax/
│
├── 📄 pipeline.py              # Main ML pipeline (train + predict)
│
├── 📁 data/
│   ├── train_full.csv          # Training data (891 samples, labeled)
│   └── test.csv                # Test data (120 samples)
│
├── 📁 models/
│   └── models.pkl              # Saved trained models
│
├── 📁 outputs/
│   └── predictions.csv         # Final predictions for 120 test samples
│
├── 📁 ui/
│   ├── app.py                  # Flask web server
│   └── templates/
│       └── index.html          # Dark-theme interactive UI
│
├── 📄 README.md
├── 📄 ERROR_ANALYSIS.md        # 11 failure cases with deep analysis
└── 📄 EDGE_PLAN.md             # On-device / mobile deployment plan
```

---

## 🧠 Approach

### Feature Engineering

**Text Features (Hand-crafted)**
- Positive / Negative / Mixed / Uncertainty word counts
- Sentiment polarity score
- Short text flag (≤5 words)
- Contrast phrase detection (`but`, `yet`, `however`)
- Body cue, task cue flags

**TF-IDF Features**
- Bigram TF-IDF on journal text (100 features)
- Captures domain phrases like *"mind kept jumping"*, *"feel lighter"*

**Metadata Features**
- Sleep hours, energy level, stress level, duration
- Time of day, ambience type, previous mood
- Face emotion hint, reflection quality
- Derived: energy-stress ratio, low_sleep, high_stress flags

---

### Part 1 — Emotional State Classification
- **Model**: Soft-voting ensemble → XGBoost (50%) + RandomForest (35%) + LogisticRegression (15%)
- **CV Accuracy**: 90%

### Part 2 — Intensity Prediction
- **Approach**: Regression (XGBoost) → round to nearest int
- **Why Regression?**: Intensity is ordinal — regression respects ordering naturally
- **MAE**: 0.81

### Part 3 — Decision Engine

Rule-based using predicted state + stress + energy + time of day:

```
overwhelmed + morning  →  box_breathing  →  NOW
focused     + morning  →  deep_work      →  NOW
restless    + night    →  sound_therapy  →  TONIGHT
calm        + night    →  journaling     →  TOMORROW_MORNING
mixed       + evening  →  journaling     →  TONIGHT
```

### Part 4 — Uncertainty Modeling

```
confidence = max_proba × 0.6 + margin × 0.3
           - uncertainty_word_penalty
           - short_text_penalty
           + quality_adjustment

uncertain_flag = 1 if confidence < 0.55
              OR text ≤ 5 words
              OR reflection = "conflicted"
              OR uncertainty_words ≥ 2
```

---

## 🔍 Error Analysis

See **[ERROR_ANALYSIS.md](ERROR_ANALYSIS.md)** for 11 detailed failure cases.

Key failure patterns:

| Pattern | Example | Fix |
|---------|---------|-----|
| Short text | `"ok"`, `"fine"` | Metadata-dominant branch |
| Contradictory signals | calm text + stress=5 | Hard metadata overrides |
| Label noise | `"kinda calm"` → labeled restless | Cleanlab noise detection |
| Class boundary confusion | overwhelmed vs restless | Intensity-aided split |
| Hedged language | `"calmer on the surface"` | Idiom detector |

---

## 📱 Edge / On-Device Deployment

See **[EDGE_PLAN.md](EDGE_PLAN.md)** for full mobile deployment plan.

| Property | Value |
|----------|-------|
| Model size (edge) | ~2 MB |
| Inference latency | <100ms |
| Network required | ❌ Zero (fully offline) |
| Privacy | ✅ On-device only |
| Platform | Android / iOS via ONNX Runtime |

---

## 💪 Robustness

| Scenario | How Handled |
|----------|-------------|
| Very short text (`"ok"`, `"fine"`) | `is_short` flag → uncertainty raised → metadata dominant |
| Missing values | `SimpleImputer(mean)` for numeric, default encoding for categorical |
| Contradictory inputs | `has_contrast` feature + stress override rules |
| Noisy labels | Confidence-based uncertain flag + ensemble smoothing |

---

## 📦 Output Format (`predictions.csv`)

| Column | Description |
|--------|-------------|
| `id` | Test sample ID |
| `predicted_state` | Emotional state class |
| `predicted_intensity` | Intensity 1–5 |
| `confidence` | Model confidence 0–1 |
| `uncertain_flag` | 1 = uncertain prediction |
| `what_to_do` | Recommended action |
| `when_to_do` | Timing recommendation |
| `support_message` | Human-like guidance message |

---

## 🛠 Tech Stack

| Library | Use |
|---------|-----|
| `XGBoost` | Primary classifier + intensity regressor |
| `scikit-learn` | RandomForest, LogisticRegression, TF-IDF, imputation |
| `pandas` / `numpy` | Data processing |
| `Flask` | Web UI backend |
| `HTML / CSS / JS` | Frontend dark-theme UI |

> ⚠️ No OpenAI / Gemini / Claude APIs used. Fully local system.

---

## 👤 Author

**Shreya Agarwal**  
Machine Learning Internship — Team ArvyaX · RevoltronX

---

<div align="center">

*Dream → Innovate → Create*

**🌿 ArvyaX — AI that doesn't just understand humans. It helps them move toward a better state.**

</div>
