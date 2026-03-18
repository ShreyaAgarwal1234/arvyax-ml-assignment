# EDGE_PLAN.md
# ArvyaX — Edge / On-Device Deployment Plan

## Overview

The ArvyaX system must run on mobile devices (Android/iOS) without internet connectivity. This document outlines the architecture, optimizations, and tradeoffs for offline-first deployment.

---

## 1. Target Deployment Environment

| Property | Value |
|---|---|
| Device | Android (≥5.0) / iOS (≥13) |
| RAM available | 100–500 MB |
| Storage budget | < 50 MB total |
| Network dependency | Zero (fully offline) |
| Latency target | < 500ms per inference |
| Battery impact | Minimal |

---

## 2. Model Architecture for Edge

### Current (Server-Side) Model
- XGBoost ensemble (3 models) + TF-IDF (80 features)
- Full Python scikit-learn + xgboost stack
- ~15 MB on disk

### Edge-Optimized Model

**Step 1 — Feature Compression**
- Keep only top 30 TF-IDF features (from importance analysis)
- Keep all 19 handcrafted text features
- Keep all 16 metadata features
- Total: ~65 features (down from 114)

**Step 2 — Model Simplification**
```
Server Model → Edge Model
XGBoost (200 trees) → XGBoost (50 trees, max_depth=3)
RandomForest (200) → Remove (not needed)
LogisticRegression → Keep as fallback (tiny size)
```

**Step 3 — Export to Portable Format**
```python
# Export XGBoost to JSON (platform-independent)
model.save_model("model_edge.json")
# Size: ~1.2 MB (vs 8 MB full model)

# OR export to ONNX for cross-platform runtime
import onnxmltools
onnx_model = onnxmltools.convert_xgboost(model)
# Size: ~800 KB
```

**Step 4 — Runtime on Device**
- **Android**: Use [XGBoost4J](https://xgboost.readthedocs.io/en/stable/jvm/index.html) or ONNX Runtime for Android
- **iOS**: Use ONNX Runtime iOS or CoreML (convert via `onnx-coreml`)
- **React Native**: Use ONNX Runtime React Native package
- **Flutter**: Embed model as asset, run via `tflite_flutter` or custom FFI bridge

---

## 3. Feature Extraction on Device

All features are rule-based — no neural network needed for this step:

```
Input: journal_text (string) + metadata fields (7 numbers)
↓
Text Features (19 hand-crafted) — pure string ops, ~1ms
↓
TF-IDF (30 terms only) — lookup in pre-built vocab dict, ~2ms
↓
Metadata Features (16 numbers) — arithmetic, ~0.1ms
↓
Total feature extraction: < 5ms on any modern phone
```

The TF-IDF vocabulary (30 terms → index mapping) is stored as a tiny JSON file (~3 KB).

---

## 4. Decision Engine on Device

The decision engine is pure rule-based logic — no ML needed:

```python
# Compiles to ~20KB of native code
def decide(state, stress, energy, time_of_day):
    # Rule lookup table — O(1)
    return what_to_do, when_to_do
```

Decision logic can be shipped as a simple JSON config file (~5 KB) and parsed at runtime. This means it can be updated without a new app release.

---

## 5. Full On-Device Stack

```
┌─────────────────────────────────┐
│        ArvyaX Mobile App        │
├─────────────────────────────────┤
│  User Input Layer               │  ← React Native / Flutter UI
├─────────────────────────────────┤
│  Feature Extraction (JS/Dart)   │  ← ~5ms, pure logic
├─────────────────────────────────┤
│  ONNX Runtime / XGBoost4J       │  ← ~50ms inference
├─────────────────────────────────┤
│  Decision Engine (JSON rules)   │  ← ~1ms
├─────────────────────────────────┤
│  Local Storage (SQLite)         │  ← history, personalization
└─────────────────────────────────┘
```

**Total latency**: < 100ms end-to-end on a mid-range 2020 phone.

---

## 6. Storage Budget

| Component | Size |
|---|---|
| ONNX model (state classifier) | 800 KB |
| ONNX model (intensity regressor) | 600 KB |
| TF-IDF vocabulary (30 terms) | 3 KB |
| Decision rules JSON | 5 KB |
| Support messages JSON | 8 KB |
| SQLite user history DB | Variable (~1 KB/session) |
| **Total static assets** | **< 2 MB** |

Well within a 50 MB budget — leaves room for UI assets, audio clips, etc.

---

## 7. Personalization (On-Device Learning)

The system can improve over time without a server:

```
After each session:
1. User provides feedback (thumbs up/down or corrects their state)
2. Store (features, true_label) in SQLite
3. After 20+ corrections: fine-tune LogisticRegression locally
   (LogReg is fast enough to re-train on 50 samples in <100ms)
4. Use fine-tuned LR as a post-processing correction layer
```

This is called **on-device federated fine-tuning** — no data ever leaves the device.

---

## 8. Tradeoffs

| Factor | Server Model | Edge Model |
|---|---|---|
| Accuracy | 92.2% CV | ~87–89% (estimated) |
| Latency | ~200ms (network + inference) | <100ms (local only) |
| Privacy | Data leaves device | Zero data transmission |
| Model updates | Instant | Requires app update or OTA config |
| Personalization | Central (with consent) | On-device (fully private) |
| Offline support | ❌ | ✅ |
| Model size | ~15 MB | ~2 MB |

The 3–5% accuracy tradeoff is acceptable for a wellness use case. Privacy and reliability (offline-first) outweigh marginal accuracy gains.

---

## 9. Battery & Performance Considerations

- **No continuous inference**: Model runs only on explicit user action (post-session)
- **No background processing**: No location, sensor polling, or background ML
- **Model stays in memory**: After first inference, keep ONNX session warm for the session
- **CPU only**: No GPU needed; XGBoost on CPU is fast and efficient
- Estimated battery cost: **< 0.1% per inference** on a modern phone

---

## 10. Update Strategy

| Component | Update Method |
|---|---|
| Decision rules | OTA JSON fetch (no app update) |
| Support messages | OTA JSON fetch |
| ML model | App update (rare, only for major retraining) |
| TF-IDF vocab | Bundled with model update |

---

## 11. Future: Tiny Language Model Option

If a small local language model becomes feasible (e.g., Gemma 2B quantized to 4-bit, ~1.5 GB):

- Use it for **text understanding only** (replace TF-IDF)
- Run as a background task, pre-warm on app open
- Latency: ~300–500ms on flagship phones
- Would improve accuracy from ~87% to likely ~94–96%
- Not recommended for now — too large and slow for mid-range devices

Current approach (XGBoost + rule features) is the right choice for 2025 edge deployment.
