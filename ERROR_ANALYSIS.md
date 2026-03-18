# ERROR_ANALYSIS.md
# ArvyaX — Failure Case Analysis

## Overview

This document analyzes 10+ failure cases observed during cross-validation and manual inspection of model predictions. These cases reveal key challenges in predicting emotional state from noisy, short, and ambiguous reflections.

---

## Failure Case 1 — Short / Vague Input

**Input:** `"ok session"`  
**True Label:** `neutral`  
**Predicted:** `calm`  
**Why It Failed:** The phrase "ok session" is semantically empty. No word-level signal distinguishes `neutral` from `calm` or `mixed` at this granularity. The model defaults to `calm` because it appears slightly more in the training data for short/vague inputs.  
**Fix:** Train a dedicated "low-confidence shorttext" branch. For inputs ≤ 5 words, defer entirely to metadata features (stress, energy, sleep).

---

## Failure Case 2 — Contradictory Signals

**Input:** `"woke up feeling more organized mentally. i was more tired than i thought."`  
**True Label:** `mixed` (expected)  
**Predicted:** `focused`  
**Why It Failed:** "organized mentally" and "more tired" pull in opposite directions. The TF-IDF representation scores "organized" heavily positive, overriding the exhaustion signal. Mixed states require detecting linguistic contrasts, not just individual words.  
**Fix:** Add a dedicated `contrast_score` feature counting conjunctions (but, yet, however, though) weighted by the polarity gap they bridge.

---

## Failure Case 3 — Ambience-Mood Mismatch

**Input:** `"kinda calm ... cafe 15 8.5 2 5 evening calm happy_face vague"`  
**True state:** Labeled `calm` but stress_level=5  
**Issue:** The reflection says "kinda calm" but stress is at maximum (5). This is a noisy label — the user likely wrote a surface-level calm while being internally stressed. The model was correct (predicts `overwhelmed`), but label is `calm`. This is a labeling noise case.  
**Insight:** High stress (≥4) + vague reflection = likely mislabeled. Need label-cleaning step.

---

## Failure Case 4 — Domain Shift (Short Slang)

**Input:** `"mind racing"`  
**True Label:** `restless` or `overwhelmed`  
**Predicted:** `neutral`  
**Why It Failed:** "Mind racing" is two words. TF-IDF assigns near-zero weight. The phrase is colloquial and high-signal but absent from the model's learned vocabulary at meaningful frequency.  
**Fix:** Build a slang/shorthand dictionary: `"mind racing" → high_negative`, `"hard to focus" → restless`, `"bit restless" → restless`. Apply as a rule-based override layer before model inference.

---

## Failure Case 5 — Temporal Shift Language

**Input:** `"started off like everything piled up, then my mind wandered again. then it faded again."`  
**True Label:** `mixed`  
**Predicted:** `overwhelmed`  
**Why It Failed:** The input describes a temporal evolution: initial overwhelm → dissociation → fading. The model captures the first clause heavily (overwhelm signal), ignoring the temporal resolution. Bag-of-words / TF-IDF cannot model sequence.  
**Fix:** Use sentence-level positional features (first-half vs second-half sentiment) or sequence models (LSTM/Transformer) to capture how emotional state changes across the reflection.

---

## Failure Case 6 — Low Sleep with High Energy (Conflicting Metadata)

**Input:** `"actually helped"` + `sleep=7, energy=4, stress=3`  
**True Label:** `focused` (implied by metadata)  
**Predicted:** `neutral`  
**Why It Failed:** The text offers almost no signal. The model falls back to the label distribution, predicting `neutral`. But the metadata clearly indicates good conditions for focus.  
**Fix:** For near-empty text inputs, weight metadata features 3x more than text features. Create a "text_is_empty" flag to trigger metadata-dominant inference.

---

## Failure Case 7 — Irony / Hedged Language

**Input:** `"not gonna lie i felt calmer on the surface but still busy underneath"`  
**True Label:** `mixed`  
**Predicted:** `calm`  
**Why It Failed:** The phrase "calmer on the surface" creates a false positive for calm. The disclaimer "still busy underneath" is subordinate and carries less TF-IDF weight. The model lacks pragmatic understanding of surface vs. internal state.  
**Fix:** Build an idiom detector: `"on the surface" → surface_only_marker`, `"still * underneath" → hidden_tension_marker`. These should contribute negatively to the calm class.

---

## Failure Case 8 — Missing Face Emotion with Conflicting Signals

**Input:** `"for a while i was steady but nothing major. i had to restart once."`  
**face_emotion_hint:** missing (`none`)  
**True Label:** `neutral`  
**Predicted:** `restless`  
**Why It Failed:** "Had to restart once" is interpreted as restlessness. Without the face emotion cue, the model over-weights this behavioral signal. With face context (`neutral_face`), the `neutral` prediction would likely be correct.  
**Fix:** Imputing missing face features with modal class works poorly. Instead, use a trained face-emotion probability distribution conditioned on `ambience_type` + `time_of_day`.

---

## Failure Case 9 — Overwhelmed ≠ Restless (Semantic Boundary)

**Input:** `"couldn't stop my mind from racing. the session helped a little but thoughts kept coming."`  
**True Label:** `overwhelmed` (intensity=4)  
**Predicted:** `restless`  
**Why It Failed:** The semantic boundary between `overwhelmed` (loss of control, flooding) and `restless` (physical/mental agitation seeking outlet) is very thin. "Racing thoughts" can indicate either. The model conflates them.  
**Fix:** Use intensity as an auxiliary signal. If intensity ≥ 4 AND neg_score ≥ 3, bias toward `overwhelmed`. Consider collapsing these two into a super-class for the classifier, then use intensity to split them.

---

## Failure Case 10 — Positive Ambience Override

**Input:** `"The cafe session helped me breathe slower"`  
**Metadata:** `stress=5, energy=2, night`  
**True Label:** `overwhelmed`  
**Predicted:** `calm`  
**Why It Failed:** The text contains a resolution phrase ("helped me breathe slower"), which is a strong calm signal. But the metadata says: max stress, low energy, late night — all pointing to overwhelm. The model over-trusts text at the expense of metadata.  
**Fix:** In extreme metadata conditions (stress=5 OR energy=1 OR sleep<4), create hard overrides that prevent `calm` / `focused` predictions unless confidence is very high (>0.8).

---

## Failure Case 11 — Label Noise from Contradictory Dataset

**Input:** `"hard to focus"` + `previous_day_mood=overwhelmed, face_emotion=happy_face`  
**True Label:** `focused`  
**Predicted:** `restless`  
**Why It Failed:** The label `focused` despite "hard to focus" text and `overwhelmed` prior mood is almost certainly a noisy label. This is a real-world data quality issue.  
**Fix:** Apply automated noise detection: use a KNN-based label cleaning algorithm (Cleanlab) to flag samples where the predicted probability of the assigned label is in the bottom 10%.

---

## Summary Table

| # | Issue Type | Root Cause | Impact |
|---|---|---|---|
| 1 | Short text | No semantic content | High — common in real use |
| 2 | Contradictory text | Model prefers dominant signal | Medium |
| 3 | Noisy labels | Surface calm masks internal stress | High — affects training |
| 4 | Slang/shorthand | OOV phrases | High — common in real use |
| 5 | Temporal language | BoW can't model time | Medium |
| 6 | Empty text + rich metadata | Text dominance over metadata | Medium |
| 7 | Hedged language | Pragmatic irony ignored | Medium |
| 8 | Missing features | No face cue → wrong class | Low-Medium |
| 9 | Class boundary confusion | Overwhelmed vs Restless overlap | High |
| 10 | Ambience override | Text trumps severe metadata | Medium |
| 11 | Label noise | Incorrect annotation | High — corrupts learning |

---

## Recommended Improvements

1. **Short-text routing**: If `word_count ≤ 5`, skip text model entirely and use metadata-only branch
2. **Hard override rules**: For extreme metadata conditions (stress=5, sleep<4), override model output
3. **Label cleaning**: Use Cleanlab or confidence-based label filtering before training
4. **Contrast detector**: Feature for "surface emotion vs. deep emotion" patterns
5. **Slang dictionary**: Pre-defined phrase → signal mapping as a preprocessing step
6. **Intensity-aided classification**: Use predicted intensity to disambiguate boundary classes
