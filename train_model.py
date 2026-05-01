"""
=============================================================
AI2002 - Artificial Intelligence | Spring 2026
Project: AI-Driven Hospital Surgery Scheduling System

train_model.py  —  Run this ONCE before launching app.py
                   It generates training data, trains a
                   Logistic Regression model, and saves it
                   as  emergency_model.pkl

Members:
     Qadeer Raza   , 23i-3059
     Abdullah Tariq, 23i-3011
     Malik Usman   , 23i-3020
=============================================================

HOW TO RUN:
    python train_model.py

OUTPUT:
    emergency_model.pkl   ← loaded automatically by app.py
=============================================================
"""

import random
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix)

random.seed(42)
np.random.seed(42)


# =============================================================
# FEATURE DEFINITIONS
# =============================================================
#
#  Feature 1 : hour_of_day        (0–23)
#              Emergencies peak in evening / night hours
#
#  Feature 2 : is_weekend         (0 or 1)
#              Weekend days have higher emergency rates
#
#  Feature 3 : num_surgeries      (0–20)
#              More surgeries → more complex environment
#
#  Feature 4 : num_urgent         (0–10)
#              High urgent load often precedes emergencies
#
#  Feature 5 : room_occupancy_pct (0.0–1.0)
#              How full the OR schedule is right now
#
#  Feature 6 : avg_surgery_duration (1–8)
#              Longer average durations → tighter schedule
#
#  Label     : emergency_incoming  (0 or 1)
#              1 = another emergency likely in next 6 hours
#
# =============================================================


def generate_sample(hour, is_weekend, num_surgeries,
                    num_urgent, room_occupancy, avg_duration):
    """
    Rule-based probability formula used to assign ground-truth
    labels to each synthetic sample.  Designed so that:
      - Evening hours (17-23) raise probability significantly
      - Weekends raise it moderately
      - High urgent count raises it
      - High occupancy raises it
      - Low hour (early morning 0-6) also raises it slightly
    """
    p = 0.10   # base probability

    # time-of-day effect
    if 17 <= hour <= 23:
        p += 0.30
    elif 0 <= hour <= 6:
        p += 0.15
    elif 7 <= hour <= 9:
        p += 0.05

    # weekend effect
    if is_weekend:
        p += 0.15

    # urgent surgery load
    p += min(num_urgent * 0.05, 0.20)

    # room occupancy pressure
    p += room_occupancy * 0.20

    # long surgeries make slots scarce → higher emergency impact
    p += max(0, (avg_duration - 3)) * 0.03

    # clip to valid probability range
    p = min(max(p, 0.0), 1.0)

    # stochastic label to simulate real-world noise
    return 1 if random.random() < p else 0


def generate_dataset(n_samples=1200):
    """
    Generate n_samples rows of synthetic hospital context data.
    Returns a pandas DataFrame.
    """
    rows = []

    for _ in range(n_samples):
        hour             = random.randint(0, 23)
        is_weekend       = random.randint(0, 1)
        num_surgeries    = random.randint(0, 20)
        num_urgent       = random.randint(0, min(num_surgeries, 10))
        room_occupancy   = round(random.uniform(0.0, 1.0), 2)
        avg_duration     = round(random.uniform(1.0, 8.0), 1)

        label = generate_sample(
            hour, is_weekend, num_surgeries,
            num_urgent, room_occupancy, avg_duration
        )

        rows.append({
            "hour_of_day"          : hour,
            "is_weekend"           : is_weekend,
            "num_surgeries"        : num_surgeries,
            "num_urgent"           : num_urgent,
            "room_occupancy_pct"   : room_occupancy,
            "avg_surgery_duration" : avg_duration,
            "emergency_incoming"   : label,
        })

    return pd.DataFrame(rows)


# =============================================================
# TRAIN
# =============================================================

def train_and_save(model_path="emergency_model.pkl"):

    print("=" * 60)
    print("  Hospital Emergency Prediction — Model Training")
    print("=" * 60)

    # ── 1. Generate data ──────────────────────────────────────
    print("\n[1/4]  Generating synthetic training data (1200 samples)...")
    df = generate_dataset(n_samples=1200)

    pos = df["emergency_incoming"].sum()
    neg = len(df) - pos
    print(f"        Label distribution:  Emergency={pos}  |  No-Emergency={neg}")

    # ── 2. Split ──────────────────────────────────────────────
    print("\n[2/4]  Splitting into train / test sets (80 / 20)...")
    X = df.drop(columns=["emergency_incoming"])
    y = df["emergency_incoming"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"        Train samples : {len(X_train)}")
    print(f"        Test  samples : {len(X_test)}")

    # ── 3. Build pipeline and train ───────────────────────────
    print("\n[3/4]  Training Logistic Regression model...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter    = 1000,
            random_state= 42,
            solver      = "lbfgs",
            C           = 1.0,
        ))
    ])

    pipeline.fit(X_train, y_train)

    # ── 4. Evaluate ───────────────────────────────────────────
    print("\n[4/4]  Evaluating on held-out test set...")

    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n        Accuracy : {acc * 100:.1f}%")
    print("\n        Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["No Emergency", "Emergency"]))

    cm = confusion_matrix(y_test, y_pred)
    print("        Confusion Matrix:")
    print(f"          True Negative  (TN): {cm[0][0]}")
    print(f"          False Positive (FP): {cm[0][1]}")
    print(f"          False Negative (FN): {cm[1][0]}")
    print(f"          True Positive  (TP): {cm[1][1]}")

    # ── Save model ────────────────────────────────────────────
    joblib.dump(pipeline, model_path)
    print(f"\n  ✅  Model saved to  →  {model_path}")
    print("=" * 60)
    print("  You can now run:   streamlit run app.py")
    print("=" * 60)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    train_and_save(model_path="emergency_model.pkl")