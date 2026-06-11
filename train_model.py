"""
STEP 3 — Train the Model
=========================
Run after step 2:  python train_model.py

ONE XGBoost model with 5 outputs.
Trained on real Delhi weather + real citizen complaint labels.

Saves: models/storm_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

FEATURES = [
    # Area fixed
    'drain', 'elev', 'road_age', 'infra_age', 'wp', 'pop',
    # Temporal
    'month', 'hour',
    # Live weather — THIS IS THE CORE
    'rain_1h', 'rain_3h', 'rain_6h', 'rain_24h',
    'temp', 'wind', 'gust', 'humid',
    'pressure', 'press_trend', 'visibility',
    'weathercode', 'storm_now', 'thunder_now',
    # Open issues context — multiplies risk
    'open_water', 'open_sewage', 'open_pothole',
    'open_garbage', 'open_elec',
    # Real-time signal
    'complaint_vel',
]

LABELS = [
    'label_flood',
    'label_pothole_worsen',
    'label_sewage_overflow',
    'label_garbage_flood',
    'label_elec_hazard',
]

def load_data():
    path = 'data/labelled_training_data.csv'
    if not os.path.exists(path):
        print("ERROR: Run create_labels.py first")
        return None
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows")
    return df

def encode_area(df):
    le = LabelEncoder()
    df['area_enc'] = le.fit_transform(df['area'])
    return df, le

def train(df):
    df, le = encode_area(df)

    # Add area_enc to features
    feats = ['area_enc'] + FEATURES
    avail = [f for f in feats if f in df.columns]

    X = df[avail].fillna(0)
    Y = df[LABELS].fillna(0).astype(int)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print(f"Training: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Features: {len(avail)}")
    print(f"Labels: {LABELS}")

    # Base XGBoost classifier
    base_clf = XGBClassifier(
        n_estimators    = 300,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        eval_metric     = 'logloss',
        random_state    = 42,
        n_jobs          = -1,
        verbosity       = 0,
    )

    # MultiOutput wrapper — one model, 5 outputs
    model = MultiOutputClassifier(base_clf, n_jobs=-1)

    print("\nTraining MultiOutput XGBoost...")
    model.fit(X_train, Y_train)
    print("Training complete")

    # Evaluate
    print(f"\n{'='*50}")
    print("EVALUATION (on test set)")
    Y_pred = model.predict(X_test)
    Y_prob = np.array(model.predict_proba(X_test))

    for i, label in enumerate(LABELS):
        try:
            probs = Y_prob[i][:, 1]
            auc   = roc_auc_score(Y_test.iloc[:, i], probs)
            pos_rate = Y_test.iloc[:, i].mean() * 100
            print(f"  {label:<30} AUC={auc:.3f}  pos_rate={pos_rate:.1f}%")
        except Exception as e:
            print(f"  {label:<30} could not evaluate: {e}")

    # Feature importance (averaged across all outputs)
    print(f"\nTop 10 most important features:")
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    feat_imp = sorted(zip(avail, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:10]:
        bar = '█' * int(imp * 80)
        print(f"  {feat:<25} {bar} {imp:.3f}")

    return model, le, avail

def save_model(model, le, features):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/storm_model.pkl')
    joblib.dump(le,    'models/area_encoder.pkl')
    meta = {
        'features': features,
        'labels':   LABELS,
        'version':  '3.0',
        'type':     'MultiOutputXGBoost',
        'trained_on': 'Open-Meteo historical Delhi 2019-2024 + AreaPulse issues',
    }
    json.dump(meta, open('models/model_meta.json', 'w'), indent=2)
    print(f"\nSaved models/storm_model.pkl")
    print(f"Saved models/area_encoder.pkl")
    print(f"Saved models/model_meta.json")

def main():
    df = load_data()
    if df is None: return
    model, le, features = train(df)
    save_model(model, le, features)

    # Quick sanity check
    print(f"\n{'='*50}")
    print("SANITY CHECK — Chandni Chowk, thunderstorm, open sewage")
    import pandas as pd
    try:
        area_enc = le.transform(['Chandni Chowk'])[0]
    except:
        area_enc = 0

    sample = pd.DataFrame([{
        'area_enc':    area_enc,
        'drain':       0.0,   # very_poor
        'elev':        0.0,   # very_low
        'road_age':    1.0,   # old
        'infra_age':   1.0,
        'wp':          0.0,
        'pop':         1.0,   # very_high
        'month':       7,     # July = peak monsoon
        'hour':        14,    # 2pm
        'rain_1h':     18.0,  # heavy rain NOW
        'rain_3h':     45.0,
        'rain_6h':     65.0,
        'rain_24h':    80.0,
        'temp':        32.0,
        'wind':        45.0,
        'gust':        70.0,
        'humid':       92.0,
        'pressure':    998.0,
        'press_trend': -4.5,  # falling = storm arriving
        'visibility':  400.0,
        'weathercode': 95,    # thunderstorm
        'storm_now':   1,
        'thunder_now': 1,
        'open_water':  2,     # 2 open water issues
        'open_sewage': 3,     # 3 open sewage issues
        'open_pothole':4,     # 4 open potholes
        'open_garbage':2,
        'open_elec':   1,
        'complaint_vel': 4,   # 4 complaints in last 2h
    }])[features]

    probs = np.array(model.predict_proba(sample))
    print(f"\nPredicted risks:")
    for i, label in enumerate(LABELS):
        prob = probs[i][0][1] * 100
        bar  = '█' * int(prob / 5)
        print(f"  {label:<30} {prob:5.1f}% {bar}")

if __name__ == '__main__':
    main()