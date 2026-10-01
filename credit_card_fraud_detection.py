"""
CODSOFT ML Internship - Task 2: Credit Card Fraud Detection
Works with either Kaggle dataset:
  * creditcard.csv               (target column: Class)
  * fraudTrain.csv/fraudTest.csv (target column: is_fraud)

Usage:  python credit_card_fraud_detection.py --train creditcard.csv
        python credit_card_fraud_detection.py --train fraudTrain.csv --test fraudTest.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)

TARGETS = ["Class", "is_fraud", "isFraud"]


def prepare(df):
    target = next(t for t in TARGETS if t in df.columns)
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # engineered features for the fraudTrain-style dataset
    if "trans_date_trans_time" in X:
        t = pd.to_datetime(X["trans_date_trans_time"])
        X["hour"], X["dayofweek"] = t.dt.hour, t.dt.dayofweek
        if "dob" in X:
            X["age"] = (t - pd.to_datetime(X["dob"])).dt.days // 365
        if {"lat", "long", "merch_lat", "merch_long"} <= set(X.columns):
            X["distance"] = np.hypot(X["lat"] - X["merch_lat"], X["long"] - X["merch_long"])

    # drop identifiers / high-cardinality text; one-hot the rest
    X = X.drop(columns=[c for c in X.columns if X[c].dtype == "object" and X[c].nunique() > 50]
               + [c for c in ["Unnamed: 0", "trans_date_trans_time", "dob", "cc_num", "unix_time"]
                  if c in X.columns])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def evaluate(name, model, X_te, y_te):
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(f"\n=== {name} ===")
    print(confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, digits=4, zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_te, proba):.4f}   PR-AUC: {average_precision_score(y_te, proba):.4f}")


def main(a):
    df = pd.read_csv(a.train)
    if a.sample and len(df) > a.sample:
        df = df.sample(a.sample, random_state=42)
    X, y = prepare(df)

    if a.test:
        dt = pd.read_csv(a.test)
        Xt, yt = prepare(dt)
        Xt = Xt.reindex(columns=X.columns, fill_value=0)
        X_tr, y_tr, X_te, y_te = X, y, Xt, yt
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"Fraud rate (train): {y_tr.mean():.4%}  |  rows: {len(X_tr):,}")
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

    # class_weight='balanced' handles the extreme class imbalance
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs_tr, y_tr)
    dt = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42).fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=150, max_depth=14, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=42).fit(X_tr, y_tr)

    evaluate("Logistic Regression", lr, Xs_te, y_te)
    evaluate("Decision Tree", dt, X_te, y_te)
    evaluate("Random Forest", rf, X_te, y_te)

    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 Random Forest features:\n", imp.head(10))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test")
    p.add_argument("--sample", type=int, default=0, help="subsample rows for speed")
    main(p.parse_args())
