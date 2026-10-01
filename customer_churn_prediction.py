"""
CODSOFT ML Internship - Task 3: Customer Churn Prediction
Dataset: Kaggle "Churn Modelling" (Churn_Modelling.csv, target: Exited)

Usage:  python customer_churn_prediction.py --data Churn_Modelling.csv
"""
import argparse, joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def main(a):
    df = pd.read_csv(a.data)
    target = "Exited" if "Exited" in df.columns else "Churn"
    df = df.drop(columns=[c for c in ["RowNumber", "CustomerId", "Surname", "customerID"] if c in df])
    y = df[target]
    if y.dtype == object:
        y = (y.str.lower() == "yes").astype(int)
    X = df.drop(columns=[target])

    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Churn rate: {y.mean():.2%}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    best, best_auc, best_pipe = None, 0, None
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc").mean()
        print(f"\n=== {name} ===  test ROC-AUC {auc:.4f} | 5-fold CV ROC-AUC {cv:.4f}")
        print(confusion_matrix(y_te, pipe.predict(X_te)))
        print(classification_report(y_te, pipe.predict(X_te), digits=3))
        if auc > best_auc:
            best, best_auc, best_pipe = name, auc, pipe

    print(f"Best model: {best} (ROC-AUC {best_auc:.4f})")
    joblib.dump(best_pipe, "churn_model.joblib")

    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        names = best_pipe.named_steps["pre"].get_feature_names_out()
        imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False)
        print("\nTop drivers of churn:\n", imp.head(8))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Churn_Modelling.csv")
    main(p.parse_args())
