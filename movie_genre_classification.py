"""
CODSOFT ML Internship - Task 1: Movie Genre Classification
Dataset: Kaggle "Genre Classification Dataset (IMDb)"
  files: train_data.txt, test_data.txt, test_data_solution.txt
  line format: ID ::: TITLE ::: GENRE ::: DESCRIPTION   (test_data has no GENRE)

Usage:  python movie_genre_classification.py --data_dir ./data
"""
import argparse, os, re, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def load(path, has_genre=True):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ::: ")
            if has_genre and len(parts) == 4:
                rows.append(parts)
            elif not has_genre and len(parts) == 3:
                rows.append(parts)
    cols = ["id", "title", "genre", "description"] if has_genre else ["id", "title", "description"]
    return pd.DataFrame(rows, columns=cols)


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main(a):
    train = load(os.path.join(a.data_dir, "train_data.txt"))
    train["text"] = (train["title"] + " " + train["description"]).map(clean)

    test_path = os.path.join(a.data_dir, "test_data_solution.txt")
    if os.path.exists(test_path):
        test = load(test_path)
        test["text"] = (test["title"] + " " + test["description"]).map(clean)
        X_tr, y_tr, X_te, y_te = train["text"], train["genre"], test["text"], test["genre"]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            train["text"], train["genre"], test_size=0.2, stratify=train["genre"], random_state=42)

    tfidf = lambda: TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                    min_df=2, max_features=200_000, sublinear_tf=True)
    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=10),
        "Linear SVM": LinearSVC(C=0.5),
    }

    best_name, best_acc, best_pipe = None, 0, None
    for name, clf in models.items():
        pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, pipe.predict(X_te))
        print(f"{name:22s} accuracy = {acc:.4f}")
        if acc > best_acc:
            best_name, best_acc, best_pipe = name, acc, pipe

    print(f"\nBest model: {best_name} ({best_acc:.4f})\n")
    print(classification_report(y_te, best_pipe.predict(X_te), zero_division=0))
    joblib.dump(best_pipe, "genre_model.joblib")
    print("Saved model -> genre_model.joblib")

    demo = "A group of astronauts travel through a wormhole to save humanity from extinction."
    print("\nDemo prediction:", best_pipe.predict([clean(demo)])[0])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data")
    main(p.parse_args())
