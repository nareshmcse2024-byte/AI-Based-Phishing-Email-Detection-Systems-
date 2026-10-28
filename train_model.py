"""Train baseline classifiers with duplicate removal before a stratified holdout split."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .preprocessing import compose_email_text
from config import ARTIFACT_PATH, METRICS_PATH, SYNTHETIC_DATA_PATH, LABELS

REQUIRED = {"subject", "body", "label"}

def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    missing = REQUIRED - set(df.columns)
    if missing: raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    df["label"] = df["label"].astype(str).str.strip()
    invalid = sorted(set(df["label"]) - set(LABELS))
    if invalid: raise ValueError(f"Invalid labels: {invalid}. Allowed labels: {list(LABELS)}")
    if df["label"].eq("").any(): raise ValueError("Dataset contains empty labels")
    for col in ("subject", "body", "sender", "urls"):
        if col not in df: df[col] = ""
    df["model_text"] = df.apply(lambda r: compose_email_text(r.subject, r.body, r.sender, str(r.urls).split()), axis=1)
    # Remove repeated subject/body records before split; sender/URL metadata differences
    # must not let the same message wording leak between partitions.
    df["_dedup"] = (df["subject"].astype(str) + "\x1f" + df["body"].astype(str)).str.lower().str.replace(r"\s+", " ", regex=True)
    conflicting = df.groupby("_dedup")["label"].nunique()
    if (conflicting > 1).any():
        raise ValueError("Duplicate email text has conflicting labels; resolve these records before training")
    df = df.drop_duplicates("_dedup").drop(columns="_dedup").reset_index(drop=True)
    if df["label"].nunique() < 3: raise ValueError("Training requires all three labels")
    counts = df["label"].value_counts()
    if (counts < 5).any(): raise ValueError("Each label needs at least five unique examples for train/validation/test splits")
    return df

def evaluate_models(df: pd.DataFrame, test_size: float = 0.20, validation_size: float = 0.20, random_state: int = 42) -> tuple[dict, dict]:
    X_dev, X_test, y_dev, y_test = train_test_split(df.model_text, df.label, test_size=test_size, random_state=random_state, stratify=df.label)
    validation_fraction_of_dev = validation_size / (1.0 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_dev, y_dev, test_size=validation_fraction_of_dev, random_state=random_state, stratify=y_dev)
    models = {
        "Logistic Regression": Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000, sublinear_tf=True)), ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state))]),
        "Random Forest": Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=12000, sublinear_tf=True)), ("clf", RandomForestClassifier(n_estimators=160, min_samples_leaf=1, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1))]),
    }
    results = {}
    best_name, best_validation_f1 = None, -1.0
    fitted_candidates = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        validation_pred = model.predict(X_validation)
        validation_f1 = float(f1_score(y_validation, validation_pred, average="macro", zero_division=0))
        fitted_candidates[name] = model
        # Model choice uses validation only; test labels are not consulted for selection.
        if validation_f1 > best_validation_f1:
            best_name, best_validation_f1 = name, validation_f1
    for name, model in fitted_candidates.items():
        pred = model.predict(X_test)
        f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
        results[name] = {"validation_macro_f1": float(f1_score(y_validation, model.predict(X_validation), average="macro", zero_division=0)), "accuracy": float(accuracy_score(y_test, pred)), "macro_precision": float(precision_score(y_test, pred, average="macro", zero_division=0)), "macro_recall": float(recall_score(y_test, pred, average="macro", zero_division=0)), "macro_f1": f1, "classification_report": classification_report(y_test, pred, labels=list(LABELS), output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(y_test, pred, labels=list(LABELS)).tolist()}
    metrics = {"dataset_type": "user-provided / source not verified", "evaluation_note": "Final metrics use an untouched test split. Scores describe this dataset only and are not evidence of real-world performance.", "rows_after_deduplication": len(df), "label_order": list(LABELS), "validation_size": validation_size, "test_size": test_size, "random_state": random_state, "selected_model": best_name, "selection_metric": "validation macro F1", "validation_macro_f1": best_validation_f1, "models": results}
    return fitted_candidates[best_name], metrics

def train_and_save(path: str | Path = SYNTHETIC_DATA_PATH, output_path: str | Path = ARTIFACT_PATH, metrics_path: str | Path = METRICS_PATH, dataset_type: str = "synthetic demonstration") -> dict:
    df = load_dataset(path)
    model, metrics = evaluate_models(df)
    metrics["dataset_type"] = dataset_type
    # Retrain selected baseline on all deduplicated approved training rows for deployment.
    model.fit(df.model_text, df.label)
    output_path, metrics_path = Path(output_path), Path(metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True); metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": list(LABELS), "model_name": metrics["selected_model"]}, output_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

if __name__ == "__main__":
    result = train_and_save()
    print(json.dumps({"selected_model": result["selected_model"], "metrics": {k: {m: round(v[m], 3) for m in ("accuracy", "macro_precision", "macro_recall", "macro_f1")} for k, v in result["models"].items()}, "note": result["evaluation_note"]}, indent=2))
