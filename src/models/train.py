"""
train.py
--------
Model eğitimi ve cross-validation yardımcı fonksiyonları.
Scikit-learn uyumlu her model ile çalışır.
"""

import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

OUTPUTS_DIR = Path("outputs/models")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, Any] = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "svm": SVC(kernel="rbf", probability=True, random_state=42),
}


def get_model(name: str):
    """İsme göre model döndürür."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Bilinmeyen model: '{name}'. Mevcut: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


# ─── Training ─────────────────────────────────────────────────────────────────

def train_cv(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    scoring: list[str] | None = None,
) -> pd.DataFrame:
    """
    Stratified K-Fold cross-validation ile model performansını ölçer.

    Parameters
    ----------
    pipeline : Preprocessing + model pipeline
    X        : Özellik matrisi
    y        : Etiket vektörü
    n_splits : K-fold sayısı
    scoring  : Değerlendirme metrikleri listesi

    Returns
    -------
    pd.DataFrame  – Her fold için metrik sonuçları
    """
    if scoring is None:
        scoring = ["roc_auc", "average_precision", "f1", "accuracy"]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    df = pd.DataFrame(results)
    print(f"\n{'='*50}")
    print(f"Cross-Validation Sonuçları ({n_splits}-Fold)")
    print(f"{'='*50}")
    for metric in scoring:
        col = f"test_{metric}"
        print(f"{metric:25s}: {df[col].mean():.4f} ± {df[col].std():.4f}")
    return df


def save_model(pipeline: Pipeline, name: str) -> Path:
    """Eğitilmiş pipeline'ı diske kaydeder."""
    path = OUTPUTS_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    print(f"Model kaydedildi: {path}")
    return path


def load_model(name: str) -> Pipeline:
    """Diskten pipeline yükler."""
    path = OUTPUTS_DIR / f"{name}.joblib"
    return joblib.load(path)
