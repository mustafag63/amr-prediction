"""
train.py
--------
Model eğitimi ve cross-validation yardımcı fonksiyonları.
Scikit-learn uyumlu her model ile çalışır.
"""

import joblib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs" / "models"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, Callable] = {
    # CV değerlendirmesi için fixed-C: nested CV problemini önler, n_jobs güvenli
    "logistic_regression": lambda: LogisticRegression(
        C=0.1, solver="saga", penalty="l1",
        max_iter=5000, random_state=42, class_weight="balanced",
    ),
    # Son model olarak kaydetmek / tek seferlik eğitim için kullan
    "logistic_regression_cv": lambda: LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
        solver="saga", penalty="l1", cv=5,
        max_iter=5000, random_state=42, class_weight="balanced", n_jobs=-1,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=2,
        random_state=42, n_jobs=-1, class_weight="balanced",
    ),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        random_state=42, class_weight="balanced",
    ),
    "svm": lambda: SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
}


def get_model(name: str):
    """İsme göre taze bir model örneği döndürür."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Bilinmeyen model: '{name}'. Mevcut: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()


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
        scoring = ["roc_auc", "average_precision", "f1_macro", "accuracy"]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # n_jobs=1: parallelism modelin kendi n_jobs'una bırakılır.
    # cross_validate + model iç içe n_jobs=-1 kullanırsa nested spawning olur.
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)

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
