"""
metrics.py
----------
Model değerlendirme metrikleri ve görselleştirme fonksiyonları.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

FIGURES_DIR = Path(__file__).parent.parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """PR eğrisinde F1'i maksimize eden threshold'u döndürür."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, label: str = "", threshold: float | None = None) -> dict:
    """
    Tüm temel metrikleri hesaplar ve yazdırır.
    Threshold belirtilmezse PR eğrisinden F1-optimal threshold otomatik seçilir.

    Returns
    -------
    dict  – {auroc, auprc, threshold, ...}
    """
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{'='*50}  {label}")
    print(f"  AUROC     : {auroc:.4f}")
    print(f"  AUPRC     : {auprc:.4f}")
    print(f"  Threshold : {threshold:.4f}  (PR-optimal)")
    print(classification_report(y_true, y_pred, target_names=["S (Duyarlı)", "R (Dirençli)"]))

    return {"auroc": auroc, "auprc": auprc, "threshold": threshold}


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, label: str = "model", save: bool = True):
    """ROC ve Precision-Recall eğrilerini çizer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[0], name=label)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("ROC Curve")

    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=axes[1], name=label)
    axes[1].set_title("Precision-Recall Curve")

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"roc_pr_{label}.png"
        plt.savefig(path, dpi=150)
        print(f"Grafik kaydedildi: {path}")
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label: str = "model", save: bool = True):
    """Confusion matrix görselleştirir."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["S (Duyarlı)", "R (Dirençli)"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix – {label}")
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"confusion_{label}.png"
        plt.savefig(path, dpi=150)
        print(f"Grafik kaydedildi: {path}")
    plt.show()
