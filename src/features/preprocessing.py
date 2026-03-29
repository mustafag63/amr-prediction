"""
preprocessing.py
----------------
Spektrum özelliklerini ML modeline hazırlamak için
ön işleme (preprocessing) pipeline'ları.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


def build_preprocessing_pipeline(
    variance_threshold: float = 0.0,
    use_pca: bool = False,
    pca_components: int = 200,
    scale: bool = True,
) -> Pipeline:
    """
    Scikit-learn Pipeline olarak ön işleme adımlarını oluşturur.

    Parameters
    ----------
    variance_threshold : Sıfır (veya düşük) varyanslı özellikleri kaldır
    use_pca            : PCA boyut azaltma uygula
    pca_components     : PCA bileşen sayısı
    scale              : StandardScaler uygula

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps = []

    if variance_threshold > 0:
        steps.append(("var_filter", VarianceThreshold(threshold=variance_threshold)))

    if scale:
        steps.append(("scaler", StandardScaler()))

    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=42)))

    if not steps:
        # Pipeline boş olamaz, identity scaler ekle
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)
