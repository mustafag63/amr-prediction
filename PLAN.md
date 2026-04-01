# Proje Planı – AMR Prediction

> MALDI-TOF spektrumlarından antibiyotik direnci tahmini yapan
> bir makine öğrenmesi pipeline'ı geliştirme yol haritası.

---

## Aşama 1 – Veri Keşfi (EDA)

**Hedef:** Veriyi anlamak, kalite sorunlarını tespit etmek.

- [x] DRIAMS-A 2018 yılı verilerini yükle
- [x] Sınıf dağılımlarını incele (R/S oranları antibiyotiklere göre)
- [x] Spektrum görselleştirmesi yap (ortalama R vs S spektrumu)
- [x] Eksik veri analizi (hangi antibiyotiklerde kaç örnek var?)
- [x] `notebooks/01_eda.ipynb` dosyasına kaydet

---

## Aşama 2 – Baseline Model

**Hedef:** Hızla çalışan ilk modeli kur ve benchmark oluştur.

- [x] `Logistic Regression` ile DRIAMS-A / Ciprofloxacin için baseline eğit
- [ ] 5-Fold Stratified CV ile değerlendir
- [ ] AUROC, AUPRC, F1 sonuçlarını `outputs/results/` altına kaydet
- [ ] Referans makaledeki sonuçlarla karşılaştır
- [ ] `notebooks/02_baseline.ipynb`

---

## Aşama 3 – Model Geliştirme

**Hedef:** Farklı modeller dene, en iyisini seç.

- [ ] `Random Forest` eğit ve karşılaştır
- [ ] `Gradient Boosting` / `XGBoost` dene
- [ ] `SVM (RBF kernel)` dene
- [ ] Sonuçları tek tabloda karşılaştır
- [ ] `notebooks/03_model_comparison.ipynb`

---

## Aşama 4 – Özellik Mühendisliği

**Hedef:** Modelin girdisini iyileştir.

- [ ] Düşük varyanslı özellikleri filtrele (`VarianceThreshold`)
- [ ] PCA boyut azaltma dene (6000 → 200 bileşen)
- [ ] Özellik önemi analizi (Random Forest feature importances)
- [ ] `notebooks/04_feature_engineering.ipynb`

---

## Aşama 5 – Hiperparametre Optimizasyonu

**Hedef:** En iyi modelin parametrelerini ayarla.

- [ ] `GridSearchCV` veya `RandomizedSearchCV` uygula
- [ ] En iyi parametreleri `configs/config.yaml`'a kaydet
- [ ] Optimizasyon sonrası performansı ölç

---

## Aşama 6 – Çapraz Site Değerlendirmesi

**Hedef:** Modelin farklı hastanelere genelleme kapasitesini test et.

- [ ] DRIAMS-A üzerinde eğit → DRIAMS-B/C/D üzerinde test et
- [ ] Site bazlı AUROC sonuçlarını karşılaştır
- [ ] Domain shift etkisini analiz et
- [ ] `notebooks/05_cross_site.ipynb`

---

## Aşama 7 – Çoklu Antibiyotik Tahmini

**Hedef:** Her antibiyotik için ayrı model eğit, sonuçları toplu gör.

- [ ] En az 10 antibiyotik için model eğit (döngü ile)
- [ ] Sonuçları `outputs/results/all_antibiotics.csv` olarak kaydet
- [ ] Heatmap görselleştirmesi: antibiyotik × metrik
- [ ] `notebooks/06_multi_antibiotic.ipynb`

---

## Notlar

- Her aşama tamamlandıkça ilgili checkbox'ı işaretle.
- Model sonuçlarını MLflow ile izlemek için `mlflow ui` komutu kullanılabilir.
- Veri klasörü (`data/`) değiştirilmemeli; tüm çıktılar `outputs/` altına yazılmalı.
