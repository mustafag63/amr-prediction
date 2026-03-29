# AMR Prediction – MALDI-TOF + Scikit-learn

> **Antimicrobial Resistance (AMR) tahmini** için makine öğrenmesi projesi.
> DRIAMS veri seti kullanılarak MALDI-TOF kütle spektrometresi verilerinden antibiyotik direnci (R/S) sınıflandırması yapılır.

---

## Proje Yapısı

```
amr-prediction/
│
├── data/                          # Ham veri (değiştirilmez)
│   └── driams/
│       ├── DRIAMS-A/
│       │   ├── binned_6000/       # Önişlenmiş spektrum dosyaları (.txt)
│       │   └── id/                # Etiket dosyaları (.csv) – R/S değerleri
│       ├── DRIAMS-B/
│       ├── DRIAMS-C/şim
│       └── DRIAMS-D/
│
├── src/                           # Kaynak kod (Python modülleri)
│   ├── data/
│   │   └── load_driams.py         # Veri yükleme fonksiyonları
│   ├── features/
│   │   └── preprocessing.py       # Sklearn preprocessing pipeline
│   ├── models/
│   │   └── train.py               # Model eğitimi ve CV
│   ├── evaluation/
│   │   └── metrics.py             # Metrikler ve görselleştirme
│   └── utils/                     # Genel yardımcı fonksiyonlar
│
├── notebooks/                     # Jupyter notebook'lar (EDA, denemeler)
│
├── configs/
│   └── config.yaml                # Deney parametreleri
│
├── outputs/
│   ├── models/                    # Kaydedilen modeller (.joblib)
│   ├── results/                   # CSV metrik sonuçları
│   └── figures/                   # ROC, PR, confusion matrix grafikleri
│
├── tests/                         # Birim testler
│
├── .env.example                   # Ortam değişkenleri şablonu
├── requirements.txt               # Python bağımlılıkları
└── PLAN.md                        # Proje yol haritası
```

---

## Hızlı Başlangıç

```bash
# 1. Bağımlılıkları kur
pip install -r requirements.txt

# 2. Ortam değişkenlerini ayarla
cp .env.example .env
# .env içindeki DRIAMS_ROOT yolunu düzenle

# 3. Veri yükle ve model eğit
python -c "
from src.data.load_driams import load_driams
X, y, meta = load_driams('DRIAMS-A', '2018', 'Ciprofloxacin')
print(f'X shape: {X.shape}, y dağılımı: R={y.sum()}, S={(1-y).sum()}')
"
```

---

## Veri Seti – DRIAMS

| Site | Hastane | Yıllar |
|------|---------|--------|
| DRIAMS-A | University Hospital Basel, İsviçre | 2015–2018 |
| DRIAMS-B | Canton Hospital Basel-Land, İsviçre | 2018 |
| DRIAMS-C | Canton Hospital Aarau, İsviçre | 2018 |
| DRIAMS-D | Viollier AG Laboratory, İsviçre | 2018 |

Her örnek için:
- **binned_6000/**: 6000 özellikli MALDI-TOF spektrum vektörü
- **id/**: Tür bilgisi + antibiyotik direnci (R = dirençli, S = duyarlı)

Referans makale: [https://doi.org/10.1101/2020.07.30.228411](https://doi.org/10.1101/2020.07.30.228411)

---

## Modeller

`configs/config.yaml` içinde `model.name` ile seçilir:

| İsim | Açıklama |
|------|----------|
| `logistic_regression` | Baseline, hızlı ve yorumlanabilir |
| `random_forest` | Güçlü ensemble, özellik önemi verir |
| `gradient_boosting` | Yüksek doğruluk, yavaş eğitim |
| `svm` | Yüksek boyutlu veriler için etkili |

---

## Değerlendirme Metrikleri

- **AUROC** – Ana metrik (sınıf dengesizliği için güvenilir)
- **AUPRC** – Precision-Recall eğrisi altı alan
- **F1 Score** – Precision ve Recall dengesi
- **Accuracy** – Genel doğruluk

---

## Bağımlılıklar

Tüm bağımlılıklar `requirements.txt` içinde listelenmiştir.
Temel kütüphaneler: `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `mlflow`.
