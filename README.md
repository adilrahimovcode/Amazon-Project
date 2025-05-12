<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg" alt="Amazon Logo" width="150"/>
</p>
# Amazon Məhsul Təsnifatlandırma Sistemi

Bu layihə məhsul adlarına əsaslanaraq kateqoriyanı avtomatik müəyyən etmək üçün maşın öyrənməsi modelindən istifadə edir.

## Layihə haqqında

Bu sistem məhsul adlarını təhlil edərək onları dörd əsas kateqoriyaya təsnif edir:

- Electronics (Elektronika)
- Clothing (Geyim)
- Sports (İdman)
- Home & Kitchen (Ev və Mətbəx)

## Xüsusiyyətlər

- Məhsul adlarının avtomatik təmizlənməsi və normallaşdırılması
- TF-IDF vektorizasiyası ilə mətn xüsusiyyətlərinin çıxarılması
- RandomForest alqoritmi ilə təsnifat
- Yeni məhsullar üçün kateqoriya və doğruluq faizi proqnozu
- Modelin yadda saxlanması və yüklənməsi imkanı

## Nəticələr

Hazırki model 50% dəqiqliklə işləyir və aşağıdakı nəticələri göstərir:

```
--- Model Qiymətləndirmə ---
                precision    recall  f1-score   support

      Clothing       1.00      0.50      0.67         4
   Electronics       1.00      0.50      0.67         4
Home & Kitchen       0.14      1.00      0.25         1
        Sports       1.00      0.33      0.50         3

      accuracy                           0.50        12
     macro avg       0.79      0.58      0.52        12
  weighted avg       0.93      0.50      0.59        12
```

Test məhsulları üçün proqnozlar:

- Wireless Earbuds -> Electronics (Doğruluq: 55.79%)
- Summer Dress -> Clothing (Doğruluq: 48.26%)
- Smart Watch -> Electronics (Doğruluq: 33.99%)
- Cooking Pan -> Home & Kitchen (Doğruluq: 39.76%)
- Gaming Mouse -> Electronics (Doğruluq: 34.85%)
- Leather Wallet -> Clothing (Doğruluq: 45.03%)
- Basketball Hoop -> Sports (Doğruluq: 45.98%)
- Microwave Oven -> Home & Kitchen (Doğruluq: 36.36%)
- Wireless Charger -> Electronics (Doğruluq: 55.56%)
- Hiking Boots -> Sports (Doğruluq: 34.95%)

## İstifadə qaydası

1. Layihəni yükləyin
2. Asılılıqları quraşdırın: `pip install -r requirements.txt`
3. Proqramı işə salın: `python amazon.py`

## Texniki detallar

- **Mətn emalı**: Xüsusi simvolların təmizlənməsi, kiçik hərflərə çevrilmə və stopwords təmizlənməsi
- **Feature engineering**: Məhsul adının uzunluğu və söz sayı kimi əlavə xüsusiyyətlər
- **Model**: RandomForest klassifikatoru (200 ağac, 15 maksimum dərinlik)
- **Vektorizasiya**: TF-IDF vektorizasiyası (5000 xüsusiyyət, 1-3 n-gram)

## Gələcək təkmilləşdirmələr

- Daha çox məhsul məlumatı əlavə etmək
- Daha mürəkkəb modellər sınamaq (XGBoost, Deep Learning)
- Daha çox feature engineering tətbiq etmək
- Hyperparameter tənzimləməsi ilə modeli optimallaşdırmaq

## Tələb olunan kitabxanalar

- pandas
- numpy
- scikit-learn
- joblib

## Layihə Strukturu

- **amazon.ipynb**: Jupyter notebook faylı, modelin öyrədilməsi və qiymətləndirilməsi üçün istifadə olunur.
- **visualize.py**: Məhsul kateqoriyalarının paylanmasını, hər kateqoriyada olan məhsulların sayını və hər kateqoriya üçün söz buludunu (word cloud) yaradan vizuallaşdırma skripti.
- **product_classifier.joblib**: Öyrədilmiş model faylı.
- **vectorizer.joblib**: TF-IDF vektorizatoru faylı.

## Tələblər

Layihəni işə salmaq üçün aşağıdakı Python paketlərini quraşdırın:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud joblib
```

## İstifadə

1. **Model Öyrətmə və Qiymətləndirmə**:

   - `amazon.ipynb` faylını açın və bütün hücrələri işə salın. Bu, modeli öyrədir və qiymətləndirir.

2. **Vizuallaşdırma**:
   - `visualize.py` faylını işə salın:
     ```bash
     python visualize.py
     ```
   - Bu, aşağıdakı vizuallaşdırma fayllarını yaradacaq:
     - `category_distribution.png`: Məhsul kateqoriyalarının paylanması.
     - `category_counts.png`: Hər kateqoriyada olan məhsulların sayı.
     - `wordcloud_*.png`: Hər kateqoriya üçün söz buludu.

## Nəticələr

Layihə, məhsul adlarına əsaslanaraq kateqoriyaları proqnozlaşdırır və vizuallaşdırma vasitəsilə məlumatları daha yaxşı başa düşməyə kömək edir.

## Əlavə Məlumat

Layihə haqqında əlavə məlumat və ya suallarınız varsa, xahiş edirəm, mənimlə əlaqə saxlayın.
