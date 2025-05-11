<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg" alt="Amazon Logo" width="80"/>
</p>
# Amazon Məhsul Təsnifatlandırma Sistemi

Bu layihə, məhsul adlarına əsaslanaraq kateqoriyaları avtomatik müəyyən etmək üçün süni intellektdən istifadə edən bir sistemdir. Layihə, məhsul adlarını təmizləyir, TF-IDF vektorizasiyasından istifadə edərək mətnləri vektorlaşdırır və Logistic Regression modeli ilə kateqoriyaları proqnozlaşdırır.

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
