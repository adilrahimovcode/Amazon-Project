<p align="center">
  <img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fmagzoid.com%2Famazon-unveils-first-major-logo-redesign-in-20-years%2F&psig=AOvVaw1V6QxfErPuI-TKPAMnR2oE&ust=1747077067441000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCJihx7WPnI0DFQAAAAAdAAAAABAE" alt="Amazon Logo" width="150"/>
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
