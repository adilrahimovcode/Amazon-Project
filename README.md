<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg" alt="Amazon Logo" width="150"/>
</p>
# Amazon Məhsul Təsnifatlandırma Sistemi

Bu layihə, məhsul adlarına əsaslanaraq kateqoriyaları avtomatik müəyyən etmək üçün süni intellektdən istifadə edən bir sistemdir. Layihə, məhsul adlarını təmizləyir, TF-IDF vektorizasiyasından istifadə edərək mətnləri vektorlaşdırır və Logistic Regression modeli ilə kateqoriyaları proqnozlaşdırır.

## Layihə Strukturu

- **amazon.ipynb**: Jupyter notebook faylı, modelin öyrədilməsi və qiymətləndirilməsi üçün istifadə olunur.
- **product_classifier.joblib**: Öyrədilmiş model faylı.
- **vectorizer.joblib**: TF-IDF vektorizatoru faylı.

## Tələblər

Layihəni işə salmaq üçün aşağıdakı Python paketlərini quraşdırın:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud joblib
```

## İstifadə

**Model Öyrətmə və Qiymətləndirmə**:

   - `amazon.ipynb` faylını açın və bütün hücrələri işə salın. Bu, modeli öyrədir və qiymətləndirir.

## Nəticələr

Layihə, məhsul adlarına əsaslanaraq kateqoriyaları proqnozlaşdırır.

## Əlavə Məlumat

Layihə haqqında əlavə məlumat və ya suallarınız varsa, xahiş edirəm, mənimlə əlaqə saxlayın.
