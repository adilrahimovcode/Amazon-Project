"""
Amazon Məhsul Təsnifatlandırma Sistemi

Məqsəd: Məhsul adlarına əsaslanaraq kateqoriyanı avtomatik müəyyən etmək üçün süni intellektdən istifadə nümunəsi.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import re

# Text cleaning
def preprocess_text(text):
    """Məhsul adını təmizləyir və kiçik hərflərə çevirir."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# Database
def get_product_data():
    """Məhsul məlumatlarını qaytarır."""
    return [
        # Electronics
        {"title": "Wireless Bluetooth Headphones", "category": "Electronics"},
        {"title": "Smartphone Case", "category": "Electronics"},
        {"title": "Laptop Backpack", "category": "Electronics"},
        {"title": "Fitness Tracker", "category": "Electronics"},
        {"title": "Wireless Mouse", "category": "Electronics"},
        {"title": "USB Flash Drive", "category": "Electronics"},
        {"title": "Power Bank", "category": "Electronics"},
        {"title": "HDMI Cable", "category": "Electronics"},
        {"title": "Wireless Keyboard", "category": "Electronics"},
        {"title": "Bluetooth Speaker", "category": "Electronics"},
        {"title": "Gaming Headset", "category": "Electronics"},
        {"title": "Smart Watch", "category": "Electronics"},
        {"title": "Wireless Earbuds", "category": "Electronics"},
        {"title": "External Hard Drive", "category": "Electronics"},
        {"title": "Webcam", "category": "Electronics"},
        # Clothing
        {"title": "Organic Cotton T-Shirt", "category": "Clothing"},
        {"title": "Denim Jeans", "category": "Clothing"},
        {"title": "Summer Dress", "category": "Clothing"},
        {"title": "Winter Jacket", "category": "Clothing"},
        {"title": "Running Shorts", "category": "Clothing"},
        {"title": "Formal Shirt", "category": "Clothing"},
        {"title": "Leather Belt", "category": "Clothing"},
        {"title": "Wool Sweater", "category": "Clothing"},
        {"title": "Swim Trunks", "category": "Clothing"},
        {"title": "Hiking Boots", "category": "Clothing"},
        {"title": "Leather Wallet", "category": "Clothing"},
        {"title": "Sunglasses", "category": "Clothing"},
        {"title": "Winter Gloves", "category": "Clothing"},
        {"title": "Running Socks", "category": "Clothing"},
        {"title": "Baseball Cap", "category": "Clothing"},
        # Sports
        {"title": "Running Shoes", "category": "Sports"},
        {"title": "Yoga Mat", "category": "Sports"},
        {"title": "Tennis Racket", "category": "Sports"},
        {"title": "Basketball", "category": "Sports"},
        {"title": "Dumbbells Set", "category": "Sports"},
        {"title": "Swimming Goggles", "category": "Sports"},
        {"title": "Football", "category": "Sports"},
        {"title": "Jump Rope", "category": "Sports"},
        {"title": "Resistance Bands", "category": "Sports"},
        {"title": "Sports Water Bottle", "category": "Sports"},
        {"title": "Basketball Hoop", "category": "Sports"},
        {"title": "Tennis Balls", "category": "Sports"},
        {"title": "Golf Clubs", "category": "Sports"},
        {"title": "Soccer Ball", "category": "Sports"},
        {"title": "Baseball Bat", "category": "Sports"},
        # Home & Kitchen
        {"title": "Coffee Maker", "category": "Home & Kitchen"},
        {"title": "Kitchen Knife Set", "category": "Home & Kitchen"},
        {"title": "Non-stick Pan", "category": "Home & Kitchen"},
        {"title": "Blender", "category": "Home & Kitchen"},
        {"title": "Toaster", "category": "Home & Kitchen"},
        {"title": "Dinner Plates Set", "category": "Home & Kitchen"},
        {"title": "Bed Sheets", "category": "Home & Kitchen"},
        {"title": "Vacuum Cleaner", "category": "Home & Kitchen"},
        {"title": "Table Lamp", "category": "Home & Kitchen"},
        {"title": "Bath Towels", "category": "Home & Kitchen"},
        {"title": "Microwave Oven", "category": "Home & Kitchen"},
        {"title": "Cooking Pot", "category": "Home & Kitchen"},
        {"title": "Dish Rack", "category": "Home & Kitchen"},
        {"title": "Cutting Board", "category": "Home & Kitchen"},
        {"title": "Mixing Bowl Set", "category": "Home & Kitchen"}
    ]

# Data pre..
def prepare_data():
    """Məlumatı DataFrame-ə çevirir və təmizləyir."""
    products = get_product_data()
    df = pd.DataFrame(products)
    df['title'] = df['title'].apply(preprocess_text)
    return df

# Model qur.. 
def train_model(X_train, y_train):
    """Modeli öyrədir və qaytarır."""
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 3),
        stop_words='english',
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs'
    )
    classifier.fit(X_train_tfidf, y_train)
    return classifier, vectorizer

# Model dəyər.
def evaluate_model(classifier, vectorizer, X_test, y_test):
    """Modelin nəticələrini çap edir."""
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)
    print("\n--- Model Qiymətləndirmə ---")
    print(classification_report(y_test, y_pred))

# Future çıxarmaq
def predict_category(product_title, classifier, vectorizer):
    """Yeni məhsul üçün kateqoriya və ehtimalı qaytarır."""
    processed_title = preprocess_text(product_title)
    product_tfidf = vectorizer.transform([processed_title])
    prediction = classifier.predict(product_tfidf)
    probabilities = classifier.predict_proba(product_tfidf)[0]
    max_prob = max(probabilities)
    return prediction[0], max_prob

# Main function
def main():
    print("Amazon Məhsul Təsnifatlandırma Sisteminə xoş gəlmisiniz!")
    df = prepare_data()
    X = df['title']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier, vectorizer = train_model(X_train, y_train)
    evaluate_model(classifier, vectorizer, X_test, y_test)
    # Test datası
    test_products = [
        "Wireless Earbuds",
        "Summer Dress",
        "Smart Watch",
        "Cooking Pan",
        "Gaming Mouse",
        "Leather Wallet",
        "Basketball Hoop",
        "Microwave Oven",
        "Wireless Charger",
        "Hiking Boots"
    ]
    print("\n--- Yeni məhsullar üçün proqnozlar ---")
    for product in test_products:
        category, confidence = predict_category(product, classifier, vectorizer)
        print(f"Məhsul: {product} -> Kateqoriya: {category} (Doğruluq: {confidence:.2%})")
    # Model save
    joblib.dump(classifier, 'product_classifier.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

if __name__ == "__main__":
    main()
