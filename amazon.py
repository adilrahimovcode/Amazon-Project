"""
Amazon Məhsul Təsnifatlandırma Sistemi
Məqsəd: Məhsul adlarına əsaslanaraq kateqoriyanı avtomatik müəyyən etmək üçün süni intellektdən istifadə nümunəsi.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import re

# Sadə stopwords siyahısı
STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'which', 
             'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 
             'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from'}

# Text cleaning
def preprocess_text(text):
    """Məhsul adını təmizləyir və normallaşdırır."""
    # Kiçik hərflərə çevir
    text = text.lower()
    
    # Xüsusi simvolları təmizlə
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Sözləri ayır və stopwords təmizlə
    words = [word for word in text.split() if word not in STOPWORDS]
    
    # Yenidən birləşdir
    text = ' '.join(words)
    return text

# Feature extraction
def extract_features(df):
    """Məhsul adlarından əlavə xüsusiyyətlər çıxarır."""
    # Məhsul adının uzunluğu
    df['title_length'] = df['title'].apply(len)
    
    # Söz sayı
    df['word_count'] = df['title'].apply(lambda x: len(x.split()))
    
    return df

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
        {"title": "Mechanical Keyboard", "category": "Electronics"},
        {"title": "Gaming Mouse", "category": "Electronics"},
        {"title": "Wireless Charger", "category": "Electronics"},
        {"title": "USB Hub", "category": "Electronics"},
        {"title": "Monitor Stand", "category": "Electronics"},
        {"title": "Tablet Holder", "category": "Electronics"},
        {"title": "Smartphone Tripod", "category": "Electronics"},
        {"title": "Bluetooth Adapter", "category": "Electronics"},
        {"title": "Computer Monitor", "category": "Electronics"},
        {"title": "Laptop Cooling Pad", "category": "Electronics"},
        {"title": "Portable SSD", "category": "Electronics"},
        {"title": "Wireless Router", "category": "Electronics"},
        {"title": "Digital Camera", "category": "Electronics"},
        {"title": "Smartphone Gimbal", "category": "Electronics"},
        {"title": "Noise Cancelling Headphones", "category": "Electronics"},
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
        {"title": "Leather Jacket", "category": "Clothing"},
        {"title": "Dress Shirt", "category": "Clothing"},
        {"title": "Casual Sneakers", "category": "Clothing"},
        {"title": "Winter Scarf", "category": "Clothing"},
        {"title": "Formal Tie", "category": "Clothing"},
        {"title": "Silk Pajamas", "category": "Clothing"},
        {"title": "Cotton Underwear", "category": "Clothing"},
        {"title": "Wool Coat", "category": "Clothing"},
        {"title": "Denim Shorts", "category": "Clothing"},
        {"title": "Leather Shoes", "category": "Clothing"},
        {"title": "Rain Jacket", "category": "Clothing"},
        {"title": "Fashion Watch", "category": "Clothing"},
        {"title": "Dress Pants", "category": "Clothing"},
        {"title": "Knit Beanie", "category": "Clothing"},
        {"title": "Designer Handbag", "category": "Clothing"},
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
        {"title": "Fitness Tracker", "category": "Sports"},
        {"title": "Sports Bag", "category": "Sports"},
        {"title": "Yoga Block Set", "category": "Sports"},
        {"title": "Sports Headband", "category": "Sports"},
        {"title": "Weight Lifting Gloves", "category": "Sports"},
        {"title": "Treadmill", "category": "Sports"},
        {"title": "Exercise Bike", "category": "Sports"},
        {"title": "Hiking Backpack", "category": "Sports"},
        {"title": "Volleyball", "category": "Sports"},
        {"title": "Badminton Set", "category": "Sports"},
        {"title": "Protein Shaker", "category": "Sports"},
        {"title": "Boxing Gloves", "category": "Sports"},
        {"title": "Cycling Helmet", "category": "Sports"},
        {"title": "Kettlebell", "category": "Sports"},
        {"title": "Hiking Boots", "category": "Sports"},
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
        {"title": "Mixing Bowl Set", "category": "Home & Kitchen"},
        {"title": "Food Processor", "category": "Home & Kitchen"},
        {"title": "Coffee Grinder", "category": "Home & Kitchen"},
        {"title": "Kitchen Scale", "category": "Home & Kitchen"},
        {"title": "Wine Glasses", "category": "Home & Kitchen"},
        {"title": "Baking Sheet", "category": "Home & Kitchen"},
        {"title": "Air Fryer", "category": "Home & Kitchen"},
        {"title": "Slow Cooker", "category": "Home & Kitchen"},
        {"title": "Electric Kettle", "category": "Home & Kitchen"},
        {"title": "Bathroom Scale", "category": "Home & Kitchen"},
        {"title": "Shower Curtain", "category": "Home & Kitchen"},
        {"title": "Throw Pillows", "category": "Home & Kitchen"},
        {"title": "Duvet Cover", "category": "Home & Kitchen"},
        {"title": "Trash Can", "category": "Home & Kitchen"},
        {"title": "Cooking Utensils Set", "category": "Home & Kitchen"},
        {"title": "Spice Rack", "category": "Home & Kitchen"}
    ]

# Data pre..
def prepare_data():
    """Məlumatı DataFrame-ə çevirir və təmizləyir."""
    products = get_product_data()
    df = pd.DataFrame(products)
    df['original_title'] = df['title']  # Orijinal adı saxla
    df['title'] = df['title'].apply(preprocess_text)
    df = extract_features(df)  # Əlavə xüsusiyyətlər çıxar
    return df

# Model qur.. 
def train_model(X_train, y_train):
    """Modeli öyrədir və qaytarır."""
    # TF-IDF vektorizatoru
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        stop_words='english',
        min_df=1,
        max_df=0.9,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # RandomForest modeli - daha sadə parametrlərlə
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    print("Model öyrədilir...")
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
    
    # Xüsusiyyətləri və hədəfi ayırırıq
    X = df['title']
    y = df['category']
    
    # Məlumatı bölürük - daha çox təlim məlumatı istifadə edirik
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    print(f"\nÖyrənmə üçün {len(X_train)} məhsul, test üçün {len(X_test)} məhsul istifadə edilir.")
    classifier, vectorizer = train_model(X_train, y_train)
    
    print("\nModel qiymətləndirilir...")
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
    print("\nModel uğurla yadda saxlanıldı!")

if __name__ == "__main__":
    main()
