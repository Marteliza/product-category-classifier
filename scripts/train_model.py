"""
train_model.py
Script pentru antrenarea modelului de clasificare produse
"""

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import os

def extract_features(df):
    """Extrage caracteristici din Product Title"""
    df['title_length'] = df['Product Title'].str.len()
    df['word_count'] = df['Product Title'].str.split().str.len()
    df['has_numbers'] = df['Product Title'].str.contains(r'\d').astype(int)
    df['number_count'] = df['Product Title'].str.count(r'\d')
    df['max_word_length'] = df['Product Title'].apply(
        lambda x: max([len(w) for w in x.split()]) if x.split() else 0
    )
    brands = ['samsung', 'apple', 'sony', 'lg', 'bosch', 'canon', 'hp']
    df['has_brand'] = df['Product Title'].apply(
        lambda x: any(brand in x for brand in brands)
    ).astype(int)
    df['special_char_count'] = df['Product Title'].apply(
        lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x))
    )
    return df

def main():
    print("=" * 70)
    print("ANTRENARE MODEL PRODUCT CATEGORY CLASSIFIER")
    print("=" * 70)
    
    # Incarc date
    print("\n[1/6] Încărcare date...")
    df = pd.read_csv('data/IMLP4_TASK_03-products.csv')
    print(f"   ✓ Încărcate {len(df)} produse")
    
    # Curatare
    print("\n[2/6] Curățare date...")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Product Title'])
    df = df.drop_duplicates(subset=['Product Title'])
    df['Product Title'] = df['Product Title'].str.lower().str.strip()
    
    category_counts = df['Category Label'].value_counts()
    valid_categories = category_counts[category_counts >= 10].index
    df = df[df['Category Label'].isin(valid_categories)].copy()
    print(f"   ✓ Date curate: {len(df)} produse, {df['Category Label'].nunique()} categorii")
    
    # Feature engineering
    print("\n[3/6] Feature engineering...")
    df = extract_features(df)
    print(f"   ✓ Create 7 features custom")
    
    # Vectorizare
    print("\n[4/6] Vectorizare TF-IDF...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_tfidf = tfidf.fit_transform(df['Product Title'])
    
    feature_cols = ['title_length', 'word_count', 'has_numbers', 
                    'number_count', 'max_word_length', 'has_brand', 'special_char_count']
    X_features = df[feature_cols].values
    X_combined = hstack([X_tfidf, X_features])
    y = df['Category Label']
    print(f"   ✓ Matrice features: {X_combined.shape}")
    
    # Split date
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Antrenare
    print("\n[5/6] Antrenare model...")
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluare
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   ✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Salvare
    print("\n[6/6] Salvare model...")
    os.makedirs('models', exist_ok=True)
    model_package = {
        'model': model,
        'tfidf': tfidf,
        'feature_columns': feature_cols,
        'accuracy': accuracy,
        'categories': sorted(y.unique())
    }
    
    with open('models/product_classifier.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    print(f"   ✓ Model salvat: models/product_classifier.pkl")
    
    print("\n" + "=" * 70)
    print(" ANTRENARE COMPLETĂ!")
    print("=" * 70)
    print(f"\nRezultat final: {accuracy*100:.2f}% accuracy")
    print("\nRaport de clasificare:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
