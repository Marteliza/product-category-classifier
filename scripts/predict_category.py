"""
predict_category.py
Script interactiv pentru prezicerea categoriei produselor
"""

import pickle
import re
from scipy.sparse import hstack

def load_model():
    """Incarc modelul antrenat"""
    print("Incarc modelul...")
    with open('models/product_classifier.pkl', 'rb') as f:
        model_package = pickle.load(f)
    print(f" Model incarcat (Accuracy: {model_package['accuracy']:.2%})\n")
    return model_package

def predict_product(title, model_package):
    """Prezice categoria pentru un produs"""
    model = model_package['model']
    tfidf = model_package['tfidf']
    feature_cols = model_package['feature_columns']
    
    # Pregatire
    title_clean = title.lower().strip()
    
    # Features
    features = {
        'title_length': len(title_clean),
        'word_count': len(title_clean.split()),
        'has_numbers': int(bool(re.search(r'\d', title_clean))),
        'number_count': len(re.findall(r'\d', title_clean)),
        'max_word_length': max([len(w) for w in title_clean.split()]) if title_clean.split() else 0,
        'has_brand': int(any(b in title_clean for b in ['samsung', 'apple', 'sony', 'lg', 'bosch', 'canon', 'hp'])),
        'special_char_count': len(re.findall(r'[^a-zA-Z0-9\s]', title_clean))
    }
    
    # Transform
    X_tfidf = tfidf.transform([title_clean])
    X_feat = [[features[col] for col in feature_cols]]
    X_final = hstack([X_tfidf, X_feat])
    
    # Predict
    prediction = model.predict(X_final)[0]
    probabilities = model.predict_proba(X_final)[0]
    confidence = max(probabilities)
    
    # Top 3
    top_3_idx = probabilities.argsort()[-3:][::-1]
    top_3 = [(model.classes_[i], probabilities[i]) for i in top_3_idx]
    
    return prediction, confidence, top_3

def main():
    """Functia principala"""
    print("=" * 70)
    print("         SISTEM DE CLASIFICARE AUTOMATA A PRODUSELOR")
    print("=" * 70)
    print()
    
    # Incarca model
    try:
        model_package = load_model()
    except FileNotFoundError:
        print(" Eroare: Fisierul 'models/product_classifier.pkl' nu a fost gasit!")
        print("   Ruleaza mai intai: python scripts/train_model.py")
        return
    
    print("Categorii disponibile:", ", ".join(model_package['categories']))
    print("\nIntrodu titlul produsului pentru a prezice categoria.")
    print("Scrie 'exit' pentru a iesi.\n")
    print("-" * 70)
    
    # Loop interactiv
    while True:
        title = input("\n Titlu produs: ").strip()
        
        if title.lower() == 'exit':
            print("\n La revedere!")
            break
        
        if not title:
            print("  Te rog introdu un titlu valid.")
            continue
        
        try:
            category, confidence, top_3 = predict_product(title, model_package)
            
            print(f"\n Categorie prezisa: {category}")
            print(f" Incredere: {confidence:.1%}")
            print(f"\n Top 3 categorii:")
            for i, (cat, prob) in enumerate(top_3, 1):
                bar = " " * int(prob * 20)
                print(f"   {i}. {cat:20s} {prob:6.1%} {bar}")
            print("-" * 70)
            
        except Exception as e:
            print(f"\n Eroare: {e}")

if __name__ == "__main__":
    main()
