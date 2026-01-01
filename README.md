##  Product Category Classifier

Sistem inteligent de Machine Learning pentru clasificarea automată a produselor pe categorii pe baza titlului.

##  Despre Proiect

Acest proiect dezvoltă un model ML care poate prezice categoria unui produs doar citind titlul său. Elimină necesitatea clasificării manuale și accelerează procesul de introducere a produselor noi pe platforme de e-commerce.

##  Rezultate

- **Acuratețe**: 95-99% (variază per categorie)
- **Produse analizate**: 30,651 (după curățare)
- **Categorii**: 13 categorii principale
- **Model**: Logistic Regression cu TF-IDF vectorization

### Categorii suportate:
- Mobile Phones
- Washing Machines
- Fridge Freezers
- CPUs
- TVs
- Fridges
- Dishwashers
- Digital Cameras
- Microwaves
- Freezers
- și altele

##  Tehnologii și Metode

### Machine Learning:
- **Algoritm**: Logistic Regression (sklearn)
- **Vectorizare text**: TF-IDF (max 3000 features, n-grams 1-2)
- **Features custom**: 
  - Lungime titlu
  - Număr cuvinte
  - Prezență numere
  - Detecție brand-uri
  - Lungime cuvânt maxim
  - Caractere speciale

### Stack Tehnologic:
- Python 3.12
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

##  Structura Proiectului

product-category-classifier/
├── notebooks/
│   └── 01_explorare_date.ipynb    # Notebook complet cu analiza și antrenare
├── models/
│   └── product_classifier.pkl      # Model antrenat salvat
├── data/
│   └── IMLP4_TASK_03-products.csv # Dataset original
└── README.md

##  Cum Rulezi Proiectul

### Varianta 1: Google Colab (recomandat)
1. Deschide `notebooks/01_explorare_date.ipynb` în Google Colab
2. Uploadează fișierul `IMLP4_TASK_03-products.csv`
3. Rulează toate celulele în ordine (Runtime → Run all)
4. Așteaptă ~2-3 minute pentru antrenare

### Varianta 2: Local
```bash
pip install pandas scikit-learn matplotlib seaborn
jupyter notebook notebooks/01_explorare_date.ipynb
```

##  Exemple de Testare

| Titlu Produs | Categorie Prezisă | Confidence |
|-------------|-------------------|------------|
| iphone 7 32gb gold | Mobile Phones | 95.7% |
| samsung galaxy a52 128gb | Mobile Phones | 96.5% |
| bosch washing machine 8kg | Washing Machines | 99.8% |

##  Process Flow

1. **Explorare Date** → Analiză 35,311 produse
2. **Curățare** → Eliminare duplicate și valori lipsă → 30,651 produse
3. **Feature Engineering** → Extragere 7 caracteristici din titluri
4. **Vectorizare** → TF-IDF transformation
5. **Training** → Split 80/20, Logistic Regression
6. **Evaluare** → Accuracy 95%+, Classification Report, Confusion Matrix
7. **Testing** → Predicții interactive

##  Ce am Învățat

- Procesare și curățare date reale (cu valori lipsă, duplicate)
- Feature engineering pentru text
- TF-IDF vectorization și n-grams
- Antrenare și evaluare modele de clasificare
- Salvare și încărcare modele cu pickle
- Documentare și organizare proiect ML

##  Observații Tehnice

- Dataset-ul avea 172 titluri lipsă și 4,450 duplicate
- Categoriile rare (<10 produse) au fost eliminate
- Brandurile detectate: samsung, apple, sony, lg, bosch, canon, hp
- TF-IDF a identificat automat cuvinte-cheie relevante pentru fiecare categorie

##  Autor

**Marteliza**  
Proiect dezvoltat ca parte a cursului de Data Science & Machine Learning

## 📅 Data

Ianuarie 2026
