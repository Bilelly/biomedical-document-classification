# 🧬 Classification de Publications Biomédicales par Type de Cancer  
*Par Bilal Sayoud – Data Scientist*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-red)
![Gensim](https://img.shields.io/badge/Gensim-4.0%2B-purple)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)

---

## 🎯 Objectif du projet

Ce projet vise à **classifier automatiquement des publications scientifiques biomédicales** selon le type de cancer traité :
- **Thyroid Cancer** (2810 documents)
- **Colon Cancer** (2580 documents)
- **Lung Cancer** (2180 documents)

L’objectif est de proposer une solution NLP robuste pour **aider les professionnels de santé, chercheurs ou systèmes d’information** à trier efficacement la littérature médicale — un enjeu crucial dans un domaine saturé de données.

---

## 📊 Données

- **Source** : Dataset public de 7 570 publications scientifiques complètes (full-text)
- **Caractéristiques** :
  - Textes longs (> 6 pages, ~25 000 mots/document)
  - Vocabulaire hautement technique
  - Classes équilibrées (~20–37 % par classe)
- **Format brut** : texte non structuré, encodage ISO-8859-1

> 💡 *La longueur et la complexité des textes rendent ce dataset particulièrement exigeant pour les modèles NLP.*

---

## 🔧 Pipeline de traitement

Le projet suit une **démarche NLP structurée en 5 étapes** :

1. **Analyse exploratoire (EDA)**  
   - Statistiques descriptives, wordcloud, répartition des classes
2. **Prétraitement textuel**  
   - Tokenization, lemmatisation (WordNet + POS tagging), suppression des stop words
3. **Représentation textuelle**  
   - **TF-IDF** (vectorisation statistique)
   - **Word2Vec** (embedding dense par moyenne)
   - **FastText** (embedding robuste aux mots rares)
4. **Modélisation**  
   - **SVM linéaire** et **Random Forest** comparés sur chaque représentation
5. **Évaluation & Visualisation**  
   - Métriques (accuracy, F1, recall), matrices de confusion, UMAP, analyse d’erreurs

---

## 📈 Résultats clés

### 🏆 Meilleur modèle : **SVM + TF-IDF**

| Métrique | Valeur |
|--------|--------|
| **Accuracy (test)** | **96.4 %** |
| **F1-score (moyen)** | **> 95 %** |
| **Lung_Cancer recall** | **100 %** |
| **Overfitting** | Aucun (écart train/test ≈ 1.4 %) |

### 🔍 Comparaison des méthodes d’embedding

| Représentation | Accuracy (SVM) |
|----------------|----------------|
| **TF-IDF**     | **96.4 %** ✅ |
| FastText       | 78.7 % |
| Word2Vec       | 74.1 % |

> **Conclusion** : Malgré sa simplicité, **TF-IDF capte mieux les termes discriminants** (ex. : "thyroidectomy", "colorectal", "adenocarcinoma") que les embeddings contextuels sur ce type de données longues et spécialisées.

---

## 🗂️ Structure du projet
``` medical-text-classification/
├── data/
│ └── medical_text.csv # Données brutes
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_text_preprocessing.ipynb
│ ├── 03_embedding_comparison.ipynb
│ ├── 04_classification.ipynb
│ └── 05_results_analysis.ipynb
├── models/
│ └── fastext_model.ft # Modèle FastText sauvegardé
├── report/
│ └── figures/ # Visualisations (UMAP, barplots, etc.)
├── requirements.txt
└── README.md

```
---

## ▶️ Comment reproduire le projet

```bash
# 1. Cloner le dépôt
git clone https://github.com/Bilelly/medical-text-classification.git

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer Jupyter et exécuter les notebooks dans l'ordre
jupyter notebook
