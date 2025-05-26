# Guide des Features Numériques pour le Clustering

## 🔢 Vue d'ensemble

Le système de clustering a été enrichi avec des **features numériques** qui permettent d'améliorer significativement la qualité du clustering pour les mots-clés contenant des éléments numériques (prix, tailles, années, quantités, etc.).

## 🎯 Objectif

Les embeddings sémantiques traditionnels peuvent parfois regrouper des mots-clés sémantiquement similaires mais numériquement très différents. Par exemple :
- "smartphone 199€" et "smartphone 999€" 
- "écran 24 pouces" et "écran 43 pouces"
- "ssd 500gb" et "ssd 2tb"

Les features numériques permettent de tenir compte de ces différences pour créer des clusters plus cohérents.

## 🔧 Types de Features Détectées

### 💰 Prix et Monnaies
- **Formats supportés** : `299€`, `$199`, `£149`, `999 euros`, `50 centimes`
- **Normalisation** : Conversion vers EUR (€1.00 = $0.85 = £1.15)
- **Exemples** : "smartphone 299€", "laptop $1299", "casque 149£"

### 📅 Années et Dates
- **Formats supportés** : `2023`, `2021 année`, `2020 an`
- **Normalisation** : Centré sur l'an 2000 (2023 → +23)
- **Exemples** : "voiture 2018", "macbook 2021", "iphone 2023"

### 📦 Quantités et Mesures
- **Poids** : `1kg`, `500g`, `2 kilos`
- **Volume** : `1l`, `500ml`, `2 litres`
- **Distance** : `15m`, `30cm`, `5mm`
- **Unités** : `3 pièces`, `pack de 10`, `lot de 5`
- **Exemples** : "farine 1kg", "écran 27 pouces", "ram 16gb"

### 📏 Tailles et Capacités
- **Informatique** : `128gb`, `1to`, `500mo`
- **Écrans** : `24 pouces`, `15.6"`, `32 inches`
- **Normalisation** : Conversion vers unités de base
- **Exemples** : "ssd 1tb", "écran 27 pouces", "smartphone 128gb"

### 📊 Pourcentages et Notations
- **Pourcentages** : `20%`, `50 pourcent`
- **Notations** : `4.5/5`, `8/10`, `5 étoiles`
- **Exemples** : "réduction 30%", "note 4.5/5 étoiles"

### 🔢 Versions et Modèles
- **Versions** : `v2.0`, `version 3.1`, `bluetooth 5.0`
- **Modèles** : `iphone 13`, `galaxy s21`
- **Exemples** : "bluetooth 5.0", "usb 3.1", "wifi 6"

## ⚙️ Méthodes d'Intégration

### 🔗 Embeddings Enrichis (Recommandé)
**Principe** : Concatène les features numériques normalisées aux embeddings sémantiques.

**Avantages** :
- ✅ Simple à implémenter
- ✅ Compatible avec tous les algorithmes de clustering
- ✅ Contrôle précis du poids numérique
- ✅ Pas de modification des algorithmes existants

**Configuration** :
```python
# Interface utilisateur
enable_numeric_features = True
numeric_method = "enhanced_embeddings"
numeric_weight = 0.1  # 10% du poids total
```

**Fonctionnement** :
1. Génération des embeddings sémantiques (384 dimensions)
2. Extraction et normalisation des features numériques (17 dimensions)
3. Concaténation pondérée : `[embeddings ; numeric_weight * features]`
4. Clustering sur les embeddings enrichis (401 dimensions)

### 📏 Distance Hybride
**Principe** : Ajoute un terme de distance numérique à la matrice de distance existante.

**Avantages** :
- ✅ Préserve les embeddings originaux
- ✅ Contrôle indépendant de la sensibilité numérique
- ✅ Peut être combiné avec SERP

**Configuration** :
```python
# Interface utilisateur
enable_numeric_features = True
numeric_method = "hybrid_distance"
numeric_sensitivity = 0.5  # Sensibilité aux différences numériques
```

**Fonctionnement** :
1. Calcul de la distance sémantique (cosine)
2. Calcul de la distance numérique par paires
3. Combinaison : `distance_finale = distance_semantique + β * distance_numerique`

## 🎛️ Paramètres de Configuration

### Interface Utilisateur

#### 🔢 Activation
- **Toggle** : "Sensibilité numérique"
- **Description** : Active/désactive les features numériques
- **Défaut** : `false`

#### 🔧 Méthode d'Intégration
- **Options** :
  - `enhanced_embeddings` : Embeddings enrichis (recommandé)
  - `hybrid_distance` : Distance hybride
- **Défaut** : `enhanced_embeddings`

#### ⚖️ Poids Numérique
- **Plage** : 0.05 - 0.50
- **Défaut** : 0.10 (10%)
- **Description** : Impact des features numériques sur le clustering
- **Recommandations** :
  - `0.05-0.10` : Influence légère (mots-clés mixtes)
  - `0.15-0.25` : Influence modérée (e-commerce, tech)
  - `0.30-0.50` : Influence forte (données très numériques)

#### 🎯 Sensibilité Distance (Distance Hybride)
- **Plage** : 0.1 - 2.0
- **Défaut** : 0.5
- **Description** : Sensibilité aux différences numériques
- **Recommandations** :
  - `0.1-0.3` : Faible sensibilité (tolère les différences)
  - `0.4-0.7` : Sensibilité modérée (équilibrée)
  - `0.8-2.0` : Haute sensibilité (sépare fortement)

## 📈 Cas d'Usage Recommandés

### 🛒 E-commerce
**Contexte** : Mots-clés produits avec prix, tailles, capacités
```
Exemples : "iphone 13 128gb 999€", "tv samsung 55 pouces 4k"
Configuration recommandée :
- Méthode : enhanced_embeddings
- Poids : 0.20-0.25
- Bénéfice : Sépare les gammes de prix et tailles
```

### 💻 Technologie
**Contexte** : Spécifications techniques, versions, capacités
```
Exemples : "ssd 1tb nvme", "ram 16gb ddr4", "bluetooth 5.0"
Configuration recommandée :
- Méthode : enhanced_embeddings
- Poids : 0.15-0.20
- Bénéfice : Groupe par spécifications similaires
```

### 🏠 Immobilier
**Contexte** : Surfaces, nombres de pièces, années
```
Exemples : "appartement 3 pièces 75m2", "maison 2018 150m2"
Configuration recommandée :
- Méthode : hybrid_distance
- Sensibilité : 0.3-0.5
- Bénéfice : Clusters par taille et époque
```

### 🚗 Automobile
**Contexte** : Années, kilométrages, prix
```
Exemples : "voiture 2018 50000km", "bmw 2020 30000€"
Configuration recommandée :
- Méthode : enhanced_embeddings
- Poids : 0.25-0.30
- Bénéfice : Sépare par âge et gamme de prix
```

## 🔍 Algorithme de Normalisation

### Étape 1 : Extraction
```python
# Regex patterns pour chaque type
patterns = {
    'price': [r'(\d+(?:[,.]?\d+)*)\s*(?:€|euros?|eur)\b', ...],
    'year': [r'\b(19\d{2}|20\d{2})\b', ...],
    'quantity': [r'(\d+(?:[,.]?\d+)*)\s*(?:kg|kilogrammes?)\b', ...],
    # ... autres types
}
```

### Étape 2 : Normalisation des Unités
```python
# Conversion vers unités de référence
unit_normalization = {
    'price': {'CENT': 0.01, 'EUR': 1.0, 'USD': 0.85, 'GBP': 1.15},
    'quantity': {'G': 0.001, 'KG': 1.0, 'ML': 0.001, 'L': 1.0},
    'size': {'MB': 1.0, 'GB': 1024.0, 'TB': 1024*1024}
}
```

### Étape 3 : Transformation Logarithmique
```python
# Pour les valeurs avec large plage dynamique
if feature_type in ['price', 'quantity', 'size'] and value > 0:
    value = np.log1p(value)  # log(1 + x)
elif feature_type == 'year' and value > 1900:
    value = value - 2000  # Centré sur l'an 2000
```

### Étape 4 : Z-Score Normalization
```python
# Normalisation statistique
z_score = (value - mean) / std
z_score = np.clip(z_score, -3, 3)  # Limite les valeurs extrêmes
```

### Étape 5 : Vectorisation
```python
# Vecteur final (17 dimensions)
vector = {
    'num_price': aggregated_price_score,
    'num_year': aggregated_year_score,
    'num_quantity': aggregated_quantity_score,
    'num_percentage': aggregated_percentage_score,
    'num_size': aggregated_size_score,
    'num_version': aggregated_version_score,
    'num_rating': aggregated_rating_score,
    'has_numbers': 1.0 if features else 0.0,
    'num_count': min(len(features) / 5.0, 1.0),
    'unit_EUR': 1.0 if has_eur else 0.0,
    'unit_USD': 1.0 if has_usd else 0.0,
    # ... autres unités principales
}
```

## 📊 Métriques de Performance

### Distance Numérique
```python
def calculate_pairwise_numeric_distance(features1, features2):
    """
    Calcule la distance entre deux ensembles de features numériques
    
    Retourne:
    - 0.0 : Identiques numériquement
    - 0.5 : Partiellement différents
    - 1.0 : Complètement différents
    """
```

### Exemples de Distances
```
"iphone 13 128gb" ↔ "iphone 13 256gb" : 0.028 (très proche)
"smartphone 299€" ↔ "smartphone 349€" : 0.143 (proche)
"tv 55 pouces" ↔ "tv 65 pouces" : 0.097 (modérément proche)
"smartphone 299€" ↔ "smartphone 999€" : 0.701 (très différent)
```

## 🚀 Intégration dans le Pipeline

### Backend (app/tasks.py)
```python
# Génération des embeddings avec features numériques
embeddings = clustering_service.generate_embeddings(
    keyword_texts,
    enable_numeric_features=parameters.enable_numeric_features,
    numeric_weight=parameters.numeric_weight
)

# Clustering avec distance hybride
use_numeric_distance = (parameters.enable_numeric_features and 
                       parameters.numeric_method == "hybrid_distance")

labels, clusters, probabilities = clustering_service.cluster_keywords(
    keyword_texts,
    embeddings,
    enable_numeric_distance=use_numeric_distance,
    numeric_sensitivity=parameters.numeric_sensitivity
)
```

### Frontend (templates/index.html)
```javascript
const parameters = {
    // ... autres paramètres
    enable_numeric_features: document.getElementById('enableNumericFeatures').checked,
    numeric_method: document.getElementById('numericMethod').value,
    numeric_weight: parseFloat(document.getElementById('numericWeight').value),
    numeric_sensitivity: parseFloat(document.getElementById('numericSensitivity').value)
};
```

## 🎯 Recommandations d'Usage

### ✅ Quand Utiliser
- **Mots-clés e-commerce** avec prix, tailles, capacités
- **Données techniques** avec spécifications numériques
- **Contenu temporel** avec années, versions
- **Mesures physiques** avec dimensions, poids

### ⚠️ Limitations
- **Coût computationnel** : +17 dimensions aux embeddings
- **Complexité** : Paramètres supplémentaires à ajuster
- **Efficacité variable** : Dépend de la densité numérique des données

### 🔧 Optimisation
1. **Commencer léger** : `numeric_weight = 0.1`
2. **Tester progressivement** : Augmenter si bénéfique
3. **Analyser les résultats** : Vérifier la cohérence des clusters
4. **Adapter par domaine** : E-commerce vs Tech vs Immobilier

## 📝 Exemples Pratiques

### Configuration E-commerce Standard
```json
{
    "enable_numeric_features": true,
    "numeric_method": "enhanced_embeddings",
    "numeric_weight": 0.20,
    "numeric_sensitivity": 0.5
}
```

### Configuration Tech/Spécifications
```json
{
    "enable_numeric_features": true,
    "numeric_method": "enhanced_embeddings", 
    "numeric_weight": 0.15,
    "numeric_sensitivity": 0.4
}
```

### Configuration Prix Sensible
```json
{
    "enable_numeric_features": true,
    "numeric_method": "hybrid_distance",
    "numeric_weight": 0.1,
    "numeric_sensitivity": 0.6
}
```

---

## 🔬 Tests et Validation

Pour tester les features numériques :

```bash
# Test complet des features
python test_numeric_features.py

# Test d'amélioration du clustering
python test_numeric_clustering_improvement.py
```

Les tests valident :
- ✅ Extraction correcte des features
- ✅ Normalisation et vectorisation
- ✅ Intégration dans les embeddings
- ✅ Calcul de distance hybride
- ✅ Amélioration de la cohérence des clusters 