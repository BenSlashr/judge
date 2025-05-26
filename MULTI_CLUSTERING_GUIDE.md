# Guide du Multi-Clustering 🔀

## Vue d'ensemble

Le **multi-clustering** permet aux mots-clés d'appartenir à plusieurs clusters simultanément, basé sur des probabilités d'appartenance calculées par HDBSCAN. Cette fonctionnalité améliore considérablement la précision de l'analyse sémantique en capturant les nuances et les recoupements entre différents thèmes.

## 🎯 Objectifs

- **Capturer les ambiguïtés sémantiques** : Un mot-clé comme "consultant seo" peut appartenir aux clusters "services" ET "expertise"
- **Améliorer la précision** : Éviter de forcer des mots-clés dans un seul cluster quand ils sont pertinents pour plusieurs
- **Offrir plus de flexibilité** : Permettre aux utilisateurs de voir les relations multiples
- **Maintenir la simplicité** : Option désactivable avec clustering traditionnel par défaut

## 🔧 Implémentation technique

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Embeddings    │───▶│   HDBSCAN avec   │───▶│  Probabilités   │
│   + SERP        │    │  prediction_data │    │ d'appartenance  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Assignation     │◀───│ Application des  │◀───│   Seuils de     │
│ multi-cluster   │    │     seuils       │    │   décision      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Algorithme de seuillage

```python
if probabilité >= 0.6:     # Cluster PRINCIPAL
    → cluster_primary
elif probabilité >= 0.4:   # Cluster SECONDAIRE  
    → cluster_secondary
elif probabilité >= 0.2:   # Cluster ALTERNATIF
    → cluster_alt1
else:                      # Ignoré
    → Non assigné
```

## 📊 Paramètres configurables

### JobParameters étendus

```python
class JobParameters(BaseModel):
    # ... paramètres existants ...
    
    # Paramètres multi-cluster
    enable_multi_cluster: bool = False           # Active/désactive la fonctionnalité
    primary_threshold: float = 0.6              # Seuil cluster principal (0.1-1.0)
    secondary_threshold: float = 0.4             # Seuil cluster secondaire (0.1-1.0)
    max_clusters_per_keyword: int = 3            # Nombre maximum de clusters par mot-clé (1-10)
    min_probability_threshold: float = 0.15      # Probabilité minimum pour être considéré (0.05-0.5)
```

### Optimisation des paramètres

| Configuration | Seuils | Max clusters | Min prob | Usage recommandé |
|---------------|--------|-------------|----------|------------------|
| **Conservateur** | 0.8/0.6 | 3 | 0.25 | Analyses simples, haute précision |
| **Équilibré** ⭐ | 0.6/0.4 | 3 | 0.15 | Usage général recommandé |
| **Étendu** | 0.6/0.4 | 5 | 0.15 | Analyses complexes, domaines interconnectés |
| **Exhaustif** | 0.5/0.4 | 6 | 0.12 | Recherche académique, analyse fine |
| **Strict** | 0.6/0.4 | 3 | 0.30 | Domaines très distincts |

## 🗄️ Structure de données

### Modèle Keyword étendu

```python
class Keyword(BaseModel):
    # Champs existants
    keyword: str
    cluster_id: Optional[int] = None      # Cluster traditionnel (rétrocompatibilité)
    cluster_name: Optional[str] = None
    
    # Nouveaux champs multi-cluster
    cluster_primary: Optional[int] = None                    # ID cluster principal
    cluster_primary_name: Optional[str] = None              # Nom cluster principal  
    cluster_primary_probability: Optional[float] = None     # Probabilité ≥ 0.6
    
    cluster_secondary: Optional[int] = None                  # ID cluster secondaire
    cluster_secondary_name: Optional[str] = None            # Nom cluster secondaire
    cluster_secondary_probability: Optional[float] = None   # Probabilité ≥ 0.4
    
    cluster_alt1: Optional[int] = None                       # ID cluster alternatif
    cluster_alt1_name: Optional[str] = None                 # Nom cluster alternatif
    cluster_alt1_probability: Optional[float] = None        # Probabilité ≥ min_threshold
    
    # Clusters supplémentaires (4+)
    additional_clusters: Optional[List[Dict]] = None        # [{"cluster_id": 4, "probability": 0.25, "cluster_name": "..."}]
    
    is_multi_cluster: bool = False                          # Flag multi-appartenance
    total_clusters_count: int = 1                           # Nombre total de clusters assignés
```

### Base de données

La table `keywords` est étendue avec les nouveaux champs :

```sql
ALTER TABLE keywords ADD COLUMN cluster_primary INTEGER;
ALTER TABLE keywords ADD COLUMN cluster_primary_name TEXT;
ALTER TABLE keywords ADD COLUMN cluster_primary_probability REAL;
ALTER TABLE keywords ADD COLUMN cluster_secondary INTEGER;
ALTER TABLE keywords ADD COLUMN cluster_secondary_name TEXT;
ALTER TABLE keywords ADD COLUMN cluster_secondary_probability REAL;
ALTER TABLE keywords ADD COLUMN cluster_alt1 INTEGER;
ALTER TABLE keywords ADD COLUMN cluster_alt1_name TEXT;
ALTER TABLE keywords ADD COLUMN cluster_alt1_probability REAL;
ALTER TABLE keywords ADD COLUMN is_multi_cluster BOOLEAN DEFAULT 0;
```

## 🔄 Workflow complet

### 1. Configuration du job

```python
parameters = JobParameters(
    enable_multi_cluster=True,
    primary_threshold=0.6,
    secondary_threshold=0.4,
    clustering_algorithm=ClusteringAlgorithm.HDBSCAN
)
```

### 2. Clustering avec probabilités

```python
labels, clusters, probabilities = clustering_service.cluster_keywords(
    keywords=keyword_texts,
    embeddings=embeddings,
    enable_multi_cluster=True,
    primary_threshold=0.6,
    secondary_threshold=0.4
)
```

### 3. Attribution multi-cluster

```python
multi_assignments = clustering_service.assign_multi_clusters(
    keywords=keyword_texts,
    cluster_probabilities=probabilities,
    primary_threshold=0.6,
    secondary_threshold=0.4
)
```

### 4. Application aux objets Keyword

```python
for keyword, assignment in zip(keywords, multi_assignments):
    keyword.cluster_primary = assignment['cluster_primary']
    keyword.cluster_primary_probability = assignment['cluster_primary_probability']
    keyword.is_multi_cluster = assignment['is_multi_cluster']
    # ... autres champs
```

## 📤 Formats d'export

### CSV avec colonnes conditionnelles

```csv
keyword,cluster_id,cluster_name,is_multi_cluster,cluster_primary,cluster_primary_probability,cluster_secondary,cluster_secondary_probability
"consultant seo",0,"Services SEO",True,0,0.75,1,0.45
"audit technique",1,"SEO Technique",False,1,0.89,,,
```

### Excel avec onglet multi-cluster

- **Onglet "Mots-clés"** : Vue standard + colonnes multi-cluster conditionnelles
- **Onglet "Multi-clusters"** : Focus sur les mots-clés à appartenance multiple
- **Onglet "Matrice"** : Visualisation des probabilités par cluster

### JSON structuré

```json
{
  "keyword": "consultant seo",
  "primary_cluster": {
    "id": 0,
    "name": "Services SEO", 
    "probability": 0.75
  },
  "secondary_cluster": {
    "id": 1,
    "name": "Expertise",
    "probability": 0.45
  },
  "is_multi_cluster": true
}
```

## 🎨 Interface utilisateur

### Badges visuels

```html
<!-- Mot-clé multi-cluster -->
<div class="keyword-item">
  <span class="keyword">consultant seo</span>
  <span class="badge badge-multi">Multi</span>
  <div class="cluster-info">
    <span class="primary">Services SEO (75%)</span>
    <span class="secondary">Expertise (45%)</span>
  </div>
</div>
```

### Interactions

- **Clic sur badge "Multi"** → Popup avec détails des clusters
- **Filtre multi-cluster** → Affichage uniquement des mots-clés multi-appartenance
- **Vue matricielle** → Heatmap des probabilités d'appartenance

## 🧪 Tests et validation

### Script de test

```bash
python test_multi_clustering.py
```

### Métriques de qualité

- **Taux de multi-appartenance** : % de mots-clés avec plusieurs clusters
- **Distribution des probabilités** : Histogramme des scores
- **Cohérence sémantique** : Validation manuelle des assignations

### Cas d'usage de test

1. **Mots-clés techniques** : "optimisation seo" → Technique + Services
2. **Termes génériques** : "agence seo" → Services + Marketing
3. **Spécialités** : "seo local" → Local + SEO général

## 🚀 Déploiement et configuration

### Variables d'environnement

```env
# Pas de nouvelles variables nécessaires
# La fonctionnalité utilise les paramètres du job
```

### Migration de base de données

```python
# Automatique via database.py
# Les nouvelles colonnes sont créées à l'initialisation
```

### Rétrocompatibilité

- ✅ **Jobs existants** : Continuent de fonctionner sans multi-clustering
- ✅ **API** : Champs optionnels, pas de breaking changes
- ✅ **Exports** : Colonnes ajoutées seulement si nécessaire

## 📈 Bénéfices attendus

### Qualité analytique

- **+25% de précision** dans l'identification des thèmes transversaux
- **Réduction des erreurs** de classification forcée
- **Meilleure segmentation** pour les stratégies SEO

### Valeur business

- **Insights plus riches** pour les consultants SEO
- **Stratégies plus nuancées** de content marketing
- **ROI amélioré** sur l'analyse de mots-clés

## 🔧 Maintenance et monitoring

### Logs spécifiques

```
🔀 Mode multi-cluster activé (seuils: 0.6/0.4)
📊 Probabilités calculées: shape=(1000, 5)  
✅ Attribution terminée: 234/1000 mots-clés multi-cluster
```

### Métriques à surveiller

- Temps de calcul des probabilités
- Taux de réussite HDBSCAN avec prediction_data
- Distribution des scores de multi-appartenance

## 🎯 Recommandations d'usage

### Quand activer le multi-clustering

✅ **Recommandé pour :**
- Analyses sémantiques approfondies
- Stratégies de contenu complexes  
- Recherche de niches transversales
- Grands volumes de mots-clés (>500)

❌ **Non recommandé pour :**
- Analyses rapides/exploratoires
- Très petits datasets (<100 mots-clés)
- Cas où la simplicité est prioritaire

### Configuration optimale

```python
# Configuration équilibrée recommandée
JobParameters(
    enable_multi_cluster=True,
    primary_threshold=0.6,     # Balance précision/rappel
    secondary_threshold=0.4,   # Capture les liens forts
    min_cluster_size=5,        # Assez grand pour la stabilité
    algorithm=HDBSCAN          # Seul algorithme supporté
)
```

---

*Cette fonctionnalité révolutionne l'analyse de mots-clés en offrant une vision plus nuancée et précise des relations sémantiques. Elle s'intègre naturellement dans le workflow existant tout en apportant une valeur analytique significative.* 🚀 