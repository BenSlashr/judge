# Interface Utilisateur Multi-Clustering

## 📋 Résumé des modifications

Le front-end a été entièrement mis à jour pour supporter les nouvelles fonctionnalités de multi-clustering. Voici les principales améliorations :

## 🔧 Modifications apportées

### 1. **Formulaire de configuration** (`templates/index.html`)

#### Nouveaux paramètres ajoutés :
- **Toggle Multi-clustering** : Activation/désactivation du mode multi-cluster
- **Seuil principal** : Probabilité minimum pour le cluster principal (défaut: 0.6)
- **Seuil secondaire** : Probabilité minimum pour les clusters secondaires (défaut: 0.4)
- **Max clusters par mot-clé** : De 3 à 10 clusters possibles (défaut: 3)
- **Probabilité minimum** : Seuil de filtrage des faibles probabilités (défaut: 0.15)

#### Interface utilisateur :
```html
<!-- Section Multi-clustering avec toggle switch animé -->
<div class="border-t border-gray-700 pt-4 mt-4">
    <label class="flex items-center cursor-pointer mb-3">
        <div class="toggle-switch">
            <input type="checkbox" id="enableMultiCluster">
            <span class="toggle-slider"></span>
        </div>
        <div class="ml-3">
            <span class="text-sm font-medium">Multi-clustering</span>
            <p class="text-xs text-gray-400">Permet aux mots-clés d'appartenir à plusieurs clusters</p>
        </div>
    </label>
    
    <!-- Paramètres avancés (affichés seulement si activé) -->
    <div id="multiClusterParams" class="hidden space-y-3">
        <!-- Contrôles de seuils et paramètres -->
    </div>
</div>
```

### 2. **Page de résultats** (`templates/results.html`)

#### Nouveau contenu :
- **Onglet Multi-clusters** : Dédié aux mots-clés appartenant à plusieurs clusters
- **Badge "Multi"** : Dans la colonne Multi du tableau principal
- **Affichage détaillé** : Probabilités et appartenance par cluster

#### Fonctionnalités ajoutées :
```javascript
// Badge dans le tableau principal
let multiClusterBadge = '';
if (kw.is_multi_cluster) {
    const totalClusters = kw.total_clusters_count || 2;
    multiClusterBadge = `<span class="bg-purple-600 text-white px-2 py-1 rounded text-xs">
                             Multi (${totalClusters})
                         </span>`;
}

// Onglet dédié avec affichage détaillé
function displayMultiClusterKeywords() {
    // Filtre et affiche les mots-clés multi-clusters
    // Avec probabilités par cluster
}
```

### 3. **Styles CSS** (`templates/base.html`)

#### Nouveaux styles :
```css
/* Toggle switch animé */
.toggle-switch {
    position: relative;
    width: 56px;
    height: 32px;
}

.toggle-slider {
    background-color: #64748b;
    transition: 0.3s;
    border-radius: 16px;
}

input:checked + .toggle-slider {
    background-color: #2563eb;
}

/* Animations pour les badges multi-cluster */
.multi-cluster-badge {
    animation: pulse 2s infinite;
}
```

## 🎨 Interface utilisateur

### **1. Formulaire de configuration**
- Section "Multi-clustering" avec toggle switch élégant
- Paramètres avancés qui s'affichent dynamiquement
- Sliders interactifs pour les seuils
- Select pour le nombre maximum de clusters

### **2. Affichage des résultats**
- **Onglet "Clusters"** : Vue classique avec informations étendues
- **Onglet "Tous les mots-clés"** : Nouvelle colonne "Multi" avec badges
- **Onglet "Multi-clusters"** : Vue détaillée des mots-clés multi-clusters
- **Onglet "Visualisation"** : Graphiques et charts

### **3. Détails multi-cluster**
Chaque mot-clé multi-cluster affiche :
- **Cluster principal** : Nom et probabilité (couleur bleue)
- **Cluster secondaire** : Nom et probabilité (couleur verte)  
- **Cluster alternatif** : Nom et probabilité (couleur jaune)
- **Clusters supplémentaires** : Jusqu'à 7 clusters additionnels (couleur violette)

### **4. Interactions**
- **Clic sur badge "Multi"** : Popup avec détails des clusters
- **Toggle activation** : Affichage dynamique des paramètres
- **Sliders temps réel** : Mise à jour immédiate des valeurs
- **Onglets fluides** : Navigation intuitive entre les vues

## 📊 Données affichées

### **Statistiques générales**
- Nombre de mots-clés total
- Nombre de clusters créés
- Volume de recherche total
- Score d'opportunité moyen

### **Informations par cluster**
- Nom du cluster (généré par IA)
- Nombre de mots-clés
- Mot-clé pivot (étoile dorée)
- Statistiques agrégées (volume, CPC, difficulté, score)

### **Détails multi-cluster**
- Probabilités d'appartenance précises
- Hiérarchie des clusters (principal → secondaire → alternatif)
- Compteur de clusters par mot-clé
- Classification visuelle par couleur

## 🔄 Flux de données

```
Frontend Form → JobParameters → Clustering Service → Multi-cluster Assignment → Database → API Response → UI Display
```

### **Envoi des paramètres**
```javascript
const parameters = {
    // Paramètres classiques
    serp_mode: "none",
    alpha: 0.5,
    clustering_algorithm: "hdbscan",
    min_cluster_size: 5,
    export_format: "csv",
    
    // Nouveaux paramètres multi-clustering
    enable_multi_cluster: true,
    primary_threshold: 0.6,
    secondary_threshold: 0.4,
    max_clusters_per_keyword: 5,
    min_probability_threshold: 0.15
};
```

### **Réception des résultats**
```javascript
// Structure keyword avec multi-clustering
{
    keyword: "consultant seo",
    cluster_primary: 0,
    cluster_primary_name: "Services SEO",
    cluster_primary_probability: 0.75,
    cluster_secondary: 1,
    cluster_secondary_name: "Expertise",  
    cluster_secondary_probability: 0.45,
    is_multi_cluster: true,
    total_clusters_count: 2,
    additional_clusters: [...]
}
```

## ✅ Tests et validation

### **Tests intégrés**
- Validation des paramètres frontend
- Sérialisation/désérialisation JSON
- Structure des modèles de données
- Format d'export étendu

### **Tests d'interface**
- Toggle switch fonctionnel
- Affichage conditionnel des paramètres
- Navigation entre onglets
- Popup modal détaillé

## 🚀 Utilisation

1. **Activation** : Cocher "Multi-clustering" dans le formulaire
2. **Configuration** : Ajuster les seuils selon vos besoins
3. **Analyse** : Lancer l'analyse normalement
4. **Exploration** : Utiliser l'onglet "Multi-clusters" pour voir les détails
5. **Export** : Télécharger avec colonnes étendues

## 🎯 Recommandations

### **Paramètres suggérés selon le cas d'usage**

| Cas d'usage | Primary | Secondary | Max clusters | Min prob |
|-------------|---------|-----------|--------------|----------|
| **Standard** | 0.6 | 0.4 | 3 | 0.15 |
| **Précis** | 0.7 | 0.5 | 3 | 0.20 |
| **Étendu** | 0.5 | 0.3 | 5 | 0.10 |
| **Exhaustif** | 0.4 | 0.2 | 7 | 0.05 |

### **Interprétation des résultats**
- **Probabilité ≥ 60%** : Appartenance forte au cluster
- **Probabilité 40-60%** : Appartenance modérée (clusters secondaires)
- **Probabilité < 40%** : Appartenance faible (filtrée par défaut)

---

**L'interface multi-clustering est maintenant entièrement fonctionnelle et prête à l'utilisation !** 🎉 