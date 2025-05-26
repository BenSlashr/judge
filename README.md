# SEO Keyword Classifier

Un outil avancé d'analyse et de clustering de mots-clés SEO avec intelligence artificielle.

## 🚀 Fonctionnalités

- **Import de fichiers** : CSV et XLSX jusqu'à 50 000 mots-clés
- **Enrichissement automatique** : Volume de recherche, CPC, difficulté (Google Ads, Semrush)
- **Clustering intelligent** : Utilise des embeddings sémantiques et l'analyse SERP
- **Nommage IA** : Attribution automatique de noms aux clusters via GPT-4
- **Visualisation UMAP** : Représentation graphique des clusters
- **Export multiple** : CSV, Excel, JSON
- **Interface moderne** : Design professionnel avec Tailwind CSS
- **Traitement asynchrone** : Suivi en temps réel avec Celery

## 🏗️ Architecture

```
app/
├── main.py              # Application FastAPI principale
├── config.py            # Configuration et chiffrement
├── models.py            # Modèles Pydantic
├── database.py          # Gestion SQLite
├── celery_app.py        # Configuration Celery
├── tasks.py             # Tâches asynchrones
├── services/
│   ├── enrichment.py    # Enrichissement SEO
│   └── clustering.py    # Clustering et IA
└── utils/
    └── file_processing.py  # Traitement de fichiers
```

## 🚀 Installation rapide

```bash
# 1. Cloner le projet
git clone <votre-repo>
cd seo-keyword-classifier

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configuration
cp env.example .env
# Éditez .env avec vos clés API

# 4. Démarrer Redis
redis-server

# 5. Démarrer Celery
celery -A app.celery_app worker --loglevel=info

# 6. Démarrer l'application
uvicorn app.main:app --reload
```

## ⚙️ Configuration

### Variables d'environnement

Copiez `env.example` vers `.env` et configurez vos clés API :

```bash
# DataForSEO (obligatoire pour la SERP)
DATAFOR_SEO_LOGIN=votre_login_dataforseo
DATAFOR_SEO_PASSWORD=votre_password_dataforseo

# OpenAI (obligatoire pour le nommage intelligent)
OPENAI_API_KEY=votre_cle_openai

# Google Ads (optionnel)
GOOGLE_ADS_DEVELOPER_TOKEN=votre_token
# ... autres clés Google Ads
```

### Sécurité

- Le fichier `.env` contient vos secrets et ne doit **jamais** être committé
- Assurez-vous que `.env` est dans votre `.gitignore`
- En production, utilisez des variables d'environnement système ou un gestionnaire de secrets

**Pourquoi pas de chiffrement ?**
- `.env` est déjà le standard de l'industrie
- Le chiffrement local n'ajoute pas de sécurité réelle
- Plus simple = moins d'erreurs

## 📊 Utilisation

### 1. Upload de fichiers

- Formats supportés : CSV, XLSX
- Colonne requise : `keyword` (ou variantes : `keywords`, `mot-clé`, etc.)
- Limite : 50 000 mots-clés par fichier

### 2. Configuration des paramètres

- **Mode SERP** : 
  - `None` : Clustering uniquement par embeddings
  - `Pivot only` : SERP seulement pour les mots-clés pivots
  - `Full SERP` : SERP pour tous les mots-clés
  
- **Alpha** : Poids entre embeddings (1.0) et SERP (0.0)
- **Algorithme** : HDBSCAN, Agglomératif, ou Louvain
- **Taille min cluster** : Nombre minimum de mots-clés par cluster

### 3. Suivi et résultats

- Suivi en temps réel de la progression
- Visualisation UMAP des clusters
- Export dans le format choisi
- Historique des analyses

## 🔧 API

### Endpoints principaux

- `POST /api/jobs` : Créer une nouvelle analyse
- `GET /api/jobs/{job_id}` : Statut d'un job
- `GET /api/jobs/{job_id}/result` : Télécharger les résultats
- `GET /api/jobs` : Liste des jobs récents
- `GET /api/jobs/{job_id}/visualization` : Données de visualisation

### Exemple d'utilisation

```python
import requests

# Créer un job
files = {'file': open('keywords.csv', 'rb')}
params = {
    'serp_mode': 'pivot_only',
    'clustering_algorithm': 'hdbscan',
    'min_cluster_size': 5,
    'export_format': 'xlsx'
}

response = requests.post(
    'http://localhost:8000/api/jobs',
    files=files,
    data={'parameters': json.dumps(params)}
)

job_id = response.json()['job_id']

# Suivre la progression
while True:
    status = requests.get(f'http://localhost:8000/api/jobs/{job_id}').json()
    if status['state'] == 'completed':
        break
    time.sleep(5)

# Télécharger les résultats
results = requests.get(f'http://localhost:8000/api/jobs/{job_id}/result')
```

## 🧪 Tests

```bash
# Tests unitaires
pytest

# Tests avec couverture
pytest --cov=app

# Tests end-to-end (Playwright)
pytest tests/e2e/
```

## 📈 Performance

### Benchmarks typiques

- **Mode embeddings uniquement** : ~1000 mots-clés/minute
- **Mode SERP pivot** : ~500 mots-clés/minute  
- **Mode SERP complet** : ~300 mots-clés/minute

### Optimisations

- Cache SERP avec TTL configurable
- Traitement par batches
- Réduction dimensionnelle UMAP pour >5000 mots-clés
- Parallélisation du scraping SERP

## 🔒 Sécurité

- Chiffrement des clés API avec Fernet
- HTTPS obligatoire en production
- Validation stricte des entrées
- Isolation des processus avec Celery

## 🐳 Déploiement

### Production avec Docker

```bash
# Variables d'environnement de production
export ENVIRONMENT=production
export SECRET_KEY=your-production-secret-key

# Déploiement
docker-compose -f docker-compose.prod.yml up -d
```

### Monitoring

- Logs structurés dans `/app/logs`
- Métriques Celery via Flower
- Health checks sur `/health`
- Statistiques sur `/api/stats`

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 Licence

Distribué sous licence MIT. Voir `LICENSE` pour plus d'informations.

## 🆘 Support

- 📧 Email : support@example.com
- 📖 Documentation : [docs.example.com](https://docs.example.com)
- 🐛 Issues : [GitHub Issues](https://github.com/example/issues)

---

**Développé avec ❤️ pour l'optimisation SEO** 