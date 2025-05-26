import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os

# Désactive le multiprocessing de PyTorch pour éviter les segfaults avec Celery
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import umap
from openai import OpenAI

from app.config import settings
from app.models import Keyword, Cluster, ClusteringAlgorithm, UMAPVisualization
from app.services.numeric_features import numeric_features_service

logger = logging.getLogger(__name__)

class ClusteringService:
    def __init__(self):
        logger.info("🔧 Initialisation du service de clustering")
        
        # Configuration sécurisée pour éviter les segfaults
        import torch
        torch.set_num_threads(1)
        
        try:
            # Charge le modèle avec des paramètres sécurisés
            logger.info("📥 Chargement du modèle SentenceTransformer...")
            self.embedding_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device='cpu'  # Force l'utilisation du CPU pour éviter les conflits
            )
            logger.info("✅ Modèle SentenceTransformer chargé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise
            
        self.openai_client = self._init_openai_client()
    
    def _init_openai_client(self) -> Optional[OpenAI]:
        """Initialise le client OpenAI pour le nommage des clusters"""
        openai_key = settings.openai_api_key
        if openai_key:
            try:
                return OpenAI(api_key=openai_key)
            except Exception as e:
                logger.error(f"Erreur d'initialisation OpenAI: {e}")
        return None
    
    def generate_embeddings(self, keywords: List[str], enable_numeric_features: bool = False, numeric_weight: float = 0.1) -> np.ndarray:
        """Génère les embeddings pour une liste de mots-clés"""
        logger.info(f"🧠 Génération des embeddings pour {len(keywords)} mots-clés")
        logger.debug(f"📝 Premiers mots-clés: {keywords[:5]}")
        
        try:
            logger.info(f"⚙️ Utilisation du modèle: paraphrase-multilingual-MiniLM-L12-v2")
            
            # Traite par petits batches pour éviter les problèmes de mémoire
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(keywords), batch_size):
                batch = keywords[i:i+batch_size]
                logger.debug(f"📦 Traitement du batch {i//batch_size + 1}/{(len(keywords)-1)//batch_size + 1}")
                
                batch_embeddings = self.embedding_model.encode(
                    batch, 
                    show_progress_bar=False,
                    batch_size=16,
                    normalize_embeddings=True
                )
                all_embeddings.append(batch_embeddings)
            
            # Combine tous les embeddings
            embeddings = np.vstack(all_embeddings)
            logger.info(f"✅ Embeddings générés avec succès: {embeddings.shape}")
            
            # Enrichissement avec features numériques si activé
            if enable_numeric_features and numeric_weight > 0:
                logger.info(f"🔢 Enrichissement avec features numériques (poids: {numeric_weight})")
                embeddings = numeric_features_service.create_enhanced_embeddings(
                    embeddings, keywords, numeric_weight
                )
                logger.info(f"✅ Embeddings enrichis: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération d'embeddings: {e}")
            # Fallback: utilise des embeddings aléatoires pour tester
            logger.warning("🔄 Utilisation d'embeddings aléatoires en fallback")
            return np.random.rand(len(keywords), 384)  # Dimension du modèle MiniLM
    
    def reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Réduit la dimensionnalité avec UMAP"""
        if len(embeddings) <= 5000:
            logger.info("Pas de réduction de dimension nécessaire (<= 5000 mots-clés)")
            return embeddings
        
        logger.info(f"Réduction de dimension UMAP de {embeddings.shape[1]} à {n_components}")
        
        reducer = umap.UMAP(
            n_components=n_components,
            metric='cosine',
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        logger.info(f"Dimension réduite: {reduced_embeddings.shape}")
        
        return reduced_embeddings
    
    def create_umap_visualization(self, embeddings: np.ndarray, keywords: List[str], cluster_labels: List[int]) -> UMAPVisualization:
        """Crée une visualisation UMAP 2D pour le frontend"""
        logger.info("Création de la visualisation UMAP")
        
        # Réduction à 2D pour la visualisation
        reducer = umap.UMAP(
            n_components=2,
            metric='cosine',
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )
        
        umap_2d = reducer.fit_transform(embeddings)
        
        # Génère des couleurs pour les clusters
        unique_labels = list(set(cluster_labels))
        colors = self._generate_cluster_colors(unique_labels)
        
        # Mappe les labels aux couleurs
        keyword_colors = [colors.get(label, "#cccccc") for label in cluster_labels]
        
        # Noms des clusters
        cluster_names = [f"Cluster {label}" if label >= 0 else "Bruit" for label in cluster_labels]
        
        return UMAPVisualization(
            x=umap_2d[:, 0].tolist(),
            y=umap_2d[:, 1].tolist(),
            labels=cluster_names,
            keywords=keywords,
            colors=keyword_colors
        )
    
    def _generate_cluster_colors(self, cluster_ids: List[int]) -> Dict[int, str]:
        """Génère des couleurs distinctes pour les clusters"""
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Utilise une palette de couleurs distinctes
        cmap = cm.get_cmap('tab20')
        colors = {}
        
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id == -1:  # Bruit dans HDBSCAN
                colors[cluster_id] = "#cccccc"
            else:
                color = cmap(i % 20)
                colors[cluster_id] = mcolors.to_hex(color)
        
        return colors
    
    def assign_multi_clusters(
        self, 
        keywords: List[str], 
        cluster_probabilities: np.ndarray,
        primary_threshold: float = 0.6,
        secondary_threshold: float = 0.4,
        max_clusters_per_keyword: int = 3,
        min_probability_threshold: float = 0.15
    ) -> List[Dict]:
        """Assigne les clusters multiples basés sur les probabilités d'appartenance"""
        logger.info(f"🔀 Attribution multi-cluster pour {len(keywords)} mots-clés")
        logger.info(f"📊 Seuils: principal={primary_threshold}, secondaire={secondary_threshold}")
        logger.info(f"📊 Limites: max_clusters={max_clusters_per_keyword}, min_prob={min_probability_threshold}")
        
        multi_cluster_assignments = []
        
        for i, keyword in enumerate(keywords):
            # Probabilités pour ce mot-clé sur tous les clusters
            keyword_probs = cluster_probabilities[i]
            
            # Trouve les indices triés par probabilité décroissante
            sorted_cluster_indices = np.argsort(keyword_probs)[::-1]
            sorted_probs = keyword_probs[sorted_cluster_indices]
            
            # Filtre les probabilités trop faibles
            valid_indices = sorted_probs >= min_probability_threshold
            sorted_cluster_indices = sorted_cluster_indices[valid_indices]
            sorted_probs = sorted_probs[valid_indices]
            
            # Limite au nombre maximum de clusters
            max_to_assign = min(len(sorted_probs), max_clusters_per_keyword)
            sorted_cluster_indices = sorted_cluster_indices[:max_to_assign]
            sorted_probs = sorted_probs[:max_to_assign]
            
            assignment = {
                'keyword': keyword,
                'cluster_primary': None,
                'cluster_primary_probability': None,
                'cluster_secondary': None,
                'cluster_secondary_probability': None,
                'cluster_alt1': None,
                'cluster_alt1_probability': None,
                'additional_clusters': [],
                'is_multi_cluster': False,
                'total_clusters_count': 1
            }
            
            # Attribution du cluster principal (plus haute probabilité)
            if len(sorted_probs) > 0:
                assignment['cluster_primary'] = int(sorted_cluster_indices[0])
                assignment['cluster_primary_probability'] = float(sorted_probs[0])
                assignment['total_clusters_count'] = 1
            
            # Attribution du cluster secondaire (≥ secondary_threshold)
            if len(sorted_probs) > 1 and sorted_probs[1] >= secondary_threshold:
                assignment['cluster_secondary'] = int(sorted_cluster_indices[1])
                assignment['cluster_secondary_probability'] = float(sorted_probs[1])
                assignment['is_multi_cluster'] = True
                assignment['total_clusters_count'] = 2
            
            # Attribution du cluster alternatif 1
            if len(sorted_probs) > 2 and sorted_probs[2] >= min_probability_threshold:
                assignment['cluster_alt1'] = int(sorted_cluster_indices[2])
                assignment['cluster_alt1_probability'] = float(sorted_probs[2])
                assignment['is_multi_cluster'] = True
                assignment['total_clusters_count'] = 3
            
            # Attribution des clusters supplémentaires (4+)
            if len(sorted_probs) > 3:
                additional_clusters = []
                for j in range(3, len(sorted_probs)):
                    if sorted_probs[j] >= min_probability_threshold:
                        additional_clusters.append({
                            'cluster_id': int(sorted_cluster_indices[j]),
                            'probability': float(sorted_probs[j])
                        })
                        assignment['total_clusters_count'] += 1
                
                if additional_clusters:
                    assignment['additional_clusters'] = additional_clusters
                    assignment['is_multi_cluster'] = True
            
            multi_cluster_assignments.append(assignment)
        
        multi_count = sum(1 for a in multi_cluster_assignments if a['is_multi_cluster'])
        logger.info(f"✅ Attribution terminée: {multi_count}/{len(keywords)} mots-clés multi-cluster")
        
        return multi_cluster_assignments
    
    def cluster_keywords(
        self, 
        keywords: List[str], 
        embeddings: np.ndarray,
        serp_similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        algorithm: ClusteringAlgorithm = ClusteringAlgorithm.HDBSCAN,
        min_cluster_size: int = 5,
        alpha: float = 0.5,
        enable_multi_cluster: bool = False,
        primary_threshold: float = 0.6,
        secondary_threshold: float = 0.4,
        enable_numeric_distance: bool = False,
        numeric_sensitivity: float = 0.5
    ) -> Tuple[List[int], List[Cluster], Optional[np.ndarray]]:
        """Clustering principal avec distance hybride embeddings + SERP"""
        
        logger.info(f"🎯 Début du clustering avec {algorithm.value}")
        if enable_multi_cluster:
            logger.info(f"🔀 Mode multi-cluster activé (seuils: {primary_threshold}/{secondary_threshold})")
        
        # Applique l'algorithme de clustering sélectionné
        distance_matrix = self._compute_hybrid_distance_matrix(
            keywords, embeddings, serp_similarity_matrix, alpha, enable_numeric_distance, numeric_sensitivity
        )
        
        logger.info(f"📊 Matrice calculée - Shape: {distance_matrix.shape}, dtype: {distance_matrix.dtype}")
        
        # Application de l'algorithme choisi avec probabilités si multi-cluster activé
        cluster_probabilities = None
        
        if algorithm == ClusteringAlgorithm.HDBSCAN:
            if enable_multi_cluster:
                cluster_labels, cluster_probabilities = self._hdbscan_clustering_with_probabilities(
                    distance_matrix, min_cluster_size
                )
            else:
                cluster_labels = self._hdbscan_clustering(distance_matrix, min_cluster_size)
        elif algorithm == ClusteringAlgorithm.AGGLOMERATIVE:
            cluster_labels = self._agglomerative_clustering(distance_matrix, min_cluster_size)
        elif algorithm == ClusteringAlgorithm.LOUVAIN:
            cluster_labels = self._louvain_clustering(distance_matrix, min_cluster_size)
        else:
            raise ValueError(f"Algorithme de clustering non supporté: {algorithm}")
        
        # Crée les objets Cluster
        clusters = self._create_cluster_objects(keywords, cluster_labels)
        
        logger.info(f"✅ Clustering terminé: {len(clusters)} clusters créés")
        return cluster_labels, clusters, cluster_probabilities
    
    def _compute_hybrid_distance_matrix(
        self,
        keywords: List[str],
        embeddings: np.ndarray,
        serp_similarity_matrix: Optional[Dict[str, Dict[str, float]]],
        alpha: float,
        enable_numeric_distance: bool = False,
        numeric_sensitivity: float = 0.5
    ) -> np.ndarray:
        """Calcule la matrice de distance hybride"""
        
        n = len(keywords)
        
        # Distance basée sur les embeddings (cosine)
        embedding_similarity = cosine_similarity(embeddings)
        embedding_distance = 1 - embedding_similarity
        
        # Assure le bon type de données pour HDBSCAN
        embedding_distance = embedding_distance.astype(np.float64)
        
        if serp_similarity_matrix is None or alpha == 1.0:
            # Pas de données SERP ou alpha = 1 (100% embeddings)
            logger.info(f"📊 Matrice de distance basée uniquement sur les embeddings")
            
            # Ajoute la distance numérique si activée
            final_distance = embedding_distance
            if enable_numeric_distance and numeric_sensitivity > 0:
                logger.info(f"🔢 Ajout de la distance numérique (sensibilité: {numeric_sensitivity})")
                final_distance = numeric_features_service.calculate_numeric_distance(
                    keywords, final_distance, numeric_sensitivity
                )
            
            return final_distance
        
        # Distance basée sur la similarité SERP
        serp_distance_matrix = np.ones((n, n), dtype=np.float64)
        
        for i, kw1 in enumerate(keywords):
            for j, kw2 in enumerate(keywords):
                if kw1 in serp_similarity_matrix and kw2 in serp_similarity_matrix[kw1]:
                    serp_similarity = serp_similarity_matrix[kw1][kw2]
                    serp_distance_matrix[i][j] = 1 - serp_similarity
        
        # Combine les deux distances selon alpha
        hybrid_distance = alpha * embedding_distance + (1 - alpha) * serp_distance_matrix
        
        # Ajoute la distance numérique si activée
        if enable_numeric_distance and numeric_sensitivity > 0:
            logger.info(f"🔢 Ajout de la distance numérique (sensibilité: {numeric_sensitivity})")
            hybrid_distance = numeric_features_service.calculate_numeric_distance(
                keywords, hybrid_distance, numeric_sensitivity
            )
        
        logger.info(f"📊 Matrice de distance hybride calculée (alpha={alpha})")
        return hybrid_distance.astype(np.float64)  # Force le bon type
    
    def _hdbscan_clustering(self, distance_matrix: np.ndarray, min_cluster_size: int) -> List[int]:
        """Clustering HDBSCAN"""
        logger.info(f"🔗 HDBSCAN avec min_cluster_size={min_cluster_size}")
        logger.debug(f"📊 Matrice de distance: shape={distance_matrix.shape}, dtype={distance_matrix.dtype}")
        
        # Assure le bon type de données
        if distance_matrix.dtype != np.float64:
            logger.warning(f"⚠️ Conversion dtype de {distance_matrix.dtype} vers float64")
            distance_matrix = distance_matrix.astype(np.float64)
        
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='precomputed',
                cluster_selection_method='eom'
            )
            
            cluster_labels = clusterer.fit_predict(distance_matrix)
            logger.info(f"✅ HDBSCAN terminé: {len(set(cluster_labels))} clusters uniques")
            return cluster_labels.tolist()
            
        except Exception as e:
            logger.error(f"❌ Erreur HDBSCAN: {e}")
            # Fallback vers clustering agglomératif
            logger.warning("🔄 Fallback vers clustering agglomératif")
            return self._agglomerative_clustering(distance_matrix, min_cluster_size)
    
    def _hdbscan_clustering_with_probabilities(self, distance_matrix: np.ndarray, min_cluster_size: int) -> Tuple[List[int], np.ndarray]:
        """Clustering HDBSCAN avec calcul des probabilités pour multi-cluster"""
        logger.info(f"🔗 HDBSCAN multi-cluster avec min_cluster_size={min_cluster_size}")
        logger.debug(f"📊 Matrice de distance: shape={distance_matrix.shape}, dtype={distance_matrix.dtype}")
        
        # Assure le bon type de données
        if distance_matrix.dtype != np.float64:
            logger.warning(f"⚠️ Conversion dtype de {distance_matrix.dtype} vers float64")
            distance_matrix = distance_matrix.astype(np.float64)
        
        try:
            # Configuration HDBSCAN pour obtenir les probabilités d'appartenance
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='precomputed',
                cluster_selection_method='eom',
                prediction_data=True  # Active le calcul des données de prédiction
            )
            
            # Fit du modèle
            clusterer.fit(distance_matrix)
            cluster_labels = clusterer.labels_
            
            logger.info(f"✅ HDBSCAN multi-cluster terminé: {len(set(cluster_labels))} clusters uniques")
            
            # Calcule les probabilités d'appartenance pour tous les clusters
            try:
                from hdbscan.prediction import all_points_membership_vectors
                
                # Génère les vecteurs de probabilité pour tous les points
                membership_vectors = all_points_membership_vectors(clusterer)
                
                logger.info(f"📊 Probabilités calculées: shape={membership_vectors.shape}")
                return cluster_labels.tolist(), membership_vectors
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur calcul probabilités: {e}, utilisation des probabilités simples")
                # Fallback: utilise les probabilités simples de HDBSCAN
                simple_probs = clusterer.probabilities_.reshape(-1, 1)
                return cluster_labels.tolist(), simple_probs
            
        except Exception as e:
            logger.error(f"❌ Erreur HDBSCAN multi-cluster: {e}")
            # Fallback vers clustering standard
            logger.warning("🔄 Fallback vers clustering HDBSCAN standard")
            cluster_labels = self._hdbscan_clustering(distance_matrix, min_cluster_size)
            # Probabilités factices pour compatibility
            fake_probs = np.zeros((len(cluster_labels), 1))
            return cluster_labels, fake_probs
    
    def _agglomerative_clustering(self, distance_matrix: np.ndarray, min_cluster_size: int) -> List[int]:
        """Clustering agglomératif"""
        logger.info(f"🔗 Clustering agglomératif avec min_cluster_size={min_cluster_size}")
        
        # Assure le bon type de données
        distance_matrix = distance_matrix.astype(np.float64)
        
        # Estime le nombre de clusters basé sur min_cluster_size
        n_samples = distance_matrix.shape[0]
        estimated_clusters = max(2, min(n_samples // min_cluster_size, n_samples - 1))
        
        logger.info(f"📊 Estimation: {estimated_clusters} clusters pour {n_samples} échantillons")
        
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=estimated_clusters,
                linkage='average',
                metric='precomputed'
            )
            
            cluster_labels = clusterer.fit_predict(distance_matrix)
            logger.info(f"✅ Clustering agglomératif terminé: {estimated_clusters} clusters")
            return cluster_labels.tolist()
            
        except Exception as e:
            logger.error(f"❌ Erreur clustering agglomératif: {e}")
            # Dernier fallback: assigne tous les éléments au même cluster
            logger.warning("🔄 Fallback: cluster unique")
            return [0] * n_samples
    
    def _louvain_clustering(self, distance_matrix: np.ndarray, min_cluster_size: int) -> List[int]:
        """Clustering Louvain (nécessite networkx et python-louvain)"""
        try:
            import networkx as nx
            import community  # python-louvain
            
            # Crée un graphe à partir de la matrice de distance
            G = nx.Graph()
            n = distance_matrix.shape[0]
            
            # Ajoute les arêtes avec des poids (inverse de la distance)
            threshold = np.percentile(distance_matrix, 20)  # Garde seulement les 20% plus similaires
            
            for i in range(n):
                for j in range(i + 1, n):
                    if distance_matrix[i][j] < threshold:
                        weight = 1 / (distance_matrix[i][j] + 1e-8)
                        G.add_edge(i, j, weight=weight)
            
            # Applique l'algorithme de Louvain
            partition = community.best_partition(G)
            
            # Convertit en liste de labels
            cluster_labels = [partition.get(i, -1) for i in range(n)]
            return cluster_labels
            
        except ImportError:
            logger.error("Les packages networkx et python-louvain sont requis pour Louvain")
            # Fallback vers HDBSCAN
            return self._hdbscan_clustering(distance_matrix, min_cluster_size)
    
    def _create_cluster_objects(self, keywords: List[str], cluster_labels: List[int]) -> List[Cluster]:
        """Crée les objets Cluster à partir des labels"""
        cluster_dict = {}
        
        # Groupe les mots-clés par cluster
        for keyword, label in zip(keywords, cluster_labels):
            if label >= 0:  # Ignore le bruit (label -1)
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append(keyword)
        
        clusters = []
        for cluster_id, cluster_keywords in cluster_dict.items():
            # Sélectionne le mot-clé pivot (le plus court ou central)
            pivot_keyword = min(cluster_keywords, key=len)
            
            cluster = Cluster(
                cluster_id=cluster_id,
                cluster_name=f"Cluster {cluster_id}",  # Sera renommé plus tard
                pivot_keyword=pivot_keyword,
                keywords_count=len(cluster_keywords)
            )
            clusters.append(cluster)
        
        return clusters
    
    def name_clusters_with_ai(self, clusters: List[Cluster], keywords_data: List[Keyword]) -> List[Cluster]:
        """Nomme les clusters en utilisant OpenAI GPT-4"""
        if not self.openai_client:
            logger.warning("Client OpenAI non disponible - nommage automatique des clusters")
            return clusters
        
        logger.info(f"Nommage AI de {len(clusters)} clusters")
        
        named_clusters = []
        
        for cluster in clusters:
            # Récupère les mots-clés du cluster
            cluster_keywords = [
                kw.keyword for kw in keywords_data 
                if kw.cluster_id == cluster.cluster_id
            ][:10]  # Limite à 10 mots-clés pour l'API
            
            try:
                cluster_name = self._generate_cluster_name(cluster_keywords)
                cluster.cluster_name = cluster_name
                logger.debug(f"Cluster {cluster.cluster_id} nommé: {cluster_name}")
                
            except Exception as e:
                logger.error(f"Erreur lors du nommage du cluster {cluster.cluster_id}: {e}")
                cluster.cluster_name = f"Cluster {cluster.cluster_id}"
            
            named_clusters.append(cluster)
        
        return named_clusters
    
    def _generate_cluster_name(self, keywords: List[str]) -> str:
        """Génère un nom de cluster avec OpenAI"""
        keywords_str = ", ".join(keywords)
        
        prompt = f"""Analyser cette liste de mots-clés et proposer un nom de cluster descriptif en français (maximum 4 mots):

Mots-clés: {keywords_str}

Règles:
- Le nom doit capturer la thématique principale
- Maximum 4 mots
- En français
- Pas de guillemets
- Format: nom commercial ou description courte

Nom du cluster:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.3
            )
            
            cluster_name = response.choices[0].message.content.strip()
            # Nettoie le nom généré
            cluster_name = cluster_name.replace('"', '').replace("'", "")
            
            return cluster_name[:50]  # Limite la longueur
            
        except Exception as e:
            logger.error(f"Erreur OpenAI: {e}")
            # Fallback: utilise le premier mot-clé comme nom
            return keywords[0] if keywords else "Cluster sans nom"
    
    def calculate_opportunity_scores(self, keywords: List[Keyword]) -> List[Keyword]:
        """Calcule le score d'opportunité pour chaque mot-clé"""
        logger.info("Calcul des scores d'opportunité")
        
        scored_keywords = []
        
        for kw in keywords:
            # Score basé sur volume de recherche, difficulté et CPC
            # Formule: (Volume * CPC) / (Difficulté + 1)
            
            volume = kw.search_volume or 0
            cpc = kw.cpc or 0
            difficulty = kw.keyword_difficulty or 50
            
            # Normalise le score entre 0 et 100
            raw_score = (volume * cpc) / (difficulty + 1) if difficulty > 0 else 0
            opportunity_score = min(100, max(0, raw_score / 100))  # Normalisation simple
            
            kw.opportunity_score = round(opportunity_score, 2)
            scored_keywords.append(kw)
        
        return scored_keywords

# Instance globale du service de clustering
clustering_service = ClusteringService() 