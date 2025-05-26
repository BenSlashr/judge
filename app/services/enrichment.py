import logging
import asyncio
import httpx
import json
from typing import List, Optional, Dict, Any, Tuple
from app.config import settings
from app.models import Keyword, SerpResult
from app.database import db_manager

logger = logging.getLogger(__name__)

class EnrichmentService:
    def __init__(self):
        self.datafor_seo_credentials = self._get_datafor_seo_credentials()
    
    def _get_datafor_seo_credentials(self) -> Optional[Dict[str, str]]:
        """Récupère les credentials DataForSEO depuis .env"""
        login = settings.datafor_seo_login
        password = settings.datafor_seo_password
        
        if login and password:
            return {"login": login, "password": password}
        return None
    
    async def enrich_keywords_metrics(self, keywords: List[Keyword]) -> List[Keyword]:
        """Enrichit les mots-clés avec les métriques SEO"""
        logger.info(f"💰 Début de l'enrichissement de {len(keywords)} mots-clés")
        
        # Les métriques sont maintenant directement lues depuis le fichier CSV
        # On ne fait plus d'enrichissement automatique
        enriched_keywords = []
        
        logger.debug(f"📊 Échantillon des mots-clés à traiter:")
        for i, kw in enumerate(keywords[:3]):
            logger.debug(f"  {i+1}. {kw.keyword} (vol: {kw.search_volume}, cpc: {kw.cpc})")
        
        for i, kw in enumerate(keywords):
            # Garde les données existantes du fichier
            enriched_kw = kw.model_copy()
            enriched_keywords.append(enriched_kw)
            
            if i % 100 == 0:  # Log tous les 100 mots-clés
                logger.debug(f"⚙️ Traité {i+1}/{len(keywords)} mots-clés")
        
        logger.info(f"✅ Enrichissement terminé pour {len(enriched_keywords)} mots-clés (données du fichier utilisées)")
        return enriched_keywords
    
    async def scrape_serp_results(self, keyword: str, country_code: str = "FR") -> List[SerpResult]:
        """Scrape les résultats SERP pour un mot-clé avec DataForSEO"""
        # Vérifier le cache d'abord
        cached_results = db_manager.get_cached_serp_results(keyword, country_code)
        if cached_results:
            logger.debug(f"Résultats SERP trouvés en cache pour '{keyword}'")
            return cached_results
        
        logger.info(f"Scraping SERP DataForSEO pour '{keyword}' ({country_code})")
        
        try:
            if self.datafor_seo_credentials:
                results = await self._scrape_with_datafor_seo(keyword, country_code)
            else:
                logger.warning("Aucun credential DataForSEO configuré")
                return []
            
            # Mettre en cache les résultats
            if results:
                db_manager.cache_serp_results(
                    keyword, 
                    country_code, 
                    results, 
                    settings.default_serp_cache_ttl_days
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du scraping SERP pour '{keyword}': {e}")
            return []
    
    async def _scrape_with_datafor_seo(self, keyword: str, country_code: str) -> List[SerpResult]:
        """Scrape avec DataForSEO API"""
        try:
            # Configuration DataForSEO
            base_url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
            
            # Payload pour l'API DataForSEO
            payload = [{
                "keyword": keyword,
                "location_code": self._get_location_code(country_code),
                "language_code": "fr",
                "device": "desktop",
                "os": "windows"
            }]
            
            # Authentification Basic Auth
            auth = (
                self.datafor_seo_credentials["login"],
                self.datafor_seo_credentials["password"]
            )
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    base_url,
                    json=payload,
                    auth=auth,
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Erreur DataForSEO HTTP {response.status_code}: {response.text}")
                    return []
                
                try:
                    data = response.json()
                except Exception as e:
                    logger.error(f"Erreur lors du parsing JSON DataForSEO: {e}")
                    return []
                
                # Parse les résultats
                serp_results = []
                if data.get("status_code") == 20000 and data.get("tasks"):
                    task_result = data["tasks"][0]
                    if task_result.get("status_code") == 20000 and task_result.get("result"):
                        organic_results = task_result["result"][0].get("items", [])
                        
                        for i, result in enumerate(organic_results, 1):
                            if result.get("type") == "organic":
                                # Protection ultra-robuste contre les valeurs None et autres types
                                description = result.get("description")
                                if description is None:
                                    description = ""
                                elif not isinstance(description, str):
                                    description = str(description) if description else ""
                                
                                # Protection pour les autres champs également
                                url = result.get("url")
                                if url is None:
                                    url = ""
                                elif not isinstance(url, str):
                                    url = str(url) if url else ""
                                
                                title = result.get("title")
                                if title is None:
                                    title = ""
                                elif not isinstance(title, str):
                                    title = str(title) if title else ""
                                
                                # Log de debug pour comprendre ce qui se passe
                                logger.debug(f"🔍 Création SerpResult pour '{keyword}' - description: {repr(description)}, type: {type(description)}")
                                
                                # Création du SerpResult avec données nettoyées
                                try:
                                    # Double vérification avant création
                                    clean_data = {
                                        "keyword": str(keyword),
                                        "url": str(url),
                                        "title": str(title),
                                        "description": str(description),
                                        "position": int(i),
                                        "domain": self._extract_domain(str(url))
                                    }
                                    
                                    serp_result = SerpResult(**clean_data)
                                    serp_results.append(serp_result)
                                    logger.debug(f"✅ SerpResult créé avec succès pour '{keyword}' - position {i}")
                                    
                                except Exception as e:
                                    logger.error(f"❌ Erreur lors de la création de SerpResult pour '{keyword}': {e}")
                                    logger.error(f"   Données: {clean_data}")
                                    continue
                
                logger.info(f"✅ DataForSEO: {len(serp_results)} résultats pour '{keyword}'")
                return serp_results
            
        except Exception as e:
            logger.error(f"Erreur DataForSEO pour '{keyword}': {e}")
            return []
    
    def _get_location_code(self, country_code: str) -> int:
        """Convertit un code pays en location_code DataForSEO"""
        # Mapping des codes pays vers les location codes DataForSEO
        location_codes = {
            "FR": 2250,  # France
            "US": 2840,  # United States
            "UK": 2826,  # United Kingdom
            "DE": 2276,  # Germany
            "ES": 2724,  # Spain
            "IT": 2380,  # Italy
            "BE": 2056,  # Belgium
            "CH": 2756,  # Switzerland
            "CA": 2124,  # Canada
        }
        return location_codes.get(country_code.upper(), 2250)  # Default: France
    
    def _extract_domain(self, url: str) -> str:
        """Extrait le domaine d'une URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return ""
    
    async def calculate_serp_similarity(
        self, 
        keywords: List[str], 
        country_code: str = "FR",
        max_concurrent: int = 10,
        enable_sampling: bool = False,
        sampling_ratio: float = 0.3,
        enable_smart_clustering: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Calcule la similarité entre les SERPs de différents mots-clés avec optimisations"""
        logger.info(f"🚀 Calcul de la similarité SERP pour {len(keywords)} mots-clés avec DataForSEO")
        logger.info(f"⚙️ Optimisations: concurrent={max_concurrent}, sampling={enable_sampling}, smart_clustering={enable_smart_clustering}")
        
        # 🎯 OPTIMISATION 1: Échantillonnage intelligent si activé
        keywords_to_process = keywords.copy()
        if enable_sampling and len(keywords) > 100:
            import random
            sample_size = max(50, int(len(keywords) * sampling_ratio))
            
            # Échantillonnage stratégique: garde les mots-clés importants
            # Tri par longueur pour garder des mots-clés représentatifs
            sorted_keywords = sorted(keywords, key=len)
            step = len(sorted_keywords) // sample_size
            keywords_to_process = [sorted_keywords[i] for i in range(0, len(sorted_keywords), max(1, step))][:sample_size]
            
            logger.info(f"📊 Échantillonnage: {len(keywords_to_process)}/{len(keywords)} mots-clés sélectionnés")
        
        # 🎯 OPTIMISATION 2: Clustering préliminaire pour éviter les doublons proches
        if enable_smart_clustering and len(keywords_to_process) > 50:
            keywords_to_process = await self._smart_keyword_selection(keywords_to_process)
            logger.info(f"🧠 Sélection intelligente: {len(keywords_to_process)} mots-clés représentatifs")
        
        # 🚀 OPTIMISATION 3: Appels parallèles avec semaphore pour limiter la concurrence
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_limit(keyword: str):
            async with semaphore:
                try:
                    results = await self.scrape_serp_results(keyword, country_code)
                    domains = [r.domain for r in results[:10]]  # Top 10 results
                    return keyword, set(domains)
                except Exception as e:
                    logger.error(f"❌ Erreur SERP pour '{keyword}': {e}")
                    return keyword, set()
        
        # Lancement de tous les appels en parallèle
        logger.info(f"🔄 Lancement de {len(keywords_to_process)} appels SERP parallèles...")
        tasks = [scrape_with_limit(keyword) for keyword in keywords_to_process]
        
        # Traitement par batches pour éviter l'overload
        batch_size = 50
        serp_data = {}
        
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i+batch_size]
            logger.info(f"📦 Traitement batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"❌ Erreur dans le batch: {result}")
                    continue
                keyword, domains = result
                serp_data[keyword] = domains
        
        logger.info(f"✅ Récupération SERP terminée: {len(serp_data)} mots-clés traités")
        
        # 🎯 OPTIMISATION 4: Extension de la matrice pour les mots-clés non traités
        if enable_sampling or enable_smart_clustering:
            serp_data = await self._extend_similarity_matrix(keywords, serp_data, keywords_to_process)
        
        # Calcule la similarité Jaccard entre chaque paire (optimisé)
        similarity_matrix = self._calculate_jaccard_matrix_optimized(keywords, serp_data)
        
        logger.info(f"✅ Matrice de similarité SERP calculée pour {len(keywords)} mots-clés")
        return similarity_matrix
    
    async def _smart_keyword_selection(self, keywords: List[str]) -> List[str]:
        """Sélection intelligente de mots-clés représentatifs"""
        try:
            # Clustering rapide basé sur la similarité textuelle
            from sentence_transformers import SentenceTransformer
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Utilise un modèle plus petit pour la sélection rapide
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            embeddings = model.encode(keywords, show_progress_bar=False)
            
            # Clustering en groupes représentatifs
            # Objectif : réduire à environ 30-50% des mots-clés originaux
            target_reduction = 0.4  # Garde 40% des mots-clés
            n_clusters = max(5, int(len(keywords) * target_reduction))
            n_clusters = min(n_clusters, len(keywords) - 1)  # Au maximum len(keywords)-1 clusters
            
            logger.debug(f"🧠 Clustering {len(keywords)} mots-clés en {n_clusters} groupes représentatifs")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Sélectionne le représentant de chaque cluster (plus proche du centroïde)
            selected_keywords = []
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_embeddings = embeddings[cluster_mask]
                cluster_keywords = [kw for i, kw in enumerate(keywords) if cluster_mask[i]]
                
                if len(cluster_keywords) > 0:
                    # Trouve le mot-clé le plus proche du centroïde
                    centroid = cluster_embeddings.mean(axis=0).reshape(1, -1)
                    similarities = cosine_similarity(cluster_embeddings, centroid).flatten()
                    best_idx = similarities.argmax()
                    selected_keywords.append(cluster_keywords[best_idx])
            
            return selected_keywords
        
        except Exception as e:
            logger.warning(f"⚠️ Erreur sélection intelligente: {e}, utilisation de l'échantillonnage simple")
            # Fallback: échantillonnage simple
            import random
            sample_size = min(50, len(keywords))
            return random.sample(keywords, sample_size)
    
    async def _extend_similarity_matrix(
        self, 
        all_keywords: List[str], 
        serp_data: Dict[str, set], 
        processed_keywords: List[str]
    ) -> Dict[str, set]:
        """Étend la matrice de similarité par approximation pour les mots-clés non traités"""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Mots-clés non traités
            missing_keywords = [kw for kw in all_keywords if kw not in serp_data]
            
            if not missing_keywords:
                return serp_data
            
            logger.info(f"🔮 Extension par approximation pour {len(missing_keywords)} mots-clés")
            
            # Calcule les embeddings pour l'approximation
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            processed_embeddings = model.encode(processed_keywords, show_progress_bar=False)
            missing_embeddings = model.encode(missing_keywords, show_progress_bar=False)
            
            # Pour chaque mot-clé manquant, trouve le plus similaire dans les traités
            similarities = cosine_similarity(missing_embeddings, processed_embeddings)
            
            extended_serp_data = serp_data.copy()
            for i, missing_kw in enumerate(missing_keywords):
                # Trouve le mot-clé traité le plus similaire
                best_match_idx = similarities[i].argmax()
                best_match_keyword = processed_keywords[best_match_idx]
                
                # Copie les domaines du mot-clé le plus similaire avec un facteur de bruit
                original_domains = serp_data[best_match_keyword]
                
                # Ajoute un peu de variation pour simuler la différence
                similarity_score = similarities[i][best_match_idx]
                keep_ratio = 0.6 + (similarity_score * 0.3)  # Entre 60% et 90% des domaines
                
                import random
                domains_to_keep = random.sample(
                    list(original_domains), 
                    max(1, int(len(original_domains) * keep_ratio))
                )
                
                extended_serp_data[missing_kw] = set(domains_to_keep)
            
            logger.info(f"✅ Extension terminée: {len(extended_serp_data)} mots-clés au total")
            return extended_serp_data
            
        except Exception as e:
            logger.error(f"❌ Erreur extension matrice: {e}")
            # Fallback: domaines vides pour les mots-clés manquants
            extended_serp_data = serp_data.copy()
            missing_keywords = [kw for kw in all_keywords if kw not in serp_data]
            for kw in missing_keywords:
                extended_serp_data[kw] = set()
            return extended_serp_data
    
    def _calculate_jaccard_matrix_optimized(
        self, 
        keywords: List[str], 
        serp_data: Dict[str, set]
    ) -> Dict[str, Dict[str, float]]:
        """Calcul optimisé de la matrice de similarité Jaccard"""
        similarity_matrix = {}
        
        # Pré-calcule les données pour éviter les recalculs
        keyword_domains = {kw: serp_data.get(kw, set()) for kw in keywords}
        
        for i, kw1 in enumerate(keywords):
            similarity_matrix[kw1] = {}
            domains1 = keyword_domains[kw1]
            
            for j, kw2 in enumerate(keywords):
                if i == j:
                    similarity_matrix[kw1][kw2] = 1.0
                elif kw2 in similarity_matrix and kw1 in similarity_matrix[kw2]:
                    # Utilise la symétrie pour éviter le recalcul
                    similarity_matrix[kw1][kw2] = similarity_matrix[kw2][kw1]
                else:
                    # Calcule la similarité Jaccard
                    domains2 = keyword_domains[kw2]
                    intersection = len(domains1.intersection(domains2))
                    union = len(domains1.union(domains2))
                    similarity = intersection / union if union > 0 else 0.0
                    similarity_matrix[kw1][kw2] = similarity
        
        return similarity_matrix

    async def batch_scrape_serp_results(
        self, 
        keywords: List[str], 
        country_code: str = "FR",
        max_concurrent: int = 10
    ) -> Dict[str, List[SerpResult]]:
        """Récupère les résultats SERP pour plusieurs mots-clés en parallèle avec gestion du cache"""
        logger.info(f"🔍 Batch SERP pour {len(keywords)} mots-clés")
        
        # Sépare les mots-clés cachés et non-cachés
        cached_results = {}
        keywords_to_scrape = []
        
        for keyword in keywords:
            cached = db_manager.get_cached_serp_results(keyword, country_code)
            if cached:
                cached_results[keyword] = cached
                logger.debug(f"✅ Cache hit pour '{keyword}'")
            else:
                keywords_to_scrape.append(keyword)
        
        logger.info(f"📊 Cache: {len(cached_results)} hits, {len(keywords_to_scrape)} mots-clés à scraper")
        
        if not keywords_to_scrape:
            return cached_results
        
        # Scraping parallèle des mots-clés manquants
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_single(keyword: str) -> Tuple[str, List[SerpResult]]:
            async with semaphore:
                results = await self.scrape_serp_results(keyword, country_code)
                return keyword, results
        
        # Lance tous les appels en parallèle
        tasks = [scrape_single(keyword) for keyword in keywords_to_scrape]
        scraped_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine les résultats
        all_results = cached_results.copy()
        
        for result in scraped_results:
            if isinstance(result, Exception):
                logger.error(f"❌ Erreur batch scraping: {result}")
                continue
            keyword, serp_results = result
            all_results[keyword] = serp_results
        
        logger.info(f"✅ Batch terminé: {len(all_results)} mots-clés traités")
        return all_results

# Instance globale du service d'enrichissement
enrichment_service = EnrichmentService() 