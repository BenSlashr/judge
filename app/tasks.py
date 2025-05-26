import logging
import asyncio
import uuid
import pandas as pd
from typing import List, Dict, Any
from celery import current_task
from celery.exceptions import Ignore

from app.celery_app import celery_app
from app.models import JobParameters, Keyword, SerpMode, JobStatus
from app.database import db_manager
from app.services.enrichment import enrichment_service
from app.services.clustering import clustering_service
from app.utils.file_processing import process_uploaded_file, export_results

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def process_keywords_task(self, job_id: str, file_path: str, parameters_dict: Dict[str, Any]):
    """Tâche principale de traitement des mots-clés"""
    
    logger.info(f"🚀 DÉBUT JOB {job_id} - Paramètres: {parameters_dict}")
    
    try:
        # Parse les paramètres
        parameters = JobParameters(**parameters_dict)
        logger.info(f"✅ JOB {job_id} - Paramètres parsés: {parameters}")
        
        # Met à jour le statut
        db_manager.update_job_status(job_id, JobStatus.PROCESSING)
        logger.info(f"📝 JOB {job_id} - Statut mis à jour: PROCESSING")
        
        # Étape 1: Lecture et nettoyage du fichier
        logger.info(f"📁 JOB {job_id} - ÉTAPE 1: Lecture du fichier {file_path}")
        self.update_state(
            state='PROGRESS',
            meta={'progress': 5, 'message': 'Lecture du fichier de mots-clés...'}
        )
        
        keywords = process_uploaded_file(file_path)
        logger.info(f"✅ JOB {job_id} - ÉTAPE 1 TERMINÉE: {len(keywords)} mots-clés extraits")
        
        # Étape 2: Enrichissement des métriques
        logger.info(f"💰 JOB {job_id} - ÉTAPE 2: Enrichissement des métriques")
        self.update_state(
            state='PROGRESS',
            meta={'progress': 20, 'message': 'Enrichissement des métriques SEO...'}
        )
        
        # Utilise asyncio pour les opérations asynchrones
        logger.info(f"🔄 JOB {job_id} - Création de la boucle asyncio")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            logger.info(f"⚙️ JOB {job_id} - Lancement enrichissement asynchrone")
            enriched_keywords = loop.run_until_complete(
                enrichment_service.enrich_keywords_metrics(keywords)
            )
            logger.info(f"✅ JOB {job_id} - ÉTAPE 2 TERMINÉE: Enrichissement de {len(enriched_keywords)} mots-clés")
        finally:
            loop.close()
            logger.info(f"🔚 JOB {job_id} - Boucle asyncio fermée")
        
        # Étape 3: Génération des embeddings
        logger.info(f"🧠 JOB {job_id} - ÉTAPE 3: Génération des embeddings")
        self.update_state(
            state='PROGRESS',
            meta={'progress': 40, 'message': 'Génération des embeddings...'}
        )
        
        keyword_texts = [kw.keyword for kw in enriched_keywords]
        logger.info(f"📝 JOB {job_id} - Liste de {len(keyword_texts)} mots-clés pour embeddings")
        
        embeddings = clustering_service.generate_embeddings(
            keyword_texts,
            enable_numeric_features=parameters.enable_numeric_features,
            numeric_weight=parameters.numeric_weight
        )
        logger.info(f"✅ JOB {job_id} - ÉTAPE 3 TERMINÉE: Embeddings générés {embeddings.shape}")
        
        # Étape 4: Clustering primaire (embeddings uniquement)
        logger.info(f"🔗 JOB {job_id} - ÉTAPE 4: Clustering primaire")
        self.update_state(
            state='PROGRESS',
            meta={'progress': 55, 'message': 'Clustering primaire...'}
        )
        
        # Détermine si on utilise la distance numérique pour le clustering primaire
        use_numeric_distance = (parameters.enable_numeric_features and 
                               parameters.numeric_method == "hybrid_distance")
        
        primary_labels, primary_clusters, _ = clustering_service.cluster_keywords(
            keyword_texts,
            embeddings,
            serp_similarity_matrix=None,
            algorithm=parameters.clustering_algorithm,
            min_cluster_size=parameters.min_cluster_size,
            alpha=1.0,  # 100% embeddings pour le clustering primaire
            enable_numeric_distance=use_numeric_distance,
            numeric_sensitivity=parameters.numeric_sensitivity
        )
        logger.info(f"✅ JOB {job_id} - ÉTAPE 4 TERMINÉE: {len(primary_clusters)} clusters créés")
        
        # Étape 5: Scraping SERP (si activé)
        logger.info(f"🌐 JOB {job_id} - ÉTAPE 5: Analyse SERP (mode: {parameters.serp_mode})")
        serp_similarity_matrix = None
        
        if parameters.serp_mode != SerpMode.NONE:
            logger.info(f"🔍 JOB {job_id} - SERP activé, début du scraping")
            self.update_state(
                state='PROGRESS',
                meta={'progress': 65, 'message': 'Scraping des SERPs...'}
            )
            
            # Détermine quels mots-clés scraper
            keywords_to_scrape = []
            
            if parameters.serp_mode == SerpMode.FULL_SERP:
                keywords_to_scrape = keyword_texts
                logger.info(f"📊 JOB {job_id} - Mode FULL_SERP: {len(keywords_to_scrape)} mots-clés à scraper")
            elif parameters.serp_mode == SerpMode.PIVOT_ONLY:
                # Scrape seulement les mots-clés pivots
                pivot_keywords = [cluster.pivot_keyword for cluster in primary_clusters]
                keywords_to_scrape = pivot_keywords
                logger.info(f"🎯 JOB {job_id} - Mode PIVOT_ONLY: {len(keywords_to_scrape)} mots-clés pivots à scraper")
            
            if keywords_to_scrape:
                logger.info(f"🔄 JOB {job_id} - Création nouvelle boucle asyncio pour SERP")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    serp_similarity_matrix = loop.run_until_complete(
                        enrichment_service.calculate_serp_similarity(
                            keywords_to_scrape, 
                            parameters.country_code,
                            max_concurrent=parameters.serp_max_concurrent,
                            enable_sampling=parameters.serp_enable_sampling,
                            sampling_ratio=parameters.serp_sampling_ratio,
                            enable_smart_clustering=parameters.serp_enable_smart_clustering
                        )
                    )
                    logger.info(f"✅ JOB {job_id} - ÉTAPE 5 TERMINÉE: Matrice SERP calculée")
                finally:
                    loop.close()
                    logger.info(f"🔚 JOB {job_id} - Boucle asyncio SERP fermée")
        else:
            logger.info(f"⏭️ JOB {job_id} - ÉTAPE 5 IGNORÉE: SERP désactivé")
        
        # Étape 6: Clustering final avec données SERP
        logger.info(f"🎯 JOB {job_id} - ÉTAPE 6: Clustering final")
        self.update_state(
            state='PROGRESS',
            meta={'progress': 80, 'message': 'Clustering final...'}
        )
        
        final_labels, final_clusters, cluster_probabilities = clustering_service.cluster_keywords(
            keyword_texts,
            embeddings,
            serp_similarity_matrix=serp_similarity_matrix,
            algorithm=parameters.clustering_algorithm,
            min_cluster_size=parameters.min_cluster_size,
            alpha=parameters.alpha,
            enable_multi_cluster=parameters.enable_multi_cluster,
            primary_threshold=parameters.primary_threshold,
            secondary_threshold=parameters.secondary_threshold,
            enable_numeric_distance=use_numeric_distance,
            numeric_sensitivity=parameters.numeric_sensitivity
        )
        logger.info(f"✅ JOB {job_id} - ÉTAPE 6 TERMINÉE: {len(final_clusters)} clusters finaux")
        
        # Met à jour les mots-clés avec les labels de cluster
        logger.info(f"🏷️ JOB {job_id} - Mise à jour des labels des mots-clés")
        
        # Traitement du multi-clustering si activé
        multi_cluster_assignments = None
        if parameters.enable_multi_cluster and cluster_probabilities is not None:
            logger.info(f"🔀 JOB {job_id} - Traitement multi-cluster")
            multi_cluster_assignments = clustering_service.assign_multi_clusters(
                keyword_texts,
                cluster_probabilities,
                parameters.primary_threshold,
                parameters.secondary_threshold,
                parameters.max_clusters_per_keyword,
                parameters.min_probability_threshold
            )
        
        for i, (keyword, cluster_id) in enumerate(zip(enriched_keywords, final_labels)):
            keyword.cluster_id = cluster_id if cluster_id >= 0 else None
            
            # Trouve le nom du cluster
            cluster_name = next(
                (c.cluster_name for c in final_clusters if c.cluster_id == cluster_id),
                "Bruit" if cluster_id == -1 else f"Cluster {cluster_id}"
            )
            keyword.cluster_name = cluster_name
            
            # Détermine si c'est un mot-clé pivot
            keyword.is_pivot = any(
                c.pivot_keyword == keyword.keyword for c in final_clusters
            )
            
            # Applique les assignations multi-cluster si disponibles
            if multi_cluster_assignments and i < len(multi_cluster_assignments):
                assignment = multi_cluster_assignments[i]
                
                keyword.cluster_primary = assignment.get('cluster_primary')
                keyword.cluster_primary_probability = assignment.get('cluster_primary_probability')
                keyword.cluster_secondary = assignment.get('cluster_secondary')
                keyword.cluster_secondary_probability = assignment.get('cluster_secondary_probability')
                keyword.cluster_alt1 = assignment.get('cluster_alt1')
                keyword.cluster_alt1_probability = assignment.get('cluster_alt1_probability')
                keyword.additional_clusters = assignment.get('additional_clusters')
                keyword.is_multi_cluster = assignment.get('is_multi_cluster', False)
                keyword.total_clusters_count = assignment.get('total_clusters_count', 1)
                
                # Met à jour les noms des clusters multiples
                if keyword.cluster_primary is not None:
                    keyword.cluster_primary_name = next(
                        (c.cluster_name for c in final_clusters if c.cluster_id == keyword.cluster_primary),
                        f"Cluster {keyword.cluster_primary}"
                    )
                if keyword.cluster_secondary is not None:
                    keyword.cluster_secondary_name = next(
                        (c.cluster_name for c in final_clusters if c.cluster_id == keyword.cluster_secondary),
                        f"Cluster {keyword.cluster_secondary}"
                    )
                if keyword.cluster_alt1 is not None:
                    keyword.cluster_alt1_name = next(
                        (c.cluster_name for c in final_clusters if c.cluster_id == keyword.cluster_alt1),
                        f"Cluster {keyword.cluster_alt1}"
                    )
                
                # Met à jour les noms des clusters supplémentaires
                if keyword.additional_clusters:
                    for cluster_info in keyword.additional_clusters:
                        cluster_info['cluster_name'] = next(
                            (c.cluster_name for c in final_clusters if c.cluster_id == cluster_info['cluster_id']),
                            f"Cluster {cluster_info['cluster_id']}"
                        )
        
        # Étape 7: Nommage des clusters avec AI
        logger.info(f"🤖 JOB {job_id} - ÉTAPE 7: Nommage intelligent des clusters")
        self.update_state(
            state='PROGRESS',
            meta={'progress': 90, 'message': 'Nommage intelligent des clusters...'}
        )
        
        named_clusters = clustering_service.name_clusters_with_ai(final_clusters, enriched_keywords)
        logger.info(f"✅ JOB {job_id} - ÉTAPE 7 TERMINÉE: Clusters nommés")
        
        # Met à jour les noms dans les mots-clés
        cluster_name_map = {c.cluster_id: c.cluster_name for c in named_clusters}
        for keyword in enriched_keywords:
            if keyword.cluster_id is not None and keyword.cluster_id in cluster_name_map:
                keyword.cluster_name = cluster_name_map[keyword.cluster_id]
        
        # Étape 8: Calcul des scores d'opportunité
        logger.info(f"📊 JOB {job_id} - ÉTAPE 8: Calcul des scores d'opportunité")
        enriched_keywords = clustering_service.calculate_opportunity_scores(enriched_keywords)
        logger.info(f"✅ JOB {job_id} - ÉTAPE 8 TERMINÉE: Scores calculés")
        
        # Étape 9: Sauvegarde en base
        logger.info(f"💾 JOB {job_id} - ÉTAPE 9: Sauvegarde en base")
        self.update_state(
            state='PROGRESS',
            meta={'progress': 95, 'message': 'Sauvegarde des résultats...'}
        )
        
        db_manager.save_keywords(job_id, enriched_keywords)
        db_manager.save_clusters(job_id, named_clusters)
        logger.info(f"✅ JOB {job_id} - ÉTAPE 9 TERMINÉE: Données sauvegardées")
        
        # Étape 10: Export du fichier
        logger.info(f"📤 JOB {job_id} - ÉTAPE 10: Export du fichier")
        export_file_path = export_results(
            job_id, 
            enriched_keywords, 
            named_clusters, 
            parameters.export_format
        )
        
        db_manager.set_export_file_path(job_id, export_file_path)
        logger.info(f"✅ JOB {job_id} - ÉTAPE 10 TERMINÉE: Fichier exporté vers {export_file_path}")
        
        # Finalisation
        logger.info(f"🏁 JOB {job_id} - FINALISATION")
        db_manager.update_job_status(job_id, JobStatus.COMPLETED)
        
        logger.info(f"🎉 JOB {job_id} TERMINÉ AVEC SUCCÈS")
        
        return {
            'job_id': job_id,
            'status': 'completed',
            'keywords_count': len(enriched_keywords),
            'clusters_count': len(named_clusters),
            'export_file': export_file_path
        }
        
    except Exception as e:
        logger.error(f"❌ ERREUR DANS LE JOB {job_id}: {str(e)}", exc_info=True)
        
        # Met à jour le statut d'erreur
        db_manager.update_job_status(job_id, JobStatus.FAILED, str(e))
        
        # Relance l'exception pour Celery
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise Ignore()

@celery_app.task
def cleanup_expired_cache():
    """Tâche de nettoyage du cache SERP expiré"""
    try:
        db_manager.cleanup_expired_cache()
        logger.info("Nettoyage du cache terminé")
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage du cache: {e}")

# Configuration des tâches périodiques
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'cleanup-expired-cache': {
        'task': 'app.tasks.cleanup_expired_cache',
        'schedule': crontab(hour=2, minute=0),  # Tous les jours à 2h00
    },
} 