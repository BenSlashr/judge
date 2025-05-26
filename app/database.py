import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlite_utils import Database
from app.config import settings
from app.models import JobStatus, JobParameters, JobResult, Keyword, Cluster, SerpResult

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or "data/app.db"  # Force l'utilisation d'app.db temporairement
        
        # Configuration SQLite pour les accès concurrents
        self.db = Database(self.db_path)
        
        # Active le mode WAL pour SQLite (meilleur support concurrent)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA cache_size=1000")
        self.db.execute("PRAGMA temp_store=MEMORY")
        self.db.execute("PRAGMA busy_timeout=30000")  # 30 secondes de timeout
        
        self._init_tables()
        self._migrate_database()
        
        logger.info(f"📁 Base de données initialisée: {self.db_path}")
    
    def _init_tables(self):
        """Initialise les tables de la base de données"""
        # Table des jobs
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                parameters TEXT NOT NULL,
                keywords_count INTEGER NOT NULL DEFAULT 0,
                clusters_count INTEGER,
                error_message TEXT,
                export_file_path TEXT
            );
            
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                keyword TEXT NOT NULL,
                search_volume INTEGER,
                cpc REAL,
                keyword_difficulty REAL,
                cluster_id INTEGER,
                cluster_name TEXT,
                is_pivot BOOLEAN DEFAULT 0,
                opportunity_score REAL,
                embedding_vector TEXT,
                
                -- Champs pour le clustering multi-appartenance
                cluster_primary INTEGER,
                cluster_primary_name TEXT,
                cluster_primary_probability REAL,
                cluster_secondary INTEGER,
                cluster_secondary_name TEXT,
                cluster_secondary_probability REAL,
                cluster_alt1 INTEGER,
                cluster_alt1_name TEXT,
                cluster_alt1_probability REAL,
                is_multi_cluster BOOLEAN DEFAULT 0,
                
                -- Clusters supplémentaires et métadonnées
                additional_clusters TEXT,  -- JSON pour clusters 4+
                total_clusters_count INTEGER DEFAULT 1,
                
                FOREIGN KEY (job_id) REFERENCES jobs (job_id)
            );
            
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                cluster_id INTEGER NOT NULL,
                cluster_name TEXT NOT NULL,
                pivot_keyword TEXT NOT NULL,
                keywords_count INTEGER NOT NULL,
                avg_search_volume REAL,
                avg_cpc REAL,
                avg_difficulty REAL,
                opportunity_score REAL,
                FOREIGN KEY (job_id) REFERENCES jobs (job_id)
            );
            
            CREATE TABLE IF NOT EXISTS serp_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL,
                country_code TEXT NOT NULL DEFAULT 'FR',
                results TEXT NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                UNIQUE(keyword, country_code)
            );
            
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_keywords_job_id ON keywords(job_id);
            CREATE INDEX IF NOT EXISTS idx_clusters_job_id ON clusters(job_id);
            CREATE INDEX IF NOT EXISTS idx_serp_cache_keyword ON serp_cache(keyword, country_code);
            CREATE INDEX IF NOT EXISTS idx_serp_cache_expires ON serp_cache(expires_at);
        """)
    
    def _migrate_database(self):
        """Migration de la base de données pour ajouter les nouvelles colonnes multi-clustering"""
        try:
            # Vérifie si les nouvelles colonnes existent
            table_info = list(self.db.execute("PRAGMA table_info(keywords)"))
            existing_columns = [row[1] for row in table_info]
            
            # Colonnes à ajouter pour le multi-clustering
            new_columns = [
                ("cluster_primary", "INTEGER"),
                ("cluster_primary_name", "TEXT"),
                ("cluster_primary_probability", "REAL"),
                ("cluster_secondary", "INTEGER"),
                ("cluster_secondary_name", "TEXT"),
                ("cluster_secondary_probability", "REAL"),
                ("cluster_alt1", "INTEGER"),
                ("cluster_alt1_name", "TEXT"),
                ("cluster_alt1_probability", "REAL"),
                ("is_multi_cluster", "BOOLEAN DEFAULT 0"),
                ("additional_clusters", "TEXT"),
                ("total_clusters_count", "INTEGER DEFAULT 1")
            ]
            
            # Ajoute les colonnes manquantes
            for column_name, column_type in new_columns:
                if column_name not in existing_columns:
                    try:
                        self.db.execute(f"ALTER TABLE keywords ADD COLUMN {column_name} {column_type}")
                        logger.info(f"✅ Colonne ajoutée: {column_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Impossible d'ajouter la colonne {column_name}: {e}")
            
            logger.info("🔄 Migration de la base de données terminée")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la migration: {e}")
            # En cas d'erreur, recréer la table keywords
            logger.info("🔧 Tentative de recréation de la table keywords")
            try:
                # Sauvegarde les données existantes
                existing_data = list(self.db.execute("SELECT * FROM keywords"))
                
                # Supprime et recrée la table
                self.db.execute("DROP TABLE IF EXISTS keywords")
                
                # Recrée la table avec la nouvelle structure
                self.db.execute("""
                    CREATE TABLE keywords (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL,
                        keyword TEXT NOT NULL,
                        search_volume INTEGER,
                        cpc REAL,
                        keyword_difficulty REAL,
                        cluster_id INTEGER,
                        cluster_name TEXT,
                        is_pivot BOOLEAN DEFAULT 0,
                        opportunity_score REAL,
                        embedding_vector TEXT,
                        
                        -- Champs pour le clustering multi-appartenance
                        cluster_primary INTEGER,
                        cluster_primary_name TEXT,
                        cluster_primary_probability REAL,
                        cluster_secondary INTEGER,
                        cluster_secondary_name TEXT,
                        cluster_secondary_probability REAL,
                        cluster_alt1 INTEGER,
                        cluster_alt1_name TEXT,
                        cluster_alt1_probability REAL,
                        is_multi_cluster BOOLEAN DEFAULT 0,
                        
                        -- Clusters supplémentaires et métadonnées
                        additional_clusters TEXT,  -- JSON pour clusters 4+
                        total_clusters_count INTEGER DEFAULT 1,
                        
                        FOREIGN KEY (job_id) REFERENCES jobs (job_id)
                    )
                """)
                
                # Restaure les données existantes (avec les nouvelles colonnes à NULL)
                if existing_data:
                    # Prépare les données pour l'insertion avec les nouvelles colonnes
                    for row in existing_data:
                        # Convertit le Row en dict et ajoute les nouvelles colonnes
                        row_dict = dict(row)
                        # Ajoute les nouvelles colonnes avec des valeurs par défaut
                        new_fields = {
                            "cluster_primary": row_dict.get("cluster_id"),
                            "cluster_primary_name": row_dict.get("cluster_name"),
                            "cluster_primary_probability": 1.0 if row_dict.get("cluster_id") is not None else None,
                            "cluster_secondary": None,
                            "cluster_secondary_name": None,
                            "cluster_secondary_probability": None,
                            "cluster_alt1": None,
                            "cluster_alt1_name": None,
                            "cluster_alt1_probability": None,
                            "is_multi_cluster": False,
                            "additional_clusters": None,
                            "total_clusters_count": 1
                        }
                        row_dict.update(new_fields)
                        
                        # Insert avec gestion d'erreur
                        try:
                            self.db["keywords"].insert(row_dict)
                        except Exception as insert_error:
                            logger.warning(f"⚠️ Erreur insertion ligne: {insert_error}")
                
                # Recrée l'index
                self.db.execute("CREATE INDEX IF NOT EXISTS idx_keywords_job_id ON keywords(job_id)")
                
                logger.info("✅ Table keywords recréée avec succès")
                
            except Exception as recreate_error:
                logger.error(f"❌ Erreur lors de la recréation: {recreate_error}")
                # En dernier recours, crée juste une table vide
                self.db.execute("DROP TABLE IF EXISTS keywords")
                self._init_tables()
                logger.warning("⚠️ Table keywords recréée vide")
    
    def create_job(self, job_id: str, parameters: JobParameters, keywords_count: int) -> None:
        """Crée un nouveau job"""
        def _create():
            self.db["jobs"].insert({
                "job_id": job_id,
                "status": JobStatus.PENDING,
                "parameters": parameters.model_dump_json(),
                "keywords_count": keywords_count
            })
            
        self._execute_with_retry(_create)
        logger.info(f"✅ Job {job_id} créé avec {keywords_count} mots-clés")
    
    def update_job_status(self, job_id: str, status: JobStatus, error_message: str = None) -> None:
        """Met à jour le statut d'un job"""
        def _update():
            update_data = {"status": status}
            if status == JobStatus.COMPLETED:
                update_data["completed_at"] = datetime.now().isoformat()
            if error_message:
                update_data["error_message"] = error_message
            
            self.db["jobs"].update(job_id, update_data)
            
        self._execute_with_retry(_update)
        logger.info(f"📝 Job {job_id} mis à jour: {status}")
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un job par son ID"""
        try:
            return dict(self.db["jobs"].get(job_id))
        except:
            return None
    
    def get_recent_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Récupère les jobs récents"""
        return list(self.db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", 
            [limit]
        ))
    
    def save_keywords(self, job_id: str, keywords: List[Keyword]) -> None:
        """Sauvegarde les mots-clés d'un job"""
        data = []
        for kw in keywords:
            data.append({
                "job_id": job_id,
                "keyword": kw.keyword,
                "search_volume": kw.search_volume,
                "cpc": kw.cpc,
                "keyword_difficulty": kw.keyword_difficulty,
                "cluster_id": kw.cluster_id,
                "cluster_name": kw.cluster_name,
                "is_pivot": kw.is_pivot,
                "opportunity_score": kw.opportunity_score,
                
                # Champs multi-cluster
                "cluster_primary": kw.cluster_primary,
                "cluster_primary_name": kw.cluster_primary_name,
                "cluster_primary_probability": kw.cluster_primary_probability,
                "cluster_secondary": kw.cluster_secondary,
                "cluster_secondary_name": kw.cluster_secondary_name,
                "cluster_secondary_probability": kw.cluster_secondary_probability,
                "cluster_alt1": kw.cluster_alt1,
                "cluster_alt1_name": kw.cluster_alt1_name,
                "cluster_alt1_probability": kw.cluster_alt1_probability,
                "additional_clusters": json.dumps(kw.additional_clusters) if kw.additional_clusters else None,
                "is_multi_cluster": kw.is_multi_cluster,
                "total_clusters_count": kw.total_clusters_count
            })
        
        self.db["keywords"].insert_all(data)
        logger.info(f"Sauvegardé {len(keywords)} mots-clés pour le job {job_id}")
    
    def save_clusters(self, job_id: str, clusters: List[Cluster]) -> None:
        """Sauvegarde les clusters d'un job"""
        data = []
        for cluster in clusters:
            data.append({
                "job_id": job_id,
                "cluster_id": cluster.cluster_id,
                "cluster_name": cluster.cluster_name,
                "pivot_keyword": cluster.pivot_keyword,
                "keywords_count": cluster.keywords_count,
                "avg_search_volume": cluster.avg_search_volume,
                "avg_cpc": cluster.avg_cpc,
                "avg_difficulty": cluster.avg_difficulty,
                "opportunity_score": cluster.opportunity_score
            })
        
        self.db["clusters"].insert_all(data)
        
        # Met à jour le nombre de clusters dans la table jobs
        self.db["jobs"].update(job_id, {"clusters_count": len(clusters)})
        logger.info(f"Sauvegardé {len(clusters)} clusters pour le job {job_id}")
    
    def get_job_keywords(self, job_id: str) -> List[Dict[str, Any]]:
        """Récupère les mots-clés d'un job"""
        try:
            rows = self.db.execute(
                """SELECT keyword, search_volume, cpc, keyword_difficulty, cluster_id, cluster_name, is_pivot, opportunity_score,
                   cluster_primary, cluster_primary_name, cluster_primary_probability,
                   cluster_secondary, cluster_secondary_name, cluster_secondary_probability,
                   cluster_alt1, cluster_alt1_name, cluster_alt1_probability, 
                   additional_clusters, is_multi_cluster, total_clusters_count
                   FROM keywords WHERE job_id = ? ORDER BY keyword""",
                [job_id]
            ).fetchall()
            
            keywords = []
            for row in rows:
                keywords.append({
                    "keyword": row[0],
                    "search_volume": row[1],
                    "cpc": row[2],
                    "keyword_difficulty": row[3],
                    "cluster_id": row[4],
                    "cluster_name": row[5],
                    "is_pivot": bool(row[6]) if row[6] is not None else False,
                    "opportunity_score": row[7],
                    
                    # Champs multi-cluster
                    "cluster_primary": row[8],
                    "cluster_primary_name": row[9],
                    "cluster_primary_probability": row[10],
                    "cluster_secondary": row[11],
                    "cluster_secondary_name": row[12],
                    "cluster_secondary_probability": row[13],
                    "cluster_alt1": row[14],
                    "cluster_alt1_name": row[15],
                    "cluster_alt1_probability": row[16],
                    "additional_clusters": json.loads(row[17]) if row[17] else None,
                    "is_multi_cluster": bool(row[18]) if row[18] is not None else False,
                    "total_clusters_count": row[19] if row[19] is not None else 1
                })
            
            return keywords
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des mots-clés {job_id}: {e}")
            return []
    
    def get_job_clusters(self, job_id: str) -> List[Dict[str, Any]]:
        """Récupère les clusters d'un job"""
        try:
            rows = self.db.execute(
                "SELECT cluster_id, cluster_name, pivot_keyword, keywords_count, avg_search_volume, avg_cpc, avg_difficulty, opportunity_score FROM clusters WHERE job_id = ? ORDER BY cluster_id",
                [job_id]
            ).fetchall()
            
            clusters = []
            for row in rows:
                clusters.append({
                    "cluster_id": row[0],
                    "cluster_name": row[1],
                    "pivot_keyword": row[2],
                    "keywords_count": row[3],
                    "avg_search_volume": row[4],
                    "avg_cpc": row[5],
                    "avg_difficulty": row[6],
                    "opportunity_score": row[7]
                })
            
            return clusters
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des clusters {job_id}: {e}")
            return []
    
    def cache_serp_results(self, keyword: str, country_code: str, results: List[SerpResult], ttl_days: int) -> None:
        """Met en cache les résultats SERP"""
        expires_at = datetime.now() + timedelta(days=ttl_days)
        
        self.db["serp_cache"].insert({
            "keyword": keyword,
            "country_code": country_code,
            "results": json.dumps([r.model_dump() for r in results]),
            "expires_at": expires_at.isoformat()
        }, replace=True)
    
    def get_cached_serp_results(self, keyword: str, country_code: str) -> Optional[List[SerpResult]]:
        """Récupère les résultats SERP depuis le cache"""
        try:
            row = self.db.execute(
                "SELECT results FROM serp_cache WHERE keyword = ? AND country_code = ? AND expires_at > ?",
                [keyword, country_code, datetime.now().isoformat()]
            ).fetchone()
            
            if row:
                # Gérer à la fois les tuples et les objets Row
                if isinstance(row, tuple):
                    results_data = json.loads(row[0])
                else:
                    results_data = json.loads(row["results"])
                
                # Création sécurisée des objets SerpResult depuis le cache
                safe_results = []
                for r in results_data:
                    try:
                        # Nettoyage des données avant création
                        clean_r = {
                            "keyword": str(r.get("keyword", "")),
                            "url": str(r.get("url", "")),
                            "title": str(r.get("title", "")),
                            "description": str(r.get("description", "")),
                            "position": int(r.get("position", 1)),
                            "domain": str(r.get("domain", ""))
                        }
                        safe_results.append(SerpResult(**clean_r))
                    except Exception as e:
                        logger.warning(f"Erreur lors de la création de SerpResult depuis le cache: {e}, données: {r}")
                        continue
                
                return safe_results
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du cache SERP pour {keyword}: {e}")
        
        return None
    
    def cleanup_expired_cache(self) -> None:
        """Nettoie le cache SERP expiré"""
        deleted = self.db.execute(
            "DELETE FROM serp_cache WHERE expires_at < ?",
            [datetime.now().isoformat()]
        ).rowcount
        
        if deleted > 0:
            logger.info(f"Nettoyé {deleted} entrées expirées du cache SERP")
    
    def set_export_file_path(self, job_id: str, file_path: str) -> None:
        """Définit le chemin du fichier d'export pour un job"""
        self.db["jobs"].update(job_id, {"export_file_path": file_path})

    def _execute_with_retry(self, operation_func, max_retries: int = 3):
        """Exécute une opération avec retry en cas de database locked"""
        import time
        
        for attempt in range(max_retries):
            try:
                return operation_func()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"⚠️ Base verrouillée, tentative {attempt + 1}/{max_retries}")
                    time.sleep(0.1 * (attempt + 1))  # Délai croissant
                    continue
                else:
                    raise

# Instance globale du gestionnaire de base de données
db_manager = DatabaseManager() 