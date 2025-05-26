from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime

class SerpMode(str, Enum):
    FULL_SERP = "full_serp"
    PIVOT_ONLY = "pivot_only"
    NONE = "none"

class ClusteringAlgorithm(str, Enum):
    AGGLOMERATIVE = "agglomerative"
    HDBSCAN = "hdbscan"
    LOUVAIN = "louvain"

class ExportFormat(str, Enum):
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobParameters(BaseModel):
    serp_mode: SerpMode = SerpMode.NONE
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)  # Poids embeddings vs SERP
    clustering_algorithm: ClusteringAlgorithm = ClusteringAlgorithm.HDBSCAN
    min_cluster_size: int = Field(default=5, ge=2)
    serp_cache_ttl_days: int = Field(default=30, ge=1)
    export_format: ExportFormat = ExportFormat.CSV
    country_code: str = Field(default="FR", min_length=2, max_length=2)
    language: str = Field(default="fr", min_length=2, max_length=5)
    enable_multi_cluster: bool = Field(default=False)  # Active le clustering multi-appartenance
    primary_threshold: float = Field(default=0.6, ge=0.1, le=1.0)  # Seuil pour cluster principal
    secondary_threshold: float = Field(default=0.4, ge=0.1, le=1.0)  # Seuil pour cluster secondaire
    max_clusters_per_keyword: int = Field(default=3, ge=1, le=10)  # Nombre maximum de clusters par mot-clé
    min_probability_threshold: float = Field(default=0.15, ge=0.05, le=0.5)  # Probabilité minimum pour être considéré
    
    # Paramètres sensibilité numérique
    enable_numeric_features: bool = Field(default=False, description="Active les features numériques")
    numeric_method: str = Field(default="enhanced_embeddings", description="Méthode: 'enhanced_embeddings' ou 'hybrid_distance'")
    numeric_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Poids des features numériques (0=désactivé, 1=maximum)")
    numeric_sensitivity: float = Field(default=0.5, ge=0.1, le=2.0, description="Sensibilité numérique pour distance hybride")
    
    # Paramètres optimisation SERP
    serp_max_concurrent: int = Field(default=10, ge=1, le=50, description="Nombre maximum d'appels SERP parallèles")
    serp_enable_sampling: bool = Field(default=False, description="Active l'échantillonnage intelligent pour SERP")
    serp_sampling_ratio: float = Field(default=0.3, ge=0.1, le=1.0, description="Ratio d'échantillonnage SERP (0.3 = 30%)")
    serp_enable_smart_clustering: bool = Field(default=True, description="Active le clustering intelligent pour sélection SERP")

class Keyword(BaseModel):
    keyword: str
    search_volume: Optional[int] = None
    cpc: Optional[float] = None
    keyword_difficulty: Optional[float] = None
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    is_pivot: bool = False
    opportunity_score: Optional[float] = None
    
    # Champs pour le clustering multi-appartenance
    cluster_primary: Optional[int] = None  # Cluster principal (prob >= primary_threshold)
    cluster_primary_name: Optional[str] = None
    cluster_primary_probability: Optional[float] = None
    
    cluster_secondary: Optional[int] = None  # Cluster secondaire (prob >= secondary_threshold)
    cluster_secondary_name: Optional[str] = None
    cluster_secondary_probability: Optional[float] = None
    
    cluster_alt1: Optional[int] = None  # Cluster alternatif 1
    cluster_alt1_name: Optional[str] = None
    cluster_alt1_probability: Optional[float] = None
    
    # Clusters supplémentaires (JSON flexible pour 4+ clusters)
    additional_clusters: Optional[List[Dict[str, Any]]] = None  # [{"id": 3, "name": "...", "probability": 0.25}, ...]
    
    is_multi_cluster: bool = False  # Indique si le mot-clé appartient à plusieurs clusters
    total_clusters_count: int = 1  # Nombre total de clusters assignés

class Cluster(BaseModel):
    cluster_id: int
    cluster_name: str
    pivot_keyword: str
    keywords_count: int
    avg_search_volume: Optional[float] = None
    avg_cpc: Optional[float] = None
    avg_difficulty: Optional[float] = None
    opportunity_score: Optional[float] = None

class JobProgress(BaseModel):
    progress: float = Field(ge=0.0, le=100.0)
    state: JobStatus
    message: str
    current_step: Optional[str] = None
    estimated_time_remaining: Optional[int] = None  # en secondes

class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    parameters: JobParameters
    keywords_count: int
    clusters_count: Optional[int] = None
    clusters: Optional[List[Cluster]] = None
    keywords: Optional[List[Keyword]] = None
    export_url: Optional[str] = None
    error_message: Optional[str] = None

class JobSummary(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    keywords_count: int
    clusters_count: Optional[int] = None
    parameters: JobParameters

class UMAPVisualization(BaseModel):
    x: List[float]
    y: List[float]
    labels: List[str]  # cluster names
    keywords: List[str]
    colors: List[str]  # hex colors for clusters

class SerpResult(BaseModel):
    keyword: str
    url: str
    title: str
    description: str = ""  # Valeur par défaut pour éviter les erreurs de validation
    position: int
    domain: str
    
    @validator('description', pre=True, always=True)
    def validate_description(cls, v):
        """Assure que la description n'est jamais None et est toujours une string"""
        if v is None:
            return ""
        if not isinstance(v, str):
            try:
                return str(v) if v else ""
            except:
                return ""
        return v
    
    @validator('title', 'url', 'domain', 'keyword', pre=True, always=True)
    def validate_string_fields(cls, v):
        """Assure que les champs string ne sont jamais None et sont toujours des strings"""
        if v is None:
            return ""
        if not isinstance(v, str):
            try:
                return str(v) if v else ""
            except:
                return ""
        return v
    
    @validator('position', pre=True)
    def validate_position(cls, v):
        """Assure que position est un entier valide"""
        if v is None:
            return 1
        try:
            return int(v)
        except:
            return 1 