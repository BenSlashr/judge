import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys (directement depuis .env)
    datafor_seo_login: Optional[str] = None
    datafor_seo_password: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_ads_developer_token: Optional[str] = None
    google_ads_client_id: Optional[str] = None
    google_ads_client_secret: Optional[str] = None
    google_ads_refresh_token: Optional[str] = None
    
    # Database
    database_url: str = "sqlite:///./data/seo_classifier.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Security
    environment: str = "development"
    
    # Logging
    log_level: str = "INFO"
    
    # Application settings
    max_keywords_per_upload: int = 50000
    default_serp_cache_ttl_days: int = 30
    processing_rate_pivot: int = 1000  # keywords per minute
    processing_rate_full_serp: int = 300  # keywords per minute
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore les champs supplémentaires

# Instance globale des paramètres
settings = Settings() 