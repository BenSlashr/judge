import os
# Configuration pour éviter les conflits avec PyTorch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from celery import Celery
from app.config import settings

# Configuration Celery
celery_app = Celery(
    "seo_keyword_classifier",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"]
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    # Configuration spécifique pour les tâches ML
    task_soft_time_limit=1800,  # 30 minutes soft limit
    task_time_limit=2100,       # 35 minutes hard limit
    worker_max_tasks_per_child=1,  # Redémarre le worker après chaque tâche (évite les fuites mémoire)
) 