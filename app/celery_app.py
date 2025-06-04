import os
# Configuration pour éviter les conflits avec PyTorch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from celery import Celery
from app.config import settings

# Configuration Celery
celery_app = Celery(
    "seo_classifier",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=['app.tasks']
)

# Configuration complète
celery_app.conf.update(
    # Configuration des tâches
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/Paris',
    enable_utc=True,
    
    # Configuration de connexion Redis
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    
    # Passe le ROOT_PATH aux tâches
    task_routes={
        'app.tasks.*': {'queue': 'default'}
    },
    
    # Variables d'environnement pour les tâches
    worker_hijack_root_logger=False,
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
    
    # Configuration des tâches et performance
    task_track_started=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    # Configuration spécifique pour les tâches ML
    task_soft_time_limit=1800,  # 30 minutes soft limit
    task_time_limit=2100,       # 35 minutes hard limit
    worker_max_tasks_per_child=1,  # Redémarre le worker après chaque tâche (évite les fuites mémoire)
)

# Ajoute ROOT_PATH comme variable globale pour les tâches
celery_app.conf.task_default_kwargs = {
    'root_path': settings.root_path
} 