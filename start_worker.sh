#!/bin/bash

echo "🚀 Démarrage du worker Celery..."
echo "ROOT_PATH: $ROOT_PATH"
echo "REDIS_URL: $REDIS_URL"

# Test de connexion Redis
echo "🔍 Test de connexion Redis..."
python -c "
import redis
try:
    r = redis.from_url('$REDIS_URL')
    r.ping()
    print('✅ Redis accessible')
except Exception as e:
    print(f'❌ Redis inaccessible: {e}')
    exit(1)
"

# Test d'import des tâches
echo "🔍 Test d'import des tâches..."
python -c "
try:
    from app.tasks import ping_task, test_task, process_keywords_task
    print('✅ Tâches importées avec succès')
except Exception as e:
    print(f'❌ Erreur import tâches: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo "🎯 Lancement du worker Celery..."
celery -A app.celery_app worker --loglevel=info --concurrency=2 