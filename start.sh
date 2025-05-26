#!/bin/bash

# Script de démarrage pour SEO Keyword Classifier
# Usage: ./start.sh [mode]
# Modes: dev (défaut), prod, docker

MODE=${1:-dev}

echo "🚀 Démarrage de SEO Keyword Classifier en mode: $MODE"

case $MODE in
    "dev")
        echo "📦 Mode développement"
        
        # Vérifie que Redis est démarré
        if ! pgrep -x "redis-server" > /dev/null; then
            echo "⚠️  Redis n'est pas démarré. Démarrage automatique..."
            redis-server --daemonize yes
        fi
        
        # Crée les dossiers nécessaires
        mkdir -p data/{uploads,exports} logs static templates
        
        # Lance l'application en mode développement
        echo "🌐 Démarrage de l'application web sur http://localhost:8000"
        uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
        
        # Lance le worker Celery
        echo "⚙️  Démarrage du worker Celery"
        celery -A app.celery_app worker --loglevel=info &
        
        echo "✅ Application démarrée en mode développement"
        echo "📖 Consultez le README.md pour plus d'informations"
        echo "🛑 Appuyez sur Ctrl+C pour arrêter"
        
        # Attend les processus
        wait
        ;;
        
    "prod")
        echo "🏭 Mode production"
        
        # Vérifie les variables d'environnement
        if [ ! -f ".env" ]; then
            echo "❌ Fichier .env manquant. Copiez env.example vers .env et configurez-le."
            exit 1
        fi
        
        # Lance avec Gunicorn pour la production
        echo "🌐 Démarrage de l'application en production"
        gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 &
        
        # Lance le worker Celery
        echo "⚙️  Démarrage du worker Celery"
        celery -A app.celery_app worker --loglevel=warning --concurrency=2 &
        
        # Lance Celery Beat pour les tâches périodiques
        echo "⏰ Démarrage de Celery Beat"
        celery -A app.celery_app beat --loglevel=warning &
        
        echo "✅ Application démarrée en mode production"
        wait
        ;;
        
    "docker")
        echo "🐳 Mode Docker"
        
        # Vérifie que Docker est installé
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker n'est pas installé"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            echo "❌ Docker Compose n'est pas installé"
            exit 1
        fi
        
        # Lance avec Docker Compose
        echo "🚀 Démarrage avec Docker Compose"
        docker-compose up --build
        ;;
        
    "stop")
        echo "🛑 Arrêt de l'application"
        
        # Arrête les processus Python
        pkill -f "uvicorn app.main:app"
        pkill -f "celery -A app.celery_app"
        pkill -f "gunicorn app.main:app"
        
        echo "✅ Application arrêtée"
        ;;
        
    *)
        echo "❌ Mode inconnu: $MODE"
        echo "Modes disponibles: dev, prod, docker, stop"
        exit 1
        ;;
esac 