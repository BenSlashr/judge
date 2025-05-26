#!/bin/bash

echo "🔄 Arrêt du serveur existant..."
pkill -f "uvicorn.*main:app" || true
sleep 2

echo "🚀 Démarrage du nouveau serveur..."
cd /Users/benoit/Judge
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &

echo "✅ Serveur redémarré" 