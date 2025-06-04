import os
import uuid
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.models import (
    JobParameters, JobProgress, JobResult, JobSummary, 
    JobStatus, UMAPVisualization, Keyword, Cluster
)
from app.database import db_manager
from app.celery_app import celery_app
# Import des tâches fait de manière lazy pour éviter les imports circulaires

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,  # Changé temporairement pour plus de détails
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="SEO Keyword Classifier",
    description="Outil d'analyse et de clustering de mots-clés SEO",
    version="1.0.0",
    root_path=settings.root_path
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware pour gérer les headers de proxy
class ProxyHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Traite les headers X-Forwarded-* pour les reverse proxies
        if "X-Forwarded-Proto" in request.headers:
            request.scope["scheme"] = request.headers["X-Forwarded-Proto"]
        if "X-Forwarded-Host" in request.headers:
            request.scope["server"] = (request.headers["X-Forwarded-Host"], 443 if request.scope["scheme"] == "https" else 80)
        
        response = await call_next(request)
        return response

app.add_middleware(ProxyHeadersMiddleware)

# Configuration des fichiers statiques et templates
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Répertoires de données
Path("data/uploads").mkdir(parents=True, exist_ok=True)
Path("data/exports").mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Événement de démarrage de l'application"""
    logger.info("Démarrage de l'application SEO Keyword Classifier")
    
    # Test de connexion Redis
    try:
        # Test de ping Redis
        inspect = celery_app.control.inspect()
        active_workers = inspect.active()
        logger.info(f"✅ Connexion Redis OK - Workers actifs: {active_workers}")
    except Exception as e:
        logger.error(f"❌ Erreur connexion Redis: {e}")
    
    # Initialise la base de données
    db_manager._init_tables()
    
    # Nettoie le cache expiré au démarrage
    db_manager.cleanup_expired_cache()
    
    logger.info(f"🚀 Application démarrée avec ROOT_PATH: '{settings.root_path}'")

@app.on_event("shutdown")
async def shutdown_event():
    """Événement d'arrêt de l'application"""
    logger.info("Arrêt de l'application")

# ===== ROUTES WEB =====

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Page d'accueil avec interface d'upload"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "root_path": settings.root_path
    })

@app.get("/parameters", response_class=HTMLResponse)
async def parameters_page(request: Request):
    """Page de configuration des paramètres"""
    return templates.TemplateResponse("parameters.html", {
        "request": request,
        "root_path": settings.root_path
    })

@app.get("/progress/{job_id}", response_class=HTMLResponse)
async def progress_page(request: Request, job_id: str):
    """Page de suivi de progression"""
    return templates.TemplateResponse("progress.html", {
        "request": request, 
        "job_id": job_id,
        "root_path": settings.root_path
    })

@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str):
    """Page d'affichage des résultats"""
    return templates.TemplateResponse("results.html", {
        "request": request, 
        "job_id": job_id,
        "root_path": settings.root_path
    })

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    """Page d'historique des jobs"""
    return templates.TemplateResponse("history.html", {
        "request": request,
        "root_path": settings.root_path
    })

@app.get("/test-links", response_class=HTMLResponse)
async def test_links_page(request: Request):
    """Page de test pour vérifier tous les liens avec root_path"""
    return templates.TemplateResponse("test_links.html", {
        "request": request,
        "root_path": settings.root_path
    })

@app.get("/test-celery")
async def test_celery():
    """Test de fonctionnement de Celery"""
    try:
        # Import lazy pour éviter les problèmes d'import circulaire
        from app.tasks import ping_task, test_task
        
        # Test de ping simple
        ping_result = ping_task.delay()
        logger.info(f"Tâche ping lancée: {ping_result.id}")
        
        # Test avec log
        test_result = test_task.delay()
        logger.info(f"Tâche test lancée: {test_result.id}")
        
        # Essaie de récupérer le résultat (avec timeout)
        try:
            ping_value = ping_result.get(timeout=10)
            test_value = test_result.get(timeout=10)
            
            return {
                "status": "success",
                "ping_task_id": ping_result.id,
                "ping_result": ping_value,
                "test_task_id": test_result.id,
                "test_result": test_value,
                "message": "Celery fonctionne correctement!"
            }
        except Exception as timeout_error:
            return {
                "status": "timeout",
                "ping_task_id": ping_result.id,
                "test_task_id": test_result.id,
                "error": str(timeout_error),
                "message": "Les tâches ont été créées mais n'ont pas été traitées dans les temps"
            }
            
    except Exception as e:
        logger.error(f"Erreur test Celery: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Erreur lors du test Celery"
        }

@app.get("/test-celery-simple")
async def test_celery_simple():
    """Test de communication avec Celery avec tâches simples"""
    logger.info("🧪 Test des tâches Celery simples")
    
    try:
        # Import des tâches de façon lazy
        from app.tasks import ping_task, test_task
        
        # Test ping
        logger.info("📤 Lancement tâche ping...")
        ping_result = ping_task.delay()
        logger.info(f"🎯 Tâche ping créée: {ping_result.id}")
        
        # Attendre 5 secondes max
        try:
            ping_response = ping_result.get(timeout=5)
            logger.info(f"✅ Ping reçu: {ping_response}")
        except Exception as e:
            logger.error(f"❌ Ping timeout: {e}")
            ping_response = f"TIMEOUT: {e}"
        
        # Test tâche simple
        logger.info("📤 Lancement tâche test...")
        test_result = test_task.delay()
        logger.info(f"🎯 Tâche test créée: {test_result.id}")
        
        try:
            test_response = test_result.get(timeout=5)
            logger.info(f"✅ Test reçu: {test_response}")
        except Exception as e:
            logger.error(f"❌ Test timeout: {e}")
            test_response = f"TIMEOUT: {e}"
        
        return {
            "ping_task_id": ping_result.id,
            "ping_result": ping_response,
            "test_task_id": test_result.id,
            "test_result": test_response
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur test Celery: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur test Celery: {str(e)}")

# ===== API ENDPOINTS =====

@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    parameters: str = Form(...)
):
    """Crée un nouveau job d'analyse de mots-clés"""
    
    try:
        # Parse les paramètres JSON
        import json
        params_dict = json.loads(parameters)
        job_parameters = JobParameters(**params_dict)
        
        # Génère un ID unique pour le job
        job_id = str(uuid.uuid4())
        
        # Sauvegarde le fichier uploadé
        upload_dir = Path("data/uploads")
        file_path = upload_dir / f"{job_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Fichier sauvegardé: {file_path}")
        
        # Validation rapide du fichier pour compter les mots-clés
        from app.utils.file_processing import process_uploaded_file
        keywords = process_uploaded_file(str(file_path))
        
        # Crée le job en base
        db_manager.create_job(job_id, job_parameters, len(keywords))
        
        # Lance la tâche asynchrone
        from app.tasks import process_keywords_task
        task = process_keywords_task.delay(
            job_id=job_id,
            file_path=str(file_path),
            parameters_dict=params_dict
        )
        
        logger.info(f"Job {job_id} créé et tâche lancée: {task.id}")
        
        return {"job_id": job_id}
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du job: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> JobProgress:
    """Récupère le statut en temps réel d'un job"""
    
    logger.debug(f"🔍 Demande de statut pour job {job_id}")
    
    try:
        # Récupère les infos du job depuis la base
        job_data = db_manager.get_job(job_id)
        
        if not job_data:
            logger.warning(f"❌ Job {job_id} non trouvé en base")
            raise HTTPException(status_code=404, detail="Job non trouvé")
        
        logger.debug(f"📊 Job {job_id} - Status DB: {job_data['status']}")
        
        # Récupère l'état de la tâche Celery
        task_id = None  # Il faudrait stocker l'ID de tâche pour un suivi précis
        
        # Pour l'instant, on se base sur le statut en base
        status = JobStatus(job_data["status"])
        
        if status == JobStatus.COMPLETED:
            logger.debug(f"✅ Job {job_id} - TERMINÉ")
            progress = JobProgress(
                progress=100.0,
                state=status,
                message="Traitement terminé avec succès"
            )
        elif status == JobStatus.FAILED:
            logger.debug(f"❌ Job {job_id} - ÉCHEC: {job_data.get('error_message', 'Erreur inconnue')}")
            progress = JobProgress(
                progress=0.0,
                state=status,
                message=job_data.get("error_message", "Erreur inconnue")
            )
        elif status == JobStatus.PROCESSING:
            logger.debug(f"⚙️ Job {job_id} - EN COURS")
            progress = JobProgress(
                progress=50.0,  # Estimation si pas d'info précise
                state=status,
                message="Traitement en cours..."
            )
        else:  # PENDING
            logger.debug(f"⏳ Job {job_id} - EN ATTENTE")
            progress = JobProgress(
                progress=0.0,
                state=status,
                message="En attente de traitement..."
            )
        
        logger.debug(f"📤 Job {job_id} - Réponse: {progress.progress}% - {progress.message}")
        return progress
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Erreur lors de la récupération du statut du job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}/result")
async def download_job_result(
    job_id: str, 
    format: str = Query("csv", regex="^(csv|xlsx|json)$")
):
    """Télécharge le résultat d'un job"""
    
    try:
        job_data = db_manager.get_job(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job non trouvé")
        
        if job_data["status"] != JobStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Job non terminé")
        
        export_file_path = job_data.get("export_file_path")
        
        if not export_file_path or not os.path.exists(export_file_path):
            raise HTTPException(status_code=404, detail="Fichier de résultat non trouvé")
        
        # Détermine le type MIME
        if format == "csv":
            media_type = "text/csv"
        elif format == "xlsx":
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:  # json
            media_type = "application/json"
        
        filename = f"keywords_clusters_{job_id}.{format}"
        
        return FileResponse(
            path=export_file_path,
            media_type=media_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du résultat {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs")
async def list_jobs(limit: int = Query(10, ge=1, le=100)) -> List[JobSummary]:
    """Liste les derniers jobs"""
    
    try:
        jobs_data = db_manager.get_recent_jobs(limit)
        
        jobs = []
        for job_data in jobs_data:
            # Parse les paramètres
            import json
            parameters = JobParameters(**json.loads(job_data["parameters"]))
            
            job_summary = JobSummary(
                job_id=job_data["job_id"],
                status=JobStatus(job_data["status"]),
                created_at=datetime.fromisoformat(job_data["created_at"]),
                completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data["completed_at"] else None,
                keywords_count=job_data["keywords_count"],
                clusters_count=job_data["clusters_count"],
                parameters=parameters
            )
            jobs.append(job_summary)
        
        return jobs
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}/details")
async def get_job_details(job_id: str):
    """Récupère les détails complets d'un job"""
    
    try:
        logger.debug(f"🔍 Récupération des détails pour job {job_id}")
        
        job_data = db_manager.get_job(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job non trouvé")
        
        logger.debug(f"📊 Job {job_id} trouvé: {job_data.get('status')}")
        
        # Parse les paramètres
        import json
        parameters_data = json.loads(job_data["parameters"])
        logger.debug(f"✅ Paramètres parsés pour job {job_id}")
        
        # Récupère les mots-clés et clusters si le job est terminé
        keywords = None
        clusters = None
        
        if job_data["status"] == JobStatus.COMPLETED:
            logger.debug(f"🔄 Récupération des données pour job terminé {job_id}")
            
            # Récupère directement les dictionnaires
            try:
                keywords = db_manager.get_job_keywords(job_id)
                logger.debug(f"📝 {len(keywords) if keywords else 0} mots-clés trouvés pour {job_id}")
            except Exception as e:
                logger.error(f"❌ Erreur récupération mots-clés {job_id}: {e}")
                keywords = []
            
            try:
                clusters = db_manager.get_job_clusters(job_id)
                logger.debug(f"🔗 {len(clusters) if clusters else 0} clusters trouvés pour {job_id}")
            except Exception as e:
                logger.error(f"❌ Erreur récupération clusters {job_id}: {e}")
                clusters = []
        
        logger.debug(f"🔄 Création de la réponse pour {job_id}")
        
        # Retourne un dictionnaire simple
        result = {
            "job_id": job_data["job_id"],
            "status": job_data["status"],
            "created_at": job_data["created_at"],
            "completed_at": job_data.get("completed_at"),
            "parameters": parameters_data,
            "keywords_count": job_data["keywords_count"],
            "clusters_count": job_data.get("clusters_count"),
            "keywords": keywords,
            "clusters": clusters,
            "export_url": f"/api/jobs/{job_id}/result" if job_data["status"] == JobStatus.COMPLETED else None,
            "error_message": job_data.get("error_message")
        }
        
        logger.debug(f"✅ Réponse créée pour {job_id}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des détails du job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}/visualization")
async def get_job_visualization(job_id: str) -> UMAPVisualization:
    """Récupère les données de visualisation UMAP pour un job"""
    
    try:
        job_data = db_manager.get_job(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job non trouvé")
        
        if job_data["status"] != JobStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Job non terminé")
        
        # Récupère les mots-clés (dictionnaires maintenant)
        keywords_raw = db_manager.get_job_keywords(job_id)
        
        if not keywords_raw:
            raise HTTPException(status_code=404, detail="Aucun mot-clé trouvé")
        
        # Génère la visualisation (à implémenter selon les besoins)
        # Pour l'instant, on retourne des données simulées
        from app.services.clustering import clustering_service
        
        keyword_texts = [kw["keyword"] for kw in keywords_raw]
        embeddings = clustering_service.generate_embeddings(keyword_texts)
        cluster_labels = [kw.get("cluster_id", -1) for kw in keywords_raw]
        
        visualization = clustering_service.create_umap_visualization(
            embeddings, keyword_texts, cluster_labels
        )
        
        return visualization
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la visualisation {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== ENDPOINTS DE SANTÉ =====

@app.get("/health")
async def health_check():
    """Point de contrôle de santé de l'application"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/stats")
async def get_application_stats():
    """Statistiques générales de l'application"""
    try:
        recent_jobs = db_manager.get_recent_jobs(100)
        
        stats = {
            "total_jobs": len(recent_jobs),
            "completed_jobs": len([j for j in recent_jobs if j["status"] == JobStatus.COMPLETED]),
            "failed_jobs": len([j for j in recent_jobs if j["status"] == JobStatus.FAILED]),
            "pending_jobs": len([j for j in recent_jobs if j["status"] == JobStatus.PENDING]),
            "processing_jobs": len([j for j in recent_jobs if j["status"] == JobStatus.PROCESSING])
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 