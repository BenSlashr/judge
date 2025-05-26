"""
Service d'embeddings simplifié en fallback
"""
import logging
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class SimpleEmbeddingService:
    """Service d'embeddings simple basé sur TF-IDF"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            stop_words=None,
            lowercase=True
        )
        logger.info("✅ Service d'embeddings simple initialisé")
    
    def generate_embeddings(self, keywords: List[str]) -> np.ndarray:
        """Génère des embeddings TF-IDF pour les mots-clés"""
        logger.info(f"📊 Génération d'embeddings TF-IDF pour {len(keywords)} mots-clés")
        
        try:
            # Génère les vecteurs TF-IDF
            embeddings = self.vectorizer.fit_transform(keywords).toarray()
            logger.info(f"✅ Embeddings TF-IDF générés: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Erreur TF-IDF: {e}")
            # Dernier fallback: embeddings aléatoires
            return np.random.rand(len(keywords), 300)

# Instance globale
simple_embedding_service = SimpleEmbeddingService() 