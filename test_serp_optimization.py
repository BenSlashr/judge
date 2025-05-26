#!/usr/bin/env python3
"""
Test des optimisations SERP pour validation des performances
"""

import asyncio
import time
import logging
from typing import List
from app.services.enrichment import enrichment_service

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_serp_optimizations():
    """Test complet des optimisations SERP"""
    
    # Mots-clés d'exemple pour les tests
    test_keywords = [
        "smartphone 128gb",
        "iphone 13 pro",
        "samsung galaxy s23",
        "pixel 7 pro",
        "oneplus 11",
        "xiaomi 13",
        "huawei p60",
        "oppo find x5",
        "realme gt neo 3",
        "nothing phone 1",
        "sony xperia 1 iv",
        "motorola edge 30",
        "asus rog phone 6",
        "google pixel 6a",
        "fairphone 4",
        "nokia x30",
        "honor magic 4",
        "vivo x90",
        "redmi note 12",
        "poco f4"
    ]
    
    logger.info(f"🧪 Test avec {len(test_keywords)} mots-clés")
    
    # Test 1: Mode séquentiel (simulation de l'ancien comportement)
    logger.info("\n📊 TEST 1: Mode séquentiel (baseline)")
    start_time = time.time()
    
    try:
        sequential_results = await enrichment_service.calculate_serp_similarity(
            test_keywords[:5],  # Limite à 5 pour les tests
            max_concurrent=1,
            enable_sampling=False,
            enable_smart_clustering=False
        )
        sequential_time = time.time() - start_time
        logger.info(f"✅ Séquentiel terminé en {sequential_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Erreur séquentiel: {e}")
        sequential_time = float('inf')
    
    # Test 2: Mode parallèle optimisé
    logger.info("\n🚀 TEST 2: Mode parallèle optimisé")
    start_time = time.time()
    
    try:
        parallel_results = await enrichment_service.calculate_serp_similarity(
            test_keywords[:5],
            max_concurrent=10,
            enable_sampling=False,
            enable_smart_clustering=True
        )
        parallel_time = time.time() - start_time
        logger.info(f"✅ Parallèle terminé en {parallel_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Erreur parallèle: {e}")
        parallel_time = float('inf')
    
    # Test 3: Mode échantillonnage
    logger.info("\n📊 TEST 3: Mode échantillonnage")
    start_time = time.time()
    
    try:
        sampling_results = await enrichment_service.calculate_serp_similarity(
            test_keywords,  # Tous les mots-clés
            max_concurrent=10,
            enable_sampling=True,
            sampling_ratio=0.3,
            enable_smart_clustering=True
        )
        sampling_time = time.time() - start_time
        logger.info(f"✅ Échantillonnage terminé en {sampling_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Erreur échantillonnage: {e}")
        sampling_time = float('inf')
    
    # Test 4: Sélection intelligente
    logger.info("\n🧠 TEST 4: Sélection intelligente seule")
    start_time = time.time()
    
    try:
        smart_results = await enrichment_service.calculate_serp_similarity(
            test_keywords,
            max_concurrent=10,
            enable_sampling=False,
            enable_smart_clustering=True
        )
        smart_time = time.time() - start_time
        logger.info(f"✅ Sélection intelligente terminée en {smart_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Erreur sélection intelligente: {e}")
        smart_time = float('inf')
    
    # Calcul des gains de performance
    logger.info("\n📈 RÉSULTATS DES PERFORMANCES:")
    logger.info("=" * 50)
    
    if sequential_time != float('inf'):
        if parallel_time != float('inf'):
            parallel_speedup = sequential_time / parallel_time
            logger.info(f"🚀 Parallélisation: {parallel_speedup:.1f}x plus rapide")
        
        if sampling_time != float('inf'):
            sampling_speedup = (sequential_time * len(test_keywords) / 5) / sampling_time
            logger.info(f"📊 Échantillonnage: {sampling_speedup:.1f}x plus rapide (estimation)")
        
        if smart_time != float('inf'):
            smart_speedup = (sequential_time * len(test_keywords) / 5) / smart_time
            logger.info(f"🧠 Sélection intelligente: {smart_speedup:.1f}x plus rapide (estimation)")
    
    # Test de la qualité des résultats
    logger.info("\n🎯 QUALITÉ DES RÉSULTATS:")
    logger.info("=" * 50)
    
    try:
        if 'sequential_results' in locals() and 'parallel_results' in locals():
            # Compare la cohérence des résultats
            common_keywords = set(sequential_results.keys()) & set(parallel_results.keys())
            if common_keywords:
                total_diff = 0
                for kw1 in list(common_keywords)[:3]:
                    for kw2 in list(common_keywords)[:3]:
                        if kw1 != kw2:
                            seq_sim = sequential_results[kw1].get(kw2, 0)
                            par_sim = parallel_results[kw1].get(kw2, 0)
                            diff = abs(seq_sim - par_sim)
                            total_diff += diff
                
                avg_diff = total_diff / max(1, len(common_keywords) * (len(common_keywords) - 1))
                logger.info(f"📊 Différence moyenne séquentiel vs parallèle: {avg_diff:.4f}")
                
                if avg_diff < 0.05:
                    logger.info("✅ Résultats cohérents entre modes séquentiel et parallèle")
                else:
                    logger.warning("⚠️ Différences notables entre modes")
    except Exception as e:
        logger.error(f"❌ Erreur comparaison qualité: {e}")
    
    logger.info("\n🎉 Tests terminés!")

def test_smart_keyword_selection():
    """Test de la sélection intelligente de mots-clés"""
    logger.info("\n🧠 TEST SÉLECTION INTELLIGENTE:")
    
    test_keywords = [
        "smartphone samsung",
        "samsung galaxy",
        "telephone samsung",
        "iphone apple",
        "apple iphone",
        "telephone apple",
        "ordinateur portable",
        "laptop ordinateur",
        "pc portable",
        "macbook apple"
    ]
    
    async def run_selection_test():
        try:
            selected = await enrichment_service._smart_keyword_selection(test_keywords)
            logger.info(f"📝 Mots-clés originaux ({len(test_keywords)}): {test_keywords}")
            logger.info(f"🎯 Mots-clés sélectionnés ({len(selected)}): {selected}")
            
            # Vérifie que la sélection est cohérente
            reduction_ratio = len(selected) / len(test_keywords)
            logger.info(f"📊 Ratio de réduction: {reduction_ratio:.2f}")
            
            if 0.3 <= reduction_ratio <= 0.8:
                logger.info("✅ Sélection intelligente cohérente")
            else:
                logger.warning("⚠️ Sélection trop agressive ou pas assez")
                
        except Exception as e:
            logger.error(f"❌ Erreur test sélection: {e}")
    
    asyncio.run(run_selection_test())

if __name__ == "__main__":
    print("🚀 TESTS DES OPTIMISATIONS SERP")
    print("=" * 50)
    
    # Test de sélection intelligente (sans API)
    test_smart_keyword_selection()
    
    # Tests complets (nécessitent les credentials DataForSEO)
    print("\n⚠️ Tests SERP complets nécessitent credentials DataForSEO")
    print("Définissez DATAFOR_SEO_LOGIN et DATAFOR_SEO_PASSWORD pour les tests complets")
    
    import os
    if os.getenv('DATAFOR_SEO_LOGIN') and os.getenv('DATAFOR_SEO_PASSWORD'):
        print("✅ Credentials détectés, lancement des tests complets...")
        asyncio.run(test_serp_optimizations())
    else:
        print("⏭️ Tests SERP sautés (pas de credentials)") 