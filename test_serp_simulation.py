#!/usr/bin/env python3
"""
Test de simulation des optimisations SERP pour validation des gains de performance
"""

import asyncio
import time
import logging
import random
from typing import List, Dict, Set
from app.services.enrichment import enrichment_service

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SerpSimulator:
    """Simule les appels SERP pour tester les optimisations"""
    
    def __init__(self, base_latency_ms: int = 1500):
        self.base_latency = base_latency_ms / 1000  # Conversion en secondes
        self.call_count = 0
        self.total_time = 0
        
    async def simulate_serp_call(self, keyword: str) -> Set[str]:
        """Simule un appel SERP avec latence réaliste"""
        self.call_count += 1
        
        # Simule la latence variable d'un vrai appel API (1-3 secondes)
        latency = self.base_latency + random.uniform(0, 1.0)
        await asyncio.sleep(latency)
        self.total_time += latency
        
        # Génère des domaines fictifs mais cohérents basés sur le mot-clé
        domains = set()
        hash_seed = hash(keyword.lower()) % 1000
        
        # Domaines communs pour mots-clés similaires
        common_domains = [
            "amazon.fr", "fnac.com", "cdiscount.com", "darty.com", 
            "boulanger.com", "ldlc.com", "materiel.net", "rueducommerce.fr"
        ]
        
        # Sélectionne 8-10 domaines de façon déterministe basée sur le mot-clé
        random.seed(hash_seed)
        domain_count = random.randint(8, 10)
        selected_domains = random.sample(common_domains, min(domain_count, len(common_domains)))
        
        # Ajoute quelques domaines spécifiques basés sur le mot-clé
        if "iphone" in keyword.lower() or "apple" in keyword.lower():
            selected_domains.extend(["apple.com", "iphon.fr"])
        elif "samsung" in keyword.lower():
            selected_domains.extend(["samsung.com", "samsung.fr"])
        elif "ordinateur" in keyword.lower() or "pc" in keyword.lower():
            selected_domains.extend(["grosbill.com", "topachat.com"])
        
        return set(selected_domains[:10])  # Limite à 10 domaines
    
    def reset_stats(self):
        """Remet à zéro les statistiques"""
        self.call_count = 0
        self.total_time = 0

async def test_serp_performance_simulation():
    """Test complet de simulation des performances SERP"""
    
    print("🧪 SIMULATION DES OPTIMISATIONS SERP")
    print("=" * 60)
    
    # Dataset de test réaliste (100 mots-clés e-commerce)
    test_keywords = [
        # Smartphones
        "iphone 14 128gb", "iphone 14 pro", "iphone 13 mini", "samsung galaxy s23", 
        "samsung galaxy s22", "samsung galaxy a54", "pixel 7 pro", "pixel 6a",
        "oneplus 11", "xiaomi 13", "huawei p60", "oppo find x5", "realme gt neo 3",
        
        # Ordinateurs
        "macbook air m2", "macbook pro 14", "dell xps 13", "lenovo thinkpad x1",
        "hp spectre x360", "asus zenbook", "surface laptop 5", "acer swift 3",
        "msi gaming laptop", "alienware x15", "razer blade 15",
        
        # Composants PC
        "rtx 4090", "rtx 4080", "rtx 3080", "radeon rx 7900", "intel i9 13900k",
        "amd ryzen 9 7950x", "ddr5 32gb", "ssd nvme 1tb", "carte mère b650",
        
        # Périphériques
        "clavier mécanique", "souris gaming", "écran 4k 27 pouces", "casque bluetooth",
        "webcam 4k", "micro streaming", "enceinte bluetooth", "tablette graphique",
        
        # Électroménager
        "lave linge 9kg", "lave vaisselle 60cm", "réfrigérateur combiné", "four encastrable",
        "micro ondes grill", "aspirateur robot", "cafetière expresso", "friteuse sans huile",
        
        # TV et Audio
        "tv oled 55 pouces", "tv qled 65 pouces", "barre de son", "home cinema",
        "casque audio", "platine vinyle", "ampli hifi", "enceinte colonne",
        
        # Objets connectés
        "montre connectée", "bracelet fitness", "caméra surveillance", "thermostat connecté",
        "ampoule connectée", "assistant vocal", "robot aspirateur", "sonnette connectée",
        
        # Gaming
        "ps5 console", "xbox series x", "nintendo switch oled", "manette xbox",
        "casque vr", "chaise gaming", "bureau gaming", "éclairage rgb",
        
        # Bricolage
        "perceuse visseuse", "scie circulaire", "ponceuse orbitale", "niveau laser",
        "multimètre", "poste à souder", "compresseur", "outillage professionnel",
        
        # Sport
        "vélo électrique", "trottinette électrique", "home trainer", "montre gps",
        "chaussures running", "sac de sport", "haltères", "tapis de course",
        
        # Variations avec prix/tailles
        "iphone 14 256gb", "samsung galaxy s23 128gb", "macbook air 256gb", 
        "écran 32 pouces", "ssd 2tb", "ram 16gb", "tv 75 pouces"
    ]
    
    simulator = SerpSimulator(base_latency_ms=1500)  # 1.5s de latence de base
    
    print(f"📊 Dataset de test : {len(test_keywords)} mots-clés")
    print(f"⏱️ Latence simulée : ~1.5-2.5s par appel SERP")
    print()
    
    # Test 1: Mode SÉQUENTIEL (baseline)
    print("🐌 TEST 1: Mode séquentiel (ancien comportement)")
    print("-" * 40)
    
    simulator.reset_stats()
    start_time = time.time()
    
    # Simule l'ancien comportement : appels séquentiels
    sample_keywords = test_keywords[:20]  # Limite pour le test
    serp_data_sequential = {}
    
    for keyword in sample_keywords:
        domains = await simulator.simulate_serp_call(keyword)
        serp_data_sequential[keyword] = domains
    
    sequential_time = time.time() - start_time
    sequential_calls = simulator.call_count
    
    print(f"✅ Terminé en {sequential_time:.1f}s")
    print(f"📞 Appels SERP : {sequential_calls}")
    print(f"⚡ Temps moyen par appel : {sequential_time/sequential_calls:.2f}s")
    print()
    
    # Test 2: Mode PARALLÈLE OPTIMISÉ
    print("🚀 TEST 2: Mode parallèle optimisé")
    print("-" * 40)
    
    simulator.reset_stats()
    start_time = time.time()
    
    # Simule les appels parallèles avec semaphore
    semaphore = asyncio.Semaphore(10)  # 10 appels parallèles
    
    async def parallel_scrape(keyword):
        async with semaphore:
            return keyword, await simulator.simulate_serp_call(keyword)
    
    tasks = [parallel_scrape(kw) for kw in sample_keywords]
    results = await asyncio.gather(*tasks)
    
    serp_data_parallel = {kw: domains for kw, domains in results}
    
    parallel_time = time.time() - start_time
    parallel_calls = simulator.call_count
    
    print(f"✅ Terminé en {parallel_time:.1f}s")
    print(f"📞 Appels SERP : {parallel_calls}")
    print(f"⚡ Temps moyen par appel : {parallel_time/parallel_calls:.2f}s")
    print(f"🚀 Gain de vitesse : {sequential_time/parallel_time:.1f}x plus rapide")
    print()
    
    # Test 3: Mode ÉCHANTILLONNAGE INTELLIGENT
    print("🧠 TEST 3: Mode échantillonnage intelligent")
    print("-" * 40)
    
    simulator.reset_stats()
    start_time = time.time()
    
    # Test de la sélection intelligente
    full_dataset = test_keywords  # Tous les mots-clés
    selected_keywords = await enrichment_service._smart_keyword_selection(full_dataset)
    
    print(f"🎯 Sélection intelligente : {len(selected_keywords)}/{len(full_dataset)} mots-clés")
    print(f"📊 Réduction : {(1 - len(selected_keywords)/len(full_dataset))*100:.0f}%")
    
    # Simule les appels pour les mots-clés sélectionnés
    tasks = [parallel_scrape(kw) for kw in selected_keywords]
    results = await asyncio.gather(*tasks)
    
    serp_data_smart = {kw: domains for kw, domains in results}
    
    # Simule l'extension de la matrice (plus rapide)
    extension_time = len(full_dataset) * 0.01  # 10ms par mot-clé pour l'approximation
    await asyncio.sleep(extension_time)
    
    smart_time = time.time() - start_time
    smart_calls = simulator.call_count
    
    print(f"✅ Terminé en {smart_time:.1f}s")
    print(f"📞 Appels SERP : {smart_calls}")
    print(f"⚡ Extension matrice : {extension_time:.1f}s")
    
    # Calcul du gain estimé pour le dataset complet
    estimated_full_sequential = (len(full_dataset) / len(sample_keywords)) * sequential_time
    smart_speedup = estimated_full_sequential / smart_time
    
    print(f"📈 Gain estimé (dataset complet) : {smart_speedup:.1f}x plus rapide")
    print()
    
    # Test 4: ÉCHANTILLONNAGE + PARALLÉLISATION
    print("⚡ TEST 4: Échantillonnage + Parallélisation combinés")
    print("-" * 40)
    
    simulator.reset_stats()
    start_time = time.time()
    
    # Échantillonnage agressif (30% des mots-clés)
    sample_size = int(len(full_dataset) * 0.3)
    sampled_keywords = random.sample(full_dataset, sample_size)
    
    # Appels parallèles sur l'échantillon
    tasks = [parallel_scrape(kw) for kw in sampled_keywords]
    results = await asyncio.gather(*tasks)
    
    # Extension rapide
    extension_time = len(full_dataset) * 0.005  # 5ms par mot-clé
    await asyncio.sleep(extension_time)
    
    combined_time = time.time() - start_time
    combined_calls = simulator.call_count
    
    print(f"🎯 Échantillonnage : {len(sampled_keywords)}/{len(full_dataset)} mots-clés (30%)")
    print(f"✅ Terminé en {combined_time:.1f}s")
    print(f"📞 Appels SERP : {combined_calls}")
    
    # Calcul du gain pour le dataset complet
    combined_speedup = estimated_full_sequential / combined_time
    print(f"📈 Gain estimé (dataset complet) : {combined_speedup:.1f}x plus rapide")
    print()
    
    # RÉSUMÉ FINAL
    print("📊 RÉSUMÉ DES PERFORMANCES")
    print("=" * 60)
    print(f"🐌 Séquentiel (baseline)     : {sequential_time:.1f}s pour {len(sample_keywords)} mots-clés")
    print(f"🚀 Parallèle                 : {parallel_time:.1f}s ({sequential_time/parallel_time:.1f}x plus rapide)")
    print(f"🧠 Sélection intelligente    : {smart_time:.1f}s ({smart_speedup:.1f}x plus rapide estimé)")
    print(f"⚡ Échantillonnage + Parallèle: {combined_time:.1f}s ({combined_speedup:.1f}x plus rapide estimé)")
    print()
    print("💡 RECOMMANDATIONS:")
    print(f"• Pour <500 mots-clés  : Parallélisation simple (10-20 appels concurrents)")
    print(f"• Pour 500-2000 m-c    : Sélection intelligente + parallélisation")
    print(f"• Pour >2000 mots-clés : Échantillonnage 30% + parallélisation")
    print()
    
    # Test de qualité des résultats
    print("🎯 QUALITÉ DES RÉSULTATS")
    print("=" * 30)
    
    # Compare la cohérence entre modes
    common_kw = set(serp_data_sequential.keys()) & set(serp_data_parallel.keys())
    if len(common_kw) >= 2:
        similarities_seq = []
        similarities_par = []
        
        for kw1 in list(common_kw)[:5]:
            for kw2 in list(common_kw)[:5]:
                if kw1 != kw2:
                    # Calcule similarité Jaccard
                    domains1_seq = serp_data_sequential[kw1]
                    domains2_seq = serp_data_sequential[kw2]
                    sim_seq = len(domains1_seq & domains2_seq) / len(domains1_seq | domains2_seq)
                    similarities_seq.append(sim_seq)
                    
                    domains1_par = serp_data_parallel[kw1]
                    domains2_par = serp_data_parallel[kw2]
                    sim_par = len(domains1_par & domains2_par) / len(domains1_par | domains2_par)
                    similarities_par.append(sim_par)
        
        if similarities_seq and similarities_par:
            avg_diff = sum(abs(s - p) for s, p in zip(similarities_seq, similarities_par)) / len(similarities_seq)
            print(f"📊 Différence moyenne séquentiel vs parallèle : {avg_diff:.4f}")
            
            if avg_diff < 0.02:
                print("✅ Qualité préservée - résultats très cohérents")
            elif avg_diff < 0.05:
                print("✅ Qualité acceptable - légères variations")
            else:
                print("⚠️ Qualité dégradée - variations importantes")

if __name__ == "__main__":
    asyncio.run(test_serp_performance_simulation()) 