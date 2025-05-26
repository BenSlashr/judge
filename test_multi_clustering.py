#!/usr/bin/env python3
"""
Script de test pour valider le fonctionnement du multi-clustering HDBSCAN
"""

import sys
import os
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au path pour importer les modules de l'app
sys.path.insert(0, os.path.abspath('.'))

from app.services.clustering import ClusteringService
from app.models import ClusteringAlgorithm

def test_multi_clustering():
    """Test du multi-clustering avec des données d'exemple"""
    
    print("🧪 Test du multi-clustering HDBSCAN")
    print("=" * 50)
    
    # Initialise le service de clustering
    clustering_service = ClusteringService()
    
    # Données d'exemple : mots-clés SEO avec des groupes naturels
    test_keywords = [
        # Groupe SEO technique
        "audit seo technique", "optimisation technique seo", "core web vitals", 
        "vitesse site web", "indexation google", "sitemap xml",
        
        # Groupe contenu SEO  
        "rédaction seo", "optimisation contenu", "mots clés longue traine",
        "balises title", "meta description", "structure h1 h2",
        
        # Groupe netlinking
        "backlinks qualité", "stratégie netlinking", "linkbuilding",
        "autorité domaine", "trust flow", "link juice",
        
        # Groupe analytics
        "google analytics", "search console", "suivi seo",
        "kpi seo", "mesure performance", "reporting seo",
        
        # Mots-clés qui pourraient appartenir à plusieurs groupes
        "outils seo", "consultant seo", "formation seo", "agence seo"
    ]
    
    print(f"📝 Test avec {len(test_keywords)} mots-clés")
    print(f"   Premiers mots-clés: {test_keywords[:3]}")
    print()
    
    # Génère les embeddings
    print("🧠 Génération des embeddings...")
    embeddings = clustering_service.generate_embeddings(test_keywords)
    print(f"✅ Embeddings générés: {embeddings.shape}")
    print()
    
    # Test clustering standard
    print("🔗 Test clustering HDBSCAN standard...")
    labels_std, clusters_std, probs_std = clustering_service.cluster_keywords(
        test_keywords,
        embeddings,
        algorithm=ClusteringAlgorithm.HDBSCAN,
        min_cluster_size=3,
        enable_multi_cluster=False
    )
    
    print(f"✅ Clustering standard terminé:")
    print(f"   - {len(clusters_std)} clusters trouvés")
    print(f"   - Labels: {set(labels_std)}")
    print()
    
    # Test clustering multi-appartenance
    print("🔀 Test clustering HDBSCAN multi-appartenance...")
    labels_multi, clusters_multi, probs_multi = clustering_service.cluster_keywords(
        test_keywords,
        embeddings,
        algorithm=ClusteringAlgorithm.HDBSCAN,
        min_cluster_size=3,
        enable_multi_cluster=True,
        primary_threshold=0.6,
        secondary_threshold=0.4
    )
    
    print(f"✅ Clustering multi-appartenance terminé:")
    print(f"   - {len(clusters_multi)} clusters trouvés")
    print(f"   - Labels: {set(labels_multi)}")
    if probs_multi is not None:
        print(f"   - Probabilités: shape={probs_multi.shape}")
    print()
    
    # Test attribution multi-cluster
    if probs_multi is not None:
        print("🎯 Test attribution multi-cluster...")
        multi_assignments = clustering_service.assign_multi_clusters(
            test_keywords,
            probs_multi,
            primary_threshold=0.6,
            secondary_threshold=0.4
        )
        
        # Analyse des résultats
        multi_count = sum(1 for a in multi_assignments if a['is_multi_cluster'])
        print(f"✅ Attribution terminée:")
        print(f"   - {multi_count}/{len(test_keywords)} mots-clés multi-cluster")
        print()
        
        # Affiche quelques exemples détaillés
        print("📊 Exemples de mots-clés multi-cluster:")
        print("-" * 40)
        
        for i, assignment in enumerate(multi_assignments):
            if assignment['is_multi_cluster']:
                keyword = assignment['keyword']
                primary = assignment['cluster_primary']
                primary_prob = assignment['cluster_primary_probability']
                secondary = assignment['cluster_secondary']
                secondary_prob = assignment['cluster_secondary_probability']
                
                print(f"🔸 {keyword}")
                print(f"   Principal: Cluster {primary} (prob: {primary_prob:.3f})")
                if secondary is not None:
                    print(f"   Secondaire: Cluster {secondary} (prob: {secondary_prob:.3f})")
                
                alt1 = assignment['cluster_alt1']
                alt1_prob = assignment['cluster_alt1_probability']
                if alt1 is not None:
                    print(f"   Alternatif: Cluster {alt1} (prob: {alt1_prob:.3f})")
                print()
        
        # Résumé des statistiques
        print("📈 Statistiques multi-clustering:")
        print("-" * 30)
        print(f"Taux de multi-appartenance: {multi_count/len(test_keywords)*100:.1f}%")
        
        # Distribution des probabilités principales
        primary_probs = [a['cluster_primary_probability'] for a in multi_assignments 
                        if a['cluster_primary_probability'] is not None]
        if primary_probs:
            print(f"Probabilité principale moyenne: {np.mean(primary_probs):.3f}")
            print(f"Probabilité principale min/max: {np.min(primary_probs):.3f}/{np.max(primary_probs):.3f}")
        
        # Distribution des probabilités secondaires
        secondary_probs = [a['cluster_secondary_probability'] for a in multi_assignments 
                          if a['cluster_secondary_probability'] is not None]
        if secondary_probs:
            print(f"Probabilité secondaire moyenne: {np.mean(secondary_probs):.3f}")
            print(f"Probabilité secondaire min/max: {np.min(secondary_probs):.3f}/{np.max(secondary_probs):.3f}")
    
    print()
    print("🎉 Test du multi-clustering terminé avec succès !")
    return True

if __name__ == "__main__":
    try:
        success = test_multi_clustering()
        if success:
            print("✅ TOUS LES TESTS SONT PASSÉS")
            sys.exit(0)
        else:
            print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
            sys.exit(1)
    except Exception as e:
        print(f"💥 ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 