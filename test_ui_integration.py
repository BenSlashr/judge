#!/usr/bin/env python3
"""
Test simple pour vérifier l'intégration UI des nouvelles fonctionnalités
"""

import json
import sys
import os

# Ajout du chemin pour les imports
sys.path.insert(0, os.path.abspath('.'))

def test_frontend_params():
    """Test que les nouveaux paramètres frontend sont bien structurés"""
    
    print("🧪 Test des paramètres frontend multi-clustering")
    
    # Simule exactement ce que le frontend envoie
    frontend_params = {
        "serp_mode": "none",
        "alpha": 0.5,
        "clustering_algorithm": "hdbscan",
        "min_cluster_size": 5,
        "export_format": "csv",
        "country_code": "FR", 
        "language": "fr",
        "enable_multi_cluster": True,
        "primary_threshold": 0.6,
        "secondary_threshold": 0.4,
        "max_clusters_per_keyword": 5,
        "min_probability_threshold": 0.15
    }
    
    try:
        from app.models import JobParameters
        
        # Test validation
        job_params = JobParameters(**frontend_params)
        
        print("✅ Validation des paramètres réussie")
        print(f"   Multi-clustering: {job_params.enable_multi_cluster}")
        print(f"   Seuils: {job_params.primary_threshold} / {job_params.secondary_threshold}")
        print(f"   Max clusters/mot-clé: {job_params.max_clusters_per_keyword}")
        print(f"   Probabilité min: {job_params.min_probability_threshold}")
        
        # Simule la sérialisation JSON (comme dans une vraie requête HTTP)
        json_params = json.dumps(frontend_params)
        restored_params = json.loads(json_params)
        
        job_params_2 = JobParameters(**restored_params)
        print("✅ Sérialisation/désérialisation JSON réussie")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_keyword_model():
    """Test que le modèle Keyword supporte les nouveaux champs"""
    
    print("\n🧪 Test du modèle Keyword étendu")
    
    try:
        from app.models import Keyword
        
        # Test création avec nouveaux champs
        kw = Keyword(
            keyword="consultant seo",
            cluster_primary=0,
            cluster_primary_name="Services SEO",
            cluster_primary_probability=0.75,
            cluster_secondary=1,
            cluster_secondary_name="Expertise",
            cluster_secondary_probability=0.45,
            cluster_alt1=2,
            cluster_alt1_name="Marketing",
            cluster_alt1_probability=0.35,
            is_multi_cluster=True,
            total_clusters_count=3,
            additional_clusters=[
                {"cluster_id": 3, "cluster_name": "Digital", "probability": 0.25}
            ]
        )
        
        print("✅ Création Keyword avec nouveaux champs réussie")
        print(f"   Multi-cluster: {kw.is_multi_cluster}")
        print(f"   Total clusters: {kw.total_clusters_count}")
        print(f"   Cluster principal: {kw.cluster_primary_name} ({kw.cluster_primary_probability:.1%})")
        
        # Test sérialisation
        kw_dict = kw.model_dump()
        print("✅ Sérialisation modèle réussie")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_export_structure():
    """Test que la structure d'export est correcte"""
    
    print("\n🧪 Test structure d'export")
    
    try:
        from app.models import Keyword
        
        # Crée des keywords de test
        keywords = [
            Keyword(
                keyword="audit seo",
                cluster_primary=0,
                cluster_primary_name="SEO Technique",
                cluster_primary_probability=0.85,
                is_multi_cluster=False,
                total_clusters_count=1
            ),
            Keyword(
                keyword="consultant seo",
                cluster_primary=0,
                cluster_primary_name="Services SEO", 
                cluster_primary_probability=0.75,
                cluster_secondary=1,
                cluster_secondary_name="Expertise",
                cluster_secondary_probability=0.45,
                is_multi_cluster=True,
                total_clusters_count=2
            )
        ]
        
        # Vérifie que les champs nécessaires sont présents
        expected_fields = [
            'keyword', 'is_multi_cluster', 'total_clusters_count',
            'cluster_primary', 'cluster_primary_name', 'cluster_primary_probability',
            'cluster_secondary', 'cluster_secondary_name', 'cluster_secondary_probability'
        ]
        
        for kw in keywords:
            kw_dict = kw.model_dump()
            
            missing_fields = [field for field in expected_fields if field not in kw_dict]
            if missing_fields:
                print(f"❌ Champs manquants pour {kw.keyword}: {missing_fields}")
                return False
        
        print("✅ Structure d'export validée")
        print(f"   Mots-clés testés: {len(keywords)}")
        print(f"   Champs vérifiés: {len(expected_fields)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Test principal"""
    
    print("🚀 Test intégration UI Multi-Clustering")
    print("=" * 50)
    
    tests = [
        test_frontend_params,
        test_keyword_model, 
        test_export_structure
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Exception dans test: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 RÉSULTATS:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 Tous les tests réussis ({passed}/{total})")
        print("✅ L'interface multi-clustering est prête!")
    else:
        print(f"⚠️ {passed}/{total} tests réussis")
        print("❌ Des problèmes subsistent")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    print(f"\n{'=' * 50}")
    if success:
        print("🎊 L'interface multi-clustering est fonctionnelle!")
        print("Vous pouvez maintenant tester avec l'interface web.")
    else:
        print("🔧 Des corrections sont nécessaires.") 