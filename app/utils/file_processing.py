import os
import json
import pandas as pd
import logging
from typing import List, Set
from pathlib import Path

from app.models import Keyword, Cluster, ExportFormat
from app.config import settings

logger = logging.getLogger(__name__)

def process_uploaded_file(file_path: str) -> List[Keyword]:
    """Traite un fichier uploadé et extrait les mots-clés"""
    logger.info(f"Traitement du fichier: {file_path}")
    
    # Détermine le type de fichier
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Format de fichier non supporté: {file_extension}")
    
    # Vérifie la présence de la colonne 'keyword'
    keyword_column = None
    for col in df.columns:
        if col.lower() in ['keyword', 'keywords', 'mot-clé', 'mot-cles', 'requete', 'query']:
            keyword_column = col
            break
    
    if keyword_column is None:
        raise ValueError("Aucune colonne 'keyword' trouvée dans le fichier")
    
    # Identifie les colonnes optionnelles
    volume_column = None
    cpc_column = None
    difficulty_column = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['volume', 'search_volume', 'volume_recherche', 'searches']:
            volume_column = col
        elif col_lower in ['cpc', 'cost_per_click', 'cout_clic']:
            cpc_column = col
        elif col_lower in ['difficulty', 'difficulte', 'keyword_difficulty', 'kd']:
            difficulty_column = col
    
    # Extrait les mots-clés
    raw_keywords = df[keyword_column].dropna().astype(str).tolist()
    
    # Nettoyage et déduplication
    cleaned_keywords = clean_and_deduplicate_keywords(raw_keywords)
    
    # Vérifie la limite
    if len(cleaned_keywords) > settings.max_keywords_per_upload:
        raise ValueError(
            f"Trop de mots-clés ({len(cleaned_keywords)}). "
            f"Maximum autorisé: {settings.max_keywords_per_upload}"
        )
    
    # Convertit en objets Keyword avec les données existantes
    keywords = []
    for i, kw in enumerate(cleaned_keywords):
        # Trouve l'index original dans le DataFrame
        original_index = df[df[keyword_column].astype(str).str.lower() == kw].index
        
        if len(original_index) > 0:
            idx = original_index[0]
            
            # Récupère les valeurs des colonnes si disponibles
            search_volume = None
            if volume_column:
                try:
                    vol_val = df.loc[idx, volume_column]
                    if pd.notna(vol_val):
                        search_volume = int(float(vol_val))
                except (ValueError, TypeError):
                    pass
            
            cpc = None
            if cpc_column:
                try:
                    cpc_val = df.loc[idx, cpc_column]
                    if pd.notna(cpc_val):
                        cpc = float(cpc_val)
                except (ValueError, TypeError):
                    pass
            
            keyword_difficulty = None
            if difficulty_column:
                try:
                    diff_val = df.loc[idx, difficulty_column]
                    if pd.notna(diff_val):
                        keyword_difficulty = float(diff_val)
                except (ValueError, TypeError):
                    pass
            
            keyword_obj = Keyword(
                keyword=kw,
                search_volume=search_volume,
                cpc=cpc,
                keyword_difficulty=keyword_difficulty
            )
        else:
            keyword_obj = Keyword(keyword=kw)
        
        keywords.append(keyword_obj)
    
    logger.info(f"Fichier traité: {len(keywords)} mots-clés valides extraits")
    return keywords

def clean_and_deduplicate_keywords(keywords: List[str]) -> List[str]:
    """Nettoie et déduplique une liste de mots-clés"""
    
    cleaned = []
    seen = set()
    
    for keyword in keywords:
        # Nettoyage basique
        cleaned_kw = keyword.strip().lower()
        
        # Supprime les mots-clés vides ou trop courts
        if len(cleaned_kw) < 2:
            continue
        
        # Supprime les caractères spéciaux en début/fin
        cleaned_kw = cleaned_kw.strip('.,!?;:"\'()[]{}')
        
        # Vérifie les doublons
        if cleaned_kw not in seen:
            seen.add(cleaned_kw)
            cleaned.append(cleaned_kw)
    
    logger.info(f"Nettoyage terminé: {len(keywords)} -> {len(cleaned)} mots-clés")
    return cleaned

def export_results(
    job_id: str, 
    keywords: List[Keyword], 
    clusters: List[Cluster], 
    export_format: ExportFormat
) -> str:
    """Exporte les résultats dans le format demandé"""
    
    # Crée le répertoire d'export s'il n'existe pas
    export_dir = Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"keywords_clusters_{job_id}_{timestamp}"
    
    if export_format == ExportFormat.CSV:
        return export_to_csv(keywords, clusters, export_dir, base_filename)
    elif export_format == ExportFormat.XLSX:
        return export_to_excel(keywords, clusters, export_dir, base_filename)
    elif export_format == ExportFormat.JSON:
        return export_to_json(keywords, clusters, export_dir, base_filename)
    else:
        raise ValueError(f"Format d'export non supporté: {export_format}")

def export_to_csv(keywords: List[Keyword], clusters: List[Cluster], export_dir: Path, base_filename: str) -> str:
    """Exporte en CSV avec deux fichiers: mots-clés et clusters"""
    
    # Export des mots-clés
    keywords_data = []
    for kw in keywords:
        data = {
            'keyword': kw.keyword,
            'search_volume': kw.search_volume,
            'cpc': kw.cpc,
            'keyword_difficulty': kw.keyword_difficulty,
            'cluster_id': kw.cluster_id,
            'cluster_name': kw.cluster_name,
            'is_pivot': kw.is_pivot,
            'opportunity_score': kw.opportunity_score
        }
        
        # Ajoute les champs multi-cluster si le mot-clé en fait partie
        if kw.is_multi_cluster:
            data.update({
                'is_multi_cluster': kw.is_multi_cluster,
                'total_clusters_count': kw.total_clusters_count,
                'cluster_primary': kw.cluster_primary,
                'cluster_primary_name': kw.cluster_primary_name,
                'cluster_primary_probability': kw.cluster_primary_probability,
                'cluster_secondary': kw.cluster_secondary,
                'cluster_secondary_name': kw.cluster_secondary_name,
                'cluster_secondary_probability': kw.cluster_secondary_probability,
                'cluster_alt1': kw.cluster_alt1,
                'cluster_alt1_name': kw.cluster_alt1_name,
                'cluster_alt1_probability': kw.cluster_alt1_probability
            })
            
            # Ajoute les clusters supplémentaires si présents
            if kw.additional_clusters:
                for i, cluster_info in enumerate(kw.additional_clusters, start=2):
                    data[f'cluster_alt{i}'] = cluster_info.get('cluster_id')
                    data[f'cluster_alt{i}_name'] = cluster_info.get('cluster_name')
                    data[f'cluster_alt{i}_probability'] = cluster_info.get('probability')
        
        keywords_data.append(data)
    
    keywords_df = pd.DataFrame(keywords_data)
    keywords_file = export_dir / f"{base_filename}_keywords.csv"
    keywords_df.to_csv(keywords_file, index=False, encoding='utf-8')
    
    # Export des clusters
    clusters_data = []
    for cluster in clusters:
        clusters_data.append({
            'cluster_id': cluster.cluster_id,
            'cluster_name': cluster.cluster_name,
            'pivot_keyword': cluster.pivot_keyword,
            'keywords_count': cluster.keywords_count,
            'avg_search_volume': cluster.avg_search_volume,
            'avg_cpc': cluster.avg_cpc,
            'avg_difficulty': cluster.avg_difficulty,
            'opportunity_score': cluster.opportunity_score
        })
    
    clusters_df = pd.DataFrame(clusters_data)
    clusters_file = export_dir / f"{base_filename}_clusters.csv"
    clusters_df.to_csv(clusters_file, index=False, encoding='utf-8')
    
    logger.info(f"Export CSV créé: {keywords_file}")
    return str(keywords_file)

def export_to_excel(keywords: List[Keyword], clusters: List[Cluster], export_dir: Path, base_filename: str) -> str:
    """Exporte en Excel avec plusieurs onglets"""
    
    excel_file = export_dir / f"{base_filename}.xlsx"
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Onglet mots-clés
        keywords_data = []
        for kw in keywords:
            data = {
                'Mot-clé': kw.keyword,
                'Volume de recherche': kw.search_volume,
                'CPC': kw.cpc,
                'Difficulté': kw.keyword_difficulty,
                'ID Cluster': kw.cluster_id,
                'Nom du cluster': kw.cluster_name,
                'Mot-clé pivot': kw.is_pivot,
                'Score opportunité': kw.opportunity_score
            }
            
            # Ajoute les colonnes multi-cluster si pertinent
            if kw.is_multi_cluster:
                data.update({
                    'Multi-cluster': 'Oui',
                    'Nombre total clusters': kw.total_clusters_count,
                    'Cluster principal': kw.cluster_primary_name,
                    'Prob. principale': kw.cluster_primary_probability,
                    'Cluster secondaire': kw.cluster_secondary_name,
                    'Prob. secondaire': kw.cluster_secondary_probability,
                    'Cluster alternatif': kw.cluster_alt1_name,
                    'Prob. alternative': kw.cluster_alt1_probability
                })
                
                # Ajoute les clusters supplémentaires si présents
                if kw.additional_clusters:
                    additional_names = []
                    additional_probs = []
                    for cluster_info in kw.additional_clusters:
                        additional_names.append(cluster_info.get('cluster_name', 'N/A'))
                        additional_probs.append(f"{cluster_info.get('probability', 0):.3f}")
                    
                    data['Clusters supplémentaires'] = ' | '.join(additional_names)
                    data['Probs. supplémentaires'] = ' | '.join(additional_probs)
            else:
                data['Multi-cluster'] = 'Non'
                data['Nombre total clusters'] = 1
            
            keywords_data.append(data)
        
        keywords_df = pd.DataFrame(keywords_data)
        keywords_df.to_excel(writer, sheet_name='Mots-clés', index=False)
        
        # Onglet clusters
        clusters_data = []
        for cluster in clusters:
            clusters_data.append({
                'ID Cluster': cluster.cluster_id,
                'Nom du cluster': cluster.cluster_name,
                'Mot-clé pivot': cluster.pivot_keyword,
                'Nombre de mots-clés': cluster.keywords_count,
                'Volume moyen': cluster.avg_search_volume,
                'CPC moyen': cluster.avg_cpc,
                'Difficulté moyenne': cluster.avg_difficulty,
                'Score opportunité': cluster.opportunity_score
            })
        
        clusters_df = pd.DataFrame(clusters_data)
        clusters_df.to_excel(writer, sheet_name='Clusters', index=False)
        
        # Onglet résumé
        summary_data = {
            'Métrique': [
                'Nombre total de mots-clés',
                'Nombre de clusters',
                'Taille moyenne des clusters',
                'Volume total de recherche',
                'CPC moyen'
            ],
            'Valeur': [
                len(keywords),
                len(clusters),
                len(keywords) / len(clusters) if clusters else 0,
                sum(kw.search_volume or 0 for kw in keywords),
                sum(kw.cpc or 0 for kw in keywords) / len(keywords) if keywords else 0
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Résumé', index=False)
    
    logger.info(f"Export Excel créé: {excel_file}")
    return str(excel_file)

def export_to_json(keywords: List[Keyword], clusters: List[Cluster], export_dir: Path, base_filename: str) -> str:
    """Exporte en JSON structuré"""
    
    json_file = export_dir / f"{base_filename}.json"
    
    # Structure les données
    export_data = {
        'metadata': {
            'export_date': pd.Timestamp.now().isoformat(),
            'total_keywords': len(keywords),
            'total_clusters': len(clusters)
        },
        'clusters': [],
        'keywords': []
    }
    
    # Ajoute les clusters
    for cluster in clusters:
        cluster_data = {
            'cluster_id': cluster.cluster_id,
            'cluster_name': cluster.cluster_name,
            'pivot_keyword': cluster.pivot_keyword,
            'keywords_count': cluster.keywords_count,
            'metrics': {
                'avg_search_volume': cluster.avg_search_volume,
                'avg_cpc': cluster.avg_cpc,
                'avg_difficulty': cluster.avg_difficulty,
                'opportunity_score': cluster.opportunity_score
            }
        }
        export_data['clusters'].append(cluster_data)
    
    # Ajoute les mots-clés
    for kw in keywords:
        keyword_data = {
            'keyword': kw.keyword,
            'cluster_id': kw.cluster_id,
            'cluster_name': kw.cluster_name,
            'is_pivot': kw.is_pivot,
            'metrics': {
                'search_volume': kw.search_volume,
                'cpc': kw.cpc,
                'keyword_difficulty': kw.keyword_difficulty,
                'opportunity_score': kw.opportunity_score
            }
        }
        export_data['keywords'].append(keyword_data)
    
    # Sauvegarde
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Export JSON créé: {json_file}")
    return str(json_file) 