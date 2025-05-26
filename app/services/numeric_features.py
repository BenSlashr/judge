"""
Service pour extraire et traiter les features numériques des mots-clés
"""

import re
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

@dataclass
class NumericFeature:
    """Représente une feature numérique extraite d'un mot-clé"""
    value: float
    unit: str
    type: str  # 'price', 'year', 'quantity', 'percentage', etc.
    original_text: str
    position: int  # Position dans le mot-clé

class NumericFeaturesExtractor:
    """Service d'extraction et traitement des features numériques"""
    
    def __init__(self):
        self.patterns = {
            'price': [
                r'(\d+(?:[,.]?\d+)*)\s*(?:€|euros?|eur)\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:\$|dollars?|usd)\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:£|pounds?|gbp)\b',
                r'(?:€|euros?|eur)\s*(\d+(?:[,.]?\d+)*)\b',
                r'(?:\$|dollars?|usd)\s*(\d+(?:[,.]?\d+)*)\b',
                r'(?:£|pounds?|gbp)\s*(\d+(?:[,.]?\d+)*)\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:centimes?|cents?)\b'
            ],
            'year': [
                r'\b(19\d{2}|20\d{2})\b',  # 1900-2099
                r'\b(\d{4})\b(?=\s*(?:année|an|years?|yr))',
            ],
            'quantity': [
                r'(\d+(?:[,.]?\d+)*)\s*(?:kg|kilogrammes?|kilos?)\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:g|grammes?)\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:l|litres?)\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:ml|millilitres?)\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:m|mètres?|metres?)\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:cm|centimètres?|centimetres?)\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:mm|millimètres?|millimetres?)\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:pièces?|pieces?|unités?|units?)\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:pack|lot|sets?)\b'
            ],
            'percentage': [
                r'(\d+(?:[,.]?\d+)*)\s*%',
                r'(\d+(?:[,.]?\d+)*)\s*(?:pourcent|percent)\b'
            ],
            'size': [
                r'(\d+(?:[,.]?\d+)*)\s*(?:mo|mb|go|gb|to|tb)\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:pouces?|inches?|")\b',
                r'(\d+(?:[,.]?\d+)*)\s*(?:mm|cm|m)\b(?!\w)',
            ],
            'version': [
                r'\bv?(\d+(?:\.\d+)*)\b',
                r'\b(?:version|ver)\s*(\d+(?:\.\d+)*)\b',
                r'\b(\d+(?:\.\d+)*)\s*(?:version|ver)\b'
            ],
            'rating': [
                r'(\d+(?:[,.]?\d+)*)\s*(?:étoiles?|stars?|★+)\b',
                r'(\d+(?:[,.]?\d+)*)/5\b',
                r'(\d+(?:[,.]?\d+)*)/10\b'
            ]
        }
        
        self.unit_mappings = {
            # Monnaies
            '€': 'EUR', 'euros': 'EUR', 'euro': 'EUR', 'eur': 'EUR',
            '$': 'USD', 'dollars': 'USD', 'dollar': 'USD', 'usd': 'USD',
            '£': 'GBP', 'pounds': 'GBP', 'pound': 'GBP', 'gbp': 'GBP',
            'centimes': 'CENT', 'centime': 'CENT', 'cents': 'CENT', 'cent': 'CENT',
            
            # Poids
            'kg': 'KG', 'kilogrammes': 'KG', 'kilogramme': 'KG', 'kilos': 'KG', 'kilo': 'KG',
            'g': 'G', 'grammes': 'G', 'gramme': 'G',
            
            # Volume
            'l': 'L', 'litres': 'L', 'litre': 'L',
            'ml': 'ML', 'millilitres': 'ML', 'millilitre': 'ML',
            
            # Distance
            'm': 'M', 'mètres': 'M', 'mètre': 'M', 'metres': 'M', 'metre': 'M',
            'cm': 'CM', 'centimètres': 'CM', 'centimètre': 'CM', 'centimetres': 'CM', 'centimetre': 'CM',
            'mm': 'MM', 'millimètres': 'MM', 'millimètre': 'MM', 'millimetres': 'MM', 'millimetre': 'MM',
            
            # Quantité
            'pièces': 'UNIT', 'pièce': 'UNIT', 'pieces': 'UNIT', 'piece': 'UNIT',
            'unités': 'UNIT', 'unité': 'UNIT', 'units': 'UNIT', 'unit': 'UNIT',
            'pack': 'PACK', 'lot': 'LOT', 'sets': 'SET', 'set': 'SET',
            
            # Pourcentage
            '%': 'PERCENT', 'pourcent': 'PERCENT', 'percent': 'PERCENT',
            
            # Taille informatique
            'mo': 'MB', 'mb': 'MB', 'go': 'GB', 'gb': 'GB', 'to': 'TB', 'tb': 'TB',
            
            # Taille physique
            'pouces': 'INCH', 'pouce': 'INCH', 'inches': 'INCH', 'inch': 'INCH', '"': 'INCH',
            
            # Rating
            'étoiles': 'STAR', 'étoile': 'STAR', 'stars': 'STAR', 'star': 'STAR', '★': 'STAR'
        }
        
        # Normalisation des unités par type
        self.unit_normalization = {
            'price': {'CENT': 0.01, 'EUR': 1.0, 'USD': 0.85, 'GBP': 1.15},  # Vers EUR
            'quantity': {
                'G': 0.001, 'KG': 1.0,  # Vers KG
                'ML': 0.001, 'L': 1.0,  # Vers L
                'MM': 0.001, 'CM': 0.01, 'M': 1.0,  # Vers M
                'UNIT': 1.0, 'PACK': 10.0, 'LOT': 5.0, 'SET': 3.0  # Vers UNIT
            },
            'size': {'MB': 1.0, 'GB': 1024.0, 'TB': 1024*1024},  # Vers MB
            'percentage': {'PERCENT': 1.0},
            'year': {'YEAR': 1.0},
            'version': {'VERSION': 1.0},
            'rating': {'STAR': 1.0, 'SCALE5': 1.0, 'SCALE10': 0.5}  # Vers 5 étoiles
        }
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def extract_numeric_features(self, keywords: List[str]) -> List[List[NumericFeature]]:
        """Extrait les features numériques de chaque mot-clé"""
        logger.info(f"🔢 Extraction des features numériques pour {len(keywords)} mots-clés")
        
        all_features = []
        
        for keyword in keywords:
            features = []
            keyword_lower = keyword.lower()
            
            for feature_type, patterns in self.patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, keyword_lower, re.IGNORECASE)
                    
                    for match in matches:
                        try:
                            # Extrait la valeur numérique
                            value_str = match.group(1).replace(',', '.')
                            value = float(value_str)
                            
                            # Extrait l'unité si présente
                            full_match = match.group(0)
                            unit_match = re.search(r'[a-zA-Z€$£%★"]+', full_match)
                            unit = unit_match.group(0) if unit_match else ''
                            
                            # Normalise l'unité
                            normalized_unit = self.unit_mappings.get(unit.lower(), unit.upper())
                            
                            features.append(NumericFeature(
                                value=value,
                                unit=normalized_unit,
                                type=feature_type,
                                original_text=full_match,
                                position=match.start()
                            ))
                            
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Erreur extraction numérique '{match.group(0)}': {e}")
                            continue
            
            all_features.append(features)
        
        # Statistiques
        total_features = sum(len(f) for f in all_features)
        keywords_with_numbers = sum(1 for f in all_features if f)
        
        logger.info(f"✅ Features numériques extraites: {total_features} features, "
                   f"{keywords_with_numbers}/{len(keywords)} mots-clés concernés")
        
        return all_features
    
    def normalize_features(self, all_features: List[List[NumericFeature]]) -> List[Dict[str, float]]:
        """Normalise les features numériques en vecteurs"""
        logger.info("🔧 Normalisation des features numériques")
        
        # Collecte toutes les valeurs par type/unité pour la normalisation
        value_collections = {}
        unit_collections = set()
        
        for features in all_features:
            for feature in features:
                key = f"{feature.type}_{feature.unit}"
                if key not in value_collections:
                    value_collections[key] = []
                
                # Normalise la valeur selon le type et l'unité
                normalized_value = self._normalize_value(feature)
                value_collections[key].append(normalized_value)
                unit_collections.add(feature.unit)
        
        # Calcule les statistiques pour la normalisation
        stats = {}
        for key, values in value_collections.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values) if len(values) > 1 else 1.0,
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Encode les unités
        all_units = list(unit_collections)
        if all_units:
            self.label_encoder.fit(all_units)
        
        # Génère les vecteurs de features normalisées
        normalized_vectors = []
        
        for features in all_features:
            vector = {}
            
            # Features numériques normalisées par type
            type_sums = {}
            type_counts = {}
            
            for feature in features:
                normalized_value = self._normalize_value(feature)
                key = f"{feature.type}_{feature.unit}"
                
                # Z-score normalization
                if key in stats and stats[key]['std'] > 0:
                    z_score = (normalized_value - stats[key]['mean']) / stats[key]['std']
                    # Clamp les valeurs extrêmes
                    z_score = np.clip(z_score, -3, 3)
                else:
                    z_score = 0.0
                
                # Agrège par type
                if feature.type not in type_sums:
                    type_sums[feature.type] = 0.0
                    type_counts[feature.type] = 0
                
                type_sums[feature.type] += z_score
                type_counts[feature.type] += 1
            
            # Moyennes par type
            for feature_type in ['price', 'year', 'quantity', 'percentage', 'size', 'version', 'rating']:
                if feature_type in type_sums:
                    vector[f"num_{feature_type}"] = type_sums[feature_type] / type_counts[feature_type]
                else:
                    vector[f"num_{feature_type}"] = 0.0
            
            # Features booléennes (présence)
            vector['has_numbers'] = 1.0 if features else 0.0
            vector['num_count'] = min(len(features) / 5.0, 1.0)  # Normalisé sur 5 max
            
            # Features d'unités (one-hot principales)
            main_units = ['EUR', 'USD', 'KG', 'L', 'M', 'PERCENT', 'STAR', 'YEAR']
            for unit in main_units:
                vector[f"unit_{unit}"] = 1.0 if any(f.unit == unit for f in features) else 0.0
            
            normalized_vectors.append(vector)
        
        logger.info(f"✅ Vecteurs normalisés créés: {len(normalized_vectors)} vecteurs, "
                   f"{len(normalized_vectors[0]) if normalized_vectors else 0} dimensions chacun")
        
        return normalized_vectors
    
    def _normalize_value(self, feature: NumericFeature) -> float:
        """Normalise une valeur selon son type et son unité"""
        value = feature.value
        
        # Normalisation par type
        if feature.type in self.unit_normalization:
            unit_norms = self.unit_normalization[feature.type]
            if feature.unit in unit_norms:
                value *= unit_norms[feature.unit]
        
        # Transformations logarithmiques pour certains types
        if feature.type in ['price', 'quantity', 'size'] and value > 0:
            value = np.log1p(value)  # log(1 + x)
        elif feature.type == 'year' and value > 1900:
            value = value - 2000  # Centré sur l'an 2000
        
        return value
    
    def create_enhanced_embeddings(self, 
                                 embeddings: np.ndarray, 
                                 keywords: List[str],
                                 numeric_weight: float = 0.1) -> np.ndarray:
        """Crée des embeddings enrichis avec les features numériques"""
        logger.info(f"🔗 Enrichissement des embeddings avec features numériques (poids: {numeric_weight})")
        
        # Extrait les features numériques
        all_features = self.extract_numeric_features(keywords)
        numeric_vectors = self.normalize_features(all_features)
        
        if not numeric_vectors:
            logger.warning("⚠️ Aucune feature numérique trouvée, retour embeddings originaux")
            return embeddings
        
        # Convertit en numpy array
        feature_dim = len(numeric_vectors[0])
        numeric_matrix = np.zeros((len(keywords), feature_dim))
        
        for i, vector in enumerate(numeric_vectors):
            for j, (key, value) in enumerate(vector.items()):
                numeric_matrix[i, j] = value
        
        # Normalise les features numériques
        if numeric_matrix.std() > 0:
            numeric_matrix = (numeric_matrix - numeric_matrix.mean(axis=0)) / (numeric_matrix.std(axis=0) + 1e-8)
        
        # Pondère et concatène
        weighted_numeric = numeric_matrix * numeric_weight
        enhanced_embeddings = np.concatenate([embeddings, weighted_numeric], axis=1)
        
        logger.info(f"✅ Embeddings enrichis: {embeddings.shape} → {enhanced_embeddings.shape}")
        
        return enhanced_embeddings
    
    def calculate_numeric_distance(self, 
                                 keywords: List[str],
                                 distance_matrix: np.ndarray,
                                 numeric_weight: float = 0.1) -> np.ndarray:
        """Calcule une matrice de distance hybride intégrant les distances numériques"""
        logger.info(f"📏 Calcul de distance hybride numérique (poids: {numeric_weight})")
        
        # Extrait les features numériques
        all_features = self.extract_numeric_features(keywords)
        
        if not any(all_features):
            logger.info("⚠️ Aucune feature numérique, retour matrice originale")
            return distance_matrix
        
        n = len(keywords)
        numeric_distance_matrix = np.zeros((n, n))
        
        # Calcule les distances numériques par paires
        for i in range(n):
            for j in range(i + 1, n):
                num_dist = self._calculate_pairwise_numeric_distance(all_features[i], all_features[j])
                numeric_distance_matrix[i, j] = num_dist
                numeric_distance_matrix[j, i] = num_dist
        
        # Normalise la matrice de distances numériques
        if numeric_distance_matrix.max() > 0:
            numeric_distance_matrix = numeric_distance_matrix / numeric_distance_matrix.max()
        
        # Combine avec la matrice de distance originale
        hybrid_distance = distance_matrix + numeric_weight * numeric_distance_matrix
        
        logger.info(f"✅ Distance hybride calculée: max original={distance_matrix.max():.3f}, "
                   f"max numérique={numeric_distance_matrix.max():.3f}, "
                   f"max hybride={hybrid_distance.max():.3f}")
        
        return hybrid_distance
    
    def _calculate_pairwise_numeric_distance(self, 
                                           features1: List[NumericFeature], 
                                           features2: List[NumericFeature]) -> float:
        """Calcule la distance numérique entre deux listes de features"""
        
        if not features1 and not features2:
            return 0.0
        
        if not features1 or not features2:
            return 1.0  # Distance maximale si l'un n'a pas de features
        
        # Groupe les features par type
        types1 = {f.type: f for f in features1}
        types2 = {f.type: f for f in features2}
        
        distances = []
        all_types = set(types1.keys()) | set(types2.keys())
        
        for feature_type in all_types:
            if feature_type in types1 and feature_type in types2:
                f1, f2 = types1[feature_type], types2[feature_type]
                
                # Distance basée sur les valeurs normalisées
                v1 = self._normalize_value(f1)
                v2 = self._normalize_value(f2)
                
                # Distance relative
                if max(abs(v1), abs(v2)) > 0:
                    rel_dist = abs(v1 - v2) / max(abs(v1), abs(v2))
                else:
                    rel_dist = 0.0
                
                # Bonus si même unité
                unit_bonus = 0.0 if f1.unit == f2.unit else 0.2
                
                distances.append(min(rel_dist + unit_bonus, 1.0))
            else:
                # Pénalité si l'un a le type et pas l'autre
                distances.append(0.5)
        
        return np.mean(distances) if distances else 0.0

# Instance globale
numeric_features_service = NumericFeaturesExtractor() 