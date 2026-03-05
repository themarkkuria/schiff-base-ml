import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, rdMolDescriptors
from rdkit.Chem.Descriptors import MolWt, MolLogP, TPSA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix,
    classification_report, mean_absolute_error, mean_squared_error
)
from imblearn.over_sampling import SMOTE
import joblib
import json
from datetime import datetime


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


class MolecularDescriptorCalculator:
    """Calculate molecular descriptors for small molecules using RDKit."""
    
    @staticmethod
    def calculate_descriptors(smiles: str) -> Optional[Dict]:
        """Calculate comprehensive descriptors for a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        try:
            descriptors = {
                'MW': MolWt(mol),
                'LogP': MolLogP(mol),
                'TPSA': TPSA(mol),
                'NumHDonors': Lipinski.NumHDonors(mol),
                'NumHAcceptors': Lipinski.NumHAcceptors(mol),
                'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
                'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
                'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
                'NumRings': rdMolDescriptors.CalcNumRings(mol),
                'BertzCT': Descriptors.BertzCT(mol),
                'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
            }
            return descriptors
        except Exception as e:
            print(f"Error calculating descriptors for {smiles}: {e}")
            return None
    
    @staticmethod
    def calculate_fingerprints(smiles: str, radius: int = 2, nBits: int = 2048) -> Optional[np.ndarray]:
        """Calculate Morgan fingerprints for a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))


class DataProcessor:
    """Handle data preprocessing and feature engineering."""
    
    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        return pd.read_csv(filepath)
    
    @staticmethod
    def handle_imbalance(X: np.ndarray, y: np.ndarray, 
                         strategy: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance using SMOTE."""
        if strategy == 'smote':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        return X, y
    
    @staticmethod
    def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    @staticmethod
    def train_val_test_split(X: np.ndarray, y: np.ndarray, 
                              test_size: float = 0.2, 
                              val_size: float = 0.1,
                              random_state: int = 42,
                              stratify: bool = True) -> Tuple:
        """Split data into train, validation, and test sets."""
        if stratify:
            stratify_param = y
        else:
            stratify_param = None
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + val_size, 
            random_state=random_state, stratify=stratify_param
        )
        
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            random_state=random_state, stratify=stratify_param
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class ModelEvaluator:
    """Evaluate model performance."""
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None) -> Dict:
        """Evaluate classification model."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        
        return metrics
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate regression model."""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        }
    
    @staticmethod
    def cross_validate(model, X: np.ndarray, y: np.ndarray, 
                       cv: int = 5, scoring: str = 'roc_auc') -> Dict:
        """Perform cross-validation."""
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return {
            f'mean_{scoring}': scores.mean(),
            f'std_{scoring}': scores.std(),
            'scores': scores.tolist()
        }


class PipelineRunner:
    """Run end-to-end ML pipelines."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
    def save_model(self, model, filename: str):
        """Save trained model to disk."""
        filepath = MODELS_DIR / f"{filename}_{self.timestamp}.joblib"
        joblib.dump(model, filepath)
        return filepath
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        return joblib.load(filepath)
    
    def save_results(self, results: Dict, filename: str):
        """Save results to CSV."""
        filepath = RESULTS_DIR / f"{filename}_{self.timestamp}.csv"
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
        return filepath
    
    def save_config(self, config: Dict, filename: str = "config"):
        """Save pipeline configuration."""
        filepath = RESULTS_DIR / f"{filename}_{self.timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        return filepath


def load_training_data(category: str) -> pd.DataFrame:
    """Load training data for a specific category."""
    data_path = DATA_DIR / category / "training_data.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    else:
        raise FileNotFoundError(f"Training data not found for category: {category}")


def save_predictions(predictions: pd.DataFrame, category: str, prefix: str = "predictions"):
    """Save prediction results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / category / f"{prefix}_{timestamp}.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(filepath, index=False)
    return filepath


class ScaffoldHopper:
    """Perform scaffold hopping for lead optimization."""
    
    @staticmethod
    def get_scaffolds(smiles_list: List[str]) -> List[str]:
        """Extract Murcko scaffolds from SMILES."""
        scaffolds = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                scaffold = Chem.MolToSmiles(Chem.MolFromSmiles(
                    Chem.MolToSmiles(mol, isomericSmiles=False)
                ), rootedAtAtom=0)
                scaffolds.append(scaffold)
            else:
                scaffolds.append(None)
        return scaffolds
    
    @staticmethod
    def generate_analogs(smiles: str, n_analogs: int = 10) -> List[str]:
        """Generate scaffold analogs (placeholder - requires reaction templates)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        return [smiles] * n_analogs


class UnifiedRanker:
    """
    Unified ranking system for multi-category molecular candidates.
    Combines activity predictions, drug-likeness, diversity, and custom weights.
    """
    
    CATEGORY_CONFIGS = {
        'amp': {
            'activity_col': 'probability_active',
            'structure_col': 'sequence',
            'mol_weight_col': 'molecular_weight',
            'hydrophobicity_col': 'hydrophobicity',
            'charge_col': 'net_charge',
            'target_col': 'target',
        },
        'schiff_base': {
            'activity_col': 'probability_active',
            'structure_col': 'product',
            'mol_weight_col': 'MW',
            'logp_col': 'LogP',
            'tpsa_col': 'TPSA',
        },
        'repurposed': {
            'activity_col': 'probability_active',
            'structure_col': 'smiles',
            'similarity_col': 'similarity',
            'mol_weight_col': 'MW',
            'logp_col': 'LogP',
        },
        'polyphenol': {
            'activity_col': 'probability_active',
            'structure_col': 'smiles',
            'mol_weight_col': 'MW',
            'logp_col': 'LogP',
            'solubility_col': 'LogS_estimate',
        },
        'protac': {
            'activity_col': 'probability_active',
            'structure_col': 'smiles',
            'mol_weight_col': 'MW',
            'degrader_score_col': 'degrader_score',
        }
    }
    
    @staticmethod
    def rank_single_category(df: pd.DataFrame, category: str,
                            weights: Optional[Dict] = None) -> pd.DataFrame:
        """
        Rank candidates within a single category.
        
        Args:
            df: DataFrame with predictions
            category: One of 'amp', 'schiff_base', 'repurposed', 'polyphenol', 'protac'
            weights: Custom weights for scoring components
        
        Returns:
            DataFrame sorted by composite score with diversity filter applied
        """
        if weights is None:
            weights = {
                'activity': 0.5,
                'druglikeness': 0.3,
                'diversity': 0.2
            }
        
        config = UnifiedRanker.CATEGORY_CONFIGS.get(category, {})
        df = df.copy()
        
        df['activity_score'] = df.get(config.get('activity_col', 'probability_active'), 0.5)
        
        if category in ['schiff_base', 'repurposed', 'polyphenol', 'protac']:
            df['druglikeness_score'] = UnifiedRanker._calculate_druglikeness(df, config)
        elif category == 'amp':
            df['druglikeness_score'] = UnifiedRanker._calculate_amp_druglikeness(df)
        
        df['composite_score'] = (
            weights.get('activity', 0.5) * df['activity_score'] +
            weights.get('druglikeness', 0.3) * df['druglikeness_score'] +
            weights.get('diversity', 0.2) * df.get('diversity_score', 0.5)
        )
        
        df = df.sort_values('composite_score', ascending=False)
        
        return df
    
    @staticmethod
    def _calculate_druglikeness(df: pd.DataFrame, config: Dict) -> pd.Series:
        """Calculate drug-likeness score for small molecules."""
        scores = pd.Series(0.5, index=df.index)
        
        mw_col = config.get('mol_weight_col')
        if mw_col and mw_col in df.columns:
            scores += ((df[mw_col] < 500).astype(float) * 
                      (df[mw_col] > 100).astype(float) * 0.3)
        
        logp_col = config.get('logp_col')
        if logp_col and logp_col in df.columns:
            scores += ((df[logp_col] >= -1).astype(float) * 
                      (df[logp_col] <= 5).astype(float) * 0.3)
        
        tpsa_col = config.get('tpsa_col')
        if tpsa_col and tpsa_col in df.columns:
            scores += ((df[tpsa_col] < 140).astype(float) * 0.2)
        
        hbd_col = config.get('mol_weight_col')
        if hbd_col and 'NumHDonors' in df.columns:
            scores += (df['NumHDonors'] <= 5).astype(float) * 0.1
        if hbd_col and 'NumHAcceptors' in df.columns:
            scores += (df['NumHAcceptors'] <= 10).astype(float) * 0.1
        
        return scores.clip(0, 1)
    
    @staticmethod
    def _calculate_amp_druglikeness(df: pd.DataFrame) -> pd.Series:
        """Calculate drug-likeness score for AMPs."""
        scores = pd.Series(0.5, index=df.index)
        
        if 'length' in df.columns:
            scores += ((df['length'] >= 10).astype(float) * 
                      (df['length'] <= 40).astype(float) * 0.3)
        
        if 'hydrophobic_ratio' in df.columns:
            scores += ((df['hydrophobic_ratio'] >= 0.3).astype(float) * 
                      (df['hydrophobic_ratio'] <= 0.6).astype(float) * 0.3)
        
        if 'net_charge' in df.columns:
            scores += ((df['net_charge'].abs() >= 1).astype(float) * 
                      (df['net_charge'].abs() <= 8).astype(float) * 0.2)
        
        if 'aromatic_ratio' in df.columns:
            scores += (df['aromatic_ratio'] < 0.2).astype(float) * 0.2
        
        return scores.clip(0, 1)
    
    @staticmethod
    def rank_multi_category(dfs: Dict[str, pd.DataFrame],
                           weights: Optional[Dict] = None,
                           top_k: int = 50) -> pd.DataFrame:
        """
        Rank candidates across multiple categories.
        
        Args:
            dfs: Dict mapping category name to DataFrame with predictions
            weights: Category-specific weights (e.g., {'amp': 1.2, 'schiff_base': 1.0})
            top_k: Number of top candidates to return per category
        
        Returns:
            Combined DataFrame with all candidates ranked
        """
        if weights is None:
            weights = {cat: 1.0 for cat in dfs.keys()}
        
        all_candidates = []
        
        for category, df in dfs.items():
            df_cat = df.copy()
            df_cat['category'] = category
            df_cat['category_weight'] = weights.get(category, 1.0)
            
            df_ranked = UnifiedRanker.rank_single_category(df_cat, category)
            
            if len(df_ranked) > top_k:
                df_ranked = df_ranked.head(top_k)
            
            all_candidates.append(df_ranked)
        
        combined = pd.concat(all_candidates, ignore_index=True)
        
        max_activity = combined['activity_score'].max()
        combined['normalized_activity'] = combined['activity_score'] / max_activity if max_activity > 0 else 0
        
        combined['final_score'] = (
            combined['normalized_activity'] * 
            combined['category_weight'] * 
            combined['druglikeness_score']
        )
        
        combined = combined.sort_values('final_score', ascending=False)
        
        return combined.head(top_k * len(dfs))
    
    @staticmethod
    def diversity_filter(df: pd.DataFrame, 
                        category: str,
                        threshold: float = 0.8,
                        fps_col: str = 'fingerprints') -> pd.DataFrame:
        """
        Apply diversity filter using Tanimoto similarity.
        
        Args:
            df: DataFrame with candidates
            category: Molecule category
            threshold: Max similarity (remove if > threshold to existing)
            fps_col: Column containing fingerprints (if pre-computed)
        
        Returns:
            DataFrame with diverse subset
        """
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
        
        if category == 'amp':
            return df.head(100)
        
        smiles_list = df['smiles'].tolist() if 'smiles' in df.columns else []
        if not smiles_list:
            return df
        
        selected = []
        selected_fps = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            
            is_diverse = True
            for sel_fp in selected_fps:
                sim = DataStructs.TanimotoSimilarity(fp, sel_fp)
                if sim > threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(i)
                selected_fps.append(fp)
            
            if len(selected) >= 100:
                break
        
        return df.iloc[selected] if selected else df


def get_category_predictions(category: str, n_samples: int = 100) -> pd.DataFrame:
    """
    Get sample predictions for a category (for testing multi-category ranking).
    """
    if category == 'amp':
        from modules.amp_module import generate_synthetic_amp_data
        return generate_synthetic_amp_data(n_samples)
    elif category == 'schiff_base':
        pass
    elif category == 'repurposed':
        from modules.repurposed_module import generate_synthetic_repurposing_data
        return generate_synthetic_repurposing_data(n_samples)
    elif category == 'polyphenol':
        from modules.polyphenols_module import generate_synthetic_polyphenol_data
        return generate_synthetic_polyphenol_data(n_samples)
    else:
        return pd.DataFrame()
