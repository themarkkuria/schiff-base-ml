"""
Antimicrobial Peptide (AMP) Module
Generate, featurize, train, and predict AMP activity.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import json
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.pipeline import DataProcessor, ModelEvaluator, PipelineRunner


AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
HYDROPHOBIC_AA = list('AILMFVWP')
HYDROPHILIC_AA = list('RKDENQSTY')
POSITIVE_AA = list('RKH')
NEGATIVE_AA = list('DE')


class AMPGenerator:
    """Generate novel AMP sequences."""
    
    @staticmethod
    def generate_random_sequence(min_len: int = 10, max_len: int = 50) -> str:
        """Generate random amino acid sequence."""
        length = random.randint(min_len, max_len)
        return ''.join(random.choices(AMINO_ACIDS, k=length))
    
    @staticmethod
    def generate_from_motif(motif: str, n_variants: int = 100) -> List[str]:
        """Generate variants from a motif pattern."""
        variants = []
        for _ in range(n_variants):
            variant = motif
            for aa in AMINO_ACIDS:
                if aa in variant:
                    variant = variant.replace(aa, random.choice(AMINO_ACIDS), 1)
            variants.append(variant)
        return variants
    
    @staticmethod
    def generate_helical_sequences(n: int = 100, 
                                     hydrophobic_ratio: float = 0.4,
                                     min_charge: float = 1.0) -> List[str]:
        """Generate sequences optimized for alpha-helical structure."""
        sequences = []
        for _ in range(n):
            length = random.randint(15, 30)
            seq = []
            n_positive = random.randint(2, 5)
            n_hydrophobic = int(length * hydrophobic_ratio)
            
            for i in range(length):
                if i < n_positive:
                    seq.append(random.choice(POSITIVE_AA))
                elif i < n_positive + n_hydrophobic:
                    seq.append(random.choice(HYDROPHOBIC_AA))
                else:
                    seq.append(random.choice(AMINO_ACIDS))
            
            sequences.append(''.join(seq))
        
        return sequences
    
    @staticmethod
    def generate_cyclic_sequences(n: int = 100) -> List[str]:
        """Generate cyclic peptide sequences."""
        sequences = AMPGenerator.generate_helical_sequences(n)
        return ['C' + seq[1:-1] + 'C' if len(seq) > 3 else seq for seq in sequences]


class AMPFeaturizer:
    """Featurize AMP sequences for ML models."""
    
    AA_PROPERTIES = {
        'hydrophobicity': {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        },
        'charge': {
            'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
            'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
            'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
            'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
        },
        'volume': {
            'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
            'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
            'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
            'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
        }
    }
    
    @staticmethod
    def one_hot_encode(sequence: str, max_len: int = 50) -> np.ndarray:
        """One-hot encode amino acid sequence."""
        aa_to_idx = {aa: i for i, aa in enumerate(sorted(AMINO_ACIDS))}
        encoding = np.zeros((max_len, len(AMINO_ACIDS)))
        
        for i, aa in enumerate(sequence[:max_len]):
            if aa in aa_to_idx:
                encoding[i, aa_to_idx[aa]] = 1
        
        return encoding.flatten()
    
    @staticmethod
    def calculate_descriptors(sequence: str) -> Dict:
        """Calculate comprehensive physicochemical descriptors."""
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return {}
        
        counts = Counter(sequence)
        
        hydrophobicity = sum(
            AMPFeaturizer.AA_PROPERTIES['hydrophobicity'].get(aa, 0) * count 
            for aa, count in counts.items()
        ) / length
        
        net_charge = sum(
            AMPFeaturizer.AA_PROPERTIES['charge'].get(aa, 0) * count 
            for aa, count in counts.items()
        )
        
        avg_volume = sum(
            AMPFeaturizer.AA_PROPERTIES['volume'].get(aa, 0) * count 
            for aa, count in counts.items()
        ) / length
        
        hydrophobic_ratio = sum(counts.get(aa, 0) for aa in HYDROPHOBIC_AA) / length
        hydrophilic_ratio = sum(counts.get(aa, 0) for aa in HYDROPHILIC_AA) / length
        positive_ratio = sum(counts.get(aa, 0) for aa in POSITIVE_AA) / length
        negative_ratio = sum(counts.get(aa, 0) for aa in NEGATIVE_AA) / length
        aromatic_ratio = sum(counts.get(aa, 0) for aa in 'FWY') / length
        
        charge_density = net_charge / length
        
        descriptors = {
            'length': length,
            'hydrophobicity': hydrophobicity,
            'net_charge': net_charge,
            'avg_volume': avg_volume,
            'hydrophobic_ratio': hydrophobic_ratio,
            'hydrophilic_ratio': hydrophilic_ratio,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'aromatic_ratio': aromatic_ratio,
            'charge_density': charge_density,
            'net_charge_abs': abs(net_charge),
            'charge_balance': positive_ratio - negative_ratio if (positive_ratio + negative_ratio) > 0 else 0
        }
        
        return descriptors
    
    @staticmethod
    def featurize_batch(sequences: List[str]) -> pd.DataFrame:
        """Featurize a batch of sequences."""
        all_descriptors = []
        for seq in sequences:
            desc = AMPFeaturizer.calculate_descriptors(seq)
            desc['sequence'] = seq
            all_descriptors.append(desc)
        
        return pd.DataFrame(all_descriptors)


class AMPModel:
    """Train and predict AMP activity."""
    
    def __init__(self, model_type: str = 'rf'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_cols = None
        
    def train(self, X: np.ndarray, y: np.ndarray, 
              tune_hyperparams: bool = False) -> Dict:
        """Train AMP activity prediction model."""
        
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                param_grid, cv=cv, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            if self.model_type == 'rf':
                self.model = RandomForestClassifier(
                    n_estimators=200, max_depth=15, 
                    min_samples_split=2, random_state=42, n_jobs=-1
                )
            elif self.model_type == 'gb':
                self.model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1
                )
            self.model.fit(X, y)
            best_params = {}
        
        self.feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
        
        return {'best_params': best_params, 'model_type': self.model_type}
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict AMP activity (probability and class)."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        return y_pred, y_prob
    
    def predict_sequences(self, sequences: List[str]) -> pd.DataFrame:
        """Predict activity for a list of sequences."""
        features_df = AMPFeaturizer.featurize_batch(sequences)
        
        if self.feature_cols is None:
            self.feature_cols = [col for col in features_df.columns 
                                  if col != 'sequence']
        
        X = features_df[self.feature_cols].values
        
        y_pred, y_prob = self.predict(X)
        
        results = features_df.copy()
        results['predicted_activity'] = y_pred
        results['probability_active'] = y_prob
        
        return results
    
    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'AMPModel':
        """Load model from disk."""
        data = joblib.load(filepath)
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.feature_cols = data['feature_cols']
        return instance


class AMPRanker:
    """Rank AMP candidates by viability."""
    
    @staticmethod
    def rank_candidates(predictions: pd.DataFrame, 
                         weights: Optional[Dict] = None) -> pd.DataFrame:
        """Rank candidates by multiple criteria."""
        
        if weights is None:
            weights = {
                'activity': 0.5,
                'hydrophobicity': 0.2,
                'charge': 0.15,
                'length': 0.15
            }
        
        df = predictions.copy()
        
        df['activity_score'] = df['probability_active']
        
        df['hydrophobicity_score'] = (df['hydrophobicity'] - df['hydrophobicity'].min()) / \
                                      (df['hydrophobicity'].max() - df['hydrophobicity'].min() + 1e-8)
        
        df['charge_score'] = 1 - np.abs(df['net_charge']) / (np.abs(df['net_charge']).max() + 1e-8)
        
        df['length_score'] = 1 - np.abs(df['length'] - 25) / 25
        
        df['viability_score'] = (
            weights['activity'] * df['activity_score'] +
            weights['hydrophobicity'] * df['hydrophobicity_score'] +
            weights['charge'] * df['charge_score'] +
            weights['length'] * df['length_score']
        )
        
        df = df.sort_values('viability_score', ascending=False)
        
        return df


def generate_synthetic_amp_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic AMP training data with labels."""
    
    active_sequences = AMPGenerator.generate_helical_sequences(n=n_samples // 2)
    inactive_sequences = [AMPGenerator.generate_random_sequence(10, 50) 
                          for _ in range(n_samples // 2)]
    
    all_sequences = active_sequences + inactive_sequences
    labels = [1] * len(active_sequences) + [0] * len(inactive_sequences)
    
    features_df = AMPFeaturizer.featurize_batch(all_sequences)
    features_df['activity'] = labels
    
    return features_df


if __name__ == "__main__":
    print("Generating synthetic AMP training data...")
    train_data = generate_synthetic_amp_data(1000)
    print(f"Generated {len(train_data)} samples")
    print(f"Active: {train_data['activity'].sum()}, Inactive: {len(train_data) - train_data['activity'].sum()}")
    
    feature_cols = [col for col in train_data.columns if col not in ['sequence', 'activity']]
    X = train_data[feature_cols].values
    y = train_data['activity'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = AMPModel(model_type='rf')
    model.train(X_train, y_train)
    
    y_pred, y_prob = model.predict(X_test)
    
    print(f"\nModel Performance:")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
    print(classification_report(y_test, y_pred))
    
    model.save("models/amp_model.joblib")
    print("\nModel saved to models/amp_model.joblib")
