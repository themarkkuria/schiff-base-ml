"""
Peptidomimetics & PROTAC Module
Design and prediction of antibacterial PROTACs and peptidomimetics.
Focus on bacterial protein degraders (e.g., LpxC inhibitors).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import json

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, rdMolDescriptors
from rdkit.Chem import BRICS, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Descriptors import MolWt, MolLogP, TPSA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import pubchempy as pcp
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.pipeline import MolecularDescriptorCalculator


PROTAC_DB_URL = "https://protacdb.biohackers.cn"


LINKER_TYPES = {
    "short": "CCC",
    "medium": "CCCCCC",
    "long": "CCCCCCCCCC",
    "PEG": "CCOCCOCC",
    "aryl": "c1ccccc1",
}

PROTEASE_TARGETS = {
    "LpxC": {"name": "LpxC (UDP-3-O-(R-3-hydroxyacyl)-glucosamine N-deacetylase)", "bacteria": "Gram-negative"},
    "ClpP": {"name": "ClpP (Caseinolytic protease)", "bacteria": "Gram-positive"},
    "FtsH": {"name": "FtsH (membrane protease)", "bacteria": "Gram-negative"},
    "DegP": {"name": "DegP (serine protease)", "bacteria": "Gram-negative"},
}


class PROTACGenerator:
    """Generate PROTAC-like molecules for antibacterial targets."""
    
    @staticmethod
    def generate_three_part_protac(target_warhead: str, 
                                   e3_ligase: str = None,
                                   linker: str = None) -> List[str]:
        """
        Generate three-part PROTACs: Warhead-Linker-E3 Ligase.
        
        Args:
            target_warhead: SMILES of target-binding warhead
            e3_ligase: SMILES of E3 ligase binder (optional)
            linker: SMILES of linker (optional)
        
        Returns:
            List of generated PROTAC SMILES
        """
        if e3_ligase is None:
            e3_ligase = "O=C(NCC1=CC=CC=C1)C2=CC=C(O)C=C2"
        if linker is None:
            linker = "CCCCCC"
        
        protacs = []
        
        for lt, linker_smiles in LINKER_TYPES.items():
            combined = f"{target_warhead}{linker_smiles}{e3_ligase}"
            mol = Chem.MolFromSmiles(combined)
            if mol is not None:
                protacs.append(combined)
        
        return protacs
    
    @staticmethod
    def generate_peptidomimetic(simple_peptide: str, n_variants: int = 50) -> List[str]:
        """
        Generate peptidomimetic small molecules from peptide template.
        Replaces peptide backbone with non-natural scaffolds.
        """
        replacements = [
            ("C", "c1ccccc1"),
            ("CC", "C1CCCCC1"),
            ("N", "N1CCCC1"),
            ("O", "O1CCCC1"),
            ("S", "S1CCCC1"),
        ]
        
        mimetics = []
        for _ in range(n_variants):
            template = simple_peptide
            for target, replacement in random.sample(replacements, k=min(3, len(replacements))):
                template = template.replace(target, replacement, 1)
            
            mol = Chem.MolFromSmiles(template)
            if mol is not None:
                mimetics.append(template)
        
        return mimetics
    
    @staticmethod
    def fragment_based_generation(warhead: str, n_fragments: int = 20) -> List[str]:
        """Generate PROTACs by fragment-based approach."""
        from rdkit.Chem import BRICS
        
        mol = Chem.MolFromSmiles(warhead)
        if mol is None:
            return []
        
        fragments = list(BRICS.BRICSDecompose(mol))
        
        generated = []
        for _ in range(n_fragments):
            selected = random.sample(fragments, k=min(3, len(fragments)))
            combined = ".".join(selected)
            
            mol_comb = Chem.MolFromSmiles(combined)
            if mol_comb is not None:
                generated.append(combined)
        
        return generated


class PROTACFeaturizer:
    """Featurize PROTACs for ML models."""
    
    @staticmethod
    def calculate_degrader_descriptors(smiles: str) -> Optional[Dict]:
        """Calculate descriptors specific to PROTACs/degraders."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        try:
            descriptors = {
                'MW': MolWt(mol),
                'LogP': MolLogP(mol),
                'TPSA': TPSA(mol),
                'NumHBD': Lipinski.NumHDonors(mol),
                'NumHBA': Lipinski.NumHAcceptors(mol),
                'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
                'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'NumRings': rdMolDescriptors.CalcNumRings(mol),
                'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
                'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
                'NumAmideBonds': smiles.count("C(=O)N") + smiles.count("NC(=O)"),
                'ChainLength': PROTACFeaturizer._estimate_chain_length(smiles),
                'BranchingIndex': PROTACFeaturizer._calculate_branching(mol),
            }
            return descriptors
        except Exception:
            return None
    
    @staticmethod
    def _estimate_chain_length(smiles: str) -> int:
        """Estimate the linker chain length."""
        chain_chars = sum(1 for c in smiles if c in 'CNOPS')
        return max(1, chain_chars // 3)
    
    @staticmethod
    def _calculate_branching(mol) -> float:
        """Calculate branching index."""
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return 0
        
        branching = 0
        for atom in mol.GetAtoms():
            neighbors = atom.GetNeighbors()
            if len(neighbors) > 2:
                branching += 1
        
        return branching / num_atoms if num_atoms > 0 else 0
    
    @staticmethod
    def featurize_smiles(smiles: str) -> Optional[Dict]:
        """Full featurization for PROTACs."""
        desc = MolecularDescriptorCalculator.calculate_descriptors(smiles)
        degrader_desc = PROTACFeaturizer.calculate_degrader_descriptors(smiles)
        
        if desc and degrader_desc:
            return {**desc, **degrader_desc}
        return desc or degrader_desc
    
    @staticmethod
    def featurize_batch(smiles_list: List[str]) -> pd.DataFrame:
        """Featurize a batch of PROTACs."""
        all_features = []
        
        for smiles in smiles_list:
            features = PROTACFeaturizer.featurize_smiles(smiles)
            if features:
                features['smiles'] = smiles
                all_features.append(features)
        
        return pd.DataFrame(all_features)


class PROTACModel:
    """Train and predict PROTAC/degrader activity."""
    
    def __init__(self, model_type: str = 'rf'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_cols = None
        
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the model."""
        
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=15,
                min_samples_split=5, class_weight='balanced',
                random_state=42, n_jobs=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42
            )
        
        self.model.fit(X, y)
        
        return {'model_type': self.model_type}
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict activity."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        y_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        return y_pred, y_prob
    
    def predict_smiles(self, smiles_list: List[str]) -> pd.DataFrame:
        """Predict activity for a list of SMILES."""
        features_df = PROTACFeaturizer.featurize_batch(smiles_list)
        
        if self.feature_cols is None:
            self.feature_cols = [col for col in features_df.columns 
                                  if col not in ['smiles', 'name']]
        
        X = features_df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        y_pred, y_prob = self.predict(X_scaled)
        
        results = features_df.copy()
        results['predicted_activity'] = y_pred
        results['probability_active'] = y_prob
        
        return results
    
    def calculate_degrader_score(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Calculate degrader-specific scoring (linker length, warhead presence)."""
        df = predictions.copy()
        
        df['degrader_score'] = (
            (df['NumAmideBonds'] >= 1).astype(float) * 0.3 +
            ((df['ChainLength'] >= 3) & (df['ChainLength'] <= 15)).astype(float) * 0.3 +
            (df['NumRings'] >= 2).astype(float) * 0.2 +
            (df['NumRotatableBonds'] >= 3).astype(float) * 0.2
        )
        
        return df
    
    def rank_by_viability(self, predictions: pd.DataFrame, 
                         activity_weight: float = 0.5,
                         degrader_weight: float = 0.3,
                         druglikeness_weight: float = 0.2) -> pd.DataFrame:
        """Rank PROTACs by overall viability."""
        df = self.calculate_degrader_score(predictions)
        
        df['activity_score'] = df['probability_active']
        
        df['druglikeness_score'] = (
            (df['MW'] < 800).astype(float) * 0.4 +
            (df['LogP'] < 5).astype(float) * 0.3 +
            (df['TPSA'] < 150).astype(float) * 0.3
        )
        
        df['viability_score'] = (
            activity_weight * df['activity_score'] +
            degrader_weight * df['degrader_score'] +
            druglikeness_weight * df['druglikeness_score']
        )
        
        return df.sort_values('viability_score', ascending=False)
    
    def save(self, filepath: str):
        """Save model."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'PROTACModel':
        """Load model."""
        data = joblib.load(filepath)
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_cols = data['feature_cols']
        return instance


class PROTACPipeline:
    """End-to-end PROTAC design pipeline."""
    
    def __init__(self, output_dir: Path = Path("../results/protac")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        
    def generate_degraders(self, warhead_smiles: str, n_analogs: int = 100) -> List[str]:
        """Generate degrader candidates from a warhead."""
        protacs = PROTACGenerator.generate_three_part_protac(warhead_smiles, n_variants=n_analogs)
        fragments = PROTACGenerator.fragment_based_generation(warhead_smiles, n_fragments=n_analogs // 2)
        
        return list(set(protacs + fragments))
    
    def screen_degraders(self, smiles_list: List[str]) -> pd.DataFrame:
        """Screen generated degraders."""
        if self.model is not None:
            predictions = self.model.predict_smiles(smiles_list)
            results = self.model.rank_by_viability(predictions)
        else:
            features = PROTACFeaturizer.featurize_batch(smiles_list)
            results = self.model.calculate_degrader_score(features)
        
        return results
    
    def design_for_target(self, target: str, warhead_smiles: str) -> pd.DataFrame:
        """Design PROTACs for a specific bacterial target."""
        candidates = self.generate_degraders(warhead_smiles, n_analogs=100)
        
        results = self.screen_degraders(candidates)
        results['target'] = target
        
        return results


def generate_synthetic_protac_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic training data for PROTACs."""
    
    known_warheads = [
        "CC(C)CC1=CC=C(C=C1)C(=O)O",
        "CC1=CC=C(C=C1)C(=O)O",
        "OC1=CC=C(C=C1)C(=O)O",
    ]
    
    active_smiles = []
    for w in known_warheads:
        active_smiles.extend(PROTACGenerator.generate_three_part_protac(w))
        active_smiles.extend(PROTACGenerator.generate_peptidomimetic(w, n_variants=20))
    
    inactive_smiles = [
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
        "CC(N)C(=O)O",
        "c1ccccc1",
    ] * 20
    
    target_n = min(n_samples, len(active_smiles) + len(inactive_smiles))
    all_smiles = active_smiles + inactive_smiles[:max(0, target_n - len(active_smiles))]
    
    features_df = PROTACFeaturizer.featurize_batch(all_smiles)
    
    n_active = min(len(active_smiles), len(features_df))
    features_df['activity'] = [1] * n_active + [0] * (len(features_df) - n_active)
    
    return features_df.dropna()


if __name__ == "__main__":
    print("Generating synthetic training data...")
    train_data = generate_synthetic_protac_data(1000)
    print(f"Generated {len(train_data)} samples")
    
    feature_cols = [col for col in train_data.columns 
                   if col not in ['smiles', 'activity', 'name']]
    X = train_data[feature_cols].values
    y = train_data['activity'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = PROTACModel(model_type='rf')
    model.scaler = scaler
    model.feature_cols = feature_cols
    model.train(X_train, y_train)
    
    y_pred, y_prob = model.predict(X_test)
    
    print(f"\nModel Performance:")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
    print(classification_report(y_test, y_pred))
    
    model.save("../models/protac_model.joblib")
    print("\nModel saved!")
