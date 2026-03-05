"""
Polyphenols & Natural Products Module
Design and optimization of curcumin-like hybrids and natural product derivatives.
Integrates with Schiff base pipeline for hybrid molecule generation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import json
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFMCS, BRICS, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD
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


CURCUMIN_HYBRID_MOTIFS = {
    "curcumin_core": "CC(=O)CC(C)=CC1=CC=C(O)C(OC)=C1",
    "beta_diketone": "CC(=O)CH2C(=O)CH3",
    "phenyl_butadiene": "c1ccccc1C=CC=CC",
}


POLYPHENOL_STARTING_POINTS = [
    {"name": "Curcumin", "smiles": "CC(=O)CC(C)=CC1=CC=C(O)C(OC)=C1"},
    {"name": "Resveratrol", "smiles": "OC1=CC=C(C=C1)C=CC1=CC(O)=CC(O)=C1"},
    {"name": "Quercetin", "smiles": "OC1=C(O)C=C2C(=O)C3C(O)=C(O)C(C3OC2=C1)=O"},
    {"name": "Epigallocatechin gallate", "smiles": "CC1(C)OC2=C(C1)OC3=C(C2=O)C=C(C=C3O)C4=CC(C5=C(C=C(C5)O)O)C(=O)O"},
    {"name": "Hydroxytyrosol", "smiles": "OC1=CC=C(O)C(CO)=C1"},
    {"name": "Caffeic acid", "smiles": "OC1=CC=C(C=CC(O)=O)C1"},
    {"name": "Ferulic acid", "smiles": "COC1=C(C=CC(=C1)C=CC(O)=O)C"},
    {"name": "Gallic acid", "smiles": "OC1=CC(O)=C(O)C(O)=C1"},
    {"name": "Ellagic acid", "smiles": "OC1=C(O)C2=C(C=C1O)C1=C(O)C(=O)C3=C(O)C(O)=C(O)C3=C1C2=O"},
    {"name": "Epicatechin", "smiles": "OC1C2C(OC3=C(C2=O)C=C(C=C3O)C1CO)CO"},
]


class PolyphenolGenerator:
    """Generate polyphenol derivatives and hybrids."""
    
    @staticmethod
    def generate_curcumin_analogs(base_smiles: str = None, n_analogs: int = 100) -> List[str]:
        """Generate curcumin analogs with variations at the terminal phenyl rings."""
        if base_smiles is None:
            base_smiles = CURCUMIN_HYBRID_MOTIFS["curcumin_core"]
        
        mol = Chem.MolFromSmiles(base_smiles)
        if mol is None:
            return []
        
        substitutions = [
            ("O", "OC"), ("O", "OC1=CC=CC=C1"), ("O", "OC1=CC=C(O)C=C1"),
            ("O", "OC1=CC=C(C)C=C1"), ("O", "OC1=CC(OC)=C(OC)C=C1"),
            ("O", "N"), ("O", "NCCO"), ("O", "NC(=O)C"),
        ]
        
        analogs = [base_smiles]
        
        for i in range(n_analogs):
            template = base_smiles
            for target, replacement in random.sample(substitutions, k=random.randint(1, 3)):
                template = template.replace(target, replacement, 1)
            
            if Chem.MolFromSmiles(template):
                analogs.append(template)
        
        return analogs[:n_analogs]
    
    @staticmethod
    def generate_scaffold_hop(source_smiles: str, n_hop: int = 50) -> List[str]:
        """Perform scaffold hopping from a source molecule."""
        mol = Chem.MolFromSmiles(source_smiles)
        if mol is None:
            return []
        
        replacements = [
            ("c1ccccc1", "c1ccncc1"),
            ("c1ccccc1", "c1ccoc1"),
            ("c1ccccc1", "C1CCCCC1"),
            ("c1ccccc1", "c1ccc2ccccc2c1"),
            ("C=C", "C=N"),
            ("C=C", "C#C"),
            ("C=O", "C=NN"),
        ]
        
        hopped = []
        for _ in range(n_hop):
            template = source_smiles
            for target, replacement in random.sample(replacements, k=1):
                template = template.replace(target, replacement, 1)
            
            if Chem.MolFromSmiles(template):
                hopped.append(template)
        
        return hopped
    
    @staticmethod
    def generate_hybrid_molecules(polyphenol_smiles: str, linker_smiles: str = None) -> List[str]:
        """Generate hybrid molecules by joining polyphenol with a linker."""
        if linker_smiles is None:
            linker_smiles = "CCC"
        
        return [
            f"{polyphenol_smiles.split('.')[0]}{linker_smiles}{polyphenol_smiles.split('.')[-1]}"
            for _ in range(10)
        ]


class PolyphenolFeaturizer:
    """Featurize polyphenols for ML models."""
    
    @staticmethod
    def calculate_solubility_descriptors(smiles: str) -> Optional[Dict]:
        """Calculate descriptors relevant to solubility/bioavailability."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        try:
            descriptors = {
                'MW': MolWt(mol),
                'LogP': MolLogP(mol),
                'TPSA': TPSA(mol),
                'NumHBD': CalcNumHBD(mol),
                'NumHBA': CalcNumHBA(mol),
                'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
                'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
                'NumRings': rdMolDescriptors.CalcNumRings(mol),
                'NumPhenolicOH': sum(1 for atom in mol.GetAtoms() 
                                     if atom.GetSymbol() == 'O' and 
                                     atom.GetTotalNumHs() > 0),
                'LogS_estimate': -0.65 * MolLogP(mol) + 0.07 * TPSA(mol) + 0.66,
            }
            return descriptors
        except Exception:
            return None
    
    @staticmethod
    def featurize_smiles(smiles: str) -> Optional[Dict]:
        """Full featurization for polyphenols."""
        desc = MolecularDescriptorCalculator.calculate_descriptors(smiles)
        sol_desc = PolyphenolFeaturizer.calculate_solubility_descriptors(smiles)
        
        if desc and sol_desc:
            return {**desc, **sol_desc}
        return desc or sol_desc
    
    @staticmethod
    def featurize_batch(smiles_list: List[str]) -> pd.DataFrame:
        """Featurize a batch of polyphenols."""
        all_features = []
        
        for smiles in smiles_list:
            features = PolyphenolFeaturizer.featurize_smiles(smiles)
            if features:
                features['smiles'] = smiles
                all_features.append(features)
        
        return pd.DataFrame(all_features)


class CoCrystalPredictor:
    """Predict co-crystal formation potential (simple H-bond donor/acceptor matching)."""
    
    @staticmethod
    def count_hbond_donors_acceptors(smiles: str) -> Tuple[int, int]:
        """Count H-bond donors and acceptors."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0, 0
        
        hbd = sum(1 for atom in mol.GetAtoms() 
                  if atom.GetTotalNumHs() > 0 and 
                  atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N')
        hba = sum(1 for atom in mol.GetAtoms() 
                  if atom.GetTotalNumHs() == 0 and 
                  atom.GetSymbol() == 'O' or atom.GetSymbol() == 'N')
        
        return hbd, hba
    
    @staticmethod
    def predict_cocrystal_partner(polyphenol_smiles: str, 
                                   candidate_smiles: List[str]) -> pd.DataFrame:
        """Predict co-crystal forming partners for a polyphenol."""
        hbd_poly, hba_poly = CoCrystalPredictor.count_hbond_donors_acceptors(polyphenol_smiles)
        
        results = []
        for smiles in candidate_smiles:
            hbd, hba = CoCrystalPredictor.count_hbond_donors_acceptors(smiles)
            
            hbd_match = abs(hbd - hba_poly)
            hba_match = abs(hba - hbd_poly)
            total_match = hbd_match + hba_match
            
            compatibility = 1.0 / (1.0 + total_match)
            
            results.append({
                'smiles': smiles,
                'hbd': hbd,
                'hba': hba,
                'hbd_match': hbd_match,
                'hba_match': hba_match,
                'compatibility': compatibility
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('compatibility', ascending=False)


class PolyphenolModel:
    """Train and predict polyphenol activity."""
    
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
        features_df = PolyphenolFeaturizer.featurize_batch(smiles_list)
        
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
    
    def rank_by_viability(self, predictions: pd.DataFrame, 
                         activity_weight: float = 0.5,
                         solubility_weight: float = 0.3,
                         druglikeness_weight: float = 0.2) -> pd.DataFrame:
        """Rank predictions by overall viability."""
        df = predictions.copy()
        
        df['activity_score'] = df['probability_active']
        
        df['solubility_score'] = df['LogS_estimate'].apply(
            lambda x: min(1.0, max(0.0, x / -1))
        )
        
        df['druglikeness_score'] = (
            (df['MW'] < 500).astype(float) * 0.33 +
            (df['LogP'] < 5).astype(float) * 0.33 +
            (df['TPSA'] < 140).astype(float) * 0.34
        )
        
        df['viability_score'] = (
            activity_weight * df['activity_score'] +
            solubility_weight * df['solubility_score'] +
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
    def load(cls, filepath: str) -> 'PolyphenolModel':
        """Load model."""
        data = joblib.load(filepath)
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_cols = data['feature_cols']
        return instance


class PolyphenolPipeline:
    """End-to-end polyphenol design pipeline."""
    
    def __init__(self, output_dir: Path = Path("../results/polyphenols")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        
    def generate_hybrids(self, base_name: str = "Curcumin", n_analogs: int = 100) -> List[str]:
        """Generate hybrid polyphenol derivatives."""
        polyphenol = next((p for p in POLYPHENOL_STARTING_POINTS if p['name'] == base_name), None)
        if polyphenol is None:
            return []
        
        analogs = PolyphenolGenerator.generate_curcumin_analogs(polyphenol['smiles'], n_analogs)
        hopped = PolyphenolGenerator.generate_scaffold_hop(polyphenol['smiles'], n_analogs // 2)
        
        return list(set(analogs + hopped))
    
    def screen_hybrids(self, smiles_list: List[str]) -> pd.DataFrame:
        """Screen generated hybrids."""
        if self.model is not None:
            predictions = self.model.predict_smiles(smiles_list)
            results = self.model.rank_by_viability(predictions)
        else:
            features = PolyphenolFeaturizer.featurize_batch(smiles_list)
            results = features
        
        return results


def generate_synthetic_polyphenol_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic training data for polyphenols."""
    polyphenols = POLYPHENOL_STARTING_POINTS.copy()
    for p in polyphenols:
        p['activity'] = 1
    
    inactive_smiles = [
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
        "CC(N)C(=O)O",
        "c1ccccc1",
    ] * 10
    
    random.seed(42)
    active_smiles = []
    for p in polyphenols:
        active_smiles.extend(PolyphenolGenerator.generate_scaffold_hop(p['smiles'], 50))
    
    all_smiles = active_smiles + inactive_smiles[:n_samples - len(active_smiles)]
    labels = [1] * len(active_smiles) + [0] * (n_samples - len(active_smiles))
    
    features_df = PolyphenolFeaturizer.featurize_batch(all_smiles)
    features_df['activity'] = labels[:len(features_df)]
    
    return features_df.dropna()


if __name__ == "__main__":
    print("Generating synthetic training data...")
    train_data = generate_synthetic_polyphenol_data(1000)
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
    
    model = PolyphenolModel(model_type='rf')
    model.scaler = scaler
    model.feature_cols = feature_cols
    model.train(X_train, y_train)
    
    y_pred, y_prob = model.predict(X_test)
    
    print(f"\nModel Performance:")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
    print(classification_report(y_test, y_pred))
    
    print("\nGenerating curcumin analogs...")
    analogs = PolyphenolGenerator.generate_curcumin_analogs(n_analogs=20)
    print(f"Generated {len(analogs)} analogs")
    
    model.save("../models/polyphenol_model.joblib")
    print("\nModel saved!")
