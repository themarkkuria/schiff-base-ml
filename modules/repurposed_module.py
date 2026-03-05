"""
Repurposed Drugs Module
Virtual screening of FDA-approved drugs for antibacterial repurposing.
Includes Halicin-like scaffold identification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import json

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, rdMolDescriptors, MACCSkeys
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import MolToImage
from rdkit import DataStructs
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import pubchempy as pcp
import requests
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.pipeline import MolecularDescriptorCalculator, DataProcessor, ModelEvaluator


HALICIN_SIMILAR = [
    {"name": "Halicin", "smiles": "CS(=O)(=O)N1CCN(CC1)C1=NC(=O)N(C(=O)N1)C1=CC=C(C=C1)Cl"},
    {"name": "Metronidazole", "smiles": "CC1=NC=C(N1CCO)[N+](=O)[O-]"},
    {"name": "Nitrofurantoin", "smiles": "CC1C(OC(=O)C1N2C(=O)C3C2C=CC3=O)C=O"},
]


class RepurposedDrugFetcher:
    """Fetch FDA-approved drugs for repurposing."""
    
    @staticmethod
    def get_fda_approved_drugs(max_compounds: int = 500) -> pd.DataFrame:
        """Get FDA-approved drugs from PubChem."""
        known_drugs = [
            "ciprofloxacin", "levofloxacin", "moxifloxacin", "azithromycin",
            "ampicillin", "amoxicillin", "cephalexin", "doxycycline", "minocycline",
            "sulfamethoxazole", "trimethoprim", "nitrofurantoin", "tetracycline",
            "chloramphenicol", "vancomycin", "linezolid", "daptomycin",
            "colistin", "polymyxin B", "rifampicin", "gentamicin",
            "amikacin", "tobramycin", "cefepime", "cefazolin", "meropenem",
            "imipenem", "ertapenem", "piperacillin", "ticarcillin",
            "halicin", "clofoctol", "triclosan", "triclosan",
        ]
        
        drugs = []
        for drug_name in known_drugs:
            try:
                compounds = pcp.get_compounds(drug_name, 'name')
                if compounds:
                    c = compounds[0]
                    drugs.append({
                        'name': drug_name,
                        'cid': c.cid,
                        'smiles': c.canonical_smiles,
                        'molecular_weight': c.molecular_weight,
                        'is_antibiotic': 1
                    })
                    time.sleep(0.2)
                    if len(drugs) >= max_compounds:
                        break
            except Exception as e:
                print(f"Error fetching {drug_name}: {e}")
        
        return pd.DataFrame(drugs)
    
    @staticmethod
    def get_pubchem_antibacterial(search_term: str = "antibacterial", 
                                  max_results: int = 200) -> pd.DataFrame:
        """Search PubChem for antibacterial compounds."""
        compounds = []
        
        try:
            results = pcp.get_compounds(search_term, 'name', listkey_count=max_results)
            for c in results:
                try:
                    compounds.append({
                        'name': c.iupac_name or f"CID_{c.cid}",
                        'cid': c.cid,
                        'smiles': c.canonical_smiles,
                        'molecular_weight': c.molecular_weight
                    })
                    time.sleep(0.1)
                except:
                    pass
        except Exception as e:
            print(f"PubChem search error: {e}")
        
        return pd.DataFrame(compounds)


class RepurposedFeaturizer:
    """Featurize compounds for repurposing prediction."""
    
    @staticmethod
    def featurize_smiles(smiles: str) -> Optional[Dict]:
        """Calculate molecular descriptors for a SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        desc = MolecularDescriptorCalculator.calculate_descriptors(smiles)
        
        morgan_fp = MolecularDescriptorCalculator.calculate_fingerprints(smiles, radius=2, nBits=1024)
        maccs_fp = np.array(MACCSkeys.GenMACCSKeys(mol))
        
        fp_dict = {f'morgan_{i}': v for i, v in enumerate(morgan_fp)}
        fp_dict.update({f'maccs_{i}': v for i, v in enumerate(maccs_fp)})
        
        return {**desc, **fp_dict}
    
    @staticmethod
    def featurize_batch(smiles_list: List[str]) -> pd.DataFrame:
        """Featurize a batch of SMILES."""
        all_features = []
        
        for smiles in smiles_list:
            features = RepurposedFeaturizer.featurize_smiles(smiles)
            if features:
                features['smiles'] = smiles
                all_features.append(features)
        
        return pd.DataFrame(all_features)


class ScaffoldSimilarity:
    """Find similar scaffolds to known actives (e.g., Halicin)."""
    
    @staticmethod
    def get_fingerprint(smiles: str) -> Optional[np.ndarray]:
        """Get Morgan fingerprint for a molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
    
    @staticmethod
    def find_similar(query_smiles: str, database_smiles: List[str], 
                     top_k: int = 10, threshold: float = 0.3) -> List[Dict]:
        """Find similar compounds using Tanimoto similarity."""
        query_fp = ScaffoldSimilarity.get_fingerprint(query_smiles)
        if query_fp is None:
            return []
        
        similarities = []
        for smiles in database_smiles:
            db_fp = ScaffoldSimilarity.get_fingerprint(smiles)
            if db_fp is not None:
                sim = DataStructs.TanimotoSimilarity(query_fp, db_fp)
                if sim >= threshold:
                    similarities.append({
                        'smiles': smiles,
                        'similarity': sim
                    })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]


class RepurposedModel:
    """Train and predict antibacterial activity for repurposed drugs."""
    
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
        features_df = RepurposedFeaturizer.featurize_batch(smiles_list)
        
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
    
    def save(self, filepath: str):
        """Save model."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'RepurposedModel':
        """Load model."""
        data = joblib.load(filepath)
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_cols = data['feature_cols']
        return instance


class DrugRepurposingPipeline:
    """End-to-end drug repurposing pipeline."""
    
    def __init__(self, output_dir: Path = Path("../results/repurposed")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.reference_smiles = HALICIN_SIMILAR
        
    def screen_compounds(self, smiles_list: List[str], 
                        top_k: int = 20) -> pd.DataFrame:
        """Screen compounds and rank by similarity to known actives + ML prediction."""
        
        all_similarities = []
        for ref in self.reference_smiles:
            similar = ScaffoldSimilarity.find_similar(
                ref['smiles'], smiles_list, top_k=50, threshold=0.2
            )
            for s in similar:
                s['reference'] = ref['name']
                all_similarities.append(s)
        
        if all_similarities:
            sim_df = pd.DataFrame(all_similarities)
            sim_df = sim_df.groupby('smiles').agg({
                'similarity': 'max',
                'reference': 'first'
            }).reset_index()
        else:
            sim_df = pd.DataFrame(columns=['smiles', 'similarity', 'reference'])
        
        if self.model is not None:
            pred_df = self.model.predict_smiles(smiles_list)
            results = pd.merge(sim_df, pred_df, on='smiles', how='outer')
        else:
            results = sim_df
            results['probability_active'] = results.get('similarity', 0)
        
        results['combined_score'] = (
            0.5 * results['probability_active'].fillna(0) + 
            0.5 * results['similarity'].fillna(0)
        )
        
        results = results.sort_values('combined_score', ascending=False)
        
        return results.head(top_k)
    
    def run_halin_screen(self, database_smiles: List[str]) -> pd.DataFrame:
        """Screen for Halicin-like compounds."""
        return self.screen_compounds(database_smiles, top_k=50)


def generate_synthetic_repurposing_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic training data for repurposing."""
    from core.data_fetch import NaturalProductFetcher
    
    np_df = NaturalProductFetcher.get_common_polyphenols()
    antibiotics = [
        "CC1=NC=C(N1CCO)[N+](=O)[O-]",
        "O=C1C2C(CC1=O)CC(=O)N2O",
        "CN1C(=O)CN=C(C1C1=CC=CC=C1Cl)C1=CC=CC=C1",
    ]
    
    active_smiles = list(np_df['smiles'].values) + antibiotics
    inactive_smiles = [
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
        "CC(N)C(=O)O",
        "c1ccccc1",
    ] * 20
    
    target_n = min(n_samples, len(active_smiles) + len(inactive_smiles))
    all_smiles = active_smiles + inactive_smiles[:target_n - len(active_smiles)]
    
    features_df = RepurposedFeaturizer.featurize_batch(all_smiles)
    
    n_active = min(len(active_smiles), len(features_df))
    features_df['activity'] = [1] * n_active + [0] * (len(features_df) - n_active)
    
    return features_df


if __name__ == "__main__":
    print("Generating synthetic training data...")
    train_data = generate_synthetic_repurposing_data(1000)
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
    
    model = RepurposedModel(model_type='rf')
    model.scaler = scaler
    model.feature_cols = feature_cols
    model.train(X_train, y_train)
    
    y_pred, y_prob = model.predict(X_test)
    
    print(f"\nModel Performance:")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
    print(classification_report(y_test, y_pred))
    
    model.save("../models/repurposed_model.joblib")
    print("\nModel saved!")
