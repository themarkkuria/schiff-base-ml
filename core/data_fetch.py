"""
Data fetching utilities for the molecular design platform.
Supports retrieval from PubChem, DBAASP, and other sources.
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pubchempy as pcp
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import time
import json
from io import StringIO


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class PubChemFetcher:
    """Fetch data from PubChem."""
    
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    @staticmethod
    def search_compounds(name: str, listkey_count: int = 100) -> List[Dict]:
        """Search compounds by name/activity."""
        url = f"{PubChemFetcher.BASE_URL}/compound/name/{name}/property/CanonicalSMILES,InChI,InChIKey,MolecularFormula,MolecularWeight/CSV"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                return df.to_dict('records')
        except Exception as e:
            print(f"Error searching PubChem for {name}: {e}")
        return []
    
    @staticmethod
    def get_compounds_by_cid(cids: List[int]) -> List[Dict]:
        """Get compound details by CID."""
        compounds = []
        for cid in cids:
            try:
                compound = pcp.Compound.from_cid(cid)
                compounds.append({
                    'cid': cid,
                    'smiles': compound.canonical_smiles,
                    'inchi': compound.inchi,
                    'inchikey': compound.inchikey,
                    'molecular_formula': compound.molecular_formula,
                    'molecular_weight': compound.molecular_weight
                })
                time.sleep(0.2)
            except Exception as e:
                print(f"Error fetching CID {cid}: {e}")
        return compounds
    
    @staticmethod
    def search_antibacterial_drugs(max_compounds: int = 500) -> pd.DataFrame:
        """Search for FDA-approved drugs with antibacterial activity."""
        compounds = []
        
        search_terms = [
            "antibacterial drug FDA approved",
            "antibiotic FDA approved", 
            "antimicrobial drug"
        ]
        
        for term in search_terms:
            if len(compounds) >= max_compounds:
                break
            results = PubChemFetcher.search_compounds(term, listkey_count=200)
            for r in results:
                if r not in compounds:
                    compounds.append(r)
                    if len(compounds) >= max_compounds:
                        break
        
        df = pd.DataFrame(compounds)
        return df
    
    @staticmethod
    def get_fda_drugs_subset() -> pd.DataFrame:
        """Get subset of FDA-approved drugs for repurposing."""
        known_antibacterial_drugs = [
            "ciprofloxacin", "levofloxacin", "azithromycin", "ampicillin",
            "amoxicillin", "cephalexin", "doxycycline", "minocycline",
            "sulfamethoxazole", "trimethoprim", "nitrofurantoin",
            "tetracycline", "chloramphenicol", "vancomycin", "linezolid"
        ]
        
        drugs = []
        for drug_name in known_antibacterial_drugs:
            results = PubChemFetcher.search_compounds(drug_name)
            if results:
                drugs.extend(results)
        
        df = pd.DataFrame(drugs)
        df = df.drop_duplicates(subset=['CanonicalSMILES'])
        return df


class DBAASPFetcher:
    """Fetch AMP data from DBAASP database."""
    
    DBAASP_URL = "https://dbaasp.org"
    
    @staticmethod
    def parse_dbaasp_file(filepath: str) -> pd.DataFrame:
        """Parse DBAASP FASTA/CSV file."""
        try:
            records = list(SeqIO.parse(filepath, "fasta"))
            data = []
            for record in records:
                seq = str(record.seq)
                data.append({
                    'sequence': seq,
                    'length': len(seq),
                    'name': record.name,
                    'description': record.description
                })
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error parsing DBAASP file: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_amp_descriptors(sequences: List[str]) -> pd.DataFrame:
        """Calculate physicochemical descriptors for AMP sequences."""
        from collections import Counter
        
        aa_props = {
            'A': (1.8, 0, 0), 'R': (-4.5, 1, 1), 'N': (-3.5, 1, 1), 'D': (-3.5, 1, 1),
            'C': (2.5, 1, 0), 'Q': (-3.5, 1, 1), 'E': (-3.5, 1, 1), 'G': (-0.4, 0, 0),
            'H': (-3.2, 1, 1), 'I': (4.5, 0, 0), 'L': (3.8, 0, 0), 'K': (-3.9, 1, 1),
            'M': (1.9, 0, 0), 'F': (2.8, 0, 0), 'P': (-1.6, 0, 0), 'S': (-0.8, 1, 1),
            'T': (-0.7, 1, 1), 'W': (-0.9, 0, 0), 'Y': (-1.3, 1, 1), 'V': (4.2, 0, 0)
        }
        
        descriptors = []
        for seq in sequences:
            seq = seq.upper()
            counts = Counter(seq)
            length = len(seq)
            
            if length == 0:
                continue
            
            hydrophobic = sum(counts.get(aa, 0) * props[0] for aa, props in aa_props.items()) / length
            charge = sum(counts.get(aa, 0) * props[1] for aa, props in aa_props.items()) / length
            polar = sum(counts.get(aa, 0) * props[2] for aa, props in aa_props.items()) / length
            
            net_charge = sum(1 for aa in seq if aa in 'RK') - sum(1 for aa in seq if aa in 'DE')
            
            hydrophobic_ratio = sum(counts.get(aa, 0) for aa in 'AILMFVWP') / length
            aromatic_ratio = sum(counts.get(aa, 0) for aa in 'FWY') / length
            
            descriptors.append({
                'sequence': seq,
                'length': length,
                'hydrophobicity': hydrophobic,
                'net_charge': net_charge,
                'hydrophobic_ratio': hydrophobic_ratio,
                'aromatic_ratio': aromatic_ratio,
                'polar_ratio': polar,
                'charge_density': net_charge / length if length > 0 else 0
            })
        
        return pd.DataFrame(descriptors)


class NaturalProductFetcher:
    """Fetch natural product data."""
    
    @staticmethod
    def get_common_polyphenols() -> pd.DataFrame:
        """Get common polyphenols for starting templates."""
        polyphenols = [
            {"name": "Curcumin", "smiles": "CC(=O)CC(C)=CC1=CC=C(O)C(OC)=C1"},
            {"name": "Resveratrol", "smiles": "OC1=CC=C(C=C1)C=CC1=CC(O)=CC(O)=C1"},
            {"name": "Quercetin", "smiles": "OC1=C(O)C=C2C(=O)C3C(O)=C(O)C(C3OC2=C1)=O"},
            {"name": "Epigallocatechin gallate", "smiles": "CC1(C)OC2=C(C1)OC3=C(C2=O)C=C(C=C3O)C4=CC(C5=C(C=C(C5)O)O)C(=O)O"},
            {"name": "Hydroxytyrosol", "smiles": "OC1=CC=C(O)C(CO)=C1"},
            {"name": "Caffeic acid", "smiles": "OC1=CC=C(C=CC(O)=O)C1"},
            {"name": "Ferulic acid", "smiles": "COC1=C(C=CC(=C1)C=CC(O)=O)C"},
            {"name": "Gallic acid", "smiles": "OC1=CC(O)=C(O)C(O)=C1"},
            {"name": "Ellagic acid", "smiles": "OC1=C(O)C2=C(C=C1O)C1=C(O)C(=O)C3=C(O)C(O)=C(O)C3=C1C2=O"},
            {"name": "Rutin", "smiles": "CC1C(C(C(C(O1)OC2C(C(C(C(O2)OC3=C(OC4=C(C3=C(C=C4)C5=C(C6=C(C=C5O)C(=O)C(O)=C6O)C(=O)O)C)CO)O)O)O)O)O)(C)O"}
        ]
        return pd.DataFrame(polyphenols)


def fetch_all_data(category: str, output_dir: Optional[Path] = None):
    """Fetch data for a specific category."""
    if output_dir is None:
        output_dir = DATA_DIR / category
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if category == "amp":
        print("Note: Please manually download DBAASP data from https://dbaasp.org")
        print("Place the file in data/amp/ directory")
        
    elif category == "repurposed":
        df = PubChemFetcher.get_fda_drugs_subset()
        df.to_csv(output_dir / "fda_drugs.csv", index=False)
        print(f"Saved {len(df)} FDA drugs to {output_dir / 'fda_drugs.csv'}")
        
    elif category == "polyphenols":
        df = NaturalProductFetcher.get_common_polyphenols()
        df.to_csv(output_dir / "polyphenols.csv", index=False)
        print(f"Saved {len(df)} polyphenols to {output_dir / 'polyphenols.csv'}")
    
    return output_dir


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        category = sys.argv[1]
    else:
        print("Usage: python data_fetch.py <category>")
        print("Categories: amp, repurposed, polyphenols")
        sys.exit(1)
    
    fetch_all_data(category)
