# Schiff Base ML: AI-Driven Molecular Design Platform

A modular platform for AI-driven molecular design and evaluation, focused on antibacterial applications. Originally built for Schiff base molecules, now expanded to support multiple categories of antibacterial candidates.

---

## Overview

This project combines **computational chemistry** and **machine learning** to accelerate the design and discovery of antibacterial molecules. It supports:

- **Schiff Bases** - Original imine-based compounds
- **Antimicrobial Peptides (AMPs)** - Short amino acid sequences
- **Repurposed Drugs** - FDA-approved drugs for antibacterial use
- **Polyphenols/Natural Products** - Curcumin-like and plant-derived compounds
- **PROTACs/Peptidomimetics** - Protein degraders and peptide mimics

---

## Architecture

```
Input → Generation → Featurization → ML Prediction → Ranking → Output
```

Each category has its own module with:
- Molecule/Sequence Generator
- Feature Extractor (RDKit/BioPython)
- ML Model (RandomForest/GradientBoosting)
- Viability Ranker

All modules integrate via the **UnifiedRanker** for cross-category comparison.

---

## Project Structure

```
schiff-base-ml/
├── core/
│   ├── pipeline.py         # Shared ML utilities + UnifiedRanker
│   ├── data_fetch.py      # PubChem/DBAASP data fetchers
│   └── __init__.py
├── modules/
│   ├── amp_module.py       # Antimicrobial Peptides
│   ├── repurposed_module.py # FDA drug repurposing
│   ├── polyphenols_module.py # Curcumin/natural products
│   ├── protac_module.py    # PROTACs & peptidomimetics
│   └── __init__.py
├── notebooks/
│   ├── 01_rdkit_intro.ipynb        # Original Schiff base generation
│   ├── 02_ml_pipeline.ipynb        # Original ML pipeline
│   └── training/
│       ├── amp_training.ipynb     # AMP model training
│       └── protac_training.ipynb   # PROTAC model training
├── data/
│   ├── amp/                # DBAASP data
│   ├── repurposed/         # FDA drugs
│   ├── polyphenols/       # Natural products
│   └── protac/             # PROTAC-DB data
├── results/                # Prediction outputs
├── models/                 # Trained models
├── app.py                  # Streamlit web interface
├── requirements.txt
└── README.md
```

---

## Categories

### 1. Schiff Bases
- Generate from aldehydes + amines via RDKit reaction SMARTS
- Descriptors: MW, LogP, TPSA, HBD, HBA
- Random Forest classifier for antibacterial prediction

### 2. Antimicrobial Peptides (AMPs)
- Generate random/helical/cyclic peptides (10-50 AA)
- Featurization: BioPython physicochemical properties
- ESM-2 embeddings option for advanced models
- Multi-task: predict activity + low toxicity

### 3. Repurposed Drugs
- Fetch FDA-approved drugs via PubChem API
- Halicin-like scaffold similarity screening
- Morgan + MACCS fingerprints

### 4. Polyphenols/Natural Products
- Curcumin analog generation
- Scaffold hopping
- Co-crystal partner prediction
- Solubility/drug-likeness scoring

### 5. PROTACs/Peptidomimetics
- Three-part PROTAC design (warhead-linker-E3 ligase)
- Degrader-specific descriptors (amide bonds, chain length)
- Target: LpxC, ClpP, FtsH, DegP

---

## Installation

```bash
git clone https://github.com/themarkkuria/schiff-base-ml.git
cd schiff-base-ml
pip install -r requirements.txt
```

### Requirements
- numpy, pandas, scikit-learn
- rdkit (cheminformatics)
- biopython (peptide analysis)
- pubchempy (data fetching)
- imbalanced-learn (class balancing)
- shap (model interpretation)
- streamlit (web interface)

---

## Usage

### Quick Test (Multi-Category Ranking)

```python
from core.pipeline import UnifiedRanker
from modules import (
    generate_synthetic_amp_data,
    generate_synthetic_repurposing_data,
    generate_synthetic_polyphenol_data,
    generate_synthetic_protac_data
)

# Generate predictions for each category
dfs = {
    'amp': generate_synthetic_amp_data(100),
    'repurposed': generate_synthetic_repurposing_data(100),
    'polyphenol': generate_synthetic_polyphenol_data(100),
    'protac': generate_synthetic_protac_data(100)
}

# Unified ranking across all categories
combined = UnifiedRanker.rank_multi_category(dfs, top_k=50)
print(combined[['category', 'final_score', 'activity_score']].head(20))
```

### Run Notebooks

```bash
# AMP Training
jupyter notebook notebooks/training/amp_training.ipynb

# PROTAC Training  
jupyter notebook notebooks/training/protac_training.ipynb

# Original Schiff Base Pipeline
jupyter notebook notebooks/01_rdkit_intro.ipynb
jupyter notebook notebooks/02_ml_pipeline.ipynb
```

### Streamlit App

```bash
streamlit run app.py
```

---

## Unified Ranking System

The `UnifiedRanker` class enables cross-category comparison:

```python
UnifiedRanker.rank_single_category(df, category)
UnifiedRanker.rank_multi_category(dfs_dict, weights, top_k)
UnifiedRanker.diversity_filter(df, category, threshold)
```

**Scoring factors:**
- Activity probability (ML prediction)
- Drug-likeness (MW, LogP, TPSA rules)
- Category-specific weights (e.g., PROTACs weighted 1.3x for antibacterial focus)

---

## Data Sources

- **DBAASP** - Antimicrobial peptide database (https://dbaasp.org)
- **PROTAC-DB** - PROTAC degrader database
- **PubChem** - FDA drugs, antibacterial compounds
- **COCONUT** - Natural products database

---

## Model Training

Each category follows the same pattern:

```python
from modules.amp_module import AMPFeaturizer, AMPModel

# Featurize
features = AMPFeaturizer.featurize_batch(sequences)

# Train
model = AMPModel(model_type='rf')
model.train(X_train, y_train)

# Predict
predictions = model.predict_sequences(new_sequences)
```

---

## Example Output

```
category   final_score   activity_score   druglikeness_score
amp           1.0            0.95              0.85
protac        0.92           0.88              0.78
amp           0.89           0.82              0.91
polyphenol    0.85           0.79              0.88
repurposed    0.82           0.75              0.90
```

---

## Future Enhancements

- [ ] Load real DBAASP/PROTAC-DB data
- [ ] ESM-2 fine-tuning for AMPs (GPU)
- [ ] GNN models (Chemprop) for small molecules
- [ ] Multi-task learning (activity + toxicity + ADMET)
- [ ] Generative models (REINVENT, AMP-GAN)
- [ ] AutoDock integration for target docking
- [ ] Active learning loop

---

## License

MIT License

---

## Credits

- Original Schiff Base ML by TheMarkKuria
- RDKit for cheminformatics
- BioPython for peptide analysis
- DBAASP, PROTAC-DB, PubChem for data
