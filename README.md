\# Schiff Base ML: Machine Learning–Assisted Design of Schiff Base Molecules



This project combines \*\*computational chemistry\*\* and \*\*machine learning\*\* to accelerate the design and discovery of Schiff base molecules. By leveraging \*\*RDKit\*\* for cheminformatics and \*\*scikit-learn\*\* for model building, the workflow enables automated generation of Schiff base ligands, calculation of molecular descriptors, and prediction of their potential bioactivity.



Aldehydes + Amines

&nbsp;       │

&nbsp;       ▼

&nbsp;Automated Schiff Base Generation

&nbsp;       │

&nbsp;       ▼

Molecular Descriptors (MW, LogP, TPSA, HBD, HBA)

&nbsp;       │

&nbsp;       ▼

&nbsp;  Machine Learning Model

&nbsp;(Random Forest Classifier)

&nbsp;       │

&nbsp;       ▼

&nbsp;Predicted Antibacterial Candidates



---

Background \& Scientific Motivation



Schiff bases are a class of imine-containing compounds widely studied in \*\*coordination chemistry, catalysis, and drug discovery\*\*. Their structural flexibility allows them to form stable complexes with metals, enabling applications as catalysts in organic synthesis and materials science. In medicinal chemistry, Schiff bases and their derivatives have shown promising \*\*antimicrobial, anticancer, and antioxidant activities\*\*.  



Traditionally, discovering active Schiff bases requires labor-intensive synthesis and screening. By integrating \*\*machine learning with cheminformatics\*\*, this project provides a \*\*green, cost-efficient, and scalable approach\*\* to designing novel Schiff bases with targeted properties, accelerating research at the interface of chemistry and AI.



---



\## 🎯 Project Goals

\- \*\*Discover novel Schiff bases with targeted bioactivity\*\*  

\- \*\*Design ligands for new catalytic platforms\*\*  

\- \*\*Reduce waste and costs through green, ML-assisted design\*\*  

\- \*\*Provide a generalizable scaffold design framework\*\* for computational chemistry  



---



\## 🧪 Workflow Overview



The pipeline consists of two stages, implemented in Jupyter notebooks:



\### 1. Schiff Base Generation \& Descriptor Calculation

\- Reads aldehydes and amines from `data/molecules.csv`  

\- Uses an RDKit reaction SMARTS template to automatically generate Schiff base products  

\- Computes descriptors: \*\*Molecular Weight, LogP, TPSA, H-bond donors/acceptors\*\*  

\- Saves product list and descriptors for downstream ML  



\*\*Key Outputs:\*\*  

\- `data/generated\_schiff\_bases.csv` (new Schiff base products)  

\- `data/generated\_schiff\_bases\_descriptors.csv` (products + descriptors)  



---



\### 2. Machine Learning Model for Antibacterial Activity

\- Loads a curated dataset (`data/schiff\_base\_complete\_dataset.csv`) containing Schiff bases with experimental MIC values  

\- Trains a \*\*Random Forest classifier\*\* to predict antibacterial activity (active vs inactive, based on MIC threshold)  

\- Evaluates model accuracy on held-out test data  

\- Applies the model to newly generated Schiff bases  

\- Ranks candidates by probability of being antibacterial  



\*\*Key Outputs:\*\*  

\- `results/predicted\_antibacterial\_candidates.csv` → ranked predictions of novel active candidates  



---



\## 📂 Project Structure



schiff-base-ml/

├── data/

│ ├── molecules.csv # starting reactants (aldehydes, amines)

│ ├── generated\_schiff\_bases.csv # auto-generated Schiff base products

│ ├── generated\_schiff\_bases\_descriptors.csv# descriptors for ML stage

│ └── schiff\_base\_complete\_dataset.csv # training dataset (with MIC values)

│

├── notebooks/

│ ├── 01\_rdkit\_intro.ipynb # generate Schiff bases + descriptors

│ └── 02\_ml\_pipeline.ipynb # ML model training + candidate prediction

│

├── results/

│ └── predicted\_antibacterial\_candidates.csv # ranked ML output

│

├── requirements.txt

└── README.md





---



\## ⚙️ Installation



Clone the repository and install dependencies:



```bash

git clone https://github.com/themarkkuria/schiff-base-ml.git

cd schiff-base-ml

pip install -r requirements.txt





🚀 Usage



Generate Schiff bases + descriptors

Run notebooks/01\_rdkit\_intro.ipynb to build new Schiff base molecules from aldehydes + amines.



Train ML model + predict activity

Run notebooks/02\_ml\_pipeline.ipynb to train a Random Forest model and predict activity of generated Schiff bases.



Check results

Open results/predicted\_antibacterial\_candidates.csv for the ranked candidate molecules.





Example Output

\[RESULTS] Top predicted antibacterial candidates:

&nbsp;aldehyde    amine    product     MW    LogP   TPSA   predicted\_activity  probability\_active

&nbsp;Benzaldehyde Aniline C1=NC=...  211.3   2.34   45.2          1                 0.87

&nbsp;...



Audience \& Future Work



This repository is intended for:



Professors and researchers in computational chemistry \& catalysis



Developers exploring cheminformatics and ML pipelines



Planned extensions:



Regression models for MIC prediction



Integration with QSAR/QSPR workflows



Expansion beyond Schiff bases to other ligand families.

