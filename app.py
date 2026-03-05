"""
Molecular Design Platform - Streamlit App
A modular platform for AI-driven molecular design and evaluation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.amp_module import (
    AMPGenerator, AMPFeaturizer, AMPModel, AMPRanker,
    generate_synthetic_amp_data
)
from core.data_fetch import NaturalProductFetcher

st.set_page_config(
    page_title="Molecular Design Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    st.title("🧪 AI-Driven Molecular Design Platform")
    st.markdown("""
    A modular platform for designing and evaluating molecules with antibacterial activity.
    """)
    
    category = st.sidebar.selectbox(
        "Select Category",
        ["AMPs", "Repurposed Drugs", "Polyphenols", "Schiff Bases", "Peptidomimetics"]
    )
    
    if category == "AMPs":
        amp_section()
    elif category == "Repurposed Drugs":
        repurposed_section()
    elif category == "Polyphenols":
        polyphenols_section()
    elif category == "Schiff Bases":
        schiff_section()
    else:
        peptidomimetics_section()


def amp_section():
    st.header("Antimicrobial Peptides (AMPs)")
    
    tab1, tab2, tab3 = st.tabs(["Generate", "Predict", "Train"])
    
    with tab1:
        st.subheader("Generate Novel AMPs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gen_method = st.selectbox(
                "Generation Method",
                ["Random", "Helical", "Cyclic", "Motif-based"]
            )
            n_sequences = st.slider("Number of sequences", 10, 500, 100)
        
        with col2:
            min_len = st.slider("Min length", 10, 30, 15)
            max_len = st.slider("Max length", 20, 50, 30)
        
        if st.button("Generate Sequences"):
            with st.spinner("Generating sequences..."):
                if gen_method == "Random":
                    sequences = [AMPGenerator.generate_random_sequence(min_len, max_len) 
                                 for _ in range(n_sequences)]
                elif gen_method == "Helical":
                    sequences = AMPGenerator.generate_helical_sequences(n_sequences)
                elif gen_method == "Cyclic":
                    sequences = AMPGenerator.generate_cyclic_sequences(n_sequences)
                else:
                    st.info("Motif-based generation coming soon!")
                    sequences = []
                
                if sequences:
                    features_df = AMPFeaturizer.featurize_batch(sequences)
                    st.session_state['generated_sequences'] = sequences
                    st.session_state['amp_features'] = features_df
                    
                    st.success(f"Generated {len(sequences)} sequences!")
                    st.dataframe(features_df.head(10), use_container_width=True)
                    
                    st.download_button(
                        "Download Sequences",
                        data=features_df.to_csv(index=False),
                        file_name="generated_amps.csv",
                        mime="text/csv"
                    )
    
    with tab2:
        st.subheader("Predict AMP Activity")
        
        if 'amp_features' in st.session_state:
            df = st.session_state['amp_features']
            
            if st.button("Predict Activity"):
                with st.spinner("Predicting..."):
                    model = AMPModel(model_type='rf')
                    
                    train_data = generate_synthetic_amp_data(1000)
                    feature_cols = [col for col in train_data.columns 
                                   if col not in ['sequence', 'activity']]
                    X_train = train_data[feature_cols].values
                    y_train = train_data['activity'].values
                    
                    model.train(X_train, y_train)
                    
                    X = df[feature_cols].values
                    y_pred, y_prob = model.predict(X)
                    
                    df['predicted_activity'] = y_pred
                    df['probability_active'] = y_prob
                    
                    ranked = AMPRanker.rank_candidates(df)
                    
                    st.session_state['amp_predictions'] = ranked
                    
                    st.success("Prediction complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Active Predictions", f"{sum(y_pred)}/{len(y_pred)}")
                    with col2:
                        avg_prob = np.mean(y_prob)
                        st.metric("Avg. Probability", f"{avg_prob:.2%}")
                    
                    st.dataframe(ranked.head(20), use_container_width=True)
                    
                    st.download_button(
                        "Download Predictions",
                        data=ranked.to_csv(index=False),
                        file_name="amp_predictions.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Generate sequences first in the Generate tab!")
    
    with tab3:
        st.subheader("Train AMP Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_samples = st.slider("Training samples", 500, 5000, 1000)
            test_size = st.slider("Test size", 0.1, 0.3, 0.2)
        
        with col2:
            model_type = st.selectbox("Model type", ["rf", "gb"])
            tune_params = st.checkbox("Tune hyperparameters", value=False)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                train_data = generate_synthetic_amp_data(n_samples)
                feature_cols = [col for col in train_data.columns 
                               if col not in ['sequence', 'activity']]
                
                X = train_data[feature_cols].values
                y = train_data['activity'].values
                
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                model = AMPModel(model_type=model_type)
                params = model.train(X_train, y_train, tune_hyperparams=tune_params)
                
                y_pred, y_prob = model.predict(X_test)
                
                from sklearn.metrics import roc_auc_score, classification_report
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AUC-ROC", f"{roc_auc_score(y_test, y_prob):.3f}")
                with col2:
                    st.metric("Accuracy", f"{(y_pred == y_test).mean():.3f}")
                with col3:
                    st.metric("Samples", n_samples)
                
                st.text("Classification Report:")
                st.code(classification_report(y_test, y_pred))
                
                model.save("models/amp_model.joblib")
                st.success("Model saved!")


def repurposed_section():
    st.header("Halicin-Like Repurposed Scaffolds")
    st.info("""
    This module allows you to search FDA-approved drugs for antibacterial repurposing.
    
    **Features coming soon:**
    - PubChem API integration
    - Similarity searching to Halicin
    - Activity prediction
    """)
    
    np_df = NaturalProductFetcher.get_common_polyphenols()
    st.subheader("Available Polyphenols (for hybrid design)")
    st.dataframe(np_df, use_container_width=True)


def polyphenols_section():
    st.header("Polyphenols & Natural Products")
    
    np_df = NaturalProductFetcher.get_common_polyphenols()
    
    st.subheader("Natural Product Templates")
    st.dataframe(np_df, use_container_width=True)
    
    st.info("""
    **Coming soon:**
    - Scaffold hopping for optimization
    - Curcumin hybrid generation
    - QSAR modeling
    """)


def schiff_section():
    st.header("Schiff Bases")
    st.info("""
    Original Schiff Base ML module.
    
    Run `notebooks/01_rdkit_intro.ipynb` to generate Schiff bases.
    Run `notebooks/02_ml_pipeline.ipynb` for ML predictions.
    """)


def peptidomimetics_section():
    st.header("Peptidomimetics & PROTAC-Like Degraders")
    st.info("""
    **Coming soon:**
    - Peptide to small-molecule conversion
    - PROTAC linker optimization
    - LpxC target docking
    - AlphaFold structure prediction
    """)


if __name__ == "__main__":
    main()
