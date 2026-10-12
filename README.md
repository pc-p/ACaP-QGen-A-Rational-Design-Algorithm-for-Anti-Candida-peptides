# ACaP-QGen-A-Rational-Design-Algorithm-for-Anti-Candida-peptides
Official repository for "ACaP-QGen: An Algorithm for the Rational Design of Anti-Candida peptides Based on Interpretable Machine Learning". Features dual-modal MIC prediction (ACaP-QPre) and SHAP-guided de novo design.

## Overview

ACaP-QGen is an interpretable machine learning framework for the rational design of anti-*Candida* peptides. It integrates quantitative activity prediction (ACaP-QPre), SHAP-based feature interpretation, and constrained sequence generation to improve peptide design efficiency.

## Workflow

1. **Activity Prediction:** Predict peptide MIC values using dual-modal features (SeqComp + AAindex) and CatBoost.
2. **SHAP Interpretation:** Identify key features and amino acid preferences associated with antifungal activity.
3. **Peptide Generation:** Generate and screen candidate sequences based on interpretable design rules.
4. **Experimental Validation:** Evaluate the antifungal activity of selected peptides.

## Results

* **Prediction performance:** R² = 0.455, MSE = 0.227.
* **Experimental validation:** All 13 designed peptides exhibited anti-*Candida* activity (MIC: 7.81–62.5 μg/mL against *C. albicans*).
* **Promising candidates:** ACaP-3, ACaP-9, and ACaP-11.

## Code and Data

This repository provides the source code and supporting datasets for ACaP-QGen.

## Citation

Please cite our associated manuscript when using this work.
