# FluvialDrainage-ML

Ensemble machine learning framework for the automated identification and classification of fluvial drainage patterns from geospatial tabular data.

> \*\*Please cite this software as:\*\*  
> D'Aniello M., Velotti C., Esposito Mocerino G., Donadio C. — \*An ensemble machine learning framework for automated classification of fluvial drainage networks\*. Computers \& Geosciences (2026).

\---

## What is this repository for?

This repository implements a dual machine learning pipeline for the large-scale classification of river network morphologies. Starting from the HydroRIVERS geospatial dataset, the pipeline:

1. **Discovers** natural morphological groupings through unsupervised clustering (K-Means, CLARANS);
2. **Classifies** approximately 57,000 river segments into seven drainage pattern categories — Dendritic, Sub-dendritic, Pinnate, Parallel, Rectangular, Trellis, and Angular — using a supervised ensemble of four tree-based models (Random Forest, XGBoost, LightGBM, HistGradientBoosting);
3. **Evaluates** the automated classifications by computing the novel Composite Modification Score (CMS), which evaluates the degree of deviation of each river segment from an idealized drainage pattern.

\---

## Repository Structure

```
FluvialDrainage-ML/
├── Clustering\_fe.ipynb         # Unsupervised clustering pipeline (K-Means, CLARANS)
├── Classifiers\_fe.ipynb        # Supervised ensemble classification pipeline
├── quick\_test.py               # Reproducible quick-test script (see Testing section)
├── README.md
├── LICENSE
└── data/
    ├── rivers.csv              # Primary HydroRIVERS dataset (\~57,000 segments)
    ├── Merged\_labels/          # Per-class label files (one CSV per morphological class)
    ├── Labeling\_for\_each\_river/# Intermediate per-river labeling data
    └── test\_items/             # Holdout test samples for individual river validation
```

\---

## Dependencies

The following Python packages are required. Versions listed are those used during development; other recent versions should work as well.

|Package|Recommended version|
|-|-|
|Python|≥ 3.10|
|pandas|≥ 2.0|
|numpy|≥ 1.24|
|scikit-learn|≥ 1.3|
|xgboost|≥ 2.0|
|lightgbm|≥ 4.0|
|pyclustering|≥ 0.10|
|optuna|≥ 3.0|
|hdbscan|≥ 0.8|
|matplotlib|≥ 3.7|
|jupyter|≥ 1.0|

Install all dependencies at once:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm pyclustering optuna hdbscan matplotlib jupyter
```

\---

## How to Use

### Step 1 — Unsupervised Clustering

Open and run `Clustering\_fe.ipynb` from top to bottom. This notebook:

* Loads `data/rivers.csv` and applies feature engineering;
* Runs K-Means and CLARANS on the full \~57,000-segment dataset;
* Evaluates clusters using Silhouette Score, Davies–Bouldin Index, and Dunn Index;
* Produces the manual label assignment interface (PCA visualization).

### Step 2 — Supervised Classification

Open and run `Classifiers\_fe.ipynb` from top to bottom. This notebook:

* Loads the labelled data from `data/Merged\_labels/`;
* Applies the same feature engineering pipeline;
* Trains an ensemble of RF, XGBoost, LightGBM, and HistGradientBoosting with pseudo-labeling;
* Reports per-class performance metrics (Precision, Recall, F1-score) on a held-out test set.

> \*\*Note on reproducibility:\*\* All stochastic components (train/test split, classifier initialization) are seeded with `random\_state=42`. Re-running the notebooks on the same data will produce identical results. The metrics reported in the associated paper correspond to the best configuration identified through an extensive grid search over hyperparameter space.

\---

## Quick Test

A standalone Python script `quick\_test.py` is provided to verify that the environment is correctly set up and that the classification pipeline runs end-to-end without errors.

### Running the test

From the root directory of the repository:

```bash
python quick\_test.py
```

The script will:

1. Load `data/rivers.csv` and apply feature engineering;
2. Run a full clustering analysis (K-Means) and evaluate it using internal metrics (Silhouette, Davies-Bouldin, Dunn);
3. Trains a complete supervised ensemble (RandomForest, XGBoost, LightGBM, HistGradientBoosting) using the labelled dataset;
4. Prints clustering performance metrics and tests the end-to-end pipeline on two sample rivers (`Garonne\_angular.csv` and `Po\_pinnate.csv`), calculating both the Delta error for clustering and the Composite Modification Score (CMS) for the ensemble classification.

**Expected output (approximate):**

```text
--- FluvialDrainage-ML Quick Test ---

--- 1. Clustering Analysis ---
Loading and engineering features for clustering test...
Running K-Means (k=7) on the full dataset...
Successfully assigned 57082 river segments to 7 clusters.

\[Clustering Metrics \& Label Mapping]
Cluster -> Geological Class (Majority Voting):
  Cluster 0: Trellis
  Cluster 1: Pinnate
  Cluster 2: Dendritic
  Cluster 3: Angular
  Cluster 4: Rectangular
  Cluster 5: Sub-dendritic
  Cluster 6: Parallel

--- 2. Supervised Classification ---
Total labelled rivers: 9907
Training pseudo-labeled models for Ensemble (RandomForest, XGBoost, LightGBM, HistGradientBoosting)...

--- 3. Testing pipeline on Garonne and Po ---

\[Clustering Analysis for Garonne\_angular.csv]
Delta (Improving distance towards 'Angular'): 0.0181

\[Supervised Classification \& CMS Test for Garonne\_angular.csv]
Original classification for Garonne\_angular.csv: angular
Classification is already correct. CMS = 0.0000
--------------------------------------------------

\[Clustering Analysis for Po\_pinnate.csv]
Delta (Improving distance towards 'Pinnate'): 0.0090

\[Supervised Classification \& CMS Test for Po\_pinnate.csv]
Original classification for Po\_pinnate.csv: pinnate
Classification is already correct. CMS = 0.0000
--------------------------------------------------
```

\---

## Contact

* Mariarca D'Aniello — mariarca.daniello@unina.it  (1)
* Claudio Velotti — cla.velotti@studenti.unina.it  (2)

(1) Department of Earth Sciences, Environment and Resources (DISTAR), University of Naples Federico II

(2) Department of Electrical Engineering and Information Technology (DIETI), University of Naples Federico II

\---

## License

This project is freely available under the **MIT License** — see the [LICENSE](LICENSE) file for details.

