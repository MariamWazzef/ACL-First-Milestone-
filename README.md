# Formula 1 Podium Prediction Companion

## Data source
This run uses [Rohan Rao’s Formula 1 World Championship dataset on Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020). The downloaded release contained the 14 supported Ergast-style CSV tables and covered 1950–2024. The dataset archive is not bundled here. Attach the dataset in Kaggle, place the CSV files beneath the notebook working folder, or set `F1_DATA_ROOT` to their directory.

## Run
1. Put the CSV tables anywhere beneath the folder containing the notebook, or set `DATA_ROOT` in Section 0. The notebook discovers filenames recursively and prints selected paths.
2. Use Python 3.10 or newer and install `requirements.txt`.
3. Open `f1_podium_prediction_milestone1.ipynb` in Jupyter and select **Run All**. On Kaggle, `/kaggle/input` is searched automatically.

The project uses scikit-learn’s `MLPClassifier` for the shallow feed-forward neural network and SHAP for global and local explanations. TensorFlow was unavailable in the verified execution environment; the project therefore uses its documented scikit-learn fallback. LIME is optional because the requirements allow local SHAP instead.

## Expected data
Required core tables: `circuits.csv`, `constructors.csv`, `drivers.csv`, `races.csv`, `results.csv`, `qualifying.csv`, and `status.csv`. Optional supported tables include constructor results/standings, driver standings, lap times, pit stops, sprint results, and seasons. Tables may be in nested folders. Missing core files raise an error naming the expected tables. Raw CSV files are read only.

## Outputs
Computed tables are written to `outputs/tables/`, figures to `outputs/figures/`, and serialized model artifacts to `outputs/artifacts/`. The run includes pre-cleaning and post-cleaning EDA, semantic duration parsing, duplicate/key/foreign-key audits, the leakage-safe modeling table, the three data-engineering answers, three model families, validation-selected thresholds, preprocessing and feature-group experiments, permutation importance, global/local SHAP, and complete/partially-missing inference examples. The analytical report is available as PDF and Markdown. A plain-language notebook walkthrough and evaluation study guide are also included.

## Verified run
All reported values were recomputed by executing the notebook top to bottom on the linked dataset. Cleaning reduced 26,759 raw result rows to 26,668 unique race-driver rows using the documented shared-drive duplicate rule. The chronological split uses training through 2019, validation in 2020–2021, and test in 2022–2024. The neural network was selected on validation PR-AUC (0.734); its held-out test PR-AUC was 0.666, ROC-AUC 0.923, and F1 0.644 at the validation-selected threshold. Random Forest had a higher test PR-AUC (0.676); model selection remains based on validation only. The MLP restores the weights from the epoch with minimum external-validation log loss (epoch 5 in this run). Global and local SHAP explanations were generated.

The home-race analysis includes an Indianapolis 500 sensitivity check: the standardized home-minus-away difference was -0.32 percentage points with all championship races and -0.20 points after excluding Indianapolis 500 events from 1950–1960. Starting grid and qualifying positions are audited separately because penalties can make them differ.

## Reproducibility and limitations
Random seeds are set to 42. The temporal split is train ≤2019, validation 2020–2021, test 2022–2024. The target is `positionOrder <= 3`; current-race results and post-race fields do not enter the predictors. Feature transformations are fitted on training rows only. Threshold and model selection use validation only; test scores are reported once as final evaluation. Historical feature coverage, rule changes, class imbalance, identity changes, and source-data quality limit generalization. Explanations and descriptive comparisons are not causal evidence.
