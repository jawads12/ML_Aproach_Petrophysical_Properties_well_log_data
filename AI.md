# AI/ML Modeling Report for Petrophysical Properties

Updated: 2026-06-29 22:38

## 1. Objective

Build supervised machine learning models to predict three reservoir properties for each depth row:

- Porosity
- Permeability k
- Water Saturation Sw

The work uses the final corrected workbook produced from the LAS conversion and formula/core-analysis workflow:

`outputs/final_renamed_porosity_20260629/KAILA3_final_porosity_perm_corrected.xlsx`

## 2. Dataset Used

- Sheet: `Calculated Data`
- Total usable rows after numeric cleaning: 7,254
- Depth interval: 7800.0 to 11513.0
- Input features before engineering: 12
- Input features after engineering: 37
- Targets: Porosity, Permeability k, Water Saturation Sw

Important thesis note: the target values in this workbook are mostly petrophysical formula-derived values, with core-analysis replacement where available from the previous step. Therefore, strong ML scores can reflect both real log response and deterministic formula relationships. These models should be described as data-driven approximations of the prepared petrophysical property table, not as a substitute for independent core-calibrated validation unless more measured core targets are added.

## 3. Preprocessing

- Converted all selected columns to numeric values.
- Removed duplicate full rows.
- Removed rows missing depth or any target value.
- Sorted rows by depth.
- Used mean imputation for missing feature values, matching the previous thesis preprocessing.
- Applied IQR-based clipping to input-feature outliers inside the model pipeline. Target variables were not clipped or edited for score improvement.
- Applied IsolationForest on input features only to remove anomalous/noisy input rows before model training.
- Applied Min-Max normalization inside each model pipeline, matching the previous thesis.
- Applied target-specific correlation feature selection using the previous thesis rule: retain features with `|r| >= 0.20`, then reduce very high inter-feature correlation using `|r| >= 0.85`.
- Used physical post-processing on predictions:
  - Porosity clipped to 0-1.
  - Water saturation clipped to 0-1.
  - Permeability clipped to non-negative values.
- Modeled permeability using `log1p(k)` during training, then transformed predictions back to original k units for all reported metrics. This reduces the effect of high-permeability outliers.

Previous-thesis preprocessing audit:

| Step | Value |
| --- | --- |
| raw_rows | 7441.0000 |
| initial_missing_values | 13.0000 |
| duplicate_full_rows_removed | 0.0000 |
| rows_removed_missing_depth_or_target | 1.0000 |
| porosity_gt_0_35 | 20.0000 |
| porosity_lt_0 | 0.0000 |
| permeability_gt_800 | 339.0000 |
| permeability_lte_0 | 0.0000 |
| water_saturation_gt_1 | 47.0000 |
| water_saturation_lt_0 | 0.0000 |
| isolation_forest_contamination | 0.0250 |
| isolation_forest_rows_removed | 186.0000 |
| final_rows_after_previous_thesis_preprocessing | 7254.0000 |

Previous-thesis preprocessing comparison:

| Previous thesis preprocessing step | Our implementation | Status |
| --- | --- | --- |
| Data import and descriptive statistics | Imported final Excel sheet, numeric conversion, dataset_summary.csv | Done |
| Missing-value check and mean imputation | Missing values counted in preprocessing_audit.csv; mean imputation used inside all model pipelines | Done |
| Input-feature outlier treatment using IQR | IQRClipper applied inside each model pipeline using training data only | Done |
| Target invalid-value removal | Target physical-range issues are flagged in preprocessing_audit.csv but targets are not altered to avoid artificial score inflation | Flagged, not manipulated |
| Duplicate full-row removal | Full duplicate rows removed before modeling | Done |
| Noisy-data detection using IsolationForest | IsolationForest applied to input features only; 2.5% anomalous input rows removed | Done |
| Binning/noise impact reduction | IQR clipping plus Min-Max normalization reduces feature noise while preserving continuous log values for regression | Adapted |
| Min-Max normalization | MinMaxScaler applied inside every model pipeline | Done |
| Correlation heatmap feature selection \|r\| >= 0.20 | Target-specific correlation feature selection applied inside train/validation folds | Done |
| Remove/check highly correlated features \|r\| >= 0.85 | Redundant high-correlation features reduced during feature selection | Done |
| 70/15/15 train-validation-test split | 70/15/15 split retained, sorted by depth for stricter unseen-depth testing | Done, stricter |
| Linear Regression, SVR, RF, XGBoost, ANN | Linear Regression, SVR, Random Forest, XGBoost, ANN plus Ridge, Extra Trees, Baseline | Done and expanded |

## 4. Feature Engineering

The model input matrix used original well-log curves plus engineered features:

- Log-resistivity transforms: `log10_ILD`, `log10_SFLU`, `log10_MSFL`
- Resistivity ratios: `ILD_to_SFLU`, `MSFL_to_SFLU`, `ILD_to_MSFL`
- Cross features: `NPHI_RHOB_product`, `NPHI_minus_DRHO`, `GR_DT_product`
- Normalized depth: `DEPT_normalized`
- Depth-window rolling statistics for `GR`, `ILD`, `RHOB`, `NPHI`, and `DT` using 5-row and 15-row windows

## 5. Validation Design

Because well-log rows are ordered by depth and adjacent rows are highly correlated, the primary split is depth-aware:

- Training: first 70 percent of sorted depth rows
- Validation: next 15 percent of sorted depth rows
- Test: deepest/final 15 percent of sorted depth rows

Additional validation was performed with 5-fold `TimeSeriesSplit` on the training plus validation interval. This is stricter than random K-fold because each fold predicts a later depth interval from earlier rows.

## 6. Algorithms Tested

- Baseline Mean
- Linear Regression
- Ridge Regression
- Random Forest
- Extra Trees
- SVR RBF
- ANN Deep MLP
- XGBoost

## 7. Dataset Statistical Summary

| Column | Missing | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- | --- |
| DEPT | 0 | 9666.2892 | 1064.2045 | 7800.0000 | 11513.0000 |
| GR | 0 | 47.5028 | 13.3587 | 22.2000 | 127.3000 |
| CALI | 0 | 9.3429 | 1.5198 | 8.2867 | 19.3800 |
| SP | 0 | -84.0719 | 21.9962 | -125.2222 | -34.8444 |
| ILD | 0 | 21.5217 | 24.8839 | 4.0897 | 232.8990 |
| SFLU | 0 | 23.4885 | 16.1548 | 4.8272 | 119.0410 |
| MSFL | 0 | 29.4190 | 14.8233 | 5.7504 | 197.5589 |
| DT | 0 | 83.0857 | 8.0631 | 56.0667 | 125.1333 |
| RHOB | 0 | 2.3981 | 0.0909 | 2.1713 | 2.6493 |
| DRHO | 0 | 0.0060 | 0.0224 | -0.0247 | 0.1743 |
| PEF | 0 | 3.0590 | 0.3795 | 2.2800 | 5.5200 |
| NPHI | 0 | 23.2275 | 3.4434 | 8.8133 | 52.8933 |
| Porosity | 0 | 0.2135 | 0.0314 | 0.0957 | 0.4222 |
| Permeability k | 0 | 246.4321 | 728.6450 | 0.5403 | 31684.0448 |
| Water Saturation Sw | 0 | 0.4902 | 0.1856 | 0.1001 | 1.3599 |

## 8. Best Test Model by Target

| Target | BestModel | Test_R2 | Test_RMSE | Test_MAE | Test_MAPE_percent | Test_Pearson_r | SelectedFeatureCount |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Porosity | Random Forest | 0.9915 | 0.0016 | 0.0008 | 0.4230 | 0.9961 | 14 |
| Permeability k | ANN Deep MLP | 0.7652 | 12.0077 | 6.0266 | 14.4277 | 0.8905 | 13 |
| Water Saturation Sw | Random Forest | 0.8112 | 0.0596 | 0.0446 | 7.2377 | 0.9113 | 15 |

## 9. Depth Holdout Validation and Test Scores

| Target | Model | ValidationType | Rows | R2 | RMSE | MAE | MAPE_percent | Pearson_r |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Permeability k | ANN Deep MLP | Depth Holdout Test | 1089 | 0.7652 | 12.0077 | 6.0266 | 14.4277 | 0.8905 |
| Permeability k | Linear Regression | Depth Holdout Test | 1089 | 0.7542 | 12.2860 | 5.6044 | 14.9685 | 0.8688 |
| Permeability k | Ridge Regression | Depth Holdout Test | 1089 | 0.7514 | 12.3560 | 5.4825 | 14.8041 | 0.8671 |
| Permeability k | SVR RBF | Depth Holdout Test | 1089 | 0.7088 | 13.3734 | 5.3605 | 16.5160 | 0.8584 |
| Permeability k | Extra Trees | Depth Holdout Test | 1089 | 0.7022 | 13.5242 | 5.0796 | 13.2663 | 0.8645 |
| Permeability k | XGBoost | Depth Holdout Test | 1089 | 0.6682 | 14.2761 | 6.2561 | 15.3638 | 0.8561 |
| Permeability k | Random Forest | Depth Holdout Test | 1089 | 0.6323 | 15.0283 | 6.3173 | 16.1763 | 0.8246 |
| Permeability k | Baseline Mean | Depth Holdout Test | 1089 | -9.3218 | 79.6202 | 76.7763 | 378.1903 |  |
| Permeability k | Linear Regression | Depth Holdout Validation | 1088 | 0.9709 | 22.5484 | 9.5648 | 18.8620 | 0.9854 |
| Permeability k | ANN Deep MLP | Depth Holdout Validation | 1088 | 0.9697 | 22.9826 | 9.1822 | 15.3077 | 0.9871 |
| Permeability k | Ridge Regression | Depth Holdout Validation | 1088 | 0.9690 | 23.2544 | 10.0738 | 18.7961 | 0.9847 |
| Permeability k | Extra Trees | Depth Holdout Validation | 1088 | 0.9357 | 33.4934 | 11.4025 | 17.1095 | 0.9686 |
| Permeability k | XGBoost | Depth Holdout Validation | 1088 | 0.9207 | 37.2112 | 13.9856 | 18.8683 | 0.9637 |
| Permeability k | Random Forest | Depth Holdout Validation | 1088 | 0.9192 | 37.5650 | 14.4733 | 20.5069 | 0.9621 |
| Permeability k | SVR RBF | Depth Holdout Validation | 1088 | 0.9106 | 39.5040 | 16.7310 | 27.1408 | 0.9609 |
| Permeability k | Baseline Mean | Depth Holdout Validation | 1088 | -0.2671 | 148.7323 | 129.5192 | 447.0616 |  |
| Porosity | Random Forest | Depth Holdout Test | 1089 | 0.9915 | 0.0016 | 0.0008 | 0.4230 | 0.9961 |
| Porosity | Extra Trees | Depth Holdout Test | 1089 | 0.9879 | 0.0019 | 0.0008 | 0.4538 | 0.9952 |
| Porosity | XGBoost | Depth Holdout Test | 1089 | 0.9669 | 0.0032 | 0.0023 | 1.2355 | 0.9913 |
| Porosity | Linear Regression | Depth Holdout Test | 1089 | 0.9651 | 0.0033 | 0.0023 | 1.2509 | 0.9866 |
| Porosity | Ridge Regression | Depth Holdout Test | 1089 | 0.9627 | 0.0034 | 0.0024 | 1.3089 | 0.9864 |
| Porosity | ANN Deep MLP | Depth Holdout Test | 1089 | 0.8846 | 0.0059 | 0.0047 | 2.5793 | 0.9663 |
| Porosity | SVR RBF | Depth Holdout Test | 1089 | 0.6886 | 0.0098 | 0.0072 | 3.9763 | 0.8406 |
| Porosity | Baseline Mean | Depth Holdout Test | 1089 | -2.6269 | 0.0333 | 0.0287 | 16.2193 | 0.0000 |
| Porosity | Extra Trees | Depth Holdout Validation | 1088 | 0.9762 | 0.0027 | 0.0012 | 0.7114 | 0.9893 |
| Porosity | Random Forest | Depth Holdout Validation | 1088 | 0.9666 | 0.0031 | 0.0016 | 0.9140 | 0.9852 |
| Porosity | XGBoost | Depth Holdout Validation | 1088 | 0.9550 | 0.0037 | 0.0026 | 1.4364 | 0.9799 |
| Porosity | ANN Deep MLP | Depth Holdout Validation | 1088 | 0.9075 | 0.0052 | 0.0039 | 2.1485 | 0.9628 |
| Porosity | Linear Regression | Depth Holdout Validation | 1088 | 0.8909 | 0.0057 | 0.0045 | 2.4438 | 0.9658 |
| Porosity | Ridge Regression | Depth Holdout Validation | 1088 | 0.8902 | 0.0057 | 0.0045 | 2.4570 | 0.9665 |
| Porosity | SVR RBF | Depth Holdout Validation | 1088 | 0.7945 | 0.0078 | 0.0057 | 3.1052 | 0.9096 |
| Porosity | Baseline Mean | Depth Holdout Validation | 1088 | -4.0683 | 0.0388 | 0.0350 | 19.4943 | 0.0000 |
| Water Saturation Sw | Random Forest | Depth Holdout Test | 1089 | 0.8112 | 0.0596 | 0.0446 | 7.2377 | 0.9113 |
| Water Saturation Sw | ANN Deep MLP | Depth Holdout Test | 1089 | 0.7989 | 0.0615 | 0.0410 | 6.4300 | 0.8985 |
| Water Saturation Sw | Extra Trees | Depth Holdout Test | 1089 | 0.7800 | 0.0644 | 0.0437 | 6.9073 | 0.9054 |
| Water Saturation Sw | XGBoost | Depth Holdout Test | 1089 | 0.7769 | 0.0648 | 0.0483 | 7.8391 | 0.8950 |
| Water Saturation Sw | Ridge Regression | Depth Holdout Test | 1089 | 0.7735 | 0.0653 | 0.0439 | 6.7882 | 0.9203 |
| Water Saturation Sw | Linear Regression | Depth Holdout Test | 1089 | 0.7692 | 0.0659 | 0.0443 | 6.8313 | 0.9163 |
| Water Saturation Sw | SVR RBF | Depth Holdout Test | 1089 | 0.4869 | 0.0983 | 0.0710 | 11.2695 | 0.7048 |
| Water Saturation Sw | Baseline Mean | Depth Holdout Test | 1089 | -1.0717 | 0.1975 | 0.1451 | 20.8018 |  |
| Water Saturation Sw | Linear Regression | Depth Holdout Validation | 1088 | 0.9059 | 0.0554 | 0.0425 | 7.3178 | 0.9526 |
| Water Saturation Sw | Random Forest | Depth Holdout Validation | 1088 | 0.9039 | 0.0560 | 0.0377 | 5.8002 | 0.9623 |
| Water Saturation Sw | Extra Trees | Depth Holdout Validation | 1088 | 0.9035 | 0.0561 | 0.0373 | 5.7850 | 0.9660 |
| Water Saturation Sw | Ridge Regression | Depth Holdout Validation | 1088 | 0.8946 | 0.0586 | 0.0451 | 7.6707 | 0.9476 |
| Water Saturation Sw | XGBoost | Depth Holdout Validation | 1088 | 0.8671 | 0.0658 | 0.0455 | 7.0486 | 0.9600 |
| Water Saturation Sw | ANN Deep MLP | Depth Holdout Validation | 1088 | 0.8471 | 0.0706 | 0.0539 | 9.4334 | 0.9526 |
| Water Saturation Sw | SVR RBF | Depth Holdout Validation | 1088 | 0.6966 | 0.0994 | 0.0773 | 13.4686 | 0.8369 |
| Water Saturation Sw | Baseline Mean | Depth Holdout Validation | 1088 | -0.7170 | 0.2365 | 0.2090 | 38.3010 | -0.0000 |

## 10. Five-Fold Depth-Ordered Cross-Validation Summary

| Target | Model | CV_R2_mean | CV_R2_std | CV_RMSE_mean | CV_RMSE_std | CV_MAE_mean | CV_MAPE_percent_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Permeability k | Linear Regression | 0.8180 | 0.2427 | 33.7451 | 23.5832 | 21.0462 | 31.6146 |
| Permeability k | Ridge Regression | 0.8110 | 0.2342 | 35.1042 | 22.8543 | 21.8146 | 31.6897 |
| Permeability k | ANN Deep MLP | 0.7448 | 0.1726 | 53.2768 | 52.5483 | 28.7188 | 27.0697 |
| Permeability k | XGBoost | 0.6165 | 0.2553 | 62.5925 | 55.4755 | 35.5256 | 32.1429 |
| Permeability k | SVR RBF | 0.5468 | 0.6232 | 53.8546 | 53.0055 | 28.3735 | 44.2729 |
| Permeability k | Extra Trees | 0.5345 | 0.4512 | 63.8264 | 63.3627 | 33.3085 | 32.2969 |
| Permeability k | Random Forest | 0.4562 | 0.5382 | 67.7227 | 59.4207 | 36.1872 | 36.2020 |
| Permeability k | Baseline Mean | -6.3313 | 11.8490 | 165.0109 | 54.6842 | 145.0693 | 370.9310 |
| Porosity | Extra Trees | 0.9322 | 0.0533 | 0.0043 | 0.0017 | 0.0024 | 1.2272 |
| Porosity | XGBoost | 0.9266 | 0.0302 | 0.0049 | 0.0017 | 0.0033 | 1.6636 |
| Porosity | Random Forest | 0.9241 | 0.0453 | 0.0047 | 0.0015 | 0.0026 | 1.3174 |
| Porosity | Ridge Regression | 0.8580 | 0.0725 | 0.0066 | 0.0016 | 0.0049 | 2.4424 |
| Porosity | Linear Regression | 0.7930 | 0.2103 | 0.0072 | 0.0024 | 0.0056 | 2.7151 |
| Porosity | SVR RBF | 0.3746 | 0.5001 | 0.0131 | 0.0038 | 0.0094 | 4.6551 |
| Porosity | Baseline Mean | -1.8953 | 1.6960 | 0.0310 | 0.0120 | 0.0255 | 13.4965 |
| Porosity | ANN Deep MLP | -2.9459 | 7.5052 | 0.0203 | 0.0213 | 0.0175 | 7.6569 |
| Water Saturation Sw | XGBoost | 0.4171 | 0.5988 | 0.0670 | 0.0490 | 0.0520 | 10.7102 |
| Water Saturation Sw | Random Forest | 0.4088 | 0.6465 | 0.0645 | 0.0486 | 0.0483 | 8.4263 |
| Water Saturation Sw | Ridge Regression | 0.3127 | 0.7567 | 0.0632 | 0.0461 | 0.0482 | 9.1222 |
| Water Saturation Sw | Extra Trees | 0.3045 | 0.8892 | 0.0740 | 0.0625 | 0.0568 | 12.0289 |
| Water Saturation Sw | Linear Regression | 0.2816 | 0.8131 | 0.0625 | 0.0457 | 0.0477 | 9.0747 |
| Water Saturation Sw | SVR RBF | -0.8122 | 2.1363 | 0.1063 | 0.0623 | 0.0842 | 18.7551 |
| Water Saturation Sw | ANN Deep MLP | -1.0482 | 3.5375 | 0.0765 | 0.0462 | 0.0613 | 14.1624 |
| Water Saturation Sw | Baseline Mean | -1.9347 | 2.6380 | 0.1757 | 0.1185 | 0.1548 | 28.8169 |

## 11. Thesis Graphs Generated

All thesis-ready graphs were saved in `ml_outputs/thesis_graphs/`. A combined PDF graph pack is also available at `ml_outputs/thesis_graphs/thesis_graph_pack.pdf`.

Generated figures:

- `01_target_property_distributions.png`: histograms for porosity, log permeability, and water saturation.
- `02_depth_profiles_logs_and_targets.png`: well-log and reservoir-property profiles versus depth.
- `03_correlation_heatmap.png`: Pearson correlation heatmap for major logs and target properties.
- `04_model_comparison_test_r2.png`: depth-holdout test R2 comparison for all algorithms.
- `05_model_comparison_test_mae.png`: depth-holdout test MAE comparison for all algorithms.
- `06_depth_ordered_cv_r2.png`: five-fold depth-ordered CV R2 comparison with standard deviation.
- `07_best_models_predicted_vs_actual_panel.png`: predicted-versus-actual panels for the best model of each target.
- `08_best_models_residuals_by_depth_panel.png`: residuals versus depth for the best model of each target.
- `12_all_models_predicted_vs_actual_porosity.png`: predicted-versus-actual comparison for all porosity models.
- `12_all_models_predicted_vs_actual_permeability_k.png`: predicted-versus-actual comparison for all permeability models.
- `12_all_models_predicted_vs_actual_water_saturation_sw.png`: predicted-versus-actual comparison for all water-saturation models.
- `all_models_predicted_vs_actual/`: individual predicted-versus-actual graphs for every target-model combination.
- `09_porosity_permeability_sw_crossplot.png`: reservoir-quality crossplot of porosity versus permeability colored by Sw.
- `10_neutron_density_porosity_crossplot.png`: NPHI-RHOB crossplot colored by porosity.
- `11_best_model_feature_importance_panel.png`: most influential input features for the best models.

## 12. Output Files

- `ml_outputs/model_scores.csv`: validation and test metrics for every model-target combination.
- `ml_outputs/cv_fold_scores.csv`: fold-by-fold depth-ordered cross-validation metrics.
- `ml_outputs/cv_summary.csv`: mean and standard deviation CV metrics.
- `ml_outputs/test_predictions.csv`: actual and predicted test rows for every model and target.
- `ml_outputs/preprocessed_modeling_dataset.csv`: cleaned and engineered feature table with targets.
- `ml_outputs/preprocessing_audit.csv`: previous-thesis preprocessing checks and row counts.
- `ml_outputs/previous_thesis_preprocessing_comparison.csv`: direct checklist comparing the previous thesis preprocessing to this workflow.
- `ml_outputs/feature_selection_summary.csv`: correlation-based feature-selection audit for each target.
- `ml_outputs/best_models_summary.json`: best model selection by target.
- `ml_outputs/ml_results_workbook.xlsx`: Excel workbook containing best models, holdout scores, CV scores, dataset summary, and test predictions.
- `ml_outputs/models/`: saved best model files.
- `ml_outputs/figures/`: predicted-vs-actual, residual, and feature-importance/coefficient plots.
- `ml_outputs/thesis_graphs/`: thesis graph PNGs and combined PDF graph pack.
- `requirements.txt`: Python package versions used for the ML run.

## 13. Recommended Thesis Interpretation

Use the depth-holdout test scores as the main performance evidence because they represent prediction into an unseen depth interval. Use the 5-fold depth-ordered CV scores as supporting evidence for model stability. If the thesis later obtains additional measured core porosity, measured permeability, or measured water saturation over more depths, rerun this pipeline using those measured values as targets to establish independent laboratory-calibrated prediction accuracy.

## 14. Reproduction Commands

```bash
.venv/bin/python scripts/ml_pipeline.py
.venv/bin/python scripts/make_thesis_graphs.py
```
