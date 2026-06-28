# AI/ML Modeling Report for Petrophysical Properties

Updated: 2026-06-29 01:36

## 1. Objective

Build supervised machine learning models to predict three reservoir properties for each depth row:

- Porosity
- Permeability k
- Water Saturation Sw

The work uses the final corrected workbook produced from the LAS conversion and formula/core-analysis workflow:

`outputs/final_renamed_porosity_20260629/KAILA3_final_porosity_perm_corrected.xlsx`

## 2. Dataset Used

- Sheet: `Calculated Data`
- Total usable rows after numeric cleaning: 7,440
- Depth interval: 7800.0 to 11519.5
- Input features before engineering: 12
- Input features after engineering: 37
- Targets: Porosity, Permeability k, Water Saturation Sw

Important thesis note: the target values in this workbook are mostly petrophysical formula-derived values, with core-analysis replacement where available from the previous step. Therefore, strong ML scores can reflect both real log response and deterministic formula relationships. These models should be described as data-driven approximations of the prepared petrophysical property table, not as a substitute for independent core-calibrated validation unless more measured core targets are added.

## 3. Preprocessing

- Converted all selected columns to numeric values.
- Removed rows missing depth or any target value.
- Sorted rows by depth.
- Used median imputation for missing feature values inside each model pipeline.
- Used standardized scaling for Ridge Regression, SVR, and ANN.
- Used physical post-processing on predictions:
  - Porosity clipped to 0-1.
  - Water saturation clipped to 0-1.
  - Permeability clipped to non-negative values.
- Modeled permeability using `log1p(k)` during training, then transformed predictions back to original k units for all reported metrics. This reduces the effect of high-permeability outliers.

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
- Ridge Regression
- Random Forest
- Extra Trees
- SVR RBF
- ANN Deep MLP
- XGBoost

## 7. Dataset Statistical Summary

| Column | Missing | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- | --- |
| DEPT | 0 | 9659.7500 | 1073.9437 | 7800.0000 | 11519.5000 |
| GR | 0 | 47.3764 | 13.4661 | 22.2000 | 144.7000 |
| CALI | 0 | 9.3350 | 1.5091 | 8.2867 | 19.3800 |
| SP | 0 | -84.3643 | 21.9982 | -125.2222 | -34.8444 |
| ILD | 0 | 23.8093 | 33.9320 | 4.0897 | 741.9307 |
| SFLU | 0 | 25.9480 | 36.5363 | 4.5119 | 1779.7676 |
| MSFL | 0 | 30.3661 | 18.1194 | 5.6802 | 408.9749 |
| DT | 0 | 83.0928 | 8.4747 | 51.6667 | 125.1333 |
| RHOB | 0 | 2.3974 | 0.0948 | 1.6053 | 2.6996 |
| DRHO | 0 | 0.0065 | 0.0239 | -0.0258 | 0.2438 |
| PEF | 0 | 3.0708 | 0.4186 | 2.2800 | 6.8667 |
| NPHI | 0 | 23.1334 | 3.7743 | 4.6800 | 52.8933 |
| Porosity | 0 | 0.2133 | 0.0342 | 0.0498 | 0.4737 |
| Permeability k | 0 | 294.9530 | 1021.5302 | 0.0151 | 31684.0448 |
| Water Saturation Sw | 0 | 0.4880 | 0.1943 | 0.0841 | 2.0208 |

## 8. Best Test Model by Target

| Target | BestModel | Test_R2 | Test_RMSE | Test_MAE | Test_MAPE_percent | Test_Pearson_r |
| --- | --- | --- | --- | --- | --- | --- |
| Porosity | Ridge Regression | 0.9567 | 0.0055 | 0.0022 | 1.0731 | 0.9853 |
| Permeability k | Ridge Regression | 0.3904 | 312.1091 | 17.4767 | 5.4775 | 0.9493 |
| Water Saturation Sw | Ridge Regression | 0.9228 | 0.0397 | 0.0220 | 3.3397 | 0.9732 |

## 9. Depth Holdout Validation and Test Scores

| Target | Model | ValidationType | Rows | R2 | RMSE | MAE | MAPE_percent | Pearson_r |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Permeability k | Ridge Regression | Depth Holdout Test | 1116 | 0.3904 | 312.1091 | 17.4767 | 5.4775 | 0.9493 |
| Permeability k | XGBoost | Depth Holdout Test | 1116 | 0.0335 | 392.9981 | 34.9081 | 17.3857 | 0.3205 |
| Permeability k | SVR RBF | Depth Holdout Test | 1116 | 0.0206 | 395.6157 | 34.0109 | 22.8515 | 0.2574 |
| Permeability k | Extra Trees | Depth Holdout Test | 1116 | 0.0167 | 396.3873 | 33.3818 | 12.5450 | 0.2378 |
| Permeability k | Random Forest | Depth Holdout Test | 1116 | 0.0146 | 396.8082 | 34.8561 | 17.2339 | 0.2083 |
| Permeability k | Baseline Mean | Depth Holdout Test | 1116 | -0.0140 | 402.5401 | 107.2764 | 405.3830 |  |
| Permeability k | ANN Deep MLP | Depth Holdout Test | 1116 | -198.3828 | 5644.5122 | 374.5925 | 32.1364 | 0.9820 |
| Permeability k | Ridge Regression | Depth Holdout Validation | 1116 | 0.9943 | 11.9189 | 4.4078 | 8.0018 | 0.9975 |
| Permeability k | XGBoost | Depth Holdout Validation | 1116 | 0.9229 | 43.7297 | 13.4206 | 22.1306 | 0.9685 |
| Permeability k | Extra Trees | Depth Holdout Validation | 1116 | 0.8958 | 50.8539 | 12.8483 | 25.0951 | 0.9581 |
| Permeability k | Random Forest | Depth Holdout Validation | 1116 | 0.8515 | 60.7105 | 17.0285 | 34.0227 | 0.9412 |
| Permeability k | SVR RBF | Depth Holdout Validation | 1116 | 0.8088 | 68.8786 | 11.9767 | 284.2956 | 0.9108 |
| Permeability k | Baseline Mean | Depth Holdout Validation | 1116 | -0.1766 | 170.8646 | 137.1706 | 2190.3854 |  |
| Permeability k | ANN Deep MLP | Depth Holdout Validation | 1116 | -12558.0042 | 17652.9754 | 567.5752 | 305947.7643 | -0.0054 |
| Porosity | Ridge Regression | Depth Holdout Test | 1116 | 0.9567 | 0.0055 | 0.0022 | 1.0731 | 0.9853 |
| Porosity | XGBoost | Depth Holdout Test | 1116 | 0.7491 | 0.0133 | 0.0029 | 1.2151 | 0.8831 |
| Porosity | ANN Deep MLP | Depth Holdout Test | 1116 | 0.7291 | 0.0138 | 0.0076 | 4.0648 | 0.8585 |
| Porosity | Extra Trees | Depth Holdout Test | 1116 | 0.6764 | 0.0151 | 0.0023 | 0.8210 | 0.8372 |
| Porosity | Random Forest | Depth Holdout Test | 1116 | 0.6560 | 0.0156 | 0.0023 | 0.8103 | 0.8231 |
| Porosity | SVR RBF | Depth Holdout Test | 1116 | 0.2743 | 0.0227 | 0.0121 | 6.1456 | 0.5373 |
| Porosity | Baseline Mean | Depth Holdout Test | 1116 | -0.9650 | 0.0373 | 0.0299 | 16.5509 |  |
| Porosity | Extra Trees | Depth Holdout Validation | 1116 | 0.9821 | 0.0027 | 0.0015 | 0.9395 | 0.9919 |
| Porosity | XGBoost | Depth Holdout Validation | 1116 | 0.9742 | 0.0033 | 0.0022 | 1.3829 | 0.9886 |
| Porosity | Random Forest | Depth Holdout Validation | 1116 | 0.9740 | 0.0033 | 0.0018 | 1.1159 | 0.9883 |
| Porosity | Ridge Regression | Depth Holdout Validation | 1116 | 0.9390 | 0.0051 | 0.0034 | 1.9393 | 0.9803 |
| Porosity | ANN Deep MLP | Depth Holdout Validation | 1116 | 0.5757 | 0.0134 | 0.0094 | 5.3159 | 0.8403 |
| Porosity | SVR RBF | Depth Holdout Validation | 1116 | 0.4937 | 0.0146 | 0.0101 | 6.1904 | 0.8022 |
| Porosity | Baseline Mean | Depth Holdout Validation | 1116 | -2.9988 | 0.0410 | 0.0357 | 21.1493 | -0.0000 |
| Water Saturation Sw | Ridge Regression | Depth Holdout Test | 1116 | 0.9228 | 0.0397 | 0.0220 | 3.3397 | 0.9732 |
| Water Saturation Sw | ANN Deep MLP | Depth Holdout Test | 1116 | 0.9170 | 0.0412 | 0.0217 | 3.5847 | 0.9605 |
| Water Saturation Sw | Extra Trees | Depth Holdout Test | 1116 | 0.9041 | 0.0443 | 0.0224 | 3.8005 | 0.9608 |
| Water Saturation Sw | XGBoost | Depth Holdout Test | 1116 | 0.8823 | 0.0491 | 0.0321 | 5.4655 | 0.9489 |
| Water Saturation Sw | Random Forest | Depth Holdout Test | 1116 | 0.8689 | 0.0518 | 0.0296 | 5.0526 | 0.9369 |
| Water Saturation Sw | SVR RBF | Depth Holdout Test | 1116 | 0.8049 | 0.0631 | 0.0303 | 4.4475 | 0.9277 |
| Water Saturation Sw | Baseline Mean | Depth Holdout Test | 1116 | -1.0128 | 0.2028 | 0.1502 | 21.8156 | 0.0000 |
| Water Saturation Sw | Extra Trees | Depth Holdout Validation | 1116 | 0.9246 | 0.0544 | 0.0224 | 3.3267 | 0.9674 |
| Water Saturation Sw | XGBoost | Depth Holdout Validation | 1116 | 0.9203 | 0.0560 | 0.0259 | 4.0395 | 0.9620 |
| Water Saturation Sw | Ridge Regression | Depth Holdout Validation | 1116 | 0.9192 | 0.0563 | 0.0267 | 4.9117 | 0.9606 |
| Water Saturation Sw | Random Forest | Depth Holdout Validation | 1116 | 0.9119 | 0.0588 | 0.0276 | 4.2729 | 0.9584 |
| Water Saturation Sw | SVR RBF | Depth Holdout Validation | 1116 | 0.8472 | 0.0775 | 0.0350 | 5.7702 | 0.9352 |
| Water Saturation Sw | ANN Deep MLP | Depth Holdout Validation | 1116 | 0.8380 | 0.0798 | 0.0499 | 8.7066 | 0.9351 |
| Water Saturation Sw | Baseline Mean | Depth Holdout Validation | 1116 | -0.6494 | 0.2545 | 0.2172 | 39.7592 |  |

## 10. Five-Fold Depth-Ordered Cross-Validation Summary

| Target | Model | CV_R2_mean | CV_R2_std | CV_RMSE_mean | CV_RMSE_std | CV_MAE_mean | CV_MAPE_percent_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Permeability k | SVR RBF | 0.5996 | 0.2504 | 65.9816 | 64.5968 | 31.7591 | 169.7631 |
| Permeability k | Extra Trees | 0.5372 | 0.4518 | 66.9277 | 69.2950 | 35.3216 | 35.5557 |
| Permeability k | Ridge Regression | 0.5044 | 0.6339 | 78.3184 | 106.2871 | 30.2413 | 15.0146 |
| Permeability k | XGBoost | 0.3913 | 0.5147 | 74.5826 | 61.7424 | 39.4452 | 35.5766 |
| Permeability k | Random Forest | 0.1007 | 1.1273 | 82.6982 | 73.6344 | 43.5581 | 43.5369 |
| Permeability k | Baseline Mean | -7.2343 | 13.0229 | 182.0667 | 55.9339 | 160.5820 | 903.9289 |
| Permeability k | ANN Deep MLP | -1215.8757 | 2334.6652 | 4097.1602 | 7226.2163 | 565.7748 | 3130.1685 |
| Porosity | Extra Trees | 0.9674 | 0.0182 | 0.0034 | 0.0004 | 0.0019 | 1.0404 |
| Porosity | XGBoost | 0.9538 | 0.0280 | 0.0041 | 0.0007 | 0.0027 | 1.4155 |
| Porosity | Random Forest | 0.9414 | 0.0427 | 0.0044 | 0.0008 | 0.0024 | 1.3140 |
| Porosity | Ridge Regression | 0.7962 | 0.3070 | 0.0070 | 0.0040 | 0.0050 | 2.4038 |
| Porosity | SVR RBF | 0.0443 | 0.3956 | 0.0193 | 0.0041 | 0.0146 | 7.6507 |
| Porosity | Baseline Mean | -1.5624 | 1.1607 | 0.0331 | 0.0129 | 0.0266 | 14.7296 |
| Porosity | ANN Deep MLP | -12.3893 | 15.3327 | 0.0607 | 0.0285 | 0.0431 | 20.6559 |
| Water Saturation Sw | Ridge Regression | 0.7015 | 0.2848 | 0.0586 | 0.0410 | 0.0345 | 7.2032 |
| Water Saturation Sw | XGBoost | 0.5333 | 0.5231 | 0.0619 | 0.0466 | 0.0399 | 7.1290 |
| Water Saturation Sw | Random Forest | 0.4905 | 0.6228 | 0.0655 | 0.0517 | 0.0416 | 6.8254 |
| Water Saturation Sw | Extra Trees | 0.4787 | 0.6163 | 0.0712 | 0.0556 | 0.0467 | 10.0170 |
| Water Saturation Sw | SVR RBF | -0.1784 | 0.9336 | 0.0936 | 0.0545 | 0.0649 | 14.5908 |
| Water Saturation Sw | ANN Deep MLP | -0.3748 | 1.5454 | 0.0896 | 0.0403 | 0.0526 | 14.3496 |
| Water Saturation Sw | Baseline Mean | -2.0216 | 2.4166 | 0.1865 | 0.1198 | 0.1623 | 30.7489 |

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
- `all_models_predicted_vs_actual/`: 21 individual predicted-versus-actual graphs, one for every target-model combination.
- `09_porosity_permeability_sw_crossplot.png`: reservoir-quality crossplot of porosity versus permeability colored by Sw.
- `10_neutron_density_porosity_crossplot.png`: NPHI-RHOB crossplot colored by porosity.
- `11_best_model_feature_importance_panel.png`: most influential input features for the best models.

## 12. Output Files

- `ml_outputs/model_scores.csv`: validation and test metrics for every model-target combination.
- `ml_outputs/cv_fold_scores.csv`: fold-by-fold depth-ordered cross-validation metrics.
- `ml_outputs/cv_summary.csv`: mean and standard deviation CV metrics.
- `ml_outputs/test_predictions.csv`: actual and predicted test rows for every model and target.
- `ml_outputs/preprocessed_modeling_dataset.csv`: cleaned and engineered feature table with targets.
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
