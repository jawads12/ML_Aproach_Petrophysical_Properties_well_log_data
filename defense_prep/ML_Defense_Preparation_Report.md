# Machine Learning Defense Preparation Report

Prepared for thesis presentation and defense  
Project: ML approach for petrophysical properties from well-log data  
Date: 2026-07-14

## 1. Purpose Of This Report

This report is a defense preparation document. It explains how the research was conducted, how machine learning was used, why the selected models were chosen, how the results should be interpreted, and what types of viva questions may be asked by faculty members.

The focus is on the machine learning part of the thesis because that is where most defense questions are likely to come from.

## 2. Research Objective

The objective of the research was to predict key reservoir properties from well-log data using machine learning models. The target reservoir properties were:

- Porosity
- Permeability
- Water saturation

The input variables were well-log responses such as depth, gamma ray, caliper, spontaneous potential, resistivity logs, sonic log, bulk density, density correction, photoelectric factor, and neutron porosity.

The main research question was:

Can machine learning models predict petrophysical properties from well-log data with acceptable accuracy, and which model performs best for each target property?

## 3. Data Used In The Study

The study used the final corrected workbook produced from LAS conversion and petrophysical calculations:

`outputs/final_renamed_porosity_20260629/KAILA3_final_porosity_perm_corrected.xlsx`

The final modeling dataset contained measured and calculated well-log curves. The important input features included:

- DEPT: depth
- GR: gamma ray
- CALI: caliper
- SP: spontaneous potential
- ILD: deep induction resistivity
- SFLU: shallow focused resistivity
- MSFL: micro-spherically focused resistivity
- DT: sonic transit time
- RHOB: bulk density
- DRHO: density correction
- PEF: photoelectric factor
- NPHI: neutron porosity

The target columns were:

- Porosity
- Permeability k
- Water Saturation Sw

## 4. Why Machine Learning Was Used

Empirical petrophysical formulas are useful, but they depend on assumptions. For example, water saturation calculations may require assumed values of formation water resistivity, cementation exponent, saturation exponent, tortuosity factor, and clean-formation conditions. In real reservoirs, these parameters can vary with depth, lithology, clay content, pore structure, salinity, and fluid distribution.

Machine learning was used because it can learn relationships from multiple well logs at the same time. It does not need to explicitly assume every petrophysical constant. Instead, it can learn the combined response of logs such as GR, RHOB, NPHI, resistivity, DT, and PEF. This does not mean that machine learning directly knows parameters such as Rw, a, m, or n. It means that machine learning can indirectly capture their effects through measured log responses.

The purpose was not to prove that complex models are always better. The purpose was to compare multiple models and select the most reliable model for each reservoir property.

## 5. Research Workflow

The research workflow followed these steps:

1. LAS well-log data was converted into Excel format.
2. Petrophysical properties were calculated using the provided formulas.
3. Core data was used where available to replace calculated values.
4. Final target columns were prepared: porosity, permeability, and water saturation.
5. Data preprocessing was performed.
6. Feature engineering was applied.
7. Several machine learning models were trained.
8. Models were validated using depth-aware validation.
9. Final models were selected using depth-holdout test results.
10. Graphs and tables were generated for result interpretation.

## 6. Preprocessing Steps

Preprocessing was performed to improve data quality before model training. The previous-year thesis preprocessing approach was reviewed and adapted to this study.

The main preprocessing steps were:

- Numeric conversion of selected columns
- Missing-value checking
- Mean imputation for missing input values
- Duplicate-row checking and removal
- Input-feature outlier treatment using IQR clipping
- Noisy-row detection using IsolationForest
- Min-Max normalization
- Correlation-based feature selection
- High-correlation feature reduction

Target values were not manipulated to artificially improve scores. Physically suspicious target values were flagged, but the target values were not changed for score inflation.

## 7. Outlier And Noisy Data Treatment

Input-feature outliers were treated using IQR-based clipping. This means that extreme input-log values were limited based on the interquartile range. The treatment was applied to input variables, not to target variables.

Noisy input rows were detected using IsolationForest. This method identifies anomalous data points based on how isolated they are from the main data distribution.

The purpose of these steps was to reduce the effect of abnormal log readings, tool error, borehole effects, or noise before model training.

## 8. Feature Engineering

Feature engineering was used to help the models capture relationships between logs. The engineered features included:

- Log transforms of resistivity curves
- Ratios between deep and shallow resistivity logs
- Interaction terms such as NPHI multiplied by RHOB
- Normalized depth
- Rolling mean and rolling standard deviation features with depth windows

These features help the models capture petrophysical behavior that may not be visible from a single raw log alone.

## 9. Feature Selection

Feature selection was performed using correlation with the target property. Features with useful correlation were retained. Highly correlated input features were checked to reduce redundancy.

The purpose of feature selection was:

- To reduce unnecessary variables
- To lower model complexity
- To reduce overfitting
- To keep only meaningful inputs

Each target property used a slightly different selected feature set because porosity, permeability, and water saturation depend on different log responses.

## 10. Models Used

The following models were applied:

- Baseline Mean
- Linear Regression
- Ridge Regression
- Random Forest
- Extra Trees
- Support Vector Regression
- XGBoost
- ANN Deep MLP

The Baseline Mean model was used only as a reference. A useful model should perform better than the baseline.

## 11. Short Explanation Of Each Model

### Linear Regression

Linear Regression assumes a straight-line relationship between input logs and target properties. It is simple, interpretable, and useful as a baseline model.

### Ridge Regression

Ridge Regression is a regularized linear regression model. It adds a penalty to reduce overfitting and stabilize coefficients. It is useful when input features are correlated.

### Random Forest

Random Forest is an ensemble of decision trees. It can capture non-linear relationships and interactions between logs. It is usually robust for tabular geological data.

### Extra Trees

Extra Trees is similar to Random Forest but introduces more randomness when splitting trees. It can reduce variance and sometimes improve generalization.

### SVR

Support Vector Regression attempts to fit the data within an acceptable error margin. It can model non-linear relationships using kernels, but it is sensitive to scaling and parameter selection.

### XGBoost

XGBoost is a gradient boosting model. It builds trees sequentially, where each new tree corrects the errors of previous trees. It is powerful for structured tabular data.

### ANN Deep MLP

ANN Deep MLP is a feed-forward neural network. It can capture complex non-linear relationships between input logs and target properties. It is useful when the target behavior is difficult to represent using simple equations.

## 12. Validation Strategy

Two validation approaches were used:

- Depth holdout validation/test
- Five-fold depth-ordered cross-validation

These methods were chosen because well-log data is ordered by depth. Neighboring depth points are often similar. Random splitting can create data leakage because adjacent depth values may appear in both training and testing data.

## 13. Depth Holdout Test

In depth holdout testing, the data was sorted by depth and divided into:

- 70 percent training interval
- 15 percent validation interval
- 15 percent final test interval

The model was trained on earlier depth data and tested on a later unseen depth interval. This gives a more realistic estimate of model performance along the well profile.

The final best model selection was based on depth-holdout test performance.

## 14. Five-Fold Depth-Ordered Cross-Validation

Five-fold depth-ordered cross-validation was used to test model stability. The model was trained and validated across five sequential depth folds.

This method answers:

Does the model perform consistently across different depth intervals?

High average CV R2 means the model performs well on average. A small CV standard deviation means the model is stable. A large CV standard deviation means the model performance changes strongly between depth intervals.

## 15. Evaluation Metrics

The following metrics were used:

- R2
- RMSE
- MAE
- MAPE
- Pearson correlation

R2 indicates how much variation in the target is explained by the model. RMSE and MAE measure prediction error. MAPE shows percentage error. Pearson correlation shows the strength of linear association between actual and predicted values.

## 16. Final Best Test Results

| Target | Best Model | Test R2 | RMSE | MAE | Pearson r |
| --- | --- | ---: | ---: | ---: | ---: |
| Porosity | Random Forest | 0.9915 | 0.0016 | 0.0008 | 0.9961 |
| Permeability k | ANN Deep MLP | 0.7652 | 12.0077 | 6.0266 | 0.8905 |
| Water Saturation Sw | Random Forest | 0.8112 | 0.0596 | 0.0446 | 0.9113 |

## 17. Result Interpretation

Random Forest was the best final test model for porosity. This means that porosity was best predicted using a tree-based ensemble model that can capture interaction between logs such as RHOB, NPHI, DT, DRHO, and GR.

ANN Deep MLP was the best final test model for permeability. Permeability is usually more difficult to predict because it depends on pore connectivity, pore throat size, sorting, fractures, and flow pathways. These relationships can be non-linear, which explains why ANN performed best on the final unseen test interval.

Random Forest was the best final test model for water saturation. Water saturation depends strongly on resistivity logs, but it is also affected by porosity, lithology, shale content, and fluid distribution. Random Forest can capture these multiple interactions effectively.

## 18. Why CV And Test Results May Differ

Cross-validation checks average stability across multiple depth intervals. The final test checks performance on one independent unseen depth interval.

Therefore, it is normal if the best CV model and best test model are not always the same.

For example, Linear Regression and Ridge Regression performed strongly in permeability cross-validation, but ANN Deep MLP performed best on the final test interval. This suggests that the average permeability trend is partly linear, but the final test interval may contain additional non-linear behavior captured better by ANN.

The final model selection should be based on test performance because the test interval represents unseen data.

## 19. Which Models Are Best And Why

### Porosity

Best model: Random Forest

Reason:

- Highest depth-holdout test R2
- Very low RMSE and MAE
- Can capture interactions between density, neutron, sonic, and shale-related logs
- More flexible than linear regression
- More stable than ANN for this target

### Permeability

Best model: ANN Deep MLP

Reason:

- Highest depth-holdout test R2
- Permeability has more complex non-linear behavior
- Neural network can learn hidden relationships between porosity, resistivity, density, and depth features
- Final test performance was better than linear models

### Water Saturation

Best model: Random Forest

Reason:

- Highest depth-holdout test R2
- Strong Pearson correlation
- Can handle non-linear relation between resistivity logs and saturation
- Robust to feature interactions and noise

## 20. Main Defense Message

The main defense message should be:

This research did not assume one model would be best. Multiple linear and non-linear models were compared. The final selection was based on depth-holdout test performance. The result shows that Random Forest was best for porosity and water saturation, while ANN was best for permeability. This is technically reasonable because different reservoir properties have different physical controls and different levels of non-linearity.

## 21. Common Viva Questions And Answers

### Q1. Why did you use machine learning?

Machine learning was used because reservoir properties depend on multiple well-log responses at the same time. Empirical formulas often require fixed assumptions, but real formations are heterogeneous. Machine learning can learn relationships from the available logs and compare whether linear or non-linear models perform better.

### Q2. Why not just use empirical formulas?

Empirical formulas are useful, but they depend on assumptions such as constant Rw, cementation exponent, saturation exponent, matrix density, and clean formation conditions. In real wells these values may vary with depth and lithology. ML can indirectly capture such variations through multiple log responses.

### Q3. Does ML directly know Rw, a, m, and n?

No. ML does not directly know these parameters unless they are provided as input. However, the effects of formation and fluid variation may be indirectly reflected in logs such as resistivity, density, neutron porosity, gamma ray, and sonic response.

### Q4. Why did you use depth holdout instead of random split?

Well-log data is depth ordered. Neighboring depth points are similar. Random splitting can place similar points in both train and test sets, causing data leakage and overly optimistic results. Depth holdout is stricter because the model must predict an unseen depth interval.

### Q5. What is five-fold depth-ordered cross-validation?

It is a validation method where the data remains in depth order. The model trains on earlier depth sections and validates on later sections across five folds. It checks model stability across different depth intervals.

### Q6. What is the difference between validation and test?

Validation is used during model comparison and tuning. Test is used only at the end for final performance evaluation. The final test result is more important for final model selection.

### Q7. Why are CV best models and test best models sometimes different?

CV measures average performance across several depth intervals. The final test measures performance on one independent unseen depth interval. Geological properties can vary with depth, so different models may perform better in different intervals.

### Q8. Why did Random Forest perform best for porosity?

Porosity depends on interactions between logs such as RHOB, NPHI, DT, DRHO, and GR. Random Forest can capture these interactions without requiring a fixed equation. It also handles non-linear behavior and noisy data well.

### Q9. Why did ANN perform best for permeability?

Permeability is controlled by pore connectivity and flow pathways, not only pore volume. This makes it more non-linear than porosity. ANN can learn hidden non-linear patterns, which likely helped it perform best on the final test interval.

### Q10. Why did Random Forest perform best for water saturation?

Water saturation depends on resistivity logs, porosity, lithology, and fluid distribution. Random Forest can combine these features and model their non-linear interaction effectively.

### Q11. What does R2 mean?

R2 shows how much variation in the target variable is explained by the model. R2 close to 1 means strong prediction. R2 near 0 means weak prediction. Negative R2 means the model performs worse than predicting the mean.

### Q12. What does RMSE mean?

RMSE is the root mean squared error. It gives higher penalty to large errors. Lower RMSE means better prediction.

### Q13. What does MAE mean?

MAE is the mean absolute error. It shows the average absolute difference between actual and predicted values. Lower MAE is better.

### Q14. What does MAPE mean?

MAPE is the mean absolute percentage error. It expresses error as a percentage. Lower MAPE means better prediction.

### Q15. What does Pearson correlation mean?

Pearson correlation shows how strongly actual and predicted values move together. A value close to 1 means strong positive agreement.

### Q16. Why was preprocessing needed?

Preprocessing was needed because raw well-log data may contain missing values, duplicate values, outliers, noisy readings, and different feature scales. Without preprocessing, model accuracy and reliability can decrease.

### Q17. Why use IQR for outliers?

IQR is a statistical method that detects values far outside the central range of data. It is simple and effective for identifying extreme input-log values.

### Q18. Did you remove target outliers?

No. Target values were flagged but not manipulated to artificially improve results. The preprocessing focused on input features to keep the result academically defensible.

### Q19. Why use IsolationForest?

IsolationForest detects anomalous rows based on how isolated they are in feature space. It is useful for detecting noisy or abnormal log responses.

### Q20. Why use Min-Max scaling?

Min-Max scaling converts features to a common range. This prevents large-scale variables from dominating scale-sensitive models such as SVR and ANN.

### Q21. Why use feature selection?

Feature selection removes weak or redundant variables. It reduces model complexity, lowers overfitting risk, and keeps the most useful logs for each target.

### Q22. Why use log transform for permeability?

Permeability often has a wide numerical range and skewed distribution. Log transformation reduces the effect of very high values and helps models learn the relationship more effectively.

### Q23. Why use multiple models?

Different reservoir properties may have different relationships with logs. Some may be linear, while others may be non-linear. Comparing multiple models gives a more reliable final selection.

### Q24. If Linear Regression performs well, why is ML still justified?

Linear Regression is also a machine learning model. Its strong performance indicates that some target relationships are approximately linear after preprocessing and feature engineering. The study is still valid because it compares linear and non-linear models objectively.

### Q25. Why not choose the most complex model?

The best model should be selected based on validation and test performance, not complexity. A simpler model is preferred if it generalizes better.

### Q26. What is overfitting?

Overfitting happens when a model learns training data too closely and performs poorly on unseen data. Depth-holdout testing and cross-validation were used to check this risk.

### Q27. What is underfitting?

Underfitting happens when a model is too simple to capture the data pattern. It performs poorly on both training and validation data.

### Q28. Why did the Baseline Mean perform poorly?

The baseline only predicts the average target value. It does not learn relationships from well logs. Its poor performance confirms that the ML models learned meaningful patterns.

### Q29. What is the limitation of this study?

The targets are mostly derived from petrophysical formulas, with core replacement only where available. More measured core data would improve independent validation.

### Q30. What is the main conclusion?

Machine learning can predict reservoir properties from well logs with good accuracy. Random Forest was best for porosity and water saturation, while ANN Deep MLP was best for permeability.

## 22. Quick One-Line Answers For Defense

Why ML?

ML was used to learn relationships between multiple well logs and reservoir properties without relying only on fixed empirical assumptions.

Why depth split?

Depth split avoids leakage from neighboring depth points and gives a more realistic unseen-depth evaluation.

Why Random Forest?

It captures non-linear interaction between logs and is robust for tabular well-log data.

Why ANN for permeability?

Permeability is controlled by complex pore connectivity, so a neural network can capture non-linear behavior better.

Why not empirical formula only?

Empirical formulas require fixed assumptions, while ML can learn combined log response from the available data.

Why CV and test results differ?

CV measures average stability across folds, while test measures final performance on one unseen depth interval.

## 23. Suggested Presentation Flow

When presenting the ML part, use this order:

1. Explain why ML was needed.
2. Show input logs and target properties.
3. Explain preprocessing.
4. Explain models used.
5. Explain depth-holdout and cross-validation.
6. Show best model table.
7. Explain why each best model makes physical sense.
8. Discuss limitations.
9. End with future work using more core data.

## 24. Final Defense Summary

This study used machine learning to predict porosity, permeability, and water saturation from well-log data. The data was cleaned, normalized, and split by depth to avoid leakage. Multiple models were compared, including linear, tree-based, boosting, support vector, and neural network models. The final depth-holdout test showed that Random Forest was best for porosity and water saturation, while ANN Deep MLP was best for permeability. The results indicate that different reservoir properties require different model structures, and model selection should be based on validation and unseen test performance rather than model complexity alone.

## 25. Must-Memorize Numbers

The presenter should memorize the following final test results:

| Target | Best model | R2 | RMSE | MAE |
| --- | --- | ---: | ---: | ---: |
| Porosity | Random Forest | 0.9915 | 0.0016 | 0.0008 |
| Permeability k | ANN Deep MLP | 0.7652 | 12.0077 | 6.0266 |
| Water Saturation Sw | Random Forest | 0.8112 | 0.0596 | 0.0446 |

The presenter should also remember:

- Random Forest was best for porosity.
- ANN Deep MLP was best for permeability.
- Random Forest was best for water saturation.
- Depth-holdout test was used for final model selection.
- Five-fold depth-ordered cross-validation was used for stability checking.

## 26. Questions That Need Careful Answers

### Why use ML if empirical formulas already exist?

Do not say empirical formulas are wrong. Say empirical formulas are useful but depend on assumptions. ML was used as a data-driven comparison method that can learn from multiple logs simultaneously.

### Did you manipulate data to improve accuracy?

Answer clearly: no. Input features were cleaned using accepted preprocessing methods. Target values were not modified to artificially increase scores.

### Why is the test result different from cross-validation?

Cross-validation measures average stability across multiple depth folds. The final test measures generalization to one unseen depth interval. Because geology changes with depth, model ranking can differ.

### Is ANN always better?

No. ANN was best only for permeability. Random Forest was best for porosity and water saturation. The best model depends on the target property and its relationship with logs.

### Why not use only Random Forest?

Because the purpose was comparative model evaluation. Different algorithms were tested so the final selection is evidence-based.

## 27. One-Day Practice Plan

Use this order for final preparation:

1. Memorize the three best models and their R2 values.
2. Practice explaining why ML was used instead of only empirical formulas.
3. Practice explaining depth holdout and five-fold depth-ordered CV.
4. Practice explaining why Random Forest was best for porosity and water saturation.
5. Practice explaining why ANN was best for permeability.
6. Practice explaining why target values were not manipulated.
7. Review the graphs: model comparison, CV R2, predicted-vs-actual, residuals, boxplots, duplicate histogram, and noisy-data scatter plots.
8. Give a two-minute summary of the whole ML workflow without looking at notes.

## 28. Two-Minute Spoken Summary

This research used well-log data to predict porosity, permeability, and water saturation using machine learning. The input logs included depth, gamma ray, caliper, spontaneous potential, resistivity logs, sonic log, bulk density, density correction, photoelectric factor, and neutron porosity. The data was cleaned using missing-value treatment, duplicate checking, IQR outlier treatment, IsolationForest noisy-data detection, Min-Max scaling, and correlation-based feature selection.

Several models were compared, including Linear Regression, Ridge Regression, Random Forest, Extra Trees, SVR, XGBoost, and ANN Deep MLP. Since well-log data is depth ordered, depth-holdout testing and five-fold depth-ordered cross-validation were used instead of only random splitting. This reduced leakage from neighboring depth points.

The final depth-holdout test showed that Random Forest was best for porosity with R2 of 0.9915, ANN Deep MLP was best for permeability with R2 of 0.7652, and Random Forest was best for water saturation with R2 of 0.8112. The result shows that different reservoir properties require different model structures. Random Forest performed well for properties controlled by multiple log interactions, while ANN performed best for permeability because permeability is controlled by more complex pore connectivity and non-linear behavior.
