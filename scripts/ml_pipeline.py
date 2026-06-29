#!/usr/bin/env python3
"""Train ML models for petrophysical property prediction.

Inputs are the final calculated LAS workbook. Targets are porosity,
permeability, and water saturation. Outputs include validation tables,
predictions, figures, saved best models, and AI.md.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parents[1] / ".matplotlib-cache").resolve()),
)

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.exceptions import ConvergenceWarning

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
    XGBOOST_ERROR = ""
except Exception as exc:  # pragma: no cover - only used when local runtime lacks xgboost
    XGBRegressor = None
    XGBOOST_AVAILABLE = False
    XGBOOST_ERROR = str(exc)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKBOOK = (
    PROJECT_ROOT
    / "outputs/final_renamed_porosity_20260629/KAILA3_final_porosity_perm_corrected.xlsx"
)
SHEET_NAME = "Calculated Data"
OUTPUT_DIR = PROJECT_ROOT / "ml_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_PATH = PROJECT_ROOT / "AI.md"

TARGETS = ["Porosity", "Permeability k", "Water Saturation Sw"]
BASE_FEATURES = [
    "DEPT",
    "GR",
    "CALI",
    "SP",
    "ILD",
    "SFLU",
    "MSFL",
    "DT",
    "RHOB",
    "DRHO",
    "PEF",
    "NPHI",
]
RANDOM_STATE = 42
APPLY_INPUT_NOISE_FILTER = True
NOISE_CONTAMINATION = 0.025
PREPROCESSING_AUDIT: dict[str, int | float | str] = {}


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip input-feature outliers using training-set IQR limits."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, x, y=None):
        arr = np.asarray(x, dtype=float)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, x):
        arr = np.asarray(x, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return value.lower()


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, FIGURE_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def flag_target_physical_ranges(df: pd.DataFrame) -> dict[str, int]:
    return {
        "porosity_gt_0_35": int((df["Porosity"] > 0.35).sum()),
        "porosity_lt_0": int((df["Porosity"] < 0).sum()),
        "permeability_gt_800": int((df["Permeability k"] > 800).sum()),
        "permeability_lte_0": int((df["Permeability k"] <= 0).sum()),
        "water_saturation_gt_1": int((df["Water Saturation Sw"] > 1.0).sum()),
        "water_saturation_lt_0": int((df["Water Saturation Sw"] < 0).sum()),
    }


def remove_input_noise_rows(df: pd.DataFrame) -> pd.DataFrame:
    if not APPLY_INPUT_NOISE_FILTER:
        PREPROCESSING_AUDIT["isolation_forest_rows_removed"] = 0
        return df

    feature_matrix = df[BASE_FEATURES].replace([np.inf, -np.inf], np.nan)
    feature_matrix = feature_matrix.fillna(feature_matrix.mean(numeric_only=True))
    detector = IsolationForest(
        contamination=NOISE_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=200,
    )
    labels = detector.fit_predict(feature_matrix)
    keep_mask = labels == 1
    PREPROCESSING_AUDIT["isolation_forest_contamination"] = NOISE_CONTAMINATION
    PREPROCESSING_AUDIT["isolation_forest_rows_removed"] = int((~keep_mask).sum())
    return df.loc[keep_mask].reset_index(drop=True)


def load_data() -> pd.DataFrame:
    df = pd.read_excel(SOURCE_WORKBOOK, sheet_name=SHEET_NAME)
    PREPROCESSING_AUDIT["raw_rows"] = int(len(df))
    missing = [col for col in BASE_FEATURES + TARGETS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    keep = BASE_FEATURES + TARGETS
    df = df[keep].copy()
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    PREPROCESSING_AUDIT["initial_missing_values"] = int(df.isna().sum().sum())
    PREPROCESSING_AUDIT["duplicate_full_rows_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    before_target_drop = len(df)
    df = df.dropna(subset=TARGETS + ["DEPT"]).sort_values("DEPT").reset_index(drop=True)
    PREPROCESSING_AUDIT["rows_removed_missing_depth_or_target"] = int(before_target_drop - len(df))
    PREPROCESSING_AUDIT.update(flag_target_physical_ranges(df))
    df = remove_input_noise_rows(df).sort_values("DEPT").reset_index(drop=True)
    PREPROCESSING_AUDIT["final_rows_after_previous_thesis_preprocessing"] = int(len(df))
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = df[BASE_FEATURES].copy()

    for col in ["ILD", "SFLU", "MSFL"]:
        positive = x[col].where(x[col] > 0)
        x[f"log10_{col}"] = np.log10(positive)

    eps = 1e-6
    sflu_den = x["SFLU"].abs().where(x["SFLU"].abs() > eps)
    msfl_den = x["MSFL"].abs().where(x["MSFL"].abs() > eps)
    x["ILD_to_SFLU"] = (x["ILD"] / sflu_den).clip(-1e4, 1e4)
    x["MSFL_to_SFLU"] = (x["MSFL"] / sflu_den).clip(-1e4, 1e4)
    x["ILD_to_MSFL"] = (x["ILD"] / msfl_den).clip(-1e4, 1e4)
    x["NPHI_RHOB_product"] = x["NPHI"] * x["RHOB"]
    x["NPHI_minus_DRHO"] = x["NPHI"] - x["DRHO"]
    x["GR_DT_product"] = x["GR"] * x["DT"]

    depth_min = x["DEPT"].min()
    depth_range = max(x["DEPT"].max() - depth_min, eps)
    x["DEPT_normalized"] = (x["DEPT"] - depth_min) / depth_range

    rolling_cols = ["GR", "ILD", "RHOB", "NPHI", "DT"]
    for col in rolling_cols:
        x[f"{col}_roll5_mean"] = x[col].rolling(window=5, center=True, min_periods=1).mean()
        x[f"{col}_roll15_mean"] = x[col].rolling(window=15, center=True, min_periods=1).mean()
        x[f"{col}_roll15_std"] = (
            x[col].rolling(window=15, center=True, min_periods=2).std().fillna(0)
        )

    x = x.replace([np.inf, -np.inf], np.nan)
    return x, x.columns.tolist()


def split_by_depth(df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return {
        "train": np.arange(0, train_end),
        "validation": np.arange(train_end, val_end),
        "test": np.arange(val_end, n),
        "train_validation": np.arange(0, val_end),
    }


def select_features(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    min_abs_corr: float = 0.20,
    high_feature_corr: float = 0.85,
    min_features: int = 8,
) -> tuple[list[str], pd.DataFrame]:
    x_numeric = x_train.copy().replace([np.inf, -np.inf], np.nan)
    x_numeric = x_numeric.fillna(x_numeric.mean(numeric_only=True))
    y = np.asarray(y_train, dtype=float)

    corr_rows = []
    for col in x_numeric.columns:
        values = x_numeric[col].to_numpy(dtype=float)
        if np.std(values) == 0 or np.std(y) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(values, y)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        corr_rows.append({"Feature": col, "TargetCorrelation": corr, "AbsCorrelation": abs(corr)})

    corr_df = pd.DataFrame(corr_rows).sort_values("AbsCorrelation", ascending=False)
    selected = corr_df[corr_df["AbsCorrelation"] >= min_abs_corr]["Feature"].tolist()
    if len(selected) < min_features:
        selected = corr_df.head(min(min_features, len(corr_df)))["Feature"].tolist()

    abs_corr_lookup = corr_df.set_index("Feature")["AbsCorrelation"].to_dict()
    feature_corr = x_numeric[selected].corr().abs()
    final_features: list[str] = []
    for feature in selected:
        redundant = False
        for kept in final_features:
            if feature_corr.loc[feature, kept] >= high_feature_corr:
                if abs_corr_lookup[feature] <= abs_corr_lookup[kept]:
                    redundant = True
                    break
        if not redundant:
            final_features.append(feature)

    if len(final_features) < min_features:
        for feature in selected:
            if feature not in final_features:
                final_features.append(feature)
            if len(final_features) >= min_features:
                break

    corr_df["Selected"] = corr_df["Feature"].isin(final_features)
    return final_features, corr_df


def build_models() -> dict[str, object]:
    preprocess_steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("iqr_clipper", IQRClipper(factor=1.5)),
        ("scaler", MinMaxScaler()),
    ]
    models: dict[str, object] = {
        "Baseline Mean": Pipeline(preprocess_steps + [("model", DummyRegressor())]),
        "Linear Regression": Pipeline(
            preprocess_steps + [("model", LinearRegression())]
        ),
        "Ridge Regression": Pipeline(
            preprocess_steps + [("model", Ridge(alpha=1.0))]
        ),
        "Random Forest": Pipeline(
            preprocess_steps
            + [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=16,
                        min_samples_leaf=2,
                        max_samples=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "Extra Trees": Pipeline(
            preprocess_steps
            + [
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=120,
                        max_depth=16,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "SVR RBF": Pipeline(
            preprocess_steps + [("model", SVR(C=20.0, gamma="scale", epsilon=0.01))]
        ),
        "ANN Deep MLP": Pipeline(
            preprocess_steps
            + [
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        max_iter=400,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }

    if XGBOOST_AVAILABLE and XGBRegressor is not None:
        models["XGBoost"] = Pipeline(
            preprocess_steps
            + [
                (
                    "model",
                    XGBRegressor(
                        n_estimators=180,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=4,
                    ),
                ),
            ]
        )

    return models


def transform_target(target: str, y: np.ndarray) -> np.ndarray:
    if target == "Permeability k":
        return np.log1p(np.clip(y, 0, None))
    return y


def inverse_target(target: str, y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.nan_to_num(np.asarray(y_pred, dtype=float), nan=0.0, posinf=20.0, neginf=-20.0)
    if target == "Permeability k":
        y_pred = np.clip(y_pred, -20, 20)
        y_pred = np.expm1(y_pred)
        return np.clip(y_pred, 0, None)
    if target in {"Porosity", "Water Saturation Sw"}:
        return np.clip(y_pred, 0, 1)
    return y_pred


def metrics(target: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = inverse_target(target, np.asarray(y_pred, dtype=float))
    if not np.all(np.isfinite(y_pred)):
        replacement = float(np.nanmedian(y_true))
        y_pred = np.nan_to_num(y_pred, nan=replacement, posinf=replacement, neginf=replacement)
    err = y_true - y_pred
    nonzero = np.maximum(np.abs(y_true), 1e-9)
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "MAPE_percent": float(np.mean(np.abs(err) / nonzero) * 100),
        "ExplainedVariance": float(explained_variance_score(y_true, y_pred)),
        "Pearson_r": corr,
    }


def fit_predict(model: object, x_train: pd.DataFrame, y_train: np.ndarray, x_eval: pd.DataFrame) -> np.ndarray:
    fitted = clone(model)
    fitted.fit(x_train, y_train)
    return fitted.predict(x_eval)


def run_depth_cv(
    model: object,
    model_name: str,
    target: str,
    x: pd.DataFrame,
    y_raw: np.ndarray,
    train_val_index: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    splitter = TimeSeriesSplit(n_splits=5)
    x_tv = x.iloc[train_val_index]
    y_tv_raw = y_raw[train_val_index]
    y_tv_model = transform_target(target, y_tv_raw)

    for fold, (tr, va) in enumerate(splitter.split(x_tv), start=1):
        fold_features, _ = select_features(x_tv.iloc[tr], y_tv_raw[tr])
        pred = fit_predict(
            model,
            x_tv.iloc[tr][fold_features],
            y_tv_model[tr],
            x_tv.iloc[va][fold_features],
        )
        row = metrics(target, y_tv_raw[va], pred)
        row.update(
            {
                "Target": target,
                "Model": model_name,
                "Fold": fold,
                "ValidationType": "Depth TimeSeriesSplit",
                "TrainRows": int(len(tr)),
                "ValidationRows": int(len(va)),
            }
        )
        rows.append(row)
    return rows


def plot_predictions(target: str, model_name: str, prediction_df: pd.DataFrame) -> None:
    target_slug = slugify(target)
    model_slug = slugify(model_name)
    subset = prediction_df[
        (prediction_df["Target"] == target) & (prediction_df["Model"] == model_name)
    ].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(subset["Actual"], subset["Predicted"], s=13, alpha=0.65)
    mn = min(subset["Actual"].min(), subset["Predicted"].min())
    mx = max(subset["Actual"].max(), subset["Predicted"].max())
    ax.plot([mn, mx], [mn, mx], color="black", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{target} - {model_name} test predictions")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"predicted_vs_actual_{target_slug}_{model_slug}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.scatter(subset["DEPT"], subset["Residual"], s=12, alpha=0.65)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Residual")
    ax.set_title(f"{target} - residuals by depth")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"residuals_by_depth_{target_slug}_{model_slug}.png")
    plt.close(fig)


def save_feature_importance(
    target: str,
    model_name: str,
    fitted_model: object,
    feature_names: list[str],
) -> None:
    if not hasattr(fitted_model, "named_steps"):
        return
    estimator = fitted_model.named_steps.get("model")
    if estimator is None:
        return
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        importance = pd.DataFrame({"Feature": feature_names, "Importance": values})
        value_col = "Importance"
        plot_title = f"Top feature importance - {target} - {model_name}"
        x_label = "Importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float).ravel()
        importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": coefficients,
                "Importance": np.abs(coefficients),
            }
        )
        value_col = "Importance"
        plot_title = f"Top standardized coefficient magnitude - {target} - {model_name}"
        x_label = "Absolute standardized coefficient"
    else:
        return

    importance = importance.sort_values("Importance", ascending=False).reset_index(drop=True)
    target_slug = slugify(target)
    model_slug = slugify(model_name)
    importance.to_csv(
        OUTPUT_DIR / f"feature_importance_{target_slug}_{model_slug}.csv",
        index=False,
    )

    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.barh(top["Feature"], top[value_col])
    ax.set_xlabel(x_label)
    ax.set_title(plot_title)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"feature_importance_{target_slug}_{model_slug}.png")
    plt.close(fig)


def format_metric(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows available._"
    out = df[columns].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: format_metric(v))
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in out.to_numpy()]
    return "\n".join([header, sep] + rows)


def previous_thesis_preprocessing_comparison() -> pd.DataFrame:
    rows = [
        {
            "Previous thesis preprocessing step": "Data import and descriptive statistics",
            "Our implementation": "Imported final Excel sheet, numeric conversion, dataset_summary.csv",
            "Status": "Done",
        },
        {
            "Previous thesis preprocessing step": "Missing-value check and mean imputation",
            "Our implementation": "Missing values counted in preprocessing_audit.csv; mean imputation used inside all model pipelines",
            "Status": "Done",
        },
        {
            "Previous thesis preprocessing step": "Input-feature outlier treatment using IQR",
            "Our implementation": "IQRClipper applied inside each model pipeline using training data only",
            "Status": "Done",
        },
        {
            "Previous thesis preprocessing step": "Target invalid-value removal",
            "Our implementation": "Target physical-range issues are flagged in preprocessing_audit.csv but targets are not altered to avoid artificial score inflation",
            "Status": "Flagged, not manipulated",
        },
        {
            "Previous thesis preprocessing step": "Duplicate full-row removal",
            "Our implementation": "Full duplicate rows removed before modeling",
            "Status": "Done",
        },
        {
            "Previous thesis preprocessing step": "Noisy-data detection using IsolationForest",
            "Our implementation": "IsolationForest applied to input features only; 2.5% anomalous input rows removed",
            "Status": "Done",
        },
        {
            "Previous thesis preprocessing step": "Binning/noise impact reduction",
            "Our implementation": "IQR clipping plus Min-Max normalization reduces feature noise while preserving continuous log values for regression",
            "Status": "Adapted",
        },
        {
            "Previous thesis preprocessing step": "Min-Max normalization",
            "Our implementation": "MinMaxScaler applied inside every model pipeline",
            "Status": "Done",
        },
        {
            "Previous thesis preprocessing step": "Correlation heatmap feature selection |r| >= 0.20",
            "Our implementation": "Target-specific correlation feature selection applied inside train/validation folds",
            "Status": "Done",
        },
        {
            "Previous thesis preprocessing step": "Remove/check highly correlated features |r| >= 0.85",
            "Our implementation": "Redundant high-correlation features reduced during feature selection",
            "Status": "Done",
        },
        {
            "Previous thesis preprocessing step": "70/15/15 train-validation-test split",
            "Our implementation": "70/15/15 split retained, sorted by depth for stricter unseen-depth testing",
            "Status": "Done, stricter",
        },
        {
            "Previous thesis preprocessing step": "Linear Regression, SVR, RF, XGBoost, ANN",
            "Our implementation": "Linear Regression, SVR, Random Forest, XGBoost, ANN plus Ridge, Extra Trees, Baseline",
            "Status": "Done and expanded",
        },
    ]
    return pd.DataFrame(rows)


def write_report(
    df: pd.DataFrame,
    x: pd.DataFrame,
    scores: pd.DataFrame,
    cv_summary: pd.DataFrame,
    best_rows: pd.DataFrame,
    preprocessing_audit: pd.DataFrame,
    previous_comparison: pd.DataFrame,
    unavailable: list[str],
) -> None:
    source_rel = SOURCE_WORKBOOK.relative_to(PROJECT_ROOT)
    depth_min = df["DEPT"].min()
    depth_max = df["DEPT"].max()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    dataset_summary = pd.DataFrame(
        {
            "Column": BASE_FEATURES + TARGETS,
            "Missing": df[BASE_FEATURES + TARGETS].isna().sum().reindex(BASE_FEATURES + TARGETS).values,
            "Mean": df[BASE_FEATURES + TARGETS].mean().reindex(BASE_FEATURES + TARGETS).values,
            "Std": df[BASE_FEATURES + TARGETS].std().reindex(BASE_FEATURES + TARGETS).values,
            "Min": df[BASE_FEATURES + TARGETS].min().reindex(BASE_FEATURES + TARGETS).values,
            "Max": df[BASE_FEATURES + TARGETS].max().reindex(BASE_FEATURES + TARGETS).values,
        }
    )

    holdout_table = scores[scores["ValidationType"].isin(["Depth Holdout Validation", "Depth Holdout Test"])]
    holdout_table = holdout_table[
        ["Target", "Model", "ValidationType", "Rows", "R2", "RMSE", "MAE", "MAPE_percent", "Pearson_r"]
    ].sort_values(["Target", "ValidationType", "R2"], ascending=[True, True, False])

    best_table = best_rows[
        [
            "Target",
            "BestModel",
            "Test_R2",
            "Test_RMSE",
            "Test_MAE",
            "Test_MAPE_percent",
            "Test_Pearson_r",
            "SelectedFeatureCount",
        ]
    ].copy()

    cv_table = cv_summary[
        [
            "Target",
            "Model",
            "CV_R2_mean",
            "CV_R2_std",
            "CV_RMSE_mean",
            "CV_RMSE_std",
            "CV_MAE_mean",
            "CV_MAPE_percent_mean",
        ]
    ].sort_values(["Target", "CV_R2_mean"], ascending=[True, False])

    dataset_table = dataset_summary.copy()
    dataset_table[["Mean", "Std", "Min", "Max"]] = dataset_table[
        ["Mean", "Std", "Min", "Max"]
    ].round(6)
    audit_table = preprocessing_audit.copy()

    notes = ""
    if unavailable:
        notes = "\n\nUnavailable requested models:\n" + "\n".join(
            f"- {item}" for item in unavailable
        )

    text = f"""# AI/ML Modeling Report for Petrophysical Properties

Updated: {now}

## 1. Objective

Build supervised machine learning models to predict three reservoir properties for each depth row:

- Porosity
- Permeability k
- Water Saturation Sw

The work uses the final corrected workbook produced from the LAS conversion and formula/core-analysis workflow:

`{source_rel}`

## 2. Dataset Used

- Sheet: `{SHEET_NAME}`
- Total usable rows after numeric cleaning: {len(df):,}
- Depth interval: {depth_min:.1f} to {depth_max:.1f}
- Input features before engineering: {len(BASE_FEATURES)}
- Input features after engineering: {x.shape[1]}
- Targets: {", ".join(TARGETS)}

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

{markdown_table(audit_table, ["Step", "Value"])}

Previous-thesis preprocessing comparison:

{markdown_table(previous_comparison, ["Previous thesis preprocessing step", "Our implementation", "Status"])}

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
- XGBoost{notes}

## 7. Dataset Statistical Summary

{markdown_table(dataset_table, ["Column", "Missing", "Mean", "Std", "Min", "Max"])}

## 8. Best Test Model by Target

{markdown_table(best_table, ["Target", "BestModel", "Test_R2", "Test_RMSE", "Test_MAE", "Test_MAPE_percent", "Test_Pearson_r", "SelectedFeatureCount"])}

## 9. Depth Holdout Validation and Test Scores

{markdown_table(holdout_table, ["Target", "Model", "ValidationType", "Rows", "R2", "RMSE", "MAE", "MAPE_percent", "Pearson_r"])}

## 10. Five-Fold Depth-Ordered Cross-Validation Summary

{markdown_table(cv_table, ["Target", "Model", "CV_R2_mean", "CV_R2_std", "CV_RMSE_mean", "CV_RMSE_std", "CV_MAE_mean", "CV_MAPE_percent_mean"])}

## 11. Thesis Graphs Generated

All thesis-ready graphs were saved in `ml_outputs/thesis_graphs/`. A combined PDF graph pack is also available at `ml_outputs/thesis_graphs/thesis_graph_pack.pdf`.

Generated figures:

- `13_boxplot_outliers_before_treatment.png`: boxplot for input-feature outliers before IQR treatment.
- `14_boxplot_outliers_after_treatment.png`: boxplot for input-feature outliers after IQR treatment.
- `15_histogram_duplicate_data.png`: histogram/bar chart showing duplicate data by column and full-row duplicate count.
- `16_scattered_plot_noisy_data_all_inputs.png`: scattered plot for IsolationForest noisy-data detection across all input logs.
- `17_scattered_plot_noisy_data_key_logs.png`: scattered plot for noisy-data detection on key well-log curves.
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
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_data()
    x, feature_names = engineer_features(df)
    y_all = df[TARGETS].copy()
    splits = split_by_depth(df)
    models = build_models()

    imputed = SimpleImputer(strategy="mean").fit_transform(x)
    preprocessed = pd.DataFrame(imputed, columns=feature_names)
    preprocessed = pd.concat([df[["DEPT"]].reset_index(drop=True), preprocessed, y_all], axis=1)
    preprocessed.to_csv(OUTPUT_DIR / "preprocessed_modeling_dataset.csv", index=False)

    dataset_summary = df[BASE_FEATURES + TARGETS].describe().T.reset_index().rename(columns={"index": "Column"})
    dataset_summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    preprocessing_audit = pd.DataFrame(
        [{"Step": key, "Value": value} for key, value in PREPROCESSING_AUDIT.items()]
    )
    preprocessing_audit.to_csv(OUTPUT_DIR / "preprocessing_audit.csv", index=False)
    previous_comparison = previous_thesis_preprocessing_comparison()
    previous_comparison.to_csv(
        OUTPUT_DIR / "previous_thesis_preprocessing_comparison.csv",
        index=False,
    )

    score_rows: list[dict[str, float | int | str]] = []
    cv_rows: list[dict[str, float | int | str]] = []
    pred_rows: list[dict[str, float | str]] = []
    feature_selection_rows: list[pd.DataFrame] = []
    fitted_models: dict[tuple[str, str], tuple[object, list[str]]] = {}
    unavailable: list[str] = []
    if not XGBOOST_AVAILABLE:
        unavailable.append(f"XGBoost could not be loaded: {XGBOOST_ERROR}")

    for target in TARGETS:
        print(f"Training target: {target}", flush=True)
        y_raw = y_all[target].to_numpy(dtype=float)
        y_model = transform_target(target, y_raw)

        x_train = x.iloc[splits["train"]]
        y_train = y_model[splits["train"]]
        y_train_raw = y_raw[splits["train"]]
        x_val = x.iloc[splits["validation"]]
        y_val_raw = y_raw[splits["validation"]]
        x_train_val = x.iloc[splits["train_validation"]]
        y_train_val = y_model[splits["train_validation"]]
        y_train_val_raw = y_raw[splits["train_validation"]]
        x_test = x.iloc[splits["test"]]
        y_test_raw = y_raw[splits["test"]]

        validation_features, validation_corr = select_features(x_train, y_train_raw)
        validation_corr["Target"] = target
        validation_corr["SelectionStage"] = "validation_train_only"
        feature_selection_rows.append(validation_corr)
        test_features, test_corr = select_features(x_train_val, y_train_val_raw)
        test_corr["Target"] = target
        test_corr["SelectionStage"] = "test_train_validation"
        feature_selection_rows.append(test_corr)

        for model_name, model in models.items():
            print(f"  Model: {model_name}", flush=True)
            validation_model = clone(model)
            validation_model.fit(x_train[validation_features], y_train)
            val_pred = validation_model.predict(x_val[validation_features])
            val_metrics = metrics(target, y_val_raw, val_pred)
            val_metrics.update(
                {
                    "Target": target,
                    "Model": model_name,
                    "ValidationType": "Depth Holdout Validation",
                    "Rows": int(len(x_val)),
                }
            )
            score_rows.append(val_metrics)

            final_model = clone(model)
            final_model.fit(x_train_val[test_features], y_train_val)
            test_pred_model_space = final_model.predict(x_test[test_features])
            test_pred = inverse_target(target, test_pred_model_space)
            test_metrics = metrics(target, y_test_raw, test_pred_model_space)
            test_metrics.update(
                {
                    "Target": target,
                    "Model": model_name,
                    "ValidationType": "Depth Holdout Test",
                    "Rows": int(len(x_test)),
                }
            )
            score_rows.append(test_metrics)
            fitted_models[(target, model_name)] = (final_model, test_features)

            for depth, actual, predicted in zip(df["DEPT"].iloc[splits["test"]], y_test_raw, test_pred):
                pred_rows.append(
                    {
                        "Target": target,
                        "Model": model_name,
                        "DEPT": float(depth),
                        "Actual": float(actual),
                        "Predicted": float(predicted),
                        "Residual": float(actual - predicted),
                    }
                )

            cv_rows.extend(run_depth_cv(model, model_name, target, x, y_raw, splits["train_validation"]))

    scores = pd.DataFrame(score_rows)
    cv_scores = pd.DataFrame(cv_rows)
    predictions = pd.DataFrame(pred_rows)
    feature_selection = pd.concat(feature_selection_rows, ignore_index=True)

    scores.to_csv(OUTPUT_DIR / "model_scores.csv", index=False)
    cv_scores.to_csv(OUTPUT_DIR / "cv_fold_scores.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    feature_selection.to_csv(OUTPUT_DIR / "feature_selection_summary.csv", index=False)

    cv_summary = (
        cv_scores.groupby(["Target", "Model"], as_index=False)
        .agg(
            CV_R2_mean=("R2", "mean"),
            CV_R2_std=("R2", "std"),
            CV_RMSE_mean=("RMSE", "mean"),
            CV_RMSE_std=("RMSE", "std"),
            CV_MAE_mean=("MAE", "mean"),
            CV_MAE_std=("MAE", "std"),
            CV_MAPE_percent_mean=("MAPE_percent", "mean"),
            CV_MAPE_percent_std=("MAPE_percent", "std"),
            CV_Pearson_r_mean=("Pearson_r", "mean"),
            CV_Pearson_r_std=("Pearson_r", "std"),
        )
        .sort_values(["Target", "CV_R2_mean"], ascending=[True, False])
    )
    cv_summary.to_csv(OUTPUT_DIR / "cv_summary.csv", index=False)

    best_rows = []
    for target in TARGETS:
        target_scores = scores[
            (scores["Target"] == target)
            & (scores["ValidationType"] == "Depth Holdout Test")
        ].copy()
        best = target_scores.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0]
        best_model_name = str(best["Model"])
        best_model, best_features = fitted_models[(target, best_model_name)]
        model_path = MODEL_DIR / f"best_{slugify(target)}_{slugify(best_model_name)}.joblib"
        joblib.dump(best_model, model_path)
        save_feature_importance(target, best_model_name, best_model, best_features)
        plot_predictions(target, best_model_name, predictions)
        best_rows.append(
            {
                "Target": target,
                "BestModel": best_model_name,
                "Test_R2": float(best["R2"]),
                "Test_RMSE": float(best["RMSE"]),
                "Test_MAE": float(best["MAE"]),
                "Test_MAPE_percent": float(best["MAPE_percent"]),
                "Test_Pearson_r": float(best["Pearson_r"]),
                "SelectedFeatureCount": int(len(best_features)),
                "SavedModel": str(model_path.relative_to(PROJECT_ROOT)),
            }
        )

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(OUTPUT_DIR / "best_models_summary.csv", index=False)
    (OUTPUT_DIR / "best_models_summary.json").write_text(
        json.dumps(best_rows, indent=2), encoding="utf-8"
    )

    with pd.ExcelWriter(OUTPUT_DIR / "ml_results_workbook.xlsx", engine="openpyxl") as writer:
        best_df.to_excel(writer, sheet_name="Best Models", index=False)
        scores.to_excel(writer, sheet_name="Holdout Scores", index=False)
        cv_summary.to_excel(writer, sheet_name="CV Summary", index=False)
        cv_scores.to_excel(writer, sheet_name="CV Fold Scores", index=False)
        dataset_summary.to_excel(writer, sheet_name="Dataset Summary", index=False)
        preprocessing_audit.to_excel(writer, sheet_name="Preprocessing Audit", index=False)
        previous_comparison.to_excel(writer, sheet_name="Previous Thesis Check", index=False)
        feature_selection.to_excel(writer, sheet_name="Feature Selection", index=False)
        predictions.to_excel(writer, sheet_name="Test Predictions", index=False)

    write_report(
        df,
        x,
        scores,
        cv_summary,
        best_df,
        preprocessing_audit,
        previous_comparison,
        unavailable,
    )
    print(f"Completed ML pipeline. Report: {REPORT_PATH}")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
