#!/usr/bin/env python3
"""Create thesis-ready graphs from the prepared ML outputs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parents[1] / ".matplotlib-cache").resolve()),
)

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKBOOK = (
    PROJECT_ROOT
    / "outputs/final_renamed_porosity_20260629/KAILA3_final_porosity_perm_corrected.xlsx"
)
SHEET_NAME = "Calculated Data"
DATA_PATH = PROJECT_ROOT / "ml_outputs/preprocessed_modeling_dataset.csv"
SCORES_PATH = PROJECT_ROOT / "ml_outputs/model_scores.csv"
CV_PATH = PROJECT_ROOT / "ml_outputs/cv_summary.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "ml_outputs/test_predictions.csv"
BEST_PATH = PROJECT_ROOT / "ml_outputs/best_models_summary.csv"
GRAPH_DIR = PROJECT_ROOT / "ml_outputs/thesis_graphs"
ALL_MODEL_DIR = GRAPH_DIR / "all_models_predicted_vs_actual"
PDF_PATH = GRAPH_DIR / "thesis_graph_pack.pdf"

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
LOG_INPUTS = ["GR", "ILD", "RHOB", "NPHI", "DT", "CALI"]
RANDOM_STATE = 42
NOISE_CONTAMINATION = 0.025


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 14,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.45,
        }
    )


def save(fig: plt.Figure, name: str, pdf: PdfPages) -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / name, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def save_png_only(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATA_PATH)
    scores = pd.read_csv(SCORES_PATH)
    cv = pd.read_csv(CV_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    best = pd.read_csv(BEST_PATH)
    return data, scores, cv, predictions, best


def load_raw_workbook() -> pd.DataFrame:
    df = pd.read_excel(SOURCE_WORKBOOK, sheet_name=SHEET_NAME)
    keep = BASE_FEATURES + TARGETS
    df = df[keep].copy()
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def minmax_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    scaled = df[cols].copy()
    for col in cols:
        values = scaled[col].astype(float)
        mn = values.min()
        mx = values.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            scaled[col] = 0.0
        else:
            scaled[col] = (values - mn) / (mx - mn)
    return scaled


def iqr_clip_frame(df: pd.DataFrame, cols: list[str], factor: float = 1.5) -> pd.DataFrame:
    clipped = df[cols].copy()
    for col in cols:
        q1 = clipped[col].quantile(0.25)
        q3 = clipped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        clipped[col] = clipped[col].clip(lower, upper)
    return clipped


def detect_noisy_rows(df: pd.DataFrame) -> np.ndarray:
    features = df[BASE_FEATURES].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.mean(numeric_only=True))
    detector = IsolationForest(
        contamination=NOISE_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_estimators=200,
    )
    return detector.fit_predict(features) == -1


def preprocessing_graphs(pdf: PdfPages) -> None:
    raw = load_raw_workbook()
    numeric = raw.dropna(subset=["DEPT"]).sort_values("DEPT").reset_index(drop=True)
    duplicate_full_rows = int(numeric.duplicated().sum())

    before = minmax_frame(numeric, BASE_FEATURES)
    after_iqr = iqr_clip_frame(numeric, BASE_FEATURES)
    after = minmax_frame(after_iqr, BASE_FEATURES)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(
        [before[col].dropna() for col in BASE_FEATURES],
        tick_labels=BASE_FEATURES,
        patch_artist=True,
        boxprops={"facecolor": "#8ecae6", "color": "#264653"},
        medianprops={"color": "#b23a48"},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.45},
    )
    ax.set_ylabel("Min-Max scaled value")
    ax.set_title("Boxplot for Input-Feature Outliers Before IQR Treatment")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "13_boxplot_outliers_before_treatment.png", pdf)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(
        [after[col].dropna() for col in BASE_FEATURES],
        tick_labels=BASE_FEATURES,
        patch_artist=True,
        boxprops={"facecolor": "#95d5b2", "color": "#264653"},
        medianprops={"color": "#b23a48"},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.45},
    )
    ax.set_ylabel("Min-Max scaled value after IQR clipping")
    ax.set_title("Boxplot for Input-Feature Outliers After IQR Treatment")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "14_boxplot_outliers_after_treatment.png", pdf)

    duplicate_counts = []
    for col in BASE_FEATURES + TARGETS:
        duplicate_counts.append(
            {
                "Column": col,
                "Duplicated values": int(numeric[col].duplicated(keep=False).sum()),
            }
        )
    duplicate_df = pd.DataFrame(duplicate_counts).sort_values("Duplicated values")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(duplicate_df["Column"], duplicate_df["Duplicated values"], color="#457b9d")
    ax.set_xlabel("Count of values duplicated within each column")
    ax.set_title(f"Histogram for Duplicate Data by Column (full-row duplicates = {duplicate_full_rows})")
    save(fig, "15_histogram_duplicate_data.png", pdf)

    noisy_mask = detect_noisy_rows(numeric)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    axes_flat = axes.ravel()
    for ax, col in zip(axes_flat, BASE_FEATURES):
        ax.scatter(
            numeric.loc[~noisy_mask, "DEPT"],
            numeric.loc[~noisy_mask, col],
            s=5,
            alpha=0.35,
            color="#457b9d",
            label="Normal",
        )
        ax.scatter(
            numeric.loc[noisy_mask, "DEPT"],
            numeric.loc[noisy_mask, col],
            s=8,
            alpha=0.75,
            color="#d62828",
            label="Noisy",
        )
        ax.set_title(col)
        ax.set_xlabel("Depth")
        ax.set_ylabel(col)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(f"Scattered Plot for Noisy Data Detected by IsolationForest (n = {int(noisy_mask.sum())})")
    save(fig, "16_scattered_plot_noisy_data_all_inputs.png", pdf)

    key_cols = ["GR", "ILD", "SFLU", "MSFL", "RHOB", "NPHI"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    axes_flat = axes.ravel()
    for ax, col in zip(axes_flat, key_cols):
        y = numeric[col]
        if col in {"ILD", "SFLU", "MSFL"}:
            y = np.log10(y.clip(lower=1e-6))
            ylabel = f"log10({col})"
        else:
            ylabel = col
        ax.scatter(numeric.loc[~noisy_mask, "DEPT"], y.loc[~noisy_mask], s=6, alpha=0.35, color="#457b9d")
        ax.scatter(numeric.loc[noisy_mask, "DEPT"], y.loc[noisy_mask], s=10, alpha=0.78, color="#d62828")
        ax.set_title(col)
        ax.set_xlabel("Depth")
        ax.set_ylabel(ylabel)
    fig.suptitle("Scattered Plot for Noisy Data on Key Well-Log Curves")
    save(fig, "17_scattered_plot_noisy_data_key_logs.png", pdf)


def target_distributions(data: pd.DataFrame, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    plot_specs = [
        ("Porosity", "Porosity", "Porosity fraction"),
        ("Permeability k", "Permeability k", "Permeability k"),
        ("Water Saturation Sw", "Water saturation", "Sw fraction"),
    ]
    for ax, (col, title, xlabel) in zip(axes, plot_specs):
        values = data[col].dropna()
        if col == "Permeability k":
            values = np.log10(values.clip(lower=1e-6))
            xlabel = "log10(k)"
        ax.hist(values, bins=45, color="#2f6f8f", edgecolor="white", linewidth=0.35)
        ax.axvline(values.mean(), color="#b23a48", linewidth=1.4, label="Mean")
        ax.axvline(values.median(), color="#2a9d8f", linewidth=1.4, label="Median")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.legend(frameon=False)
    fig.suptitle("Distribution of Target Reservoir Properties")
    save(fig, "01_target_property_distributions.png", pdf)


def depth_profiles(data: pd.DataFrame, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(15, 9), sharey=True)
    curves = [
        ("GR", "Gamma Ray", "#53777a"),
        ("ILD", "Deep Resistivity", "#b45f06"),
        ("Porosity", "Porosity", "#2a9d8f"),
        ("Permeability k", "Permeability k", "#6d597a"),
        ("Water Saturation Sw", "Water Saturation", "#b23a48"),
        ("RHOB", "Bulk Density", "#1d3557"),
    ]
    depth = data["DEPT"]
    for ax, (col, label, color) in zip(axes, curves):
        x = data[col]
        if col == "Permeability k":
            x = np.log10(x.clip(lower=1e-6))
            label = "log10(k)"
        elif col == "ILD":
            x = np.log10(x.clip(lower=1e-6))
            label = "log10(ILD)"
        ax.plot(x, depth, color=color, linewidth=0.85)
        ax.set_xlabel(label)
        ax.invert_yaxis()
        ax.set_title(label)
    axes[0].set_ylabel("Depth")
    fig.suptitle("Well Log and Reservoir Property Profiles with Depth")
    save(fig, "02_depth_profiles_logs_and_targets.png", pdf)


def correlation_heatmap(data: pd.DataFrame, pdf: PdfPages) -> None:
    cols = ["DEPT"] + LOG_INPUTS + TARGETS
    corr_data = data[cols].copy()
    corr_data["log10_Permeability k"] = np.log10(corr_data["Permeability k"].clip(lower=1e-6))
    corr_data = corr_data.drop(columns=["Permeability k"])
    corr = corr_data.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            color = "white" if abs(value) > 0.62 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    ax.set_title("Correlation Heatmap of Logs and Target Properties")
    save(fig, "03_correlation_heatmap.png", pdf)


def model_comparison(scores: pd.DataFrame, pdf: PdfPages) -> None:
    test = scores[scores["ValidationType"] == "Depth Holdout Test"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for ax, target in zip(axes, TARGETS):
        subset = test[test["Target"] == target].sort_values("R2", ascending=True)
        colors = ["#2a9d8f" if r == subset["R2"].max() else "#6c757d" for r in subset["R2"]]
        ax.barh(subset["Model"], subset["R2"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(target)
        ax.set_xlabel("Test R2")
    fig.suptitle("Model Comparison Using Depth-Holdout Test R2")
    save(fig, "04_model_comparison_test_r2.png", pdf)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for ax, target in zip(axes, TARGETS):
        subset = test[test["Target"] == target].sort_values("MAE", ascending=False)
        colors = ["#2a9d8f" if r == subset["MAE"].min() else "#6c757d" for r in subset["MAE"]]
        ax.barh(subset["Model"], subset["MAE"], color=colors)
        ax.set_title(target)
        ax.set_xlabel("Test MAE")
    fig.suptitle("Model Comparison Using Depth-Holdout Test MAE")
    save(fig, "05_model_comparison_test_mae.png", pdf)


def cv_comparison(cv: pd.DataFrame, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for ax, target in zip(axes, TARGETS):
        subset = cv[cv["Target"] == target].sort_values("CV_R2_mean", ascending=True)
        ax.barh(subset["Model"], subset["CV_R2_mean"], xerr=subset["CV_R2_std"], color="#457b9d")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(target)
        ax.set_xlabel("Mean CV R2 +/- std")
    fig.suptitle("Five-Fold Depth-Ordered Cross-Validation R2")
    save(fig, "06_depth_ordered_cv_r2.png", pdf)


def prediction_panels(predictions: pd.DataFrame, best: pd.DataFrame, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, (_, row) in zip(axes, best.iterrows()):
        subset = predictions[
            (predictions["Target"] == row["Target"]) & (predictions["Model"] == row["BestModel"])
        ]
        ax.scatter(subset["Actual"], subset["Predicted"], s=10, alpha=0.62, color="#264653")
        mn = min(subset["Actual"].min(), subset["Predicted"].min())
        mx = max(subset["Actual"].max(), subset["Predicted"].max())
        ax.plot([mn, mx], [mn, mx], color="#b23a48", linewidth=1.2)
        ax.set_title(f"{row['Target']}\n{row['BestModel']}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    fig.suptitle("Best Model Predicted vs Actual on Depth-Holdout Test Interval")
    save(fig, "07_best_models_predicted_vs_actual_panel.png", pdf)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharex=True)
    for ax, (_, row) in zip(axes, best.iterrows()):
        subset = predictions[
            (predictions["Target"] == row["Target"]) & (predictions["Model"] == row["BestModel"])
        ]
        ax.scatter(subset["DEPT"], subset["Residual"], s=10, alpha=0.62, color="#6d597a")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{row['Target']}\nResidual by depth")
        ax.set_xlabel("Depth")
        ax.set_ylabel("Actual - Predicted")
    fig.suptitle("Best Model Residuals Across the Depth-Holdout Test Interval")
    save(fig, "08_best_models_residuals_by_depth_panel.png", pdf)


def all_model_prediction_graphs(predictions: pd.DataFrame, pdf: PdfPages) -> None:
    model_order = [
        "Baseline Mean",
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "Extra Trees",
        "SVR RBF",
        "ANN Deep MLP",
        "XGBoost",
    ]

    for target in TARGETS:
        target_predictions = predictions[predictions["Target"] == target].copy()
        models = [m for m in model_order if m in set(target_predictions["Model"])]

        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        axes_flat = axes.ravel()
        for ax, model in zip(axes_flat, models):
            subset = target_predictions[target_predictions["Model"] == model]
            ax.scatter(subset["Actual"], subset["Predicted"], s=8, alpha=0.58, color="#264653")
            mn = min(subset["Actual"].min(), subset["Predicted"].min())
            mx = max(subset["Actual"].max(), subset["Predicted"].max())
            ax.plot([mn, mx], [mn, mx], color="#b23a48", linewidth=1.0)
            ax.set_title(model)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
        for ax in axes_flat[len(models) :]:
            ax.axis("off")
        fig.suptitle(f"All Models Predicted vs Actual - {target}")
        save(fig, f"12_all_models_predicted_vs_actual_{slug(target)}.png", pdf)

        for model in models:
            subset = target_predictions[target_predictions["Model"] == model]
            fig, ax = plt.subplots(figsize=(6.2, 5.4))
            ax.scatter(subset["Actual"], subset["Predicted"], s=12, alpha=0.64, color="#264653")
            mn = min(subset["Actual"].min(), subset["Predicted"].min())
            mx = max(subset["Actual"].max(), subset["Predicted"].max())
            ax.plot([mn, mx], [mn, mx], color="#b23a48", linewidth=1.2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{target} - {model} Predicted vs Actual")
            save_png_only(
                fig,
                ALL_MODEL_DIR / f"{slug(target)}__{slug(model)}__predicted_vs_actual.png",
            )


def reservoir_quality_crossplots(data: pd.DataFrame, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        data["Porosity"],
        data["Permeability k"].clip(lower=1e-6),
        c=data["Water Saturation Sw"],
        s=13,
        alpha=0.72,
        cmap="viridis_r",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Porosity")
    ax.set_ylabel("Permeability k, log scale")
    ax.set_title("Reservoir Quality Crossplot: Porosity vs Permeability")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Water Saturation Sw")
    save(fig, "09_porosity_permeability_sw_crossplot.png", pdf)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        data["NPHI"],
        data["RHOB"],
        c=data["Porosity"],
        s=13,
        alpha=0.72,
        cmap="magma",
    )
    ax.invert_yaxis()
    ax.set_xlabel("NPHI")
    ax.set_ylabel("RHOB")
    ax.set_title("Neutron-Density Crossplot Colored by Porosity")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Porosity")
    save(fig, "10_neutron_density_porosity_crossplot.png", pdf)


def best_feature_importance(best: pd.DataFrame, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    for ax, (_, row) in zip(axes, best.iterrows()):
        path = (
            PROJECT_ROOT
            / "ml_outputs"
            / f"feature_importance_{slug(row['Target'])}_{slug(row['BestModel'])}.csv"
        )
        if not path.exists():
            ax.axis("off")
            ax.set_title(row["Target"])
            continue
        imp = pd.read_csv(path).head(12).iloc[::-1]
        ax.barh(imp["Feature"], imp["Importance"], color="#2f6f8f")
        ax.set_title(row["Target"])
        ax.set_xlabel("Importance")
    fig.suptitle("Most Important Inputs for Best Models")
    save(fig, "11_best_model_feature_importance_panel.png", pdf)


def slug(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def main() -> None:
    setup_style()
    data, scores, cv, predictions, best = load()
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    with PdfPages(PDF_PATH) as pdf:
        preprocessing_graphs(pdf)
        target_distributions(data, pdf)
        depth_profiles(data, pdf)
        correlation_heatmap(data, pdf)
        model_comparison(scores, pdf)
        cv_comparison(cv, pdf)
        prediction_panels(predictions, best, pdf)
        all_model_prediction_graphs(predictions, pdf)
        reservoir_quality_crossplots(data, pdf)
        best_feature_importance(best, pdf)

    print(f"Created thesis graphs in: {GRAPH_DIR}")
    print(f"Combined PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()
