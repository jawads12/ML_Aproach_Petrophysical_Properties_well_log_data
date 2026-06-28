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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "ml_outputs/preprocessed_modeling_dataset.csv"
SCORES_PATH = PROJECT_ROOT / "ml_outputs/model_scores.csv"
CV_PATH = PROJECT_ROOT / "ml_outputs/cv_summary.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "ml_outputs/test_predictions.csv"
BEST_PATH = PROJECT_ROOT / "ml_outputs/best_models_summary.csv"
GRAPH_DIR = PROJECT_ROOT / "ml_outputs/thesis_graphs"
PDF_PATH = GRAPH_DIR / "thesis_graph_pack.pdf"

TARGETS = ["Porosity", "Permeability k", "Water Saturation Sw"]
LOG_INPUTS = ["GR", "ILD", "RHOB", "NPHI", "DT", "CALI"]


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


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATA_PATH)
    scores = pd.read_csv(SCORES_PATH)
    cv = pd.read_csv(CV_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    best = pd.read_csv(BEST_PATH)
    return data, scores, cv, predictions, best


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
        target_distributions(data, pdf)
        depth_profiles(data, pdf)
        correlation_heatmap(data, pdf)
        model_comparison(scores, pdf)
        cv_comparison(cv, pdf)
        prediction_panels(predictions, best, pdf)
        reservoir_quality_crossplots(data, pdf)
        best_feature_importance(best, pdf)

    print(f"Created thesis graphs in: {GRAPH_DIR}")
    print(f"Combined PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()
