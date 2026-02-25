"""
╔══════════════════════════════════════════════════════════════╗
║         AUTO END-TO-END MACHINE LEARNING PIPELINE           ║
║   Upload any CSV/Excel file → Auto-detects ML scenario      ║
║   → Preprocesses → Trains → Evaluates → Saves Model         ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. DEPENDENCY CHECK & INSTALL
# ─────────────────────────────────────────────
def install_if_missing(package, import_name=None):
    import importlib, subprocess
    name = import_name or package
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

print("\n🔧 Checking dependencies...")
for pkg, imp in [("pandas", None), ("numpy", None), ("scikit-learn", "sklearn"),
                 ("xgboost", None), ("lightgbm", None), ("matplotlib", None),
                 ("seaborn", None), ("openpyxl", None), ("joblib", None)]:
    install_if_missing(pkg, imp)
print("✅ All dependencies ready.\n")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless mode
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.cluster import KMeans
from xgboost import XGBClassifier, XGBRegressor

# ─────────────────────────────────────────────
# 2. FILE INPUT
# ─────────────────────────────────────────────
def get_file_path():
    print("=" * 60)
    print("  📂  DATA INPUT")
    print("=" * 60)

    # Try Google Colab upload first
    try:
        from google.colab import files
        print("  Detected Google Colab environment.")
        print("  Please upload your CSV or Excel file:\n")
        uploaded = files.upload()
        if uploaded:
            fname = list(uploaded.keys())[0]
            print(f"  ✅ Uploaded: {fname}")
            return fname
    except (ImportError, ModuleNotFoundError):
        pass

    # Jupyter / ipywidgets upload
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        import io

        print("  Detected Jupyter environment. Use the button below to upload:\n")
        uploader = widgets.FileUpload(accept=".csv,.xlsx,.xls", multiple=False)
        display(uploader)
        btn = widgets.Button(description="Continue ▶", button_style="success")
        output = widgets.Output()
        result = {"path": None}

        def on_click(b):
            with output:
                clear_output()
                if uploader.value:
                    item = list(uploader.value.values())[0]
                    fname = item["metadata"]["name"]
                    content = item["content"]
                    with open(fname, "wb") as f:
                        f.write(content)
                    result["path"] = fname
                    print(f"  ✅ File saved: {fname}")
                else:
                    print("  ⚠️  No file selected.")

        btn.on_click(on_click)
        display(btn, output)

        # Block until file is uploaded
        import time
        print("  Waiting for upload...", end="")
        for _ in range(120):
            time.sleep(1)
            if result["path"]:
                return result["path"]
        print("\n  ⚠️  Timed out. Falling back to manual path input.")
    except Exception:
        pass

    # Terminal / script fallback
    print("  Enter the full path to your CSV or Excel file.")
    print("  Example: /home/user/data.csv  or  C:\\Users\\you\\data.xlsx\n")
    while True:
        path = input("  📁 File path: ").strip().strip('"').strip("'")
        if os.path.isfile(path):
            return path
        else:
            print(f"  ❌ File not found: '{path}'. Please try again.\n")


# ─────────────────────────────────────────────
# 3. DATA LOADING
# ─────────────────────────────────────────────
def load_data(path):
    print(f"\n📥 Loading: {os.path.basename(path)}")
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        # Try to detect delimiter
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
        delimiter = "," if sample.count(",") >= sample.count(";") else ";"
        df = pd.read_csv(path, delimiter=delimiter, encoding="utf-8", errors="replace")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 4. SCENARIO AUTO-DETECTION
# ─────────────────────────────────────────────
def detect_scenario(df):
    print("\n🔍 Auto-detecting ML scenario...")
    target_keywords = ["target", "label", "class", "output", "result",
                       "price", "sales", "revenue", "amount", "score",
                       "churn", "survived", "diagnosis", "fraud", "default"]

    # Find the most likely target column
    target_col = None
    for col in reversed(df.columns):  # last column is often the target
        if any(kw in col.lower() for kw in target_keywords):
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]

    print(f"  🎯 Detected target column: '{target_col}'")

    n_unique = df[target_col].nunique()
    dtype = df[target_col].dtype

    if n_unique <= 20 and (dtype == object or dtype.name == "category" or n_unique / len(df) < 0.05):
        scenario = "classification"
    elif np.issubdtype(dtype, np.number):
        scenario = "regression"
    else:
        scenario = "clustering"

    print(f"  🤖 Scenario: {scenario.upper()}")
    return scenario, target_col


# ─────────────────────────────────────────────
# 5. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def run_eda(df, target_col, scenario, output_dir):
    print("\n📊 Running EDA...")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  📋 Dataset Info:")
    print(f"     Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"     Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string() or '     None'}")
    print(f"\n  📈 Target distribution:\n{df[target_col].value_counts().head(10).to_string()}")

    # Plot 1: Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EDA Overview", fontsize=14, fontweight="bold")

    if scenario in ("classification", "clustering"):
        df[target_col].value_counts().plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
        axes[0].set_title(f"Target Distribution: {target_col}")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", rotation=45)
    else:
        axes[0].hist(df[target_col].dropna(), bins=30, color="steelblue", edgecolor="white")
        axes[0].set_title(f"Target Distribution: {target_col}")

    # Plot 2: Missing values heatmap
    missing = df.isnull().mean().sort_values(ascending=False).head(15)
    missing.plot(kind="barh", ax=axes[1], color="tomato")
    axes[1].set_title("Missing Value Ratio (Top 15 cols)")
    axes[1].set_xlabel("Fraction Missing")

    plt.tight_layout()
    eda_path = os.path.join(output_dir, "eda_overview.png")
    plt.savefig(eda_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  💾 EDA chart saved → {eda_path}")

    # Correlation heatmap (numeric only)
    numeric_df = df.select_dtypes(include=np.number).dropna(axis=1, how="all")
    if len(numeric_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(min(16, len(numeric_df.columns) + 2),
                                        min(12, len(numeric_df.columns) + 1)))
        sns.heatmap(numeric_df.corr(), annot=len(numeric_df.columns) <= 15,
                    fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout()
        corr_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(corr_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  💾 Correlation heatmap saved → {corr_path}")


# ─────────────────────────────────────────────
# 6. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df, target_col, scenario):
    print("\n⚙️  Preprocessing data...")
    df = df.copy()

    # Drop columns with too many unique values (e.g., IDs, free text)
    drop_cols = [c for c in df.columns
                 if c != target_col and df[c].nunique() > 0.95 * len(df) and df[c].dtype == object]
    if drop_cols:
        print(f"  🗑️  Dropping high-cardinality columns: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)

    # Drop columns with >80% missing
    high_missing = [c for c in df.columns if df[c].isnull().mean() > 0.8]
    if high_missing:
        print(f"  🗑️  Dropping high-missing columns: {high_missing}")
        df.drop(columns=high_missing, inplace=True)

    # Separate features and target
    if scenario != "clustering":
        X = df.drop(columns=[target_col])
        y = df[target_col].copy()
    else:
        X = df.copy()
        y = None

    # Encode target for classification
    le = None
    if scenario == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        print(f"  🏷️  Classes: {list(le.classes_)}")

    # Split numeric / categorical
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # Impute
    if num_cols:
        num_imp = SimpleImputer(strategy="median")
        X[num_cols] = num_imp.fit_transform(X[num_cols])
    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        X[cat_cols] = cat_imp.fit_transform(X[cat_cols])
        for col in cat_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"  ✅ Features ready: {X_scaled.shape[1]} columns")
    return X_scaled, y, le, scaler


# ─────────────────────────────────────────────
# 7. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(X, y, scenario, output_dir, le=None):
    print("\n🚀 Training models...")

    if scenario == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
            "XGBoost":             XGBClassifier(n_estimators=200, eval_metric="logloss",
                                                  use_label_encoder=False, random_state=42,
                                                  verbosity=0),
        }
        metric_name = "Accuracy"

    elif scenario == "regression":
        models = {
            "Linear Regression":  LinearRegression(),
            "Ridge Regression":   Ridge(alpha=1.0),
            "Random Forest":      RandomForestRegressor(n_estimators=200, random_state=42),
            "XGBoost":            XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
        }
        metric_name = "R²"

    else:  # clustering
        return _train_clustering(X, output_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if scenario == "classification" else None
    )

    results = {}
    for name, model in models.items():
        print(f"  ⏳ {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if scenario == "classification":
            score = accuracy_score(y_test, y_pred)
        else:
            score = r2_score(y_test, y_pred)

        cv_scores = cross_val_score(model, X, y, cv=5,
                                     scoring="accuracy" if scenario == "classification" else "r2")
        results[name] = {
            "model": model, "score": score,
            "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
            "y_test": y_test, "y_pred": y_pred
        }
        print(f"{metric_name}: {score:.4f}  (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")

    # Best model
    best_name = max(results, key=lambda k: results[k]["score"])
    best = results[best_name]
    print(f"\n🏆 Best Model: {best_name}  ({metric_name}: {best['score']:.4f})")

    # Detailed report
    print("\n📋 Detailed Evaluation:")
    if scenario == "classification":
        labels = le.classes_ if le else None
        print(classification_report(best["y_test"], best["y_pred"], target_names=labels))
    else:
        rmse = np.sqrt(mean_squared_error(best["y_test"], best["y_pred"]))
        mae  = mean_absolute_error(best["y_test"], best["y_pred"])
        r2   = r2_score(best["y_test"], best["y_pred"])
        print(f"  RMSE : {rmse:.4f}")
        print(f"  MAE  : {mae:.4f}")
        print(f"  R²   : {r2:.4f}")

    # ── Plots ───────────────────────────────────
    if scenario == "classification":
        _plot_confusion_matrix(best["y_test"], best["y_pred"],
                                le.classes_ if le else None, best_name, output_dir)
    else:
        _plot_regression_results(best["y_test"], best["y_pred"], best_name, output_dir)

    _plot_model_comparison(results, metric_name, output_dir)
    _plot_feature_importance(best["model"], X.columns, best_name, output_dir)

    # Save best model
    model_path = os.path.join(output_dir, f"best_model_{best_name.replace(' ', '_')}.pkl")
    joblib.dump(best["model"], model_path)
    print(f"\n💾 Best model saved → {model_path}")

    return best["model"], best_name


def _train_clustering(X, output_dir):
    print("  🔍 Determining optimal clusters (k=2..9)...")
    inertias, silhouettes = [], []
    K_range = range(2, min(10, len(X) // 10 + 2))

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    best_k = list(K_range)[np.argmax(silhouettes)]
    print(f"  🏆 Best k={best_k}  (Silhouette: {max(silhouettes):.4f})")

    best_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    best_km.fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(K_range), inertias, "bo-")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")

    axes[1].plot(list(K_range), silhouettes, "ro-")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Score")

    plt.tight_layout()
    path = os.path.join(output_dir, "clustering_analysis.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  💾 Cluster analysis saved → {path}")

    model_path = os.path.join(output_dir, "best_model_KMeans.pkl")
    joblib.dump(best_km, model_path)
    print(f"  💾 KMeans model saved → {model_path}")
    return best_km, f"KMeans (k={best_k})"


# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────
def _plot_confusion_matrix(y_test, y_pred, labels, title, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(cm)), max(5, len(cm) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  💾 Confusion matrix saved → {path}")


def _plot_regression_results(y_test, y_pred, title, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.5, color="steelblue", edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--")
    axes[0].set_title(f"Actual vs Predicted — {title}")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")

    residuals = np.array(y_test) - np.array(y_pred)
    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Residuals Distribution")
    axes[1].set_xlabel("Residual")

    plt.tight_layout()
    path = os.path.join(output_dir, "regression_results.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  💾 Regression results saved → {path}")


def _plot_model_comparison(results, metric_name, output_dir):
    names  = list(results.keys())
    scores = [results[n]["score"] for n in names]
    cv_means = [results[n]["cv_mean"] for n in names]
    cv_stds  = [results[n]["cv_std"]  for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x - 0.2, scores, 0.35, label=f"Test {metric_name}", color="steelblue")
    ax.bar(x + 0.2, cv_means, 0.35, yerr=cv_stds, capsize=5,
           label=f"CV {metric_name}", color="seagreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_title(f"Model Comparison — {metric_name}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  💾 Model comparison saved → {path}")


def _plot_feature_importance(model, feature_names, title, output_dir):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in indices[::-1]],
            importances[indices[::-1]], color="steelblue")
    ax.set_title(f"Feature Importance — {title}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  💾 Feature importance saved → {path}")


# ─────────────────────────────────────────────
# 8. MAIN ORCHESTRATOR
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("   🤖  AUTO END-TO-END MACHINE LEARNING PIPELINE")
    print("=" * 60)

    # Step 1: Get file
    file_path = get_file_path()

    # Step 2: Load
    df = load_data(file_path)

    # Step 3: Detect scenario
    scenario, target_col = detect_scenario(df)

    # Step 4: Set output directory
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.path.dirname(file_path), f"ml_output_{base_name}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  📁 Output directory: {output_dir}")

    # Step 5: EDA
    run_eda(df, target_col, scenario, output_dir)

    # Step 6: Preprocess
    X, y, le, scaler = preprocess(df, target_col, scenario)

    # Step 7: Train & Evaluate
    best_model, best_name = train_and_evaluate(X, y, scenario, output_dir, le)

    # Save scaler too
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    print("\n" + "=" * 60)
    print("  ✅  PIPELINE COMPLETE!")
    print(f"  🏆  Best Model     : {best_name}")
    print(f"  📂  All outputs in : {output_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()