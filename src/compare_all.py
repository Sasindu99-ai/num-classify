"""
Cross-Representation Aggregator & Master Leaderboard Generator.

This script scans all representation subdirectories within `results/` (e.g. `a1`, `a2`, `a3`,
`b1`, `b2`, `b3`) and compiles unified comparative leaderboards across all supervised
classification and unsupervised clustering experiments conducted for the EE5130 Mini Project.

Usage:
    python -m src.compare_all
"""

from pathlib import Path
import pandas as pd
import sys

def aggregate_classification(results_root: Path):
    """Aggregate all available classification summary tables across representations."""
    csv_paths = list(results_root.glob("*/tables/classification_summary.csv"))
    if not csv_paths:
        print("[!] No classification summary tables found in results/*/tables/")
        return None

    dfs = []
    for p in csv_paths:
        repr_name = p.parent.parent.name.upper()
        try:
            df = pd.read_csv(p)
            df.insert(0, "Submodule", repr_name)
            dfs.append(df)
        except Exception as e:
            print(f"[!] Error reading {p}: {e}")

    master_df = pd.concat(dfs, ignore_index=True)
    
    # Sort primarily by Test Accuracy (%), secondarily by Macro F1
    if "Test Accuracy (%)" in master_df.columns:
        master_df = master_df.sort_values(by=["Test Accuracy (%)", "Macro F1"], ascending=[False, False])
    
    master_df.reset_index(drop=True, inplace=True)
    master_df.index = master_df.index + 1
    master_df.index.name = "Global Rank"
    return master_df

def aggregate_clustering(results_root: Path):
    """Aggregate all available clustering metric tables across representations."""
    csv_paths = list(results_root.glob("*/tables/clustering_metrics.csv"))
    if not csv_paths:
        print("[!] No clustering metric tables found in results/*/tables/")
        return None

    dfs = []
    for p in csv_paths:
        repr_name = p.parent.parent.name.upper()
        try:
            df = pd.read_csv(p)
            df.insert(0, "Submodule", repr_name)
            dfs.append(df)
        except Exception as e:
            print(f"[!] Error reading {p}: {e}")

    master_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by NMI and ARI descending
    if "Normalized Mutual Info (NMI)" in master_df.columns:
        master_df = master_df.sort_values(by=["Normalized Mutual Info (NMI)", "Adjusted Rand Index (ARI)"], ascending=[False, False])
        
    master_df.reset_index(drop=True, inplace=True)
    master_df.index = master_df.index + 1
    master_df.index.name = "Global Rank"
    return master_df

def main():
    root = Path(__file__).resolve().parent.parent
    results_root = root / "results"
    output_tables = results_root / "master_tables"
    output_tables.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" EE5130 MINI PROJECT: CROSS-REPRESENTATION MASTER COMPARATOR")
    print("=" * 80)
    print(f"Scanning for representation results in: {results_root}")

    # 1. Classification Leaderboard
    cls_df = aggregate_classification(results_root)
    if cls_df is not None:
        save_path = output_tables / "master_classification_leaderboard.csv"
        cls_df.to_csv(save_path)
        print(f"\n[+] Master Classification Leaderboard generated ({len(cls_df)} models):")
        display_cols = [
            "Submodule", "Model", "Representation", "Test Accuracy (%)",
            "Macro F1", "ROC-AUC (Macro OVR)", "Train Time (s)", "Inference Latency (ms/1k)"
        ]
        available_cols = [c for c in display_cols if c in cls_df.columns]
        print(cls_df[available_cols].to_string())
        print(f"    Saved to: {save_path}")

    # 2. Clustering Leaderboard
    clu_df = aggregate_clustering(results_root)
    if clu_df is not None:
        save_path = output_tables / "master_clustering_leaderboard.csv"
        clu_df.to_csv(save_path)
        print(f"\n[+] Master Clustering Leaderboard generated ({len(clu_df)} configurations):")
        display_cols = [
            "Submodule", "Algorithm", "Clusters", "Silhouette Score",
            "Davies-Bouldin Index", "Adjusted Rand Index (ARI)", "Normalized Mutual Info (NMI)"
        ]
        available_cols = [c for c in display_cols if c in clu_df.columns]
        print(clu_df[available_cols].to_string())
        print(f"    Saved to: {save_path}")

    print("\n" + "=" * 80)
    print("Integration Status:")
    present_modules = sorted({p.parent.parent.name.upper() for p in results_root.glob("*/tables/*.csv")})
    print(f"  - Active modules detected: {', '.join(present_modules)}")
    all_expected = ["A1", "A2", "A3", "B1", "B2", "B3"]
    pending = [m for m in all_expected if m not in present_modules]
    if pending:
        print(f"  - Pending teammate modules: {', '.join(pending)}")
        print("  - To integrate: Teammates should deposit classification_summary.csv")
        print("    and clustering_metrics.csv in results/<module>/tables/")
    else:
        print("  - ALL representations integrated successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
