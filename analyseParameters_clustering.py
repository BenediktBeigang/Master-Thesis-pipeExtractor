# analyze_grid.py
from __future__ import annotations
import json, re, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== Parameter-Metadaten ====
# Spaltennamen wie gewünscht (lesbar, aber eindeutig)
PARAM_COLS = [
    'config.slice_cluster_and_merge.epsilon',
    'config.slice_cluster_and_merge.rho_scale',
    'config.global_cluster_and_merge.epsilon',
    'config.global_cluster_and_merge.rho_scale',
]

# Regex: fängt die 4 Werte unmittelbar nach "_sampled_" bis vor "_config_pipes.json"
FNAME_RE = re.compile(
    r"_sampled_([^_]+)_([^_]+)_([^_]+)_([^_]+)_config_pipes\.json$",
    flags=re.IGNORECASE,
)

def coerce_number(s: str):
    """Versucht, Token in int/float zu wandeln; sonst String belassen."""
    try:
        # normale ints
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        # floats (inkl. wissenschaftliche Notation)
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s):
            return float(s)
    except Exception:
        pass
    return s

def parse_percent_from_coverage_iou(s: str | None) -> float:
    """
    Erwartet z. B. '30.21 / 45.17 m  |  66.89 %' -> 66.89
    """
    if not s:
        return np.nan
    m = re.search(r"([0-9]+[.,]?[0-9]*)\s*%", s)
    return float(m.group(1).replace(",", ".")) if m else np.nan

def extract_params_from_fname(fp: Path) -> dict[str, float | int | str] | None:
    """
    Extrahiert in der Reihenfolge:
      1) config.slice_cluster_and_merge.epsilon
      2) config.slice_cluster_and_merge.rho_scale
      3) config.global_cluster_and_merge.epsilon
      4) config.global_cluster_and_merge.rho_scale
    """
    m = FNAME_RE.search(fp.name)
    if not m:
        return None
    
    # Raw values aus Dateiname
    raw_vals = [coerce_number(g) for g in m.groups()]
    
    vals = [
        raw_vals[0] / 100.0,  # slice_epsilon
        raw_vals[1] / 10.0,   # slice_rho_scale
        raw_vals[2] / 10.0,   # global_epsilon
        raw_vals[3] / 10.0,   # global_rho_scale
    ]
    
    return dict(zip(PARAM_COLS, vals, strict=True))

def load_one(fp: Path) -> dict | None:
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] JSON-Fehler in {fp}: {e}")
        return None

    params = extract_params_from_fname(fp)
    if params is None:
        print(f"[WARN] Konnte Parameter aus Dateiname NICHT parsen: {fp.name}")
        return None

    cov_iou_pct = parse_percent_from_coverage_iou(data.get("coverage_iou"))
    if np.isnan(cov_iou_pct):
        print(f"[WARN] coverage_iou fehlt/unklar in {fp.name}")
        return None

    row = {
        "file": str(fp),
        "coverage_iou_pct": cov_iou_pct,
        "false_positives": data.get("false_positives"),
        "missed": data.get("missed"),
        "correct": data.get("correct"),
        "partial": data.get("partial"),
    }
    row.update(params)
    return row

def process_single_pc(root: Path, pc_id: int, out_base: Path) -> pd.DataFrame:
    """Verarbeitet eine einzelne Punktwolke und gibt DataFrame zurück."""
    files = list(root.rglob(f"ontras_{pc_id}_*_pipes.json"))
    if not files:
        print(f"[WARN] Keine *_pipes.json für ontras_{pc_id} in {root} gefunden.")
        return pd.DataFrame()

    rows = []
    for fp in files:
        r = load_one(fp)
        if r: 
            r['pc_id'] = pc_id  # PC-ID hinzufügen
            rows.append(r)

    if not rows:
        print(f"[WARN] Keine verwertbaren Dateien für ontras_{pc_id} gefunden.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Datentypen ordnen (numerische Parameter als numerisch; sonst kategorisch)
    for k in PARAM_COLS:
        if pd.api.types.is_numeric_dtype(df[k]):
            continue
        # Strings geordnet (alphabetisch) – nur falls nötig
        cats = sorted(df[k].astype(str).unique(), key=lambda s: s.lower())
        df[k] = pd.Categorical(df[k].astype(str), categories=cats, ordered=True)

    # Einzelne PC-Ergebnisse speichern
    out = out_base / f"ontras_{pc_id}"
    out.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(out / "all_runs.csv", index=False)

    # Gruppieren (falls Replikate pro Parameter-Kombination existieren)
    group_cols = PARAM_COLS
    agg = (df
           .groupby(group_cols, as_index=False)
           .agg(coverage_iou_pct_mean=("coverage_iou_pct", "mean"),
                coverage_iou_pct_std=("coverage_iou_pct", "std"),
                runs=("coverage_iou_pct", "count")))

    agg.to_csv(out / "grid_mean.csv", index=False)

    print(f"\n=== Ontras {pc_id} - Verarbeitet ===")
    if not agg.empty:
        # Bestes Setting (nach Mittelwert; bei Einzelruns identisch)
        best_idx = agg["coverage_iou_pct_mean"].idxmax()
        best_row = agg.loc[best_idx]
        print(f"Bestes Setting: coverage_iou = {best_row['coverage_iou_pct_mean']:.2f}%")

    return df

def create_combined_plots(all_dfs: dict[int, pd.DataFrame], out: Path):
    """Erstellt kombinierte Plots für alle Punktwolken."""
    colors = ['blue', 'green', 'red', 'purple']
    
    # 2x2 Grid für alle vier Parameter
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    axes = [ax1, ax2, ax3, ax4]
    
    param_configs = [
        ('config.slice_cluster_and_merge.epsilon', 'Slice Epsilon', ax1),
        ('config.slice_cluster_and_merge.rho_scale', 'Slice Rho Scale', ax2),
        ('config.global_cluster_and_merge.epsilon', 'Global Epsilon', ax3),
        ('config.global_cluster_and_merge.rho_scale', 'Global Rho Scale', ax4),
    ]
    
    for param_col, param_title, ax in param_configs:
        for pc_id, df in all_dfs.items():
            if df.empty:
                continue
                
            # Main Effect berechnen
            me = df.groupby(param_col, as_index=False)["coverage_iou_pct"].mean()
            
            # X-Achsen-Labels formatieren (1 Dezimalstelle)
            x_labels = [f"{val:.1f}" for val in me[param_col]]
            
            ax.plot(x_labels, me['coverage_iou_pct'], 
                   marker='o', color=colors[pc_id], 
                   label=f'Ontras {pc_id}', linewidth=2, markersize=6)
        
        ax.set_xlabel(param_title)
        ax.set_ylabel('coverage_iou [%]')
        ax.set_title(param_title)
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Clustering Parameter und ihre Auswirkung auf die Leitungsabdeckung', fontsize=16)
    plt.tight_layout()
    plt.savefig(out / "main_effects_clustering_all_ontras_combined.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    print(f"Kombinierter Plot erstellt: {out / 'main_effects_clustering_all_ontras_combined.png'}")

def main():
    ap = argparse.ArgumentParser(description="4D-Gitter-Analyse für coverage_iou")
    ap.add_argument("root", type=Path, help="Ordner mit *_pipes.json (rekursiv)")
    ap.add_argument("pc_id", type=int, help="Id der Ontras Punktwolke (-1 für alle: 0,1,2,3)")
    ap.add_argument("--top", type=int, default=10, help="Anzahl Top-K-Kombinationen")
    args = ap.parse_args()

    root: Path = args.root
    
    if args.pc_id == -1:
        # Alle Punktwolken verarbeiten
        pc_ids = [0, 1, 2, 3]
        out_base = Path("eval_clustering_ontras_all")
        out_base.mkdir(parents=True, exist_ok=True)
        
        all_dfs = {}
        combined_df = pd.DataFrame()
        
        for pc_id in pc_ids:
            print(f"\n=== Verarbeite Ontras {pc_id} ===")
            df = process_single_pc(root, pc_id, out_base)
            all_dfs[pc_id] = df
            
            if not df.empty:
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        if combined_df.empty:
            raise SystemExit("Keine verwertbaren Dateien für alle Punktwolken gefunden.")
        
        # Kombinierte Ergebnisse speichern
        combined_df.to_csv(out_base / "all_ontras_combined.csv", index=False)
        
        # Kombinierte Plots erstellen
        create_combined_plots(all_dfs, out_base)
        
        # Gesamtstatistik
        print(f"\n=== Gesamtstatistik über alle Ontras ===")
        for pc_id in pc_ids:
            if pc_id in all_dfs and not all_dfs[pc_id].empty:
                best_score = all_dfs[pc_id]['coverage_iou_pct'].max()
                print(f"Ontras {pc_id}: Beste coverage_iou = {best_score:.2f}%")
        
        print(f"\nFertig. Alle Clustering-Analyse-Ergebnisse in: {out_base.resolve()}")
        
    else:
        # Einzelne Punktwolke verarbeiten (ursprünglicher Code)
        out: Path = Path(f"eval_clustering_ontras_{args.pc_id}")
        out.mkdir(parents=True, exist_ok=True)

        files = list(root.rglob(f"ontras_{args.pc_id}_*_pipes.json"))
        if not files:
            raise SystemExit(f"Keine *_pipes.json in {root} gefunden.")

        rows = []
        for fp in files:
            r = load_one(fp)
            if r: rows.append(r)

        if not rows:
            raise SystemExit("Keine verwertbaren Dateien (coverage_iou & Parameter) gefunden.")

        df = pd.DataFrame(rows)

        # Datentypen ordnen (numerische Parameter als numerisch; sonst kategorisch)
        for k in PARAM_COLS:
            if pd.api.types.is_numeric_dtype(df[k]):
                continue
            # Strings geordnet (alphabetisch) – nur falls nötig
            cats = sorted(df[k].astype(str).unique(), key=lambda s: s.lower())
            df[k] = pd.Categorical(df[k].astype(str), categories=cats, ordered=True)

        # Volle Ergebnistabelle sichern
        df.to_csv(out / "all_runs.csv", index=False)

        # Gruppieren (falls Replikate pro Parameter-Kombination existieren)
        group_cols = PARAM_COLS
        agg = (df
               .groupby(group_cols, as_index=False)
               .agg(coverage_iou_pct_mean=("coverage_iou_pct", "mean"),
                    coverage_iou_pct_std=("coverage_iou_pct", "std"),
                    runs=("coverage_iou_pct", "count")))

        agg.to_csv(out / "grid_mean.csv", index=False)

        # Bestes Setting (nach Mittelwert; bei Einzelruns identisch)
        best_idx = agg["coverage_iou_pct_mean"].idxmax()
        best_row = agg.loc[best_idx]
        print("\n=== Bestes Setting (nach coverage_iou_pct_mean) ===")
        print(best_row.to_string())

        # Top-K auf Einzelrun-Ebene (nicht gemittelt) – für konkrete Dateien
        top_k = (df.sort_values("coverage_iou_pct", ascending=False)
                   .head(args.top))
        print(f"\n=== Top {args.top} Einzelruns ===")
        print(top_k[PARAM_COLS + ["coverage_iou_pct", "false_positives", "missed", "file"]]
              .to_string(index=False))

        # Main-Effects (über alle anderen Parameter gemittelt)
        # Epsilon-Plot (slice und global kombiniert)
        epsilon_data = []
        for prefix in ['slice', 'global']:
            col = f'config.{prefix}_cluster_and_merge.epsilon'
            me = df.groupby(col, as_index=False)["coverage_iou_pct"].mean()
            for _, row in me.iterrows():
                epsilon_data.append({
                    'value': row[col],
                    'coverage_iou_pct': row['coverage_iou_pct'],
                    'type': prefix
                })
        
        epsilon_df = pd.DataFrame(epsilon_data)
        
        plt.figure(figsize=(8, 4))
        for prefix in ['slice', 'global']:
            subset = epsilon_df[epsilon_df['type'] == prefix]
            # Epsilon hat immer 1 Dezimalstelle (0.2, 0.3, etc.)
            x_labels = [f"{val:.1f}" for val in subset['value']]
            plt.plot(x_labels, subset['coverage_iou_pct'], marker='o', label=f'{prefix}_epsilon')
        
        plt.xlabel('Epsilon')
        plt.ylabel('coverage_iou [%]')
        plt.title('Main Effect: Epsilon (Slice vs Global)')
        plt.legend()
        plt.grid(True, linestyle=":", linewidth=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out / "main_effect__epsilon.png", dpi=160)
        plt.close()
        
        # Rho_scale-Plot (slice und global kombiniert)
        rho_data = []
        for prefix in ['slice', 'global']:
            col = f'config.{prefix}_cluster_and_merge.rho_scale'
            me = df.groupby(col, as_index=False)["coverage_iou_pct"].mean()
            for _, row in me.iterrows():
                rho_data.append({
                    'value': row[col],
                    'coverage_iou_pct': row['coverage_iou_pct'],
                    'type': prefix
                })
        
        rho_df = pd.DataFrame(rho_data)
        
        plt.figure(figsize=(8, 4))
        for prefix in ['slice', 'global']:
            subset = rho_df[rho_df['type'] == prefix]
            # Rho_scale hat immer 1 Dezimalstelle (0.2, 0.4, etc.)
            x_labels = [f"{val:.1f}" for val in subset['value']]
            plt.plot(x_labels, subset['coverage_iou_pct'], marker='o', label=f'{prefix}_rho_scale')
        
        plt.xlabel('Rho Scale')
        plt.ylabel('coverage_iou [%]')
        plt.title('Main Effect: Rho Scale (Slice vs Global)')
        plt.legend()
        plt.grid(True, linestyle=":", linewidth=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out / "main_effect__rho_scale.png", dpi=160)
        plt.close()

        print(f"\nFertig. Ergebnisse in: {out.resolve()}")

if __name__ == "__main__":
    main()