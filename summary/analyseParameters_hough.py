# analyze_hough_grid.py
from __future__ import annotations
import json, re, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== Hough Parameter-Metadaten ====
# Spaltennamen für Hough-Parameter
HOUGH_PARAM_COLS = [
    'config.slice_thickness',
    'config.hough.cell_size',
    'config.hough.canny_sigma',
    'config.hough.threshold',
]

# Regex: fängt die 4 Hough-Werte unmittelbar nach "_sampled_" bis vor "_config_pipes.json"
HOUGH_FNAME_RE = re.compile(
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

def extract_hough_params_from_fname(fp: Path) -> dict[str, float | int | str] | None:
    """
    Extrahiert Hough-Parameter in der Reihenfolge:
      1) config.slice_thickness
      2) config.hough.cell_size
      3) config.hough.canny_sigma
      4) config.hough.threshold
    """
    m = HOUGH_FNAME_RE.search(fp.name)
    if not m:
        return None
    
    # Raw values aus Dateiname
    raw_vals = [coerce_number(g) for g in m.groups()]
    
    vals = [
        raw_vals[0] / 10.0,   # slice_thickness
        raw_vals[1] / 100.0,  # cell_size
        raw_vals[2] / 10.0,   # canny_sigma
        float(raw_vals[3]),   # threshold (bleibt als int/float)
    ]
    
    return dict(zip(HOUGH_PARAM_COLS, vals, strict=True))

def load_one_hough(fp: Path) -> dict | None:
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] JSON-Fehler in {fp}: {e}")
        return None

    params = extract_hough_params_from_fname(fp)
    if params is None:
        print(f"[WARN] Konnte Hough-Parameter aus Dateiname NICHT parsen: {fp.name}")
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
        r = load_one_hough(fp)
        if r: 
            r['pc_id'] = pc_id  # PC-ID hinzufügen
            rows.append(r)

    if not rows:
        print(f"[WARN] Keine verwertbaren Dateien für ontras_{pc_id} gefunden.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Datentypen ordnen
    for k in HOUGH_PARAM_COLS:
        if pd.api.types.is_numeric_dtype(df[k]):
            continue
        cats = sorted(df[k].astype(str).unique(), key=lambda s: s.lower())
        df[k] = pd.Categorical(df[k].astype(str), categories=cats, ordered=True)

    # Einzelne PC-Ergebnisse speichern
    out = out_base / f"ontras_{pc_id}"
    out.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(out / "all_hough_runs.csv", index=False)

    # Gruppieren
    group_cols = HOUGH_PARAM_COLS
    agg = (df
           .groupby(group_cols, as_index=False)
           .agg(coverage_iou_pct_mean=("coverage_iou_pct", "mean"),
                coverage_iou_pct_std=("coverage_iou_pct", "std"),
                runs=("coverage_iou_pct", "count")))

    agg.to_csv(out / "hough_grid_mean.csv", index=False)

    print(f"\n=== Ontras {pc_id} - Verarbeitet ===")
    if not agg.empty:
        best_idx = agg["coverage_iou_pct_mean"].idxmax()
        best_row = agg.loc[best_idx]
        print(f"Bestes Setting: coverage_iou = {best_row['coverage_iou_pct_mean']:.2f}%")

    return df

def create_combined_plots(all_dfs: dict[int, pd.DataFrame], out: Path):
    """Erstellt kombinierte Plots für alle Punktwolken."""
    colors = ['blue', 'green', 'red', 'purple']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    axes = [ax1, ax2, ax3, ax4]
    param_names = ['config.slice_thickness', 'config.hough.cell_size', 'config.hough.canny_sigma', 'config.hough.threshold']
    param_labels = ['Slice Thickness [m]', 'Cell Size [m]', 'Canny Sigma', 'Threshold']
    param_titles = ['Slice Thickness', 'Cell Size', 'Canny Sigma', 'Threshold']
    
    for i, (param, label, title, ax) in enumerate(zip(param_names, param_labels, param_titles, axes)):
        for pc_id, df in all_dfs.items():
            if df.empty:
                continue
                
            # Main Effect berechnen
            me = df.groupby(param, as_index=False)["coverage_iou_pct"].mean()
            
            # X-Achsen-Labels formatieren
            if param == 'config.slice_thickness':
                x_labels = [f"{val:.1f}" for val in me[param]]
            elif param == 'config.hough.cell_size':
                x_labels = [f"{val:.2f}" for val in me[param]]
            elif param == 'config.hough.canny_sigma':
                x_labels = [f"{val:.1f}" for val in me[param]]
            else:  # threshold
                x_labels = [str(int(val)) for val in me[param]]
            
            ax.plot(x_labels, me['coverage_iou_pct'], 
                   marker='o', color=colors[pc_id], 
                   label=f'Ontras {pc_id}', linewidth=2, markersize=6)
        
        ax.set_xlabel(label)
        ax.set_ylabel('coverage_iou [%]')
        ax.set_title(title)
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.legend()
        
        if param == 'config.hough.cell_size':
            ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Hough Parameter und ihre Auswirkung auf die Leitungsabdeckung', fontsize=16)
    plt.tight_layout()
    plt.savefig(out / "main_effects_all_ontras_combined.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    print(f"Kombinierter Plot erstellt: {out / 'main_effects_all_ontras_combined.png'}")

def main():
    ap = argparse.ArgumentParser(description="4D-Hough-Parameter-Analyse für coverage_iou")
    ap.add_argument("root", type=Path, help="Ordner mit *_pipes.json (rekursiv)")
    ap.add_argument("pc_id", type=int, help="Id der Ontras Punktwolke (-1 für alle: 0,1,2,3)")
    ap.add_argument("--top", type=int, default=10, help="Anzahl Top-K-Kombinationen")
    args = ap.parse_args()

    root: Path = args.root
    
    if args.pc_id == -1:
        # Alle Punktwolken verarbeiten
        pc_ids = [0, 1, 2, 3]
        out_base = Path("eval_hough_ontras_all")
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
        
        print(f"\nFertig. Alle Hough-Analyse-Ergebnisse in: {out_base.resolve()}")
        
    else:
        # Einzelne Punktwolke verarbeiten (alter Code)
        out = Path(f"eval_hough_ontras_{args.pc_id}")
        out.mkdir(parents=True, exist_ok=True)
        
        df = process_single_pc(root, args.pc_id, Path("."))
        if df.empty:
            raise SystemExit(f"Keine verwertbaren Dateien für ontras_{args.pc_id} gefunden.")

        # Main-Effects für Hough-Parameter
        
        # 1. Slice Thickness
        me_slice_thickness = df.groupby('config.slice_thickness', as_index=False)["coverage_iou_pct"].mean()
        
        plt.figure(figsize=(8, 4))
        x_labels = [f"{val:.1f}" for val in me_slice_thickness['config.slice_thickness']]
        plt.plot(x_labels, me_slice_thickness['coverage_iou_pct'], marker='o', color='blue')
        plt.xlabel('Slice Thickness [m]')
        plt.ylabel('coverage_iou [%]')
        plt.title('Main Effect: Slice Thickness')
        plt.grid(True, linestyle=":", linewidth=0.6)
        plt.tight_layout()
        plt.savefig(out / "main_effect__slice_thickness.png", dpi=160)
        plt.close()
        
        # 2. Cell Size
        me_cell_size = df.groupby('config.hough.cell_size', as_index=False)["coverage_iou_pct"].mean()
        
        plt.figure(figsize=(8, 4))
        x_labels = [f"{val:.2f}" for val in me_cell_size['config.hough.cell_size']]
        plt.plot(x_labels, me_cell_size['coverage_iou_pct'], marker='o', color='green')
        plt.xlabel('Cell Size [m]')
        plt.ylabel('coverage_iou [%]')
        plt.title('Main Effect: Hough Cell Size')
        plt.grid(True, linestyle=":", linewidth=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out / "main_effect__cell_size.png", dpi=160)
        plt.close()
        
        # 3. Canny Sigma
        me_canny_sigma = df.groupby('config.hough.canny_sigma', as_index=False)["coverage_iou_pct"].mean()
        
        plt.figure(figsize=(8, 4))
        x_labels = [f"{val:.1f}" for val in me_canny_sigma['config.hough.canny_sigma']]
        plt.plot(x_labels, me_canny_sigma['coverage_iou_pct'], marker='o', color='red')
        plt.xlabel('Canny Sigma')
        plt.ylabel('coverage_iou [%]')
        plt.title('Main Effect: Canny Sigma')
        plt.grid(True, linestyle=":", linewidth=0.6)
        plt.tight_layout()
        plt.savefig(out / "main_effect__canny_sigma.png", dpi=160)
        plt.close()
        
        # 4. Threshold
        me_threshold = df.groupby('config.hough.threshold', as_index=False)["coverage_iou_pct"].mean()
        
        plt.figure(figsize=(8, 4))
        x_labels = [str(int(val)) for val in me_threshold['config.hough.threshold']]
        plt.plot(x_labels, me_threshold['coverage_iou_pct'], marker='o', color='purple')
        plt.xlabel('Threshold')
        plt.ylabel('coverage_iou [%]')
        plt.title('Main Effect: Hough Threshold')
        plt.grid(True, linestyle=":", linewidth=0.6)
        plt.tight_layout()
        plt.savefig(out / "main_effect__threshold.png", dpi=160)
        plt.close()

        # Kombiniertes Plot aller Hough-Parameter
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Slice Thickness
        x_labels = [f"{val:.1f}" for val in me_slice_thickness['config.slice_thickness']]
        ax1.plot(x_labels, me_slice_thickness['coverage_iou_pct'], marker='o', color='blue')
        ax1.set_xlabel('Slice Thickness [m]')
        ax1.set_ylabel('coverage_iou [%]')
        ax1.set_title('Slice Thickness')
        ax1.grid(True, linestyle=":", linewidth=0.6)
        
        # Cell Size
        x_labels = [f"{val:.2f}" for val in me_cell_size['config.hough.cell_size']]
        ax2.plot(x_labels, me_cell_size['coverage_iou_pct'], marker='o', color='green')
        ax2.set_xlabel('Cell Size [m]')
        ax2.set_ylabel('coverage_iou [%]')
        ax2.set_title('Cell Size')
        ax2.grid(True, linestyle=":", linewidth=0.6)
        ax2.tick_params(axis='x', rotation=45)
        
        # Canny Sigma
        x_labels = [f"{val:.1f}" for val in me_canny_sigma['config.hough.canny_sigma']]
        ax3.plot(x_labels, me_canny_sigma['coverage_iou_pct'], marker='o', color='red')
        ax3.set_xlabel('Canny Sigma')
        ax3.set_ylabel('coverage_iou [%]')
        ax3.set_title('Canny Sigma')
        ax3.grid(True, linestyle=":", linewidth=0.6)
        
        # Threshold
        x_labels = [str(int(val)) for val in me_threshold['config.hough.threshold']]
        ax4.plot(x_labels, me_threshold['coverage_iou_pct'], marker='o', color='purple')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('coverage_iou [%]')
        ax4.set_title('Threshold')
        ax4.grid(True, linestyle=":", linewidth=0.6)
        
        plt.suptitle('Main Effects: Hough Parameters')
        plt.tight_layout()
        plt.savefig(out / "main_effects__hough_combined.png", dpi=160)
        plt.close()

        print(f"\nFertig. Hough-Analyse-Ergebnisse in: {out.resolve()}")

if __name__ == "__main__":
    main()