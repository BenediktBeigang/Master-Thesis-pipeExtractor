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
    vals = [coerce_number(g) for g in m.groups()]
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

def main():
    ap = argparse.ArgumentParser(description="4D-Gitter-Analyse für coverage_iou")
    ap.add_argument("root", type=Path, help="Ordner mit *_pipes.json (rekursiv)")
    ap.add_argument("--out", type=Path, default=Path("out"), help="Ausgabe-Ordner")
    ap.add_argument("--top", type=int, default=10, help="Anzahl Top-K-Kombinationen")
    args = ap.parse_args()

    root: Path = args.root
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    files = list(root.rglob("*_pipes.json"))
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
    for k in PARAM_COLS:
        me = df.groupby(k, as_index=False)["coverage_iou_pct"].mean()
        plt.figure(figsize=(6, 3.4))
        plt.plot(me[k].astype(str), me["coverage_iou_pct"], marker="o")
        plt.xlabel(k)
        plt.ylabel("coverage_iou [%]")
        plt.title(f"Main Effect: {k}")
        plt.grid(True, linestyle=":", linewidth=0.6)
        plt.tight_layout()
        plt.savefig(out / f"main_effect__{k.replace('.', '_')}.png", dpi=160)
        plt.close()

    print(f"\nFertig. Ergebnisse in: {out.resolve()}")

if __name__ == "__main__":
    main()
