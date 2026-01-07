import json
import glob
import numpy as np
from eval.plots.componentClassPlot import plot_componentClasses
from eval.plots.distancePlot import plot_boxplots_lineDistances


def load_component_metrics_from_json(metrics_file: str) -> tuple[int, int, int, list, list]:
    """
    Lädt Component-Metriken aus einer metrics.json Datei.
    
    Returns:
        Tuple[found, missed, false_positives, xy_samples, z_samples]
    """
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Direkter Zugriff auf die Hauptebene (nicht nested unter 'components')
        found = data.get('found', 0)
        missed = data.get('missed', 0) 
        false_positives = data.get('false_positives', 0)
        
        # Distanz-Samples laden - korrigierte Feldnamen
        xy_samples = data.get("distance_xy_samples", [])
        z_samples = data.get("distance_z_samples", [])
        
        return found, missed, false_positives, xy_samples, z_samples
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Fehler beim Laden von {metrics_file}: {e}")
        return 0, 0, 0, [], []


def combine_component_evaluations():
    """
    Liest alle metrics.json Dateien aus dem metrics-Ordner und erstellt kombinierte Plots für Components
    """
    # Alle JSON-Dateien im metrics-Ordner finden
    json_files = glob.glob("./output_component_pipeFilter/metrics/*.json")
    
    if not json_files:
        print("Keine metrics.json-Dateien gefunden in ./output_component_pipeFilter/metrics/")
        return
    
    # Daten aus allen Dateien sammeln
    total_found = 0
    total_missed = 0
    total_false_positives = 0
    all_xy_samples = []
    all_z_samples = []
    
    pointcloud_names = []
    
    print(f"Verarbeite {len(json_files)} Dateien:")
    
    for json_file in json_files:
        print(f"  - {json_file}")
        
        found, missed, false_positives, xy_samples, z_samples = load_component_metrics_from_json(json_file)
        
        # Punktwolkenname aus Pfad extrahieren
        name = json_file.split('\\')[-2] if '\\' in json_file else json_file.split('/')[-2]
        pointcloud_names.append(name)
        
        # Component-Klassifikationen sammeln
        total_found += found
        total_missed += missed
        total_false_positives += false_positives
        
        # Distanz-Samples sammeln
        all_xy_samples.extend(xy_samples)
        all_z_samples.extend(z_samples)
        
        print(f"    {name}: Found={found}, Missed={missed}, FP={false_positives}")
    
    # Kombinierte Distanz-Plots erstellen
    print(f"\nErstelle kombinierte Distanz-Plots...", len(all_xy_samples), len(all_z_samples))
    if len(all_xy_samples) > 0 or len(all_z_samples) > 0:
        plot_boxplots_lineDistances(
            all_xy_samples,
            all_z_samples,
            out_png="./output_component_pipeFilter/plots/combined_boxplot_components_distance.png",
            part="Components",
            title=f"Kombinierte Abstände der erkannten Rohrbauteile ({len(json_files)} Punktwolken)",
            show=False,
        )
        print(f"Kombinierter Boxplot erstellt: ./output_component_pipeFilter/plots/combined_boxplot_components.png")
    
    # Kombinierte Component-Klassifikations-Plots erstellen
    plot_componentClasses(
        total_found,
        total_missed,
        total_false_positives,
        out_png="./output_component_pipeFilter/plots/combined_componentClasses.png",
        show=False,
    )
    print(f"Kombiniertes Component-Klassifikations-Diagramm erstellt: ./output_component_pipeFilter/plots/combined_componentClasses.png")
    
    # Zusammenfassung ausgeben
    print(f"\nZusammenfassung über {len(json_files)} Punktwolken:")
    print(f"  - Verarbeitete Dateien: {', '.join(pointcloud_names)}")
    print(f"  - Gesamt gefundene Components: {total_found}")
    print(f"  - Gesamt verpasste Components: {total_missed}")
    print(f"  - Gesamt falsch-positive Components: {total_false_positives}")
    
    # Berechne Precision und Recall
    if total_found + total_false_positives > 0:
        precision = total_found / (total_found + total_false_positives) * 100
        print(f"  - Precision: {precision:.2f}%")
    else:
        print(f"  - Precision: N/A")
        
    if total_found + total_missed > 0:
        recall = total_found / (total_found + total_missed) * 100
        print(f"  - Recall: {recall:.2f}%")
    else:
        print(f"  - Recall: N/A")
        
    if total_found + total_missed > 0 and total_found + total_false_positives > 0:
        f1_score = 2 * (precision/100 * recall/100) / (precision/100 + recall/100) * 100
        print(f"  - F1-Score: {f1_score:.2f}%")
    
    # Distanz-Statistiken ausgeben
    if all_xy_samples:
        print(f"  - XY-Distanz Durchschnitt: {np.mean(all_xy_samples):.3f}m")
        print(f"  - XY-Distanz Median: {np.median(all_xy_samples):.3f}m")
    else:
        print("  - Keine XY-Distanzen")
    if all_z_samples:
        print(f"  - Z-Distanz Durchschnitt: {np.mean(all_z_samples):.3f}m")
        print(f"  - Z-Distanz Median: {np.median(all_z_samples):.3f}m")
    else:
        print("  - Keine Z-Distanzen")


if __name__ == "__main__":
    combine_component_evaluations()