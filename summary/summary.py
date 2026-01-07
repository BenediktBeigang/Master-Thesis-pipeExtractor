import json
import glob
import numpy as np
from eval.plots.distancePlot import plot_boxplots_lineDistances
from eval.plots.pipeClassPlot import plot_segmentClasses

def combine_pipe_evaluations():
    """
    Liest alle JSON-Dateien aus dem metrics-Ordner und erstellt kombinierte Plots
    """
    # Alle JSON-Dateien im metrics-Ordner finden
    json_files = glob.glob("./output/metrics/*_pipes.json")
    
    if not json_files:
        print("Keine JSON-Dateien gefunden in ./output/metrics/")
        return
    
    # Daten aus allen Dateien sammeln
    all_xy_samples = []
    all_z_samples = []
    total_correct = 0
    total_partial = 0
    total_missed = 0
    total_false_positives = 0
    total_coverage = 0
    total_missed_length = 0
    total_false_positive_length = 0
    total_ground_truth_length = 0
    
    pointcloud_names = []
    
    print(f"Verarbeite {len(json_files)} Dateien:")
    
    for json_file in json_files:
        print(f"  - {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Punktwolkenname aus Dateiname extrahieren
        name = json_file.split('\\')[-1].replace('_pipes.json', '')  # Windows-Pfad
        pointcloud_names.append(name)
        
        # Distanz-Samples sammeln
        all_xy_samples.extend(data.get("distance_3D_xy_samples", []))
        all_z_samples.extend(data.get("distance_3D_z_samples", []))
        
        # Segment-Klassifikationen sammeln (als Zahlen, nicht Listen)
        total_correct += data.get("correct", 0)
        total_partial += data.get("partial", 0)
        total_missed += data.get("missed", 0)
        total_false_positives += data.get("false_positives", 0)
        
        # Coverage-Daten sammeln (aus String parsen)
        coverage_str = data.get("coverage", "0 / 0 m  |  0 %")
        coverage_parts = coverage_str.split(' / ')
        if len(coverage_parts) >= 2:
            current_coverage = float(coverage_parts[0])
            total_length = float(coverage_parts[1].split(' m')[0])
            total_coverage += current_coverage
            total_missed_length += (total_length - current_coverage)
            total_ground_truth_length += total_length
        
        # False positive length aus coverage_iou extrahieren
        coverage_iou_str = data.get("coverage_iou", "0 / 0 m  |  0 %")
        iou_parts = coverage_iou_str.split(' / ')
        if len(iou_parts) >= 2:
            total_with_fp = float(iou_parts[1].split(' m')[0])
            fp_length = total_with_fp - total_length
            total_false_positive_length += fp_length
    
    # Kombinierte Distanz-Plots erstellen
    if len(all_xy_samples) > 0 or len(all_z_samples) > 0:
        plot_boxplots_lineDistances(
            all_xy_samples,
            all_z_samples,
            out_png="./output/plots/combined_boxplot_pipes.png",
            part="Endpunkte",
            title=f"Kombinierte Abstände der erkannten Rohre ({len(json_files)} Punktwolken)",
            show=False,
        )
        print(f"Kombinierter Boxplot erstellt: ./output/plots/combined_boxplot_pipes.png")
    
    # Kombinierte Segment-Klassifikations-Plots erstellen
    # Übergeben von Integer-Werten statt Listen
    plot_segmentClasses(
        total_correct,
        total_partial,
        total_missed,
        total_false_positives,
        total_coverage,
        missed_length=total_missed_length,
        false_positive_length=total_false_positive_length,
        out_png="./output/plots/combined_segmentClasses.png",
        show=False,
    )
    print(f"Kombiniertes Segment-Klassifikations-Diagramm erstellt: ./output/plots/combined_segmentClasses.png")
    
    # Zusammenfassung ausgeben
    print(f"\nZusammenfassung über {len(json_files)} Punktwolken:")
    print(f"  - Verarbeitete Dateien: {', '.join(pointcloud_names)}")
    print(f"  - Gesamt korrekte Segmente: {total_correct}")
    print(f"  - Gesamt partielle Segmente: {total_partial}")
    print(f"  - Gesamt verpasste Segmente: {total_missed}")
    print(f"  - Gesamt falsch-positive Segmente: {total_false_positives}")
    print(f"  - Gesamte Coverage: {total_coverage:.2f} / {total_ground_truth_length:.2f} m  |  {(total_coverage/total_ground_truth_length*100):.2f} %")
    print(f"  - Coverage IoU: {total_coverage:.2f} / {total_ground_truth_length + total_false_positive_length:.2f} m  |  {(total_coverage/(total_ground_truth_length + total_false_positive_length)*100):.2f} %")
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
    combine_pipe_evaluations()