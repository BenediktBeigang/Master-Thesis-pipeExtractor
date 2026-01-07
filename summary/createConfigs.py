import json
import os
from itertools import product

def config_exists(slice_eps, slice_rho, global_eps, global_rho, finished_dir="finished_configs"):
    """Überprüft, ob eine Konfiguration mit diesen Parametern bereits existiert."""
    # Dateiname nach demselben Schema erstellen
    slice_eps_str = str(int(slice_eps * 100)).zfill(2)
    slice_rho_str = str(int(slice_rho * 10)).zfill(2)
    global_eps_str = str(int(global_eps * 10)).zfill(2)
    global_rho_str = str(int(global_rho * 10)).zfill(2)
    
    filename = f"{slice_eps_str}_{slice_rho_str}_{global_eps_str}_{global_rho_str}_config.json"
    filepath = os.path.join(finished_dir, filename)
    
    return os.path.exists(filepath)

def load_base_config(config_path):
    """Lädt die Basis-Konfiguration aus der angegebenen Datei."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, filename, output_dir):
    """Speichert die Konfiguration in eine Datei."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Konfiguration gespeichert: {filepath}")

def generate_config_files():
    """Generiert alle Konfigurationsdateien mit den verschiedenen Parameterkombinationen."""
    
    # Basis-Konfiguration laden
    base_config_path = "config.json"
    base_config = load_base_config(base_config_path)
    
    # Parameter-Werte Clustering
    slice_epsilon_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    slice_rho_scale_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    global_epsilon_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    global_rho_scale_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    
    # Parameter-Werte Hough
    slice_thickness_values = [0.1, 0.2, 0.3]
    cell_size_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    canny_sigma_values = [0.6, 0.8, 1.0, 1.2, 1.4, 1.8]
    threshold_values = [5, 10, 15]
    
    # Ausgabeordner
    output_dir = "config_variations"
    
    # Alle Kombinationen generieren
    combinations_clustering = product(
        slice_epsilon_values,
        slice_rho_scale_values,
        global_epsilon_values,
        global_rho_scale_values
    )
    
    combinations_hough = product(
        slice_thickness_values,
        cell_size_values,
        canny_sigma_values,
        threshold_values
    )
    
    config_count = 0
    skipped_count = 0
    
    if True:
        for slice_thickness, cell_size, canny_sigma, threshold in combinations_hough:
            # Überprüfen, ob die Konfiguration bereits existiert
            if config_exists(slice_thickness, cell_size, canny_sigma, threshold):
                print(f"Konfiguration bereits vorhanden, überspringe: {slice_thickness}_{cell_size}_{canny_sigma}_{threshold}")
                skipped_count += 1
                continue

            # Kopie der Basis-Konfiguration erstellen
            config = base_config.copy()
            
            # Parameter anpassen
            config["slice_thickness"] = slice_thickness
            config["hough"]["cell_size"] = cell_size
            config["hough"]["canny_sigma"] = canny_sigma
            config["hough"]["threshold"] = threshold

            # Dateiname erstellen (Format: slice_eps_slice_rho_global_eps_global_rho)
            # Dezimalpunkte durch Unterstriche ersetzen und als Integer formatieren
            slice_thickness_str = str(int(slice_thickness * 10)).zfill(2)
            cell_size_str = str(int(cell_size * 100)).zfill(2)
            canny_sigma_str = str(int(canny_sigma * 10)).zfill(2)
            threshold_str = str(int(threshold)).zfill(2)
            
            filename = f"{slice_thickness_str}_{cell_size_str}_{canny_sigma_str}_{threshold_str}_config.json"
            
            # Konfiguration speichern
            save_config(config, filename, output_dir)
            config_count += 1
    else:
        for slice_eps, slice_rho, global_eps, global_rho in combinations_clustering:
            # Überprüfen, ob die Konfiguration bereits existiert
            if config_exists(slice_eps, slice_rho, global_eps, global_rho):
                print(f"Konfiguration bereits vorhanden, überspringe: {slice_eps}_{slice_rho}_{global_eps}_{global_rho}")
                skipped_count += 1
                continue

            # Kopie der Basis-Konfiguration erstellen
            config = base_config.copy()
            
            # Parameter anpassen
            config["slice_cluster_and_merge"]["epsilon"] = slice_eps
            config["slice_cluster_and_merge"]["rho_scale"] = slice_rho
            config["global_cluster_and_merge"]["epsilon"] = global_eps
            config["global_cluster_and_merge"]["rho_scale"] = global_rho
            
            # Dateiname erstellen (Format: slice_eps_slice_rho_global_eps_global_rho)
            # Dezimalpunkte durch Unterstriche ersetzen und als Integer formatieren
            slice_thickness_str = str(int(slice_eps * 100)).zfill(2)
            cell_size_str = str(int(slice_rho * 10)).zfill(2)
            canny_sigma_str = str(int(global_eps * 10)).zfill(2)
            threshold_str = str(int(global_rho * 10)).zfill(2)
            
            filename = f"{slice_thickness_str}_{cell_size_str}_{canny_sigma_str}_{threshold_str}_config.json"
            
            # Konfiguration speichern
            save_config(config, filename, output_dir)
            config_count += 1
    
    print(f"\n{config_count} Konfigurationsdateien wurden erfolgreich erstellt in '{output_dir}'")
    print(f"{skipped_count} Konfigurationen wurden übersprungen (bereits vorhanden)")

    # Übersicht der Parameter ausgeben
    print("\nParameter-Übersicht:")
    print(f"slice_cluster_and_merge.epsilon: {slice_epsilon_values}")
    print(f"slice_cluster_and_merge.rho_scale: {slice_rho_scale_values}")
    print(f"global_cluster_and_merge.epsilon: {global_epsilon_values}")
    print(f"global_cluster_and_merge.rho_scale: {global_rho_scale_values}")

if __name__ == "__main__":
    generate_config_files()