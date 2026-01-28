"""
Einfacher Parameter Optimizer für Franka Cube Stacking

Dieses Skript testet systematisch verschiedene Controller-Parameter
durch Aufruf von fcs_main_parallel.py und Auswertung der CSV-Ergebnisse.

Verwendung:
    1. Isaac Sim Umgebung aktivieren
    2. python optimize_parameters.py
    
    Oder mit Optionen:
    python optimize_parameters.py --episodes 5 --dry-run

Das Skript:
1. Erstellt eine temporäre config.yaml mit Testparametern
2. Führt fcs_main_parallel.py mit wenigen Episoden aus
3. Liest die CSV-Ergebnisse und zählt Erfolge/Fehler
4. Speichert die besten Parameter
"""

import os
import sys
import yaml
import shutil
import subprocess
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import copy

# ============================================================================
# PARAMETER KONFIGURATION
# ============================================================================

# Zu optimierende Parameter mit Suchbereich
PARAMETERS = {
    "trajectory_resolution": {
        "values": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "description": "Basis-Geschwindigkeit aller Phasen",
    },
    "air_speed_multiplier": {
        "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        "description": "Extra Speed für AIR Phasen (0,4,5,8,9)",
    },
    "critical_speed_factor": {
        "values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "description": "Speed in kritischer Zone (niedriger=langsamer)",
    },
    "critical_height_threshold": {
        "values": [0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02],
        "description": "Höhe der kritischen Zone (m)",
    },
}

# Test-Einstellungen
EPISODES_PER_TEST = 3      # Episoden pro Parameterkombination
MAX_ALLOWED_FAILURES = 0   # Max erlaubte Fehler (0 = alle müssen erfolgreich sein)


def load_config(path: Path) -> dict:
    """Lädt YAML Config."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: Path):
    """Speichert YAML Config."""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def run_episodes(config_path: Path, num_episodes: int, output_name: str) -> Path:
    """
    Führt fcs_main_parallel.py mit der gegebenen Config aus.
    
    Returns:
        Path zur CSV-Ergebnisdatei
    """
    # Temporäre Config mit angepasster Episode-Anzahl
    config = load_config(config_path)
    config["dataset"]["episodes"] = num_episodes
    config["dataset"]["name"] = output_name
    
    # Temporäre Config speichern
    temp_config = config_path.parent / "config_temp.yaml"
    save_config(config, temp_config)
    
    # fcs_main_parallel.py ausführen
    script_path = config_path.parent / "fcs_main_parallel.py"
    
    print(f"  Starte Simulation mit {num_episodes} Episoden...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--config", str(temp_config)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 Minuten Timeout
            cwd=str(config_path.parent),
        )
        
        if result.returncode != 0:
            print(f"  WARNUNG: Simulation beendet mit Code {result.returncode}")
            if result.stderr:
                print(f"  Stderr: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print("  WARNUNG: Simulation Timeout!")
    except Exception as e:
        print(f"  FEHLER: {e}")
    finally:
        # Temporäre Config löschen
        if temp_config.exists():
            temp_config.unlink()
    
    # CSV-Pfad zurückgeben
    csv_path = Path(config["dataset"]["path"]) / output_name / "episode_tracking.csv"
    return csv_path


def analyze_csv(csv_path: Path) -> Tuple[int, int, float]:
    """
    Analysiert die CSV-Ergebnisse.
    
    Returns:
        Tuple[successes, failures, avg_timesteps]
    """
    if not csv_path.exists():
        return 0, 0, float('inf')
    
    successes = 0
    failures = 0
    total_timesteps = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=';')
            rows = list(reader)
            
            if len(rows) < 2:
                return 0, 0, float('inf')
            
            # Header ist erste Zeile
            header = rows[0]
            
            # Finde Spalten-Indizes
            timesteps_row_idx = None
            validation_row_idx = None
            
            for i, row in enumerate(rows):
                if len(row) > 0:
                    if "Gesamte Timesteps" in row[0]:
                        timesteps_row_idx = i
                    elif "Validierung" in row[0]:
                        validation_row_idx = i
            
            if timesteps_row_idx is None or validation_row_idx is None:
                return 0, 0, float('inf')
            
            # Zähle Ergebnisse (Spalten = Runs)
            num_runs = len(header) - 1  # Erste Spalte ist Variable-Name
            
            for col in range(1, num_runs + 1):
                # Validierung prüfen
                if validation_row_idx < len(rows) and col < len(rows[validation_row_idx]):
                    val = rows[validation_row_idx][col]
                    if "JA" in val:
                        successes += 1
                        # Timesteps addieren
                        if timesteps_row_idx < len(rows) and col < len(rows[timesteps_row_idx]):
                            ts = rows[timesteps_row_idx][col]
                            try:
                                total_timesteps += int(ts)
                            except:
                                pass
                    else:
                        failures += 1
    except Exception as e:
        print(f"  Fehler beim CSV-Parsing: {e}")
        return 0, 0, float('inf')
    
    avg_timesteps = total_timesteps / successes if successes > 0 else float('inf')
    return successes, failures, avg_timesteps


def test_parameter_value(
    base_config: dict,
    config_path: Path,
    param_name: str,
    param_value: float,
    test_idx: int,
) -> Tuple[bool, float, int, int]:
    """
    Testet einen einzelnen Parameterwert.
    
    Returns:
        Tuple[success, avg_timesteps, successes, failures]
    """
    # Config mit neuem Parameterwert
    test_config = copy.deepcopy(base_config)
    test_config["controller"][param_name] = param_value
    
    # Temporäre Config speichern
    temp_config = config_path.parent / "config_test.yaml"
    save_config(test_config, temp_config)
    
    # Eindeutiger Output-Name
    output_name = f"opt_test_{test_idx:03d}"
    
    # Episoden ausführen
    csv_path = run_episodes(temp_config, EPISODES_PER_TEST, output_name)
    
    # Ergebnisse analysieren
    successes, failures, avg_timesteps = analyze_csv(csv_path)
    
    # Aufräumen (temporäre Config)
    if temp_config.exists():
        temp_config.unlink()
    
    # Erfolg = alle Episoden erfolgreich
    is_success = failures <= MAX_ALLOWED_FAILURES and successes > 0
    
    return is_success, avg_timesteps, successes, failures


def optimize_parameter(
    param_name: str,
    param_config: dict,
    base_config: dict,
    config_path: Path,
    test_counter: List[int],
) -> Tuple[float, float]:
    """
    Optimiert einen einzelnen Parameter.
    
    Returns:
        Tuple[best_value, best_timesteps]
    """
    print(f"\n{'='*60}")
    print(f"Optimiere: {param_name}")
    print(f"  {param_config['description']}")
    print(f"  Werte: {param_config['values']}")
    print('='*60)
    
    best_value = param_config['values'][0]
    best_timesteps = float('inf')
    last_successful_value = param_config['values'][0]
    
    for value in param_config['values']:
        test_counter[0] += 1
        print(f"\n  Test #{test_counter[0]}: {param_name} = {value}")
        
        is_success, avg_timesteps, successes, failures = test_parameter_value(
            base_config, config_path, param_name, value, test_counter[0]
        )
        
        if is_success:
            print(f"    ✓ Erfolg: {avg_timesteps:.1f} Timesteps ({successes}/{successes+failures} ok)")
            last_successful_value = value
            
            if avg_timesteps < best_timesteps:
                best_timesteps = avg_timesteps
                best_value = value
        else:
            print(f"    ✗ Fehlgeschlagen: {failures} Fehler")
            print(f"    → Stoppe, letzter erfolgreicher Wert: {last_successful_value}")
            break
    
    # Update base_config mit bestem Wert
    base_config["controller"][param_name] = best_value
    
    print(f"\n  → Bester Wert: {param_name} = {best_value} ({best_timesteps:.1f} Timesteps)")
    return best_value, best_timesteps


def main():
    """Hauptfunktion."""
    print("\n" + "="*70)
    print("  PARAMETER OPTIMIZER für Franka Cube Stacking")
    print("="*70)
    
    # Pfade
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.yaml"
    
    if not config_path.exists():
        print(f"FEHLER: Config nicht gefunden: {config_path}")
        sys.exit(1)
    
    # Basis-Config laden
    base_config = load_config(config_path)
    
    # Original Config sichern
    backup_path = script_dir / f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    shutil.copy(config_path, backup_path)
    print(f"Config-Backup: {backup_path}")
    
    # Aktuelle Parameter anzeigen
    print("\nAktuelle Controller-Parameter:")
    for param in PARAMETERS.keys():
        print(f"  {param}: {base_config['controller'].get(param, 'N/A')}")
    
    # Optimierung
    test_counter = [0]
    results = {}
    
    for param_name, param_config in PARAMETERS.items():
        best_value, best_timesteps = optimize_parameter(
            param_name, param_config, base_config, config_path, test_counter
        )
        results[param_name] = {"value": best_value, "timesteps": best_timesteps}
    
    # Finale Ergebnisse
    print("\n" + "="*70)
    print("  OPTIMIERUNG ABGESCHLOSSEN")
    print("="*70)
    
    print("\nOptimierte Parameter:")
    for param, result in results.items():
        print(f"  {param}: {result['value']}")
    
    # Optimierte Config speichern
    optimized_path = script_dir / "config_optimized.yaml"
    save_config(base_config, optimized_path)
    print(f"\nOptimierte Config gespeichert: {optimized_path}")
    
    # Frage ob Original ersetzen
    print("\n" + "-"*60)
    response = input("Original config.yaml mit optimierten Werten überschreiben? [y/N]: ")
    if response.lower() == 'y':
        save_config(base_config, config_path)
        print("✓ config.yaml aktualisiert")
    else:
        print("config.yaml unverändert")
    
    print("\nFertig!")


if __name__ == "__main__":
    main()
