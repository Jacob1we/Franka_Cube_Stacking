"""
CSV Episode Logger für Trajektorie-Analyse nach jeder Episode.

Speichert Daten im Matrix-Format:
- Reihen: Variablen (Datum, Zeit, Parameter, Phase-Metriken)
- Spalten: Einzelne Runs/Episodes

Verwendet CSV-Format (kompatibel mit Excel, keine Abhängigkeiten).

Controller Phasen (Pick & Place):
    Phase 0: Move EE above cube     [AIR]      - can be fast
    Phase 1: Lower EE to cube       [CRITICAL] - must be precise for gripping
    Phase 2: Wait for settle        [WAIT]     - fixed timing
    Phase 3: Close gripper          [GRIP]     - fixed timing
    Phase 4: Lift EE up             [AIR]      - can be fast
    Phase 5: Move to target XY      [AIR]      - can be fast
    Phase 6: Lower to place         [CRITICAL] - must be precise for placing
    Phase 7: Open gripper           [RELEASE]  - fixed timing
    Phase 8: Lift EE up             [AIR]      - can be fast
    Phase 9: Return to start        [AIR]      - can be fast
"""

import csv
from datetime import datetime
from pathlib import Path


class CSVEpisodeLogger:
    """
    Logger für Episode-Daten im CSV-Matrix-Format.
    
    Format: 
    - Reihen = Variablen (Datum, Zeit, Parameter, Phase-Daten)
    - Spalten = Einzelne Episodes/Runs
    
    Speichert alle Episode-Daten im Memory und schreibt Matrix am Ende.
    """
    
    # 10 Phasen des Pick&Place Controllers mit Beschreibung
    # Kategorie-Tags: [AIR] = schnell, [CRITICAL] = präzise, [WAIT/GRIP/RELEASE] = feste Zeit
    PHASES = [
        ("Phase 0", "MOVE_ABOVE_CUBE [AIR]"),      # Move EE above cube - can be fast
        ("Phase 1", "LOWER_TO_CUBE [CRITICAL]"),   # Lower EE to cube - must be precise for gripping
        ("Phase 2", "WAIT_SETTLE [WAIT]"),         # Wait for settle - fixed timing
        ("Phase 3", "CLOSE_GRIPPER [GRIP]"),       # Close gripper - fixed timing
        ("Phase 4", "LIFT_UP [AIR]"),              # Lift EE up - can be fast
        ("Phase 5", "MOVE_TO_TARGET [AIR]"),       # Move to target XY - can be fast
        ("Phase 6", "LOWER_TO_PLACE [CRITICAL]"),  # Lower to place - must be precise for placing
        ("Phase 7", "OPEN_GRIPPER [RELEASE]"),     # Open gripper - fixed timing
        ("Phase 8", "LIFT_AFTER_PLACE [AIR]"),     # Lift EE up - can be fast
        ("Phase 9", "RETURN_TO_START [AIR]"),      # Return to start - can be fast
    ]
    
    def __init__(self, output_dir: str = None, filename: str = None):
        """
        Initialisiert den CSV Logger im Matrix-Format.
        
        Args:
            output_dir: Verzeichnis für CSV-Datei (default: ./logs)
            filename: Name der CSV-Datei (default: episode_tracking.csv)
        """
        # output_dir kommt von logger.dataset_path, das endet mit 000000/
        # Gehe deshalb ein Verzeichnis nach oben
        self.output_dir = Path(output_dir).parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.filename = filename or "episode_tracking.csv"
        self.filepath = self.output_dir / self.filename
        
        # Sammele alle Episode-Daten im Memory (Spalten = Episodes, Reihen = Variablen)
        self.episodes_data = {}
    
    def log_episode(
        self,
        episode_seed: int,
        controller_params: dict,
        phase_data: dict,
        total_timesteps: int,
        total_time: float,
        validation_success: bool = True,
        notes: str = None,
    ):
        """
        Sammelt Episode-Daten für Matrix-Format (nicht sofort schreiben).
        
        Args:
            episode_seed: Eindeutige Episode-ID oder Seed
            controller_params: Dict mit Controller-Parametern
            phase_data: Dict mit Phase-Daten pro Phase-Index
            total_timesteps: Gesamtzahl Timesteps für die Episode
            total_time: Gesamtzeit der Episode in Sekunden
            validation_success: Ob Validierung erfolgreich war
            notes: Optionale Notizen
        """
        now = datetime.now()
        
        # Erstelle Spalte für diese Episode
        run_key = f"Run {len(self.episodes_data)}"
        
        column_data = {}
        
        # === METADATEN ===
        column_data["Datum"] = now.strftime("%d.%m.%Y")
        column_data["Zeit"] = now.strftime("%H:%M:%S")
        
        # === CONTROLLER PARAMETER ===
        column_data["trajectory_resolution"] = str(controller_params.get("trajectory_resolution", 1.0)).replace('.', ',')
        column_data["air_speed_multiplier"] = str(controller_params.get("air_speed_multiplier", 1.0)).replace('.', ',')
        column_data["height_adaptive_speed"] = "JA" if controller_params.get("height_adaptive_speed", False) else "NEIN"
        column_data["critical_height_threshold"] = str(controller_params.get("critical_height_threshold", 0.15)).replace('.', ',')
        column_data["critical_speed_factor"] = str(controller_params.get("critical_speed_factor", 0.25)).replace('.', ',')
        
        # === GESAMTE EPISODE-METRIKEN ===
        column_data["Gesamte Timesteps"] = str(total_timesteps)
        column_data["Gesamtzeit (s)"] = str(round(total_time, 4)).replace('.', ',')
        
        # === PHASE-DATEN ===
        for phase_idx, (phase_label, phase_name) in enumerate(self.PHASES):
            if phase_idx in phase_data:
                phase_info = phase_data[phase_idx]
                column_data[f"{phase_label}: {phase_name} - Wegpunkte"] = str(phase_info.get("waypoints", 0))
                column_data[f"{phase_label}: {phase_name} - Zeit (s)"] = str(round(phase_info.get("time", 0.0), 4)).replace('.', ',')
            else:
                column_data[f"{phase_label}: {phase_name} - Wegpunkte"] = "0"
                column_data[f"{phase_label}: {phase_name} - Zeit (s)"] = "0,0"
        
        # === VALIDIERUNG UND NOTIZEN ===
        column_data["Validierung erfolgreich"] = "✓ JA" if validation_success else "✗ NEIN"
        notes_safe = (notes or "").replace(';', ',')
        column_data["Notizen"] = notes_safe
        
        # Speichere Episode-Spalte
        self.episodes_data[run_key] = column_data
        
        print(f"✓ Episode {len(self.episodes_data)} gesammelt: {run_key}")
    
    def save_matrix(self):
        """
        Schreibt alle gesammelten Episode-Daten als Matrix ins CSV.
        
        Format:
        - Reihen = Variablen/Metriken
        - Spalten = Episodes (Run 0, Run 1, Run 2, ...)
        """
        if not self.episodes_data:
            print("⚠ Keine Episode-Daten zum Speichern vorhanden")
            return
        
        try:
            # Sammle alle eindeutigen Variablen-Namen (Reihen)
            all_rows = set()
            for column_data in self.episodes_data.values():
                all_rows.update(column_data.keys())
            
            # Sortiere Reihen in gewünschte Reihenfolge
            rows_order = [
                "Datum",
                "Zeit",
                "trajectory_resolution",
                "air_speed_multiplier",
                "height_adaptive_speed",
                "critical_height_threshold",
                "critical_speed_factor",
                "Gesamte Timesteps",
                "Gesamtzeit (s)",
            ]
            
            # Füge Phase-Daten in Reihenfolge hinzu
            for phase_idx, (phase_label, phase_name) in enumerate(self.PHASES):
                rows_order.append(f"{phase_label}: {phase_name} - Wegpunkte")
                rows_order.append(f"{phase_label}: {phase_name} - Zeit (s)")
            
            # Füge restliche Reihen hinzu
            rows_order.append("Validierung erfolgreich")
            rows_order.append("Notizen")
            
            # Schreibe Matrix
            with open(self.filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f, delimiter=';')
                
                # Header-Zeile: Variable Name + Episode-Namen
                column_keys = list(self.episodes_data.keys())
                header = ["Variable"] + column_keys
                writer.writerow(header)
                
                # Daten-Zeilen: Jede Reihe eine Variable
                for row_name in rows_order:
                    row = [row_name]
                    for run_key in column_keys:
                        value = self.episodes_data[run_key].get(row_name, "")
                        row.append(value)
                    writer.writerow(row)
            
            print(f"✓ Matrix-CSV gespeichert: {self.filepath}")
            print(f"  Format: {len(rows_order)} Reihen × {len(column_keys)+1} Spalten")
        except Exception as e:
            print(f"✗ FEHLER beim Speichern der Matrix: {e}")
