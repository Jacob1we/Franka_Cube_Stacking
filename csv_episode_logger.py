"""
CSV Episode Logger für Trajektorie-Analyse nach jeder Episode.

Speichert für jede Episode:
- Datum und Zeit
- Controller Parameter
- Trajektorie-Daten pro Phase (Anzahl Wegpunkte, Zeitdauer, Modifikatoren)

Verwendet CSV-Format (kompatibel mit Excel, keine Abhängigkeiten).
"""

import csv
from datetime import datetime
from pathlib import Path


class CSVEpisodeLogger:
    """
    Logger für Episode-Daten im CSV-Format.
    
    Speichert nach jeder Episode:
    - Zeitstempel (Datum, Zeit)
    - Controller Parameter (trajectory_resolution, air_speed_multiplier, etc.)
    - Phase-Daten (10 Phasen: Wegpunkte, Zeit, Modifikatoren)
    """
    
    # 10 Phasen des Pick&Place Controllers
    PHASES = [
        "GRIP_OPEN",              # 0
        "MOVE_DOWN_CRITICAL",     # 1
        "GRIP_CLOSE",             # 2
        "MOVE_UP",                # 3
        "MOVE_TO_STACK",          # 4
        "MOVE_DOWN_CRITICAL_STK", # 5
        "WAIT",                   # 6
        "GRIP_OPEN_STK",          # 7
        "MOVE_UP_STK",            # 8
        "MOVE_AWAY",              # 9
    ]
    
    def __init__(self, output_dir: str = None, filename: str = None):
        """
        Initialisiert den CSV Logger.
        
        Args:
            output_dir: Verzeichnis für CSV-Datei (default: ./logs)
            filename: Name der CSV-Datei (default: episode_tracking.csv)
        """
        self.output_dir = Path(output_dir+"/..")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.filename = filename or "episode_tracking.csv"
        self.filepath = self.output_dir / self.filename
        
        # Überprüfe ob Datei bereits existiert
        self.file_exists = self.filepath.exists()
        
        # Initialisiere Header falls neue Datei
        if not self.file_exists:
            self._create_header()
    
    def _get_header(self) -> list:
        """Erstellt Header-Liste mit allen Spalten."""
        headers = [
            "Episode ID",
            "Phase",
            "Phase Name",
            "Datum",
            "Zeit",
            "trajectory_resolution",
            "air_speed_multiplier",
            "height_adaptive_speed",
            "critical_height_threshold",
            "critical_speed_factor",
            "Gesamte Timesteps",
            "Gesamtzeit (s)",
            "Wegpunkte",
            "Zeit (s)",
            "Modifikator",
            "Validierung erfolgreich",
            "Notizen",
        ]
        
        return headers
    
    def _create_header(self):
        """Schreibt Header-Zeile in neue CSV-Datei."""
        headers = self._get_header()
        with open(self.filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(headers)
    
    def log_episode(
        self,
        episode_seed : int,
        controller_params: dict,
        phase_data: dict,
        total_timesteps: int,
        total_time: float,
        validation_success: bool = True,
        notes: str = None,
    ):
        """
        Loggt eine abgeschlossene Episode in CSV - eine Zeile pro Phase.
        
        Args:
            episode_seed: Eindeutige Episode-ID
            controller_params: Dict mit Controller-Parametern
            phase_data: Dict mit Phase-Daten pro Phase-Index
            total_timesteps: Gesamtzahl Timesteps für die Episode
            total_time: Gesamtzeit der Episode in Sekunden
            validation_success: Ob Validierung erfolgreich war
            notes: Optionale Notizen
        """
        now = datetime.now()
        
        # Basis-Daten die für alle Phasen gleich sind
        base_data = [
            now.strftime("%d.%m.%Y"),
            now.strftime("%H:%M:%S"),
            str(controller_params.get("trajectory_resolution", 1.0)).replace('.', ','),
            str(controller_params.get("air_speed_multiplier", 1.0)).replace('.', ','),
            "JA" if controller_params.get("height_adaptive_speed", False) else "NEIN",
            str(controller_params.get("critical_height_threshold", 0.15)).replace('.', ','),
            str(controller_params.get("critical_speed_factor", 0.25)).replace('.', ','),
            str(total_timesteps),
            str(round(total_time, 4)).replace('.', ','),
        ]
        
        # Schreibe eine Zeile pro Phase
        for phase_idx in range(len(self.PHASES)):
            phase_name = self.PHASES[phase_idx]
            
            row = [
                str(episode_seed),
                str(phase_idx),
                phase_name,
            ]
            row.extend(base_data)
            
            # Phase-Daten
            if phase_idx in phase_data:
                phase_info = phase_data[phase_idx]
                row.append(str(phase_info.get("waypoints", 0)))
                row.append(str(round(phase_info.get("time", 0.0), 4)).replace('.', ','))
                row.append(str(round(phase_info.get("modifier", 1.0), 4)).replace('.', ','))
            else:
                # Leere Phasen-Daten
                row.extend(["0", "0,0", "1,0"])
            
            # Validierung
            row.append("✓ JA" if validation_success else "✗ NEIN")
            
            # Notizen (Kommas escapen, falls nötig)
            notes_safe = (notes or "").replace(';', ',')
            row.append(notes_safe)
            
            # Schreibe Zeile
            try:
                with open(self.filepath, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE, escapechar='\\')
                    writer.writerow(row)
            except Exception as e:
                print(f"FEHLER beim Schreiben in CSV: {e}")
                print(f"  Datei: {self.filepath}")
                print(f"  Phase {phase_idx} ({phase_name}): {row}")
