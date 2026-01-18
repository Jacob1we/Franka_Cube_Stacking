# Excel Episode Logger - Dokumentation

## Überblick

Der `ExcelEpisodeLogger` speichert Daten zu jeder abgeschlossenen Episode automatisch in einer Excel-Datei (`episode_tracking.xlsx`). Dies ermöglicht eine kontinuierliche Nachverfolgung und Analyse von:

- **Zeitstempel**: Datum und Uhrzeit jeder Episode
- **Controller Parameter**: Die verwendeten Konfigurationen (trajectory_resolution, air_speed_multiplier, etc.)
- **Phase-Daten**: Für alle 10 Phasen des Pick&Place Controllers:
  - Anzahl Trajektorienwegpunkte pro Phase
  - Benötigte Zeit pro Phase (in Sekunden)
  - Verwendeter Modifikator (dt-Faktor)
- **Gesamtmetriken**: Gesamtzahl Timesteps und Gesamtzeit der Episode
- **Validierungsstatus**: Ob die Episode erfolgreich war oder fehlgeschlagen ist
- **Notizen**: Zusätzliche Informationen (Seed, Umgebungs-Index, Grund für Fehler)

## Installieren von openpyxl

Der Excel-Logger benötigt das Paket `openpyxl`:

```bash
pip install openpyxl
```

## Struktur der Excel-Datei

### Header (Zeile 1)

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| A | Episode ID | Eindeutige ID der Episode |
| B | Datum | Datum (TT.MM.YYYY) |
| C | Zeit | Uhrzeit (HH:MM:SS) |
| D | trajectory_resolution | Base-Auflösung (Standard: 1.0, schneller: >1.0) |
| E | air_speed_multiplier | Multiplikator für AIR-Phasen (Standard: 1.0) |
| F | height_adaptive_speed | Dynamische Geschwindigkeit aktiviert? (JA/NEIN) |
| G | critical_height_threshold | Höhen-Schwelle in Metern |
| H | critical_speed_factor | Geschwindigkeitsfaktor in kritischer Zone |
| I | Gesamte Timesteps | Summe aller Timesteps in dieser Episode |
| J | Gesamtzeit (s) | Gesamtdauer in Sekunden |

### Phase-Daten (Spalten K - AG)

Für jede der 10 Phasen (K-M für Phase 0, N-P für Phase 1, etc.):

| Phase | Spalten | Beschreibung |
|-------|---------|-------------|
| 0: GRIP_OPEN | K, L, M | Wegpunkte, Zeit (s), Modifikator |
| 1: MOVE_DOWN_CRITICAL | N, O, P | Wegpunkte, Zeit (s), Modifikator |
| 2: GRIP_CLOSE | Q, R, S | Wegpunkte, Zeit (s), Modifikator |
| 3: MOVE_UP | T, U, V | Wegpunkte, Zeit (s), Modifikator |
| 4: MOVE_TO_STACK | W, X, Y | Wegpunkte, Zeit (s), Modifikator |
| 5: MOVE_DOWN_CRITICAL_STK | Z, AA, AB | Wegpunkte, Zeit (s), Modifikator |
| 6: WAIT | AC, AD, AE | Wegpunkte, Zeit (s), Modifikator |
| 7: GRIP_OPEN_STK | AF, AG, AH | Wegpunkte, Zeit (s), Modifikator |
| 8: MOVE_UP_STK | AI, AJ, AK | Wegpunkte, Zeit (s), Modifikator |
| 9: MOVE_AWAY | AL, AM, AN | Wegpunkte, Zeit (s), Modifikator |

### Abschluss-Spalten (AO, AP)

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| AO | Validierung erfolgreich | ✓ JA oder ✗ NEIN |
| AP | Notizen | Beliebige Zusatzinformationen |

## Verwendung im Code

### Initialisierung

```python
from excel_episode_logger import ExcelEpisodeLogger

# Erstelle oder lade Excel-Logger
excel_logger = ExcelEpisodeLogger(
    output_dir="./logs",  # Verzeichnis für die Datei
    filename="episode_tracking.xlsx"  # Name der Datei
)
```

### Episode Logging

Nach jeder abgeschlossenen Episode:

```python
# Sammle Controller-Parameter
controller_params = {
    "trajectory_resolution": controller.trajectory_resolution,
    "air_speed_multiplier": controller.air_speed_multiplier,
    "height_adaptive_speed": controller.height_adaptive_speed,
    "critical_height_threshold": controller.critical_height_threshold,
    "critical_speed_factor": controller.critical_speed_factor,
}

# Sammle Phase-Daten
phase_data = {
    0: {"waypoints": 42, "time": 0.7, "modifier": 1.0},
    1: {"waypoints": 55, "time": 0.92, "modifier": 1.0},
    2: {"waypoints": 38, "time": 0.63, "modifier": 1.0},
    # ... weitere 7 Phasen
}

# Schreibe Episode
excel_logger.log_episode(
    episode_id=1,
    controller_params=controller_params,
    phase_data=phase_data,
    total_timesteps=483,
    total_time=8.05,
    validation_success=True,
    notes="Seed: 12345, Env: 0"
)
```

## Formatierung

- **Header-Zeile**: Dunkelblau mit weißem Text, fett, zentriert
- **Erfolgreiche Episoden**: Grüner Hintergrund (#D4EDDA)
- **Fehlgeschlagene Episoden**: Roter Hintergrund (#F8D7DA)
- **Spaltenbreiten**: Automatisch angepasst für Lesbarkeit
- **Header eingefrioren**: Erste Zeile bleibt sichtbar beim Scrollen

## Integration in fcs_main_parallel.py

Die Integration erfolgt automatisch:

1. **Initialisierung** (nach Logger-Setup):
   ```python
   excel_logger = ExcelEpisodeLogger(
       output_dir=str(logger.dataset_path),
       filename="episode_tracking.xlsx"
   )
   ```

2. **Nach erfolgreicher Episode**:
   - Controller-Parameter werden extrahiert
   - Phase-Daten werden aus Episode-Länge berechnet
   - `excel_logger.log_episode()` wird aufgerufen

3. **Nach fehlgeschlagener Episode**:
   - Episode wird auch mit `validation_success=False` geloggt
   - Grund für Fehler wird in Notizen gespeichert
   - Hilft bei Debugging und Statistik-Tracking

## Analyse der Daten

Mit den Excel-Daten können Sie:

1. **Parameter-Effekte visualisieren**: Vergleichen Sie trajectory_resolution und air_speed_multiplier auf Zeitdauern
2. **Phase-Bottlenecks identifizieren**: Welche Phasen sind langsam?
3. **Konsistenz überprüfen**: Sind die Zeiten für gleiche Parameter konsistent?
4. **Erfolgsrate analysieren**: Wie viele Episoden sind mit welchen Parametern erfolgreich?
5. **Trends ermitteln**: Werden die Episoden schneller oder langsamer im Zeitverlauf?

## Tipps

- Die Datei wird nach jeder Episode automatisch gespeichert
- Sie können die Datei öffnen, während die Datensammlung läuft (Excel zeigt aktuelle Daten an)
- Für große Datenmengen empfiehlt sich die Verwendung von Pivot-Tabellen in Excel
- Die numerischen Spalten sind auf 4 Dezimalstellen formatiert

## Fehlerbehebung

Falls Sie folgende Fehler sehen:

### "ImportError: No module named 'openpyxl'"
```bash
pip install openpyxl
```

### "File is locked" (Datei ist gesperrt)
- Excel-Datei ist in Excel geöffnet
- Speichern Sie die Datei oder schließen Sie sie, bevor die nächste Episode geloggt wird
- Der Logger versucht die Datei automatisch zu speichern nach jeder Episode

### Laufwerk-Fehler bei Dateispeicherung
- Stellen Sie sicher, dass das output_dir existiert und beschreibbar ist
- Überprüfen Sie Schreibrechte im Verzeichnis

