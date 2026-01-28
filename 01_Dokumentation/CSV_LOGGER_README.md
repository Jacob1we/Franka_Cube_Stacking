# CSV Episode Logger - Dokumentation

## Überblick

Der `CSVEpisodeLogger` speichert Daten zu jeder abgeschlossenen Episode automatisch in einer CSV-Datei (`episode_tracking.csv`). Dies ermöglicht eine kontinuierliche Nachverfolgung und Analyse von:

- **Zeitstempel**: Datum und Uhrzeit jeder Episode
- **Controller Parameter**: Die verwendeten Konfigurationen (trajectory_resolution, air_speed_multiplier, etc.)
- **Phase-Daten**: Für alle 10 Phasen des Pick&Place Controllers:
  - Anzahl Trajektorienwegpunkte pro Phase
  - Benötigte Zeit pro Phase (in Sekunden)
  - Verwendeter Modifikator (dt-Faktor)
- **Gesamtmetriken**: Gesamtzahl Timesteps und Gesamtzeit der Episode
- **Validierungsstatus**: Ob die Episode erfolgreich war oder fehlgeschlagen ist
- **Notizen**: Zusätzliche Informationen (Seed, Umgebungs-Index, Grund für Fehler)

## Vorteile

✅ **Keine Abhängigkeiten**: CSV ist ein Standard-Format, benötigt nur `csv` (Standard Library)  
✅ **Kompatibilität**: Öffnet direkt in Excel, LibreOffice, Google Sheets  
✅ **Einfach**: Reine Textdatei, leicht zu bearbeiten und weiterzuverarbeiten  
✅ **Effizient**: Kompakt und schnell zum Schreiben  
✅ **UTF-8 mit BOM**: Excel erkennt Umlaute korrekt  

## Struktur der CSV-Datei

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

Für jede der 10 Phasen (3 Spalten pro Phase):

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

### Abschluss-Spalten

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| AO | Validierung erfolgreich | ✓ JA oder ✗ NEIN |
| AP | Notizen | Beliebige Zusatzinformationen |

## CSV-Format Details

- **Trennzeichen**: Semikolon (`;`) - Excel öffnet es korrekt
- **Encoding**: UTF-8 mit BOM - Umlaute werden korrekt angezeigt
- **Dezimaltrennzeichen**: Punkt (`.`) - international kompatibel
- **Spalten-Anzahl**: 47 Spalten (3 + 8 + 30 + 2)

## Beispiel einer Zeile

```
1;18.01.2026;14:40:25;1.0;4.0;JA;0.05;0.8;483;8.05;42;0.7;1.0;55;0.92;1.0;...;✓ JA;Seed: 12345, Env: 0
```

## Integration in fcs_main_parallel.py

Die Integration erfolgt automatisch:

1. **Initialisierung** (nach Logger-Setup):
   ```python
   csv_logger = CSVEpisodeLogger(
       output_dir=str(logger.dataset_path),
       filename="episode_tracking.csv"
   )
   ```

2. **Nach erfolgreicher Episode**:
   - Controller-Parameter werden extrahiert
   - Phase-Daten werden aus Episode-Länge berechnet
   - `csv_logger.log_episode()` wird aufgerufen

3. **Nach fehlgeschlagener Episode**:
   - Episode wird auch mit `validation_success=False` geloggt
   - Grund für Fehler wird in Notizen gespeichert
   - Hilft bei Debugging und Statistik-Tracking

## In Excel öffnen

1. Öffne Excel
2. Datei > Öffnen > `episode_tracking.csv`
3. Im Text-in-Spalten-Dialog:
   - Trennzeichen: Semikolon (`;`)
   - Codierung: UTF-8
4. Fertig! CSV wird automatisch in Spalten aufgeteilt

Alternativ: Drag & Drop der `.csv` in Excel öffnet die Datei direkt.

## Analyse der Daten

Mit den CSV-Daten können Sie:

1. **Parameter-Effekte visualisieren**: Vergleichen Sie trajectory_resolution und air_speed_multiplier auf Zeitdauern
2. **Phase-Bottlenecks identifizieren**: Welche Phasen sind langsam?
3. **Konsistenz überprüfen**: Sind die Zeiten für gleiche Parameter konsistent?
4. **Erfolgsrate analysieren**: Wie viele Episoden sind mit welchen Parametern erfolgreich?
5. **Trends ermitteln**: Werden die Episoden schneller oder langsamer im Zeitverlauf?

### Beispiel: Pivot-Tabelle in Excel

1. Daten > Pivot Table erstellen
2. Row Labels: `trajectory_resolution`
3. Values: `Gesamtzeit (s)` (Durchschnitt)
4. Sie sehen sofort, wie dieser Parameter die Gesamtzeit beeinflusst

## Verwendung in Python (pandas)

```python
import pandas as pd

# CSV laden
df = pd.read_csv('episode_tracking.csv', sep=';', decimal='.')

# Erfolgsrate pro Parameter
success_rate = df.groupby('trajectory_resolution')['Validierung erfolgreich'].apply(
    lambda x: (x == '✓ JA').sum() / len(x) * 100
)
print(success_rate)

# Durchschnittliche Zeit pro Parameter
avg_time = df.groupby('air_speed_multiplier')['Gesamtzeit (s)'].mean()
print(avg_time)
```

## Fehlerbehebung

### CSV wird nicht korrekt in Excel angezeigt

- **Problem**: Spalten sind nicht aufgeteilt
- **Lösung**: 
  1. Datei > Text-in-Spalten
  2. Trennzeichen: Semikolon (`;`)
  3. OK

### Umlaute werden nicht korrekt angezeigt

- **Problem**: Umlaute sehen aus wie `?`
- **Lösung**: CSV-Datei hat UTF-8-BOM. Falls nicht:
  - Datei mit Notepad öffnen
  - Speichern unter > Codierung: UTF-8-BOM wählen
  - Speichern

### Datei ist gesperrt

- **Problem**: "Datei wird verwendet und kann nicht bearbeitet werden"
- **Lösung**: 
  - Simulation noch laufend? Warten oder unterbrechen
  - Excel hat Datei offen? Schließen oder Speichern
  - Der Logger speichert nach jeder Episode automatisch

## Tipps

- Die Datei wird nach jeder Episode automatisch gespeichert
- Sie können die Datei öffnen, während die Datensammlung läuft (einfach F5 drücken zum Aktualisieren)
- Für große Datenmengen empfiehlt sich die Verwendung von Pivot-Tabellen in Excel
- CSV kann auch mit Bash/PowerShell verarbeitet werden: `cat episode_tracking.csv | grep "✓ JA" | wc -l`

