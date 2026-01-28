"""
Parameter Optimizer für Franka Cube Stacking Controller

Dieses Skript optimiert automatisch die Controller-Parameter, um die minimale
Anzahl an Timesteps pro Episode zu erreichen, ohne dass das Stacking fehlschlägt.

Strategie:
1. Starte mit konservativen (langsamen) Parametern
2. Erhöhe jeden Parameter schrittweise
3. Teste mit mehreren Seeds pro Parameterkombination
4. Wenn Fehlerrate > Schwellenwert: Parameter zurücksetzen
5. Fahre mit nächstem Parameter fort

Verwendung:
    Aktiviere Isaac Sim Conda Environment, dann:
    > python parameter_optimizer.py
    
    Oder mit Custom-Config:
    > python parameter_optimizer.py --config my_config.yaml --episodes 5
"""

import os
import sys
import yaml
import json
import copy
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

# ============================================================================
# KONFIGURATION
# ============================================================================

# Parameter-Definitionen mit Suchbereichen
PARAMETER_CONFIG = {
    "trajectory_resolution": {
        "path": ["controller", "trajectory_resolution"],
        "start": 1.0,           # Konservativ (langsam)
        "end": 5.0,             # Aggressiv (schnell)
        "step": 0.5,            # Schrittgröße
        "description": "Beeinflusst ALLE Phasen gleichmäßig",
    },
    "air_speed_multiplier": {
        "path": ["controller", "air_speed_multiplier"],
        "start": 1.0,
        "end": 10.0,
        "step": 1.0,
        "description": "Beeinflusst nur AIR Phasen (0, 4, 5, 8, 9)",
    },
    "critical_speed_factor": {
        "path": ["controller", "critical_speed_factor"],
        "start": 0.2,           # Langsam in kritischer Zone
        "end": 1.0,             # Normal in kritischer Zone
        "step": 0.1,
        "description": "Geschwindigkeit in kritischer Höhe (0=sehr langsam, 1=normal)",
    },
    "critical_height_threshold": {
        "path": ["controller", "critical_height_threshold"],
        "start": 0.10,          # Große kritische Zone
        "end": 0.02,            # Kleine kritische Zone
        "step": -0.02,          # Negative Schritte (wird kleiner)
        "description": "Höhe unter der als kritisch gilt (m)",
    },
}

# Optimierungseinstellungen
DEFAULT_EPISODES_PER_TEST = 3       # Episoden pro Parameterkombination
DEFAULT_MAX_FAILURE_RATE = 0.0      # Max erlaubte Fehlerrate (0 = keine Fehler erlaubt)
DEFAULT_SEEDS = [42, 123, 456]      # Test-Seeds für Reproduzierbarkeit

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Konfiguriert Logging für Konsole und Datei."""
    log = logging.getLogger("ParameterOptimizer")
    log.setLevel(logging.DEBUG)
    
    # Konsole Handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(console)
    
    # Datei Handler
    file_handler = logging.FileHandler(output_dir / "optimization.log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    log.addHandler(file_handler)
    
    return log


# ============================================================================
# CONFIG MANAGEMENT
# ============================================================================

def load_config(config_path: Path) -> dict:
    """Lädt die YAML-Konfiguration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: Path):
    """Speichert die YAML-Konfiguration."""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_nested_value(config: dict, path: List[str]):
    """Holt einen verschachtelten Wert aus der Config."""
    value = config
    for key in path:
        value = value[key]
    return value


def set_nested_value(config: dict, path: List[str], value):
    """Setzt einen verschachtelten Wert in der Config."""
    for key in path[:-1]:
        config = config[key]
    config[path[-1]] = value


# ============================================================================
# SIMULATION RUNNER (Isaac Sim)
# ============================================================================

class SimulationRunner:
    """
    Führt einzelne Episoden mit gegebenen Parametern aus.
    
    Diese Klasse kapselt die Isaac Sim Logik und ermöglicht
    das Testen verschiedener Parameterkombinationen.
    """
    
    def __init__(self, config_path: Path, log: logging.Logger):
        self.config_path = config_path
        self.log = log
        self.initialized = False
        
        # Isaac Sim Objekte (werden bei Bedarf initialisiert)
        self.simulation_app = None
        self.world = None
        self.env = None
        self.controller = None
        
    def initialize(self):
        """Initialisiert Isaac Sim (einmalig)."""
        if self.initialized:
            return
            
        self.log.info("Initialisiere Isaac Sim...")
        
        # Isaac Sim importieren
        from isaacsim import SimulationApp
        
        # Headless für schnellere Ausführung
        self.simulation_app = SimulationApp({
            "headless": True,
            "width": 256,
            "height": 256,
        })
        
        # Weitere Imports nach SimulationApp
        from isaacsim.core.api import World
        
        self.world = World(stage_units_in_meters=1.0)
        self.initialized = True
        self.log.info("Isaac Sim initialisiert")
    
    def run_episode(self, config: dict, seed: int) -> Tuple[bool, int, str]:
        """
        Führt eine einzelne Episode mit der gegebenen Konfiguration aus.
        
        Args:
            config: Konfigurationsdict mit Controller-Parametern
            seed: Seed für Domain Randomization
            
        Returns:
            Tuple[success, timesteps, reason]:
                - success: True wenn Stacking erfolgreich
                - timesteps: Anzahl der Timesteps
                - reason: Grund bei Fehler, sonst ""
        """
        if not self.initialized:
            self.initialize()
        
        # Importiere Environment-Klasse
        from fcs_main_parallel import FrankaCubeStackEnv, validate_stacking
        
        # Controller-Parameter aus Config extrahieren
        trajectory_resolution = config["controller"]["trajectory_resolution"]
        air_speed_multiplier = config["controller"]["air_speed_multiplier"]
        height_adaptive_speed = config["controller"]["height_adaptive_speed"]
        critical_height_threshold = config["controller"]["critical_height_threshold"]
        critical_speed_factor = config["controller"]["critical_speed_factor"]
        
        try:
            # Environment erstellen (falls noch nicht vorhanden oder Reset nötig)
            if self.env is None:
                self.env = FrankaCubeStackEnv(
                    world=self.world,
                    env_idx=0,
                    trajectory_resolution=trajectory_resolution,
                    air_speed_multiplier=air_speed_multiplier,
                    height_adaptive_speed=height_adaptive_speed,
                    critical_height_threshold=critical_height_threshold,
                    critical_speed_factor=critical_speed_factor,
                )
                self.world.reset()
                
                # Warte auf Initialisierung
                for _ in range(10):
                    self.simulation_app.update()
                    self.world.step(render=False)
                
                self.controller = self.env.setup_post_load()
            else:
                # Controller Reset mit neuen Parametern
                self.controller.reset()
            
            # Domain Randomization
            cube_pos, target_pos = self.env.domain_randomization(seed)
            
            # Episode ausführen
            timesteps = 0
            max_timesteps = 2000  # Sicherheitslimit
            
            while not self.controller.is_done() and timesteps < max_timesteps:
                self.simulation_app.update()
                
                # Beobachtungen sammeln und Action berechnen
                obs = self.world.get_observations()
                action = self.controller.forward(observations=obs)
                
                # Action ausführen
                articulation = self.env.franka.get_articulation_controller()
                articulation.apply_action(action)
                
                self.world.step(render=False)
                timesteps += 1
            
            # Validierung
            success, reason = validate_stacking(self.env.task, target_pos)
            
            return success, timesteps, reason if not success else ""
            
        except Exception as e:
            self.log.error(f"Fehler bei Episode: {e}")
            import traceback
            self.log.debug(traceback.format_exc())
            return False, 0, str(e)
    
    def cleanup(self):
        """Räumt Isaac Sim auf."""
        if self.simulation_app is not None:
            try:
                self.simulation_app.close()
            except:
                pass


# ============================================================================
# PARAMETER OPTIMIZER
# ============================================================================

class ParameterOptimizer:
    """
    Optimiert Controller-Parameter für minimale Timesteps.
    
    Strategie: Greedy Search
    - Optimiere einen Parameter nach dem anderen
    - Für jeden Parameter: Erhöhe schrittweise bis Fehler auftreten
    - Behalte den letzten erfolgreichen Wert
    """
    
    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        episodes_per_test: int = DEFAULT_EPISODES_PER_TEST,
        max_failure_rate: float = DEFAULT_MAX_FAILURE_RATE,
        seeds: List[int] = None,
        dry_run: bool = False,
    ):
        self.config_path = config_path
        self.output_dir = output_dir
        self.episodes_per_test = episodes_per_test
        self.max_failure_rate = max_failure_rate
        self.seeds = seeds or DEFAULT_SEEDS[:episodes_per_test]
        self.dry_run = dry_run
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = setup_logging(output_dir)
        
        # Ergebnisse speichern
        self.results: List[Dict] = []
        self.best_config: Dict = None
        self.best_timesteps: float = float('inf')
        
        # Simulation Runner (nur wenn nicht dry_run)
        self.runner = None if dry_run else SimulationRunner(config_path, self.log)
    
    def test_parameters(self, config: dict) -> Tuple[float, float, List[int]]:
        """
        Testet eine Parameterkombination mit mehreren Seeds.
        
        Returns:
            Tuple[avg_timesteps, failure_rate, timesteps_list]
        """
        timesteps_list = []
        failures = 0
        
        for seed in self.seeds:
            if self.dry_run:
                # Dummy-Ergebnisse für Dry-Run
                success = np.random.random() > 0.1
                timesteps = int(np.random.normal(500, 100))
            else:
                success, timesteps, reason = self.runner.run_episode(config, seed)
            
            if success:
                timesteps_list.append(timesteps)
            else:
                failures += 1
                self.log.debug(f"  Seed {seed}: FEHLGESCHLAGEN - {reason if not self.dry_run else 'dry-run'}")
        
        failure_rate = failures / len(self.seeds)
        avg_timesteps = np.mean(timesteps_list) if timesteps_list else float('inf')
        
        return avg_timesteps, failure_rate, timesteps_list
    
    def optimize_parameter(
        self,
        param_name: str,
        param_config: dict,
        base_config: dict,
    ) -> Tuple[float, dict]:
        """
        Optimiert einen einzelnen Parameter.
        
        Returns:
            Tuple[best_value, updated_config]
        """
        self.log.info(f"\n{'='*60}")
        self.log.info(f"Optimiere: {param_name}")
        self.log.info(f"  Beschreibung: {param_config['description']}")
        self.log.info(f"  Bereich: {param_config['start']} → {param_config['end']} (Step: {param_config['step']})")
        self.log.info('='*60)
        
        path = param_config["path"]
        start = param_config["start"]
        end = param_config["end"]
        step = param_config["step"]
        
        # Generiere Testwerte
        if step > 0:
            values = np.arange(start, end + step/2, step)
        else:
            values = np.arange(start, end + step/2, step)
        
        best_value = start
        best_timesteps = float('inf')
        last_successful_value = start
        
        for value in values:
            # Config mit neuem Parameterwert
            test_config = copy.deepcopy(base_config)
            set_nested_value(test_config, path, float(value))
            
            self.log.info(f"\n  Teste {param_name} = {value:.3f}")
            
            # Teste diese Parameterkombination
            avg_timesteps, failure_rate, timesteps_list = self.test_parameters(test_config)
            
            # Ergebnis speichern
            result = {
                "parameter": param_name,
                "value": float(value),
                "avg_timesteps": float(avg_timesteps),
                "failure_rate": float(failure_rate),
                "timesteps_list": timesteps_list,
                "config": {k: get_nested_value(test_config, v["path"]) 
                          for k, v in PARAMETER_CONFIG.items()},
            }
            self.results.append(result)
            
            # Ausgabe
            if failure_rate <= self.max_failure_rate:
                status = "✓"
                self.log.info(f"    {status} Erfolg: {avg_timesteps:.1f} Timesteps (Fehlerrate: {failure_rate*100:.0f}%)")
                
                last_successful_value = value
                
                if avg_timesteps < best_timesteps:
                    best_timesteps = avg_timesteps
                    best_value = value
            else:
                status = "✗"
                self.log.info(f"    {status} Fehlgeschlagen: Fehlerrate {failure_rate*100:.0f}% > {self.max_failure_rate*100:.0f}%")
                self.log.info(f"    → Stoppe bei {param_name} = {last_successful_value:.3f}")
                break
        
        self.log.info(f"\n  Bester Wert für {param_name}: {best_value:.3f} ({best_timesteps:.1f} Timesteps)")
        
        # Config mit bestem Wert zurückgeben
        result_config = copy.deepcopy(base_config)
        set_nested_value(result_config, path, float(best_value))
        
        return best_value, result_config
    
    def optimize(self) -> dict:
        """
        Führt die vollständige Optimierung durch.
        
        Returns:
            Optimierte Konfiguration
        """
        self.log.info("\n" + "="*70)
        self.log.info("  PARAMETER OPTIMIZER - Start")
        self.log.info("="*70)
        self.log.info(f"Config: {self.config_path}")
        self.log.info(f"Output: {self.output_dir}")
        self.log.info(f"Episodes pro Test: {self.episodes_per_test}")
        self.log.info(f"Max Fehlerrate: {self.max_failure_rate*100:.0f}%")
        self.log.info(f"Seeds: {self.seeds}")
        self.log.info(f"Dry-Run: {self.dry_run}")
        
        # Basis-Konfiguration laden
        base_config = load_config(self.config_path)
        current_config = copy.deepcopy(base_config)
        
        # Initiale Werte loggen
        self.log.info("\nInitiale Controller-Parameter:")
        for name, pconfig in PARAMETER_CONFIG.items():
            value = get_nested_value(current_config, pconfig["path"])
            self.log.info(f"  {name}: {value}")
        
        # Baseline messen
        self.log.info("\n" + "-"*60)
        self.log.info("Baseline-Messung mit aktuellen Parametern...")
        baseline_timesteps, baseline_failure, _ = self.test_parameters(current_config)
        self.log.info(f"Baseline: {baseline_timesteps:.1f} Timesteps, Fehlerrate: {baseline_failure*100:.0f}%")
        
        # Parameter einzeln optimieren
        optimized_values = {}
        
        for param_name, param_config in PARAMETER_CONFIG.items():
            best_value, current_config = self.optimize_parameter(
                param_name, param_config, current_config
            )
            optimized_values[param_name] = best_value
        
        # Finale Messung
        self.log.info("\n" + "="*60)
        self.log.info("FINALE MESSUNG mit optimierten Parametern...")
        self.log.info("="*60)
        
        final_timesteps, final_failure, final_list = self.test_parameters(current_config)
        
        self.log.info(f"\nOptimierte Parameter:")
        for name, pconfig in PARAMETER_CONFIG.items():
            value = get_nested_value(current_config, pconfig["path"])
            self.log.info(f"  {name}: {value:.3f}")
        
        self.log.info(f"\nErgebnis:")
        self.log.info(f"  Baseline:   {baseline_timesteps:.1f} Timesteps")
        self.log.info(f"  Optimiert:  {final_timesteps:.1f} Timesteps")
        self.log.info(f"  Verbesserung: {(1 - final_timesteps/baseline_timesteps)*100:.1f}%")
        
        # Ergebnisse speichern
        self._save_results(current_config, optimized_values, baseline_timesteps, final_timesteps)
        
        return current_config
    
    def _save_results(self, config: dict, optimized_values: dict, baseline: float, final: float):
        """Speichert alle Ergebnisse."""
        
        # Optimierte Config speichern
        optimized_config_path = self.output_dir / "config_optimized.yaml"
        save_config(config, optimized_config_path)
        self.log.info(f"\nOptimierte Config gespeichert: {optimized_config_path}")
        
        # Detaillierte Ergebnisse als JSON
        results_path = self.output_dir / "optimization_results.json"
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "baseline_timesteps": baseline,
            "final_timesteps": final,
            "improvement_percent": (1 - final/baseline) * 100 if baseline > 0 else 0,
            "optimized_values": optimized_values,
            "all_tests": self.results,
        }
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        self.log.info(f"Detaillierte Ergebnisse: {results_path}")
        
        # Zusammenfassung als CSV
        csv_path = self.output_dir / "optimization_summary.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Parameter;Wert;Avg_Timesteps;Fehlerrate\n")
            for r in self.results:
                f.write(f"{r['parameter']};{r['value']:.3f};{r['avg_timesteps']:.1f};{r['failure_rate']*100:.0f}%\n")
        self.log.info(f"CSV-Zusammenfassung: {csv_path}")
    
    def cleanup(self):
        """Räumt auf."""
        if self.runner:
            self.runner.cleanup()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimiert Controller-Parameter für minimale Timesteps"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Pfad zur config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=DEFAULT_EPISODES_PER_TEST,
        help=f"Episoden pro Parameterkombination (default: {DEFAULT_EPISODES_PER_TEST})"
    )
    parser.add_argument(
        "--max-failure", "-f",
        type=float,
        default=DEFAULT_MAX_FAILURE_RATE,
        help=f"Max erlaubte Fehlerrate 0.0-1.0 (default: {DEFAULT_MAX_FAILURE_RATE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output-Verzeichnis (default: optimization_TIMESTAMP)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-Run ohne Isaac Sim (für Tests)"
    )
    
    args = parser.parse_args()
    
    # Pfade
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / f"optimization_{timestamp}"
    
    # Prüfe ob Config existiert
    if not config_path.exists():
        print(f"FEHLER: Config nicht gefunden: {config_path}")
        sys.exit(1)
    
    # Optimizer erstellen und ausführen
    optimizer = ParameterOptimizer(
        config_path=config_path,
        output_dir=output_dir,
        episodes_per_test=args.episodes,
        max_failure_rate=args.max_failure,
        dry_run=args.dry_run,
    )
    
    try:
        optimized_config = optimizer.optimize()
        
        print("\n" + "="*70)
        print("  OPTIMIERUNG ABGESCHLOSSEN")
        print("="*70)
        print(f"Ergebnisse in: {output_dir}")
        print("\nOptimierte Parameter:")
        for name, pconfig in PARAMETER_CONFIG.items():
            value = get_nested_value(optimized_config, pconfig["path"])
            print(f"  {name}: {value:.3f}")
        
    except KeyboardInterrupt:
        print("\n\nOptimierung abgebrochen.")
    finally:
        optimizer.cleanup()


if __name__ == "__main__":
    main()
