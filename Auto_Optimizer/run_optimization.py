#!/usr/bin/env python
"""
Praktisches Parameter-Optimierungsskript für Franka Cube Stacking

Dieses Skript testet verschiedene Controller-Parameter direkt in Isaac Sim
und findet die schnellste Konfiguration, die noch zuverlässig funktioniert.

WICHTIG: Muss in der Isaac Sim Python-Umgebung ausgeführt werden!

Verwendung:
    # In Isaac Sim Terminal:
    python run_optimization.py
    
    # Mit Optionen:
    python run_optimization.py --episodes 5 --verbose

Workflow:
1. Startet mit konservativen Parametern
2. Testet jeden Parameter einzeln mit steigenden Werten
3. Stoppt wenn Fehler auftreten
4. Speichert die optimierten Parameter

Ausgabe:
- config_optimized.yaml: Beste gefundene Parameter
- optimization_log.txt: Detailliertes Log
- optimization_results.csv: Tabellarische Ergebnisse
"""

import sys
import os

# ============================================================================
# ISAAC SIM INITIALISIERUNG (muss ZUERST passieren!)
# ============================================================================

print("="*60)
print("  Parameter Optimizer für Franka Cube Stacking")
print("="*60)
print("\nInitialisiere Isaac Sim...")

from isaacsim import SimulationApp

# Headless für schnellere Ausführung (kein GUI)
CONFIG = {
    "headless": True,
    "width": 256,
    "height": 256,
    "anti_aliasing": 0,
}

simulation_app = SimulationApp(CONFIG)
print("✓ Isaac Sim initialisiert")

# ============================================================================
# JETZT können wir andere Module importieren
# ============================================================================

import yaml
import copy
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Isaac Sim Module
from isaacsim.core.api import World

# Lokale Module (nach Isaac Sim Init!)
# Diese werden direkt aus fcs_main_parallel.py importiert
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from fcs_main_parallel import FrankaCubeStackEnv, validate_stacking

# ============================================================================
# KONFIGURATION
# ============================================================================

# Parameter-Suchräume (von konservativ zu aggressiv)
PARAMETER_SEARCH = {
    "trajectory_resolution": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "air_speed_multiplier": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    "critical_speed_factor": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "critical_height_threshold": [0.06, 0.05, 0.04, 0.03, 0.02],
}

# Test-Einstellungen
EPISODES_PER_TEST = 3          # Episoden pro Parameterwert
TEST_SEEDS = [42, 123, 456]    # Reproduzierbare Seeds
MAX_FAILURES_ALLOWED = 0       # Keine Fehler erlaubt


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def load_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config: dict, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


class OptimizationLogger:
    """Loggt Ergebnisse in Datei und Konsole."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(output_dir / "optimization_log.txt", 'w', encoding='utf-8')
        self.results = []
    
    def log(self, message: str, console: bool = True):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_file.write(line + "\n")
        self.log_file.flush()
        if console:
            print(message)
    
    def add_result(self, param: str, value: float, timesteps: float, 
                   successes: int, failures: int):
        self.results.append({
            "parameter": param,
            "value": value,
            "avg_timesteps": timesteps,
            "successes": successes,
            "failures": failures,
        })
    
    def save_csv(self):
        csv_path = self.output_dir / "optimization_results.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Parameter;Wert;Avg_Timesteps;Erfolge;Fehler\n")
            for r in self.results:
                f.write(f"{r['parameter']};{r['value']};{r['avg_timesteps']:.1f};")
                f.write(f"{r['successes']};{r['failures']}\n")
        self.log(f"CSV gespeichert: {csv_path}")
    
    def close(self):
        self.log_file.close()


# ============================================================================
# EPISODE RUNNER
# ============================================================================

class EpisodeRunner:
    """Führt einzelne Episoden mit verschiedenen Parametern aus."""
    
    def __init__(self, base_config: dict, logger: OptimizationLogger):
        self.base_config = base_config
        self.logger = logger
        self.world = None
        self.env = None
        self.controller = None
        self.current_params = {}
    
    def initialize(self, params: dict):
        """Initialisiert oder re-initialisiert die Simulation mit neuen Parametern."""
        
        # Prüfe ob Neuinitialisierung nötig
        if self.current_params != params:
            self.logger.log(f"  Initialisiere mit neuen Parametern...", console=False)
            
            # World erstellen/zurücksetzen
            if self.world is None:
                self.world = World(stage_units_in_meters=1.0)
            
            # Environment mit neuen Parametern
            self.env = FrankaCubeStackEnv(
                world=self.world,
                env_idx=0,
                trajectory_resolution=params["trajectory_resolution"],
                air_speed_multiplier=params["air_speed_multiplier"],
                height_adaptive_speed=params["height_adaptive_speed"],
                critical_height_threshold=params["critical_height_threshold"],
                critical_speed_factor=params["critical_speed_factor"],
            )
            
            self.world.reset()
            
            # Warmup
            for _ in range(10):
                simulation_app.update()
                self.world.step(render=False)
            
            self.controller = self.env.setup_post_load()
            self.current_params = params.copy()
    
    def run_episode(self, seed: int) -> Tuple[bool, int, str]:
        """
        Führt eine Episode aus.
        
        Returns:
            (success, timesteps, reason)
        """
        # Reset
        self.controller.reset()
        cube_pos, target_pos = self.env.domain_randomization(seed)
        
        timesteps = 0
        max_timesteps = 2000
        
        while not self.controller.is_done() and timesteps < max_timesteps:
            simulation_app.update()
            
            obs = self.world.get_observations()
            action = self.controller.forward(observations=obs)
            
            articulation = self.env.franka.get_articulation_controller()
            articulation.apply_action(action)
            
            self.world.step(render=False)
            timesteps += 1
        
        # Validierung
        success, reason = validate_stacking(self.env.task, target_pos)
        
        return success, timesteps, reason if not success else ""
    
    def test_params(self, params: dict, seeds: List[int]) -> Tuple[float, int, int]:
        """
        Testet Parameter mit mehreren Seeds.
        
        Returns:
            (avg_timesteps, successes, failures)
        """
        self.initialize(params)
        
        timesteps_list = []
        successes = 0
        failures = 0
        
        for seed in seeds:
            try:
                success, timesteps, reason = self.run_episode(seed)
                
                if success:
                    successes += 1
                    timesteps_list.append(timesteps)
                else:
                    failures += 1
                    self.logger.log(f"      Seed {seed}: ✗ {reason}", console=False)
            except Exception as e:
                failures += 1
                self.logger.log(f"      Seed {seed}: ✗ Exception: {e}", console=False)
        
        avg_timesteps = np.mean(timesteps_list) if timesteps_list else float('inf')
        return avg_timesteps, successes, failures


# ============================================================================
# OPTIMIZER
# ============================================================================

def optimize_parameters(config_path: Path, output_dir: Path):
    """Hauptoptimierungsfunktion."""
    
    logger = OptimizationLogger(output_dir)
    
    logger.log("\n" + "="*60)
    logger.log("  PARAMETER OPTIMIERUNG START")
    logger.log("="*60)
    logger.log(f"Config: {config_path}")
    logger.log(f"Output: {output_dir}")
    logger.log(f"Episodes pro Test: {EPISODES_PER_TEST}")
    logger.log(f"Test Seeds: {TEST_SEEDS[:EPISODES_PER_TEST]}")
    
    # Basis-Config laden
    base_config = load_config(config_path)
    
    # Aktuelle Parameter extrahieren
    current_params = {
        "trajectory_resolution": base_config["controller"]["trajectory_resolution"],
        "air_speed_multiplier": base_config["controller"]["air_speed_multiplier"],
        "height_adaptive_speed": base_config["controller"]["height_adaptive_speed"],
        "critical_height_threshold": base_config["controller"]["critical_height_threshold"],
        "critical_speed_factor": base_config["controller"]["critical_speed_factor"],
    }
    
    logger.log("\nAktuelle Parameter:")
    for k, v in current_params.items():
        logger.log(f"  {k}: {v}")
    
    # Episode Runner erstellen
    runner = EpisodeRunner(base_config, logger)
    
    # Baseline messen
    logger.log("\n" + "-"*60)
    logger.log("BASELINE MESSUNG...")
    baseline_ts, baseline_ok, baseline_fail = runner.test_params(
        current_params, TEST_SEEDS[:EPISODES_PER_TEST]
    )
    logger.log(f"Baseline: {baseline_ts:.1f} Timesteps ({baseline_ok}/{baseline_ok+baseline_fail} ok)")
    
    # Optimierte Parameter (starten mit aktuellen)
    best_params = current_params.copy()
    
    # Jeden Parameter optimieren
    for param_name, values in PARAMETER_SEARCH.items():
        logger.log("\n" + "="*60)
        logger.log(f"OPTIMIERE: {param_name}")
        logger.log(f"  Werte: {values}")
        logger.log("="*60)
        
        best_value = best_params[param_name]
        best_timesteps = float('inf')
        last_good_value = best_value
        
        for value in values:
            # Skip wenn Wert konservativer als aktuell bester
            # (für Parameter wo größer = schneller)
            if param_name in ["trajectory_resolution", "air_speed_multiplier", "critical_speed_factor"]:
                if value < best_value:
                    continue
            # Für critical_height_threshold ist kleiner = aggressiver
            elif param_name == "critical_height_threshold":
                if value > best_value:
                    continue
            
            # Test-Parameter
            test_params = best_params.copy()
            test_params[param_name] = value
            
            logger.log(f"\n  Teste {param_name} = {value}")
            
            avg_ts, successes, failures = runner.test_params(
                test_params, TEST_SEEDS[:EPISODES_PER_TEST]
            )
            
            logger.add_result(param_name, value, avg_ts, successes, failures)
            
            if failures <= MAX_FAILURES_ALLOWED and successes > 0:
                logger.log(f"    ✓ {avg_ts:.1f} Timesteps ({successes}/{successes+failures} ok)")
                last_good_value = value
                
                if avg_ts < best_timesteps:
                    best_timesteps = avg_ts
                    best_value = value
            else:
                logger.log(f"    ✗ Fehlgeschlagen ({failures} Fehler)")
                logger.log(f"    → Stoppe bei {param_name} = {last_good_value}")
                break
        
        # Besten Wert für diesen Parameter übernehmen
        best_params[param_name] = best_value
        logger.log(f"\n  → Bester Wert: {param_name} = {best_value}")
    
    # Finale Messung
    logger.log("\n" + "="*60)
    logger.log("FINALE MESSUNG mit optimierten Parametern...")
    logger.log("="*60)
    
    final_ts, final_ok, final_fail = runner.test_params(
        best_params, TEST_SEEDS[:EPISODES_PER_TEST]
    )
    
    # Ergebnisse
    logger.log("\n" + "="*60)
    logger.log("  ERGEBNIS")
    logger.log("="*60)
    
    logger.log("\nOptimierte Parameter:")
    for k, v in best_params.items():
        if k != "height_adaptive_speed":
            logger.log(f"  {k}: {v}")
    
    logger.log(f"\nVergleich:")
    logger.log(f"  Baseline:   {baseline_ts:.1f} Timesteps")
    logger.log(f"  Optimiert:  {final_ts:.1f} Timesteps")
    if baseline_ts > 0:
        improvement = (1 - final_ts/baseline_ts) * 100
        logger.log(f"  Verbesserung: {improvement:.1f}%")
    
    # Config speichern
    optimized_config = copy.deepcopy(base_config)
    for k, v in best_params.items():
        optimized_config["controller"][k] = v
    
    optimized_path = output_dir / "config_optimized.yaml"
    save_config(optimized_config, optimized_path)
    logger.log(f"\nOptimierte Config: {optimized_path}")
    
    # CSV speichern
    logger.save_csv()
    
    logger.close()
    
    return best_params, final_ts


# ============================================================================
# MAIN
# ============================================================================

def main():
    global EPISODES_PER_TEST
    import argparse
    
    parser = argparse.ArgumentParser(description="Parameter Optimizer")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    # Globale Einstellung anpassen
    EPISODES_PER_TEST = args.episodes
    
    # Pfade
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / f"optimization_{timestamp}"
    
    try:
        best_params, final_timesteps = optimize_parameters(config_path, output_dir)
        
        print("\n" + "="*60)
        print("  OPTIMIERUNG ABGESCHLOSSEN")
        print("="*60)
        print(f"\nBeste Parameter (in {output_dir}/config_optimized.yaml):")
        for k, v in best_params.items():
            if k != "height_adaptive_speed":
                print(f"  {k}: {v}")
        print(f"\nFinale Timesteps: {final_timesteps:.1f}")
        
    except KeyboardInterrupt:
        print("\n\nAbgebrochen.")
    finally:
        print("\nSchließe Isaac Sim...")
        simulation_app.close()


if __name__ == "__main__":
    main()
