"""
Trajektorie-Visualisierung für Franka Cube Stacking Controller

Zeigt wie die Parameter:
- trajectory_resolution
- air_speed_multiplier  
- height_adaptive_speed
- critical_height_threshold
- critical_speed_factor

die resultierenden Trajektorien und ihre Auflösung (Anzahl Interpolationspunkte) beeinflussen.

Verwendung:
    python visualize_trajectory.py

Output:
    - 3D Plot der Trajektorien mit verschiedenen Parametern
    - Punkt-Dichte-Visualisierung
    - Vergleich der Phase-Auflösungen
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib import cm
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# Trajektorie-Simulator (vereinfacht)
# ============================================================================

class SimpleTrajectorySimulator:
    """
    Simuliert die 10 Phasen eines Pick&Place und berechnet die EE-Trajektorie.
    
    Dies ist eine VEREINFACHTE Simulation - keine echte Robotik-Berechnung,
    sondern lineare Interpolationen zwischen Ziel-Positionen.
    """
    
    AIR_PHASES = [0, 4, 5, 8, 9]
    CRITICAL_PHASES = [1, 6]
    GRIPPER_PHASES = [2, 3, 7]
    
    DEFAULT_EVENTS_DT = [0.008, 0.005, 1, 0.1, 0.05, 0.05, 0.0025, 1, 0.008, 0.08]
    
    def __init__(
        self,
        trajectory_resolution: float = 1.0,
        air_speed_multiplier: float = 1.0,
        height_adaptive_speed: bool = False,
        critical_height_threshold: float = 0.15,
        critical_speed_factor: float = 0.25,
    ):
        self.trajectory_resolution = trajectory_resolution
        self.air_speed_multiplier = air_speed_multiplier
        self.height_adaptive_speed = height_adaptive_speed
        self.critical_height_threshold = critical_height_threshold
        self.critical_speed_factor = critical_speed_factor
        
        # Berechne events_dt
        self.events_dt = self._compute_events_dt()
        
    def _compute_events_dt(self) -> list:
        """Berechnet die modifizierten events_dt basierend auf Parametern."""
        events_dt = list(self.DEFAULT_EVENTS_DT)
        
        # 1. trajectory_resolution (alle Phasen)
        events_dt = [dt * self.trajectory_resolution for dt in events_dt]
        
        # 2. air_speed_multiplier (nur AIR-Phasen)
        if self.air_speed_multiplier != 1.0:
            for phase_idx in self.AIR_PHASES:
                if phase_idx < len(events_dt):
                    events_dt[phase_idx] *= self.air_speed_multiplier
        
        return events_dt
    
    def _get_effective_dt(self, phase: int, target_height: float) -> float:
        """Berechnet das effektive dt für eine Phase, inkl. Height-Adaptive Anpassung."""
        dt = self.events_dt[phase]
        
        # Height-adaptive Anpassung
        if self.height_adaptive_speed and target_height < self.critical_height_threshold:
            # Nähe am Boden → langsamer
            dt *= self.critical_speed_factor
        
        return dt
    
    def simulate_phase(
        self,
        phase: int,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        phase_duration: float = 1.0,
    ) -> np.ndarray:
        """
        Simuliert eine Phase und gibt alle Interpolationspunkte zurück.
        
        Args:
            phase: Phase-Nummer (0-9)
            start_pos: Start-Position [x, y, z]
            target_pos: Ziel-Position [x, y, z]
            phase_duration: Maximale Phase-Dauer in Sekunden
            
        Returns:
            Array der Interpolationspunkte Shape: (N, 3)
        """
        dt = self._get_effective_dt(phase, target_pos[2])
        
        # Anzahl der Interpolationspunkte = phase_duration / dt
        # Je kleiner dt, desto mehr Punkte (feinere Auflösung)
        n_points = max(int(phase_duration / dt), 2)
        
        # Lineare Interpolation
        t_values = np.linspace(0, 1, n_points)
        trajectory = start_pos[np.newaxis, :] + t_values[:, np.newaxis] * (target_pos - start_pos)
        
        return trajectory
    
    def simulate_full_pick_place(
        self,
        picking_pos: np.ndarray = np.array([0.5, 0.2, 0.05]),
        placing_pos: np.ndarray = np.array([0.5, 0.5, 0.1]),
        ee_initial_height: float = 0.3,
    ) -> dict:
        """
        Simuliert einen vollständigen Pick&Place Zyklus.
        
        Args:
            picking_pos: Position des zu greifenden Objekts
            placing_pos: Position zum Ablegen
            ee_initial_height: Initiale Höhe des Greifers über Tisch
            
        Returns:
            dict mit Trajektorie-Daten pro Phase
        """
        base_pos = np.array([0.0, 0.0, 0.0])  # Roboter-Basis
        
        # Phase-Zielposition und -Dauer
        phases_config = [
            # (name, target_pos, duration_sec, is_movement)
            ("Phase 0: Above", picking_pos + np.array([0, 0, ee_initial_height]), 1.0, True),
            ("Phase 1: Lower", picking_pos, 1.0, True),
            ("Phase 2: Wait", picking_pos, 0.1, False),
            ("Phase 3: Grip", picking_pos, 0.1, False),
            ("Phase 4: Lift", picking_pos + np.array([0, 0, ee_initial_height]), 1.0, True),
            ("Phase 5: Move XY", placing_pos + np.array([0, 0, ee_initial_height]), 1.0, True),
            ("Phase 6: Lower", placing_pos, 1.0, True),
            ("Phase 7: Release", placing_pos, 0.1, False),
            ("Phase 8: Lift", placing_pos + np.array([0, 0, ee_initial_height]), 1.0, True),
            ("Phase 9: Return", base_pos + np.array([0, 0, ee_initial_height]), 1.0, True),
        ]
        
        result = {
            "phases": [],
            "full_trajectory": [],
            "point_counts": [],
            "phase_names": [],
        }
        
        current_pos = base_pos + np.array([0, 0, ee_initial_height])
        
        for phase_idx, (name, target_pos, duration, is_movement) in enumerate(phases_config):
            if is_movement:
                trajectory = self.simulate_phase(phase_idx, current_pos, target_pos, duration)
            else:
                # Nicht-Bewegungs-Phasen: nur Start-Position
                trajectory = np.array([current_pos])
            
            result["phases"].append(trajectory)
            result["full_trajectory"].extend(trajectory.tolist())
            result["point_counts"].append(len(trajectory))
            result["phase_names"].append(name)
            
            current_pos = trajectory[-1] if len(trajectory) > 0 else target_pos
        
        result["full_trajectory"] = np.array(result["full_trajectory"])
        
        return result
    
    def simulate_two_cube_stacking(
        self,
        cube1_picking_pos: np.ndarray = np.array([0.3, 0.2, 0.05]),
        cube2_picking_pos: np.ndarray = np.array([0.7, 0.2, 0.05]),
        placing_pos: np.ndarray = np.array([0.5, 0.5, 0.1]),
        ee_initial_height: float = 0.3,
    ) -> dict:
        """
        Simuliert das Stapeln von 2 Würfeln (2x Pick&Place).
        
        Args:
            cube1_picking_pos: Position des ersten Würfels
            cube2_picking_pos: Position des zweiten Würfels
            placing_pos: Gemeinsame Ablage-Position
            ee_initial_height: Initiale Greifer-Höhe
            
        Returns:
            dict mit Daten für beide Pick&Place Zyklen
        """
        result = {
            "cycle_1": None,
            "cycle_2": None,
            "total_timesteps": 0,
            "total_points": 0,
            "total_time": 0.0,
            "phase_breakdown": {},
        }
        
        # Zyklus 1: Ersten Würfel greifen und ablegen
        result["cycle_1"] = self.simulate_full_pick_place(
            picking_pos=cube1_picking_pos,
            placing_pos=placing_pos,
            ee_initial_height=ee_initial_height,
        )
        
        # Zyklus 2: Zweiten Würfel greifen und ablegen
        result["cycle_2"] = self.simulate_full_pick_place(
            picking_pos=cube2_picking_pos,
            placing_pos=placing_pos,
            ee_initial_height=ee_initial_height,
        )
        
        # Berechne Gesamtstatistiken
        total_points = (
            len(result["cycle_1"]["full_trajectory"]) + 
            len(result["cycle_2"]["full_trajectory"])
        )
        result["total_points"] = total_points
        
        # Berechne Gesamtzeit (Summe aller Phase-Zeiten × dt)
        total_time_c1 = sum(
            result["cycle_1"]["point_counts"][i] * self.events_dt[i] 
            for i in range(10)
        )
        total_time_c2 = sum(
            result["cycle_2"]["point_counts"][i] * self.events_dt[i] 
            for i in range(10)
        )
        result["total_time"] = total_time_c1 + total_time_c2
        
        # Phase-Breakdown
        for phase_idx in range(10):
            result["phase_breakdown"][phase_idx] = {
                "cycle_1_points": result["cycle_1"]["point_counts"][phase_idx],
                "cycle_2_points": result["cycle_2"]["point_counts"][phase_idx],
                "cycle_1_time": result["cycle_1"]["point_counts"][phase_idx] * self.events_dt[phase_idx],
                "cycle_2_time": result["cycle_2"]["point_counts"][phase_idx] * self.events_dt[phase_idx],
            }
        
        return result


# ============================================================================
# Visualisierung
# ============================================================================

def plot_trajectories_3d():
    """Plottet mehrere Trajektorien mit verschiedenen Parametern in 3D."""
    
    # Parameter-Kombinationen zum Vergleichen
    configs = [
        {
            "name": "Standard",
            "params": {"trajectory_resolution": 1.0, "air_speed_multiplier": 1.0, "height_adaptive_speed": False},
            "color": "blue",
        },
        {
            "name": "Schnelle Air-Bewegungen",
            "params": {"trajectory_resolution": 1.0, "air_speed_multiplier": 4.0, "height_adaptive_speed": False},
            "color": "green",
        },
        {
            "name": "Mit Height Adaptation",
            "params": {"trajectory_resolution": 1.0, "air_speed_multiplier": 1.0, "height_adaptive_speed": True, "critical_height_threshold": 0.15, "critical_speed_factor": 0.25},
            "color": "red",
        },
        {
            "name": "Optimal (Schnell + Adaptiv)",
            "params": {"trajectory_resolution": 1.5, "air_speed_multiplier": 4.0, "height_adaptive_speed": True, "critical_height_threshold": 0.05, "critical_speed_factor": 0.8},
            "color": "purple",
        },
    ]
    
    fig = plt.figure(figsize=(16, 12))
    
    # ========================================================================
    # Subplot 1: 3D Trajektorie Vergleich
    # ========================================================================
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    trajectories = {}
    for config in configs:
        sim = SimpleTrajectorySimulator(**config["params"])
        result = sim.simulate_full_pick_place()
        traj = result["full_trajectory"]
        trajectories[config["name"]] = (traj, result)
        
        # Plot
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                label=config["name"], color=config["color"], linewidth=2, alpha=0.7)
        
        # Start und End-Punkte
        ax1.scatter(*traj[0], color=config["color"], s=100, marker='o', alpha=0.8)  # Start
        ax1.scatter(*traj[-1], color=config["color"], s=100, marker='X', alpha=0.8)  # End
    
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Trajektorie Vergleich\n(○ = Start, ✕ = End)", fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Subplot 2: Punkt-Dichte pro Phase
    # ========================================================================
    ax2 = fig.add_subplot(2, 2, 2)
    
    x_pos = np.arange(10)  # 10 Phasen
    width = 0.2
    
    for idx, config in enumerate(configs):
        sim = SimpleTrajectorySimulator(**config["params"])
        result = sim.simulate_full_pick_place()
        
        ax2.bar(x_pos + idx * width, result["point_counts"], width, 
               label=config["name"], color=config["color"], alpha=0.7)
    
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Anzahl Interpolationspunkte")
    ax2.set_title("Punkt-Dichte pro Phase\n(Höhere Dichte = Feinere Auflösung)", fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos + width * 1.5)
    ax2.set_xticklabels([f"P{i}" for i in range(10)])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Subplot 3: Events DT Vergleich
    # ========================================================================
    ax3 = fig.add_subplot(2, 2, 3)
    
    phase_labels = ["0:\nAbove", "1:\nLower", "2:\nWait", "3:\nGrip", "4:\nLift", 
                   "5:\nMove", "6:\nLower", "7:\nRelease", "8:\nLift", "9:\nReturn"]
    
    x_pos = np.arange(10)
    width = 0.2
    
    for idx, config in enumerate(configs):
        sim = SimpleTrajectorySimulator(**config["params"])
        events_dt = sim.events_dt
        
        ax3.bar(x_pos + idx * width, events_dt, width, 
               label=config["name"], color=config["color"], alpha=0.7)
    
    ax3.set_xlabel("Phase")
    ax3.set_ylabel("dt Wert (sec)")
    ax3.set_title("Effektive dt-Werte pro Phase\n(Niedriger dt = Höhere Auflösung / Langsamer)", 
                 fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos + width * 1.5)
    ax3.set_xticklabels(phase_labels, fontsize=8)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_yscale('log')  # Log-Skala weil Werte sehr unterschiedlich sind
    
    # ========================================================================
    # Subplot 4: Z-Höhen Profil (zeigt Kritische Zone)
    # ========================================================================
    ax4 = fig.add_subplot(2, 2, 4)
    
    for config in configs:
        sim = SimpleTrajectorySimulator(**config["params"])
        result = sim.simulate_full_pick_place()
        traj = result["full_trajectory"]
        
        # Entfernung entlang Trajektorie
        distances = np.zeros(len(traj))
        for i in range(1, len(traj)):
            distances[i] = distances[i-1] + np.linalg.norm(traj[i] - traj[i-1])
        
        ax4.plot(distances, traj[:, 2], label=config["name"], color=config["color"], 
                linewidth=2, alpha=0.7)
    
    # Kritische Höhe-Zone schattieren
    ax4.axhspan(0, 0.15, alpha=0.2, color='red', label='Kritische Zone (Z < 0.15m)')
    
    ax4.set_xlabel("Entfernung entlang Trajektorie (m)")
    ax4.set_ylabel("Z-Höhe (m)")
    ax4.set_title("Z-Höhen Profil\n(Zeigt wo Height-Adaptive Speed aktiv ist)", 
                 fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("trajectory_comparison.png", dpi=150, bbox_inches='tight')
    print("✅ Plot gespeichert: trajectory_comparison.png")
    plt.show()


def create_detailed_analysis_table():
    """Erstellt eine detaillierte Analyse-Tabelle."""
    
    print("\n" + "="*100)
    print("DETAILLIERTE ANALYSE - Parameter-Auswirkungen")
    print("="*100 + "\n")
    
    configs = [
        {
            "name": "Standard",
            "params": {"trajectory_resolution": 1.0, "air_speed_multiplier": 1.0, "height_adaptive_speed": False},
        },
        {
            "name": "Schnelle Air (4x)",
            "params": {"trajectory_resolution": 1.0, "air_speed_multiplier": 4.0, "height_adaptive_speed": False},
        },
        {
            "name": "Height Adaptive",
            "params": {"trajectory_resolution": 1.0, "air_speed_multiplier": 1.0, "height_adaptive_speed": True, "critical_height_threshold": 0.15, "critical_speed_factor": 0.25},
        },
        {
            "name": "Optimiert",
            "params": {"trajectory_resolution": 1.5, "air_speed_multiplier": 4.0, "height_adaptive_speed": True, "critical_height_threshold": 0.05, "critical_speed_factor": 0.8},
        },
    ]
    
    for config in configs:
        print(f"\n{'─'*100}")
        print(f"📊 KONFIGURATION: {config['name']}")
        print(f"{'─'*100}")
        
        params = config["params"]
        print(f"Parameter:")
        print(f"  • trajectory_resolution:      {params.get('trajectory_resolution', 1.0)}")
        print(f"  • air_speed_multiplier:       {params.get('air_speed_multiplier', 1.0)}")
        print(f"  • height_adaptive_speed:      {params.get('height_adaptive_speed', False)}")
        if params.get('height_adaptive_speed'):
            print(f"    - critical_height_threshold: {params.get('critical_height_threshold', 0.15)}m")
            print(f"    - critical_speed_factor:     {params.get('critical_speed_factor', 0.25)}")
        
        sim = SimpleTrajectorySimulator(**params)
        result = sim.simulate_full_pick_place()
        
        print(f"\nTrajektorie-Analyse:")
        print(f"  • Gesamte Punkte:             {len(result['full_trajectory'])}")
        print(f"  • Durchschn. Punkte/Phase:    {np.mean(result['point_counts']):.1f}")
        print(f"  • Min/Max Punkte:             {min(result['point_counts'])}/{max(result['point_counts'])}")
        
        # Events DT Analyse
        print(f"\nPhase-Zeiten (events_dt):")
        print(f"  AIR Phasen (0,4,5,8,9):")
        air_dts = [sim.events_dt[i] for i in [0, 4, 5, 8, 9]]
        print(f"    - Durchschnitt: {np.mean(air_dts):.4f}s")
        print(f"    - Min/Max:      {min(air_dts):.4f}s / {max(air_dts):.4f}s")
        
        print(f"  CRITICAL Phasen (1,6):")
        crit_dts = [sim.events_dt[i] for i in [1, 6]]
        print(f"    - Durchschnitt: {np.mean(crit_dts):.4f}s")
        print(f"    - Min/Max:      {min(crit_dts):.4f}s / {max(crit_dts):.4f}s")
        
        print(f"  WAIT/GRIP/RELEASE Phasen (2,3,7):")
        grip_dts = [sim.events_dt[i] for i in [2, 3, 7]]
        print(f"    - Durchschnitt: {np.mean(grip_dts):.4f}s")
        print(f"    - Min/Max:      {min(grip_dts):.4f}s / {max(grip_dts):.4f}s")
        
        # Geschwindigkeit Analyse
        traj = result["full_trajectory"]
        distances = np.zeros(len(traj))
        for i in range(1, len(traj)):
            distances[i] = distances[i-1] + np.linalg.norm(traj[i] - traj[i-1])
        
        print(f"\nBewegung:")
        print(f"  • Gesamte Distanz:            {distances[-1]:.3f}m")
        print(f"  • Durchschn. Schritt-Größe:   {np.mean(np.diff(distances)):.4f}m")
        
        # Höhen-Profil
        z_values = traj[:, 2]
        in_critical = np.sum(z_values < 0.15)
        print(f"\nHöhen-Profil:")
        print(f"  • Min/Max Z:                  {z_values.min():.3f}m / {z_values.max():.3f}m")
        print(f"  • Punkte in kritischer Zone:  {in_critical}/{len(z_values)} ({in_critical/len(z_values)*100:.1f}%)")


def create_two_cube_comparison():
    """Vergleicht die gesamten Timesteps für 2-Würfel-Stacking."""
    
    print("\n" + "="*120)
    print("🏗️  2-WÜRFEL STACKING - TIMESTEP VERGLEICH")
    print("="*120 + "\n")
    
    configs = [
        {
            "name": "Standard",
            "params": {"trajectory_resolution": 1.0, "air_speed_multiplier": 1.0, "height_adaptive_speed": False},
        },
        {
            "name": "Schnelle Air (4x)",
            "params": {"trajectory_resolution": 1.0, "air_speed_multiplier": 4.0, "height_adaptive_speed": False},
        },
        {
            "name": "Height Adaptive",
            "params": {"trajectory_resolution": 1.0, "air_speed_multiplier": 1.0, "height_adaptive_speed": True, "critical_height_threshold": 0.15, "critical_speed_factor": 0.25},
        },
        {
            "name": "Optimiert (1.5x + 4x Air)",
            "params": {"trajectory_resolution": 1.5, "air_speed_multiplier": 4.0, "height_adaptive_speed": True, "critical_height_threshold": 0.05, "critical_speed_factor": 0.8},
        },
    ]
    
    results_data = []
    
    for config in configs:
        print(f"\n{'─'*120}")
        print(f"📊 {config['name']}")
        print(f"{'─'*120}")
        
        params = config["params"]
        print(f"Parameter: trajectory_res={params.get('trajectory_resolution', 1.0)} | "
              f"air_speed_mult={params.get('air_speed_multiplier', 1.0)} | "
              f"height_adaptive={params.get('height_adaptive_speed', False)}")
        
        sim = SimpleTrajectorySimulator(**params)
        result = sim.simulate_two_cube_stacking()
        
        # Gesamtstatistiken
        print(f"\n📈 GESAMTSTATISTIKEN (2 Würfel):")
        print(f"  ├─ Gesamte Timesteps:        {result['total_points']:>5}")
        print(f"  ├─ Gesamtzeit:               {result['total_time']:>8.2f}s")
        print(f"  ├─ Durchschn. pro Phase:     {result['total_points'] / 20:>8.1f}")
        print(f"  └─ Durchschn. Zeit/Zyklus:   {result['total_time'] / 2:>8.2f}s")
        
        # Zyklusvergleich
        c1 = result["cycle_1"]
        c2 = result["cycle_2"]
        c1_points = len(c1["full_trajectory"])
        c2_points = len(c2["full_trajectory"])
        
        print(f"\n🔄 ZYKLUSVERGLEICH:")
        print(f"  Zyklus 1 (Würfel 1):")
        print(f"    ├─ Timesteps:  {c1_points:>5}")
        print(f"    ├─ Zeit:       {sum(c1['point_counts'][i] * sim.events_dt[i] for i in range(10)):>8.2f}s")
        print(f"    └─ Phasen:     Min={min(c1['point_counts'])}, Max={max(c1['point_counts'])}")
        
        print(f"  Zyklus 2 (Würfel 2):")
        print(f"    ├─ Timesteps:  {c2_points:>5}")
        print(f"    ├─ Zeit:       {sum(c2['point_counts'][i] * sim.events_dt[i] for i in range(10)):>8.2f}s")
        print(f"    └─ Phasen:     Min={min(c2['point_counts'])}, Max={max(c2['point_counts'])}")
        
        # Phase-Breakdown detailliert
        print(f"\n📊 PHASE-BREAKDOWN (beide Zyklen kombiniert):")
        print(f"  {'Phase':<12} {'Beschreibung':<15} {'C1 Pts':<8} {'C2 Pts':<8} {'C1 Zeit':<10} {'C2 Zeit':<10} {'Gesamt':<10}")
        print(f"  {'-'*85}")
        
        phase_names = ["Above", "Lower", "Wait", "Grip", "Lift", "Move", "Lower", "Release", "Lift", "Return"]
        total_c1_time = 0
        total_c2_time = 0
        
        for phase_idx in range(10):
            pb = result["phase_breakdown"][phase_idx]
            c1_time = pb["cycle_1_time"]
            c2_time = pb["cycle_2_time"]
            total_c1_time += c1_time
            total_c2_time += c2_time
            
            phase_type = ""
            if phase_idx in sim.AIR_PHASES:
                phase_type = "[AIR]"
            elif phase_idx in sim.CRITICAL_PHASES:
                phase_type = "[CRIT]"
            else:
                phase_type = "[GRIP]"
            
            print(f"  {phase_idx:<3} {phase_names[phase_idx]:<8} {phase_type:<7} "
                  f"{pb['cycle_1_points']:<8} {pb['cycle_2_points']:<8} "
                  f"{c1_time:<10.4f} {c2_time:<10.4f} {c1_time+c2_time:<10.4f}")
        
        print(f"  {'-'*85}")
        print(f"  {'TOTAL':<25} {'-':<8} {'-':<8} {total_c1_time:<10.4f} {total_c2_time:<10.4f} {total_c1_time+total_c2_time:<10.4f}")
        
        # Speedup berechnen
        print(f"\n⚡ SPEEDUP-ANALYSE:")
        air_time_c1 = sum(
            result["cycle_1"]["point_counts"][i] * sim.events_dt[i] 
            for i in sim.AIR_PHASES
        )
        air_time_c2 = sum(
            result["cycle_2"]["point_counts"][i] * sim.events_dt[i] 
            for i in sim.AIR_PHASES
        )
        crit_time_c1 = sum(
            result["cycle_1"]["point_counts"][i] * sim.events_dt[i] 
            for i in sim.CRITICAL_PHASES
        )
        crit_time_c2 = sum(
            result["cycle_2"]["point_counts"][i] * sim.events_dt[i] 
            for i in sim.CRITICAL_PHASES
        )
        grip_time_c1 = sum(
            result["cycle_1"]["point_counts"][i] * sim.events_dt[i] 
            for i in sim.GRIPPER_PHASES
        )
        grip_time_c2 = sum(
            result["cycle_2"]["point_counts"][i] * sim.events_dt[i] 
            for i in sim.GRIPPER_PHASES
        )
        
        print(f"  AIR Phasen:      {air_time_c1+air_time_c2:>8.4f}s ({(air_time_c1+air_time_c2)/result['total_time']*100:>5.1f}%)")
        print(f"  CRITICAL Phasen: {crit_time_c1+crit_time_c2:>8.4f}s ({(crit_time_c1+crit_time_c2)/result['total_time']*100:>5.1f}%)")
        print(f"  GRIP Phasen:     {grip_time_c1+grip_time_c2:>8.4f}s ({(grip_time_c1+grip_time_c2)/result['total_time']*100:>5.1f}%)")
        
        results_data.append({
            "name": config["name"],
            "total_time": result["total_time"],
            "total_points": result["total_points"],
            "air_time": air_time_c1 + air_time_c2,
            "crit_time": crit_time_c1 + crit_time_c2,
            "grip_time": grip_time_c1 + grip_time_c2,
        })
    
    # ========================================================================
    # Vergleichstabelle
    # ========================================================================
    print("\n\n" + "="*120)
    print("📊 ZUSAMMENFASSUNG - ALLE KONFIGURATIONEN")
    print("="*120 + "\n")
    
    print(f"{'Konfiguration':<25} {'Zeit (s)':<12} {'Punkte':<10} {'Air%':<10} {'Crit%':<10} {'Grip%':<10} {'Speedup':<10}")
    print(f"{'-'*110}")
    
    baseline_time = results_data[0]["total_time"]
    
    for data in results_data:
        speedup = baseline_time / data["total_time"]
        air_pct = data["air_time"] / data["total_time"] * 100 if data["total_time"] > 0 else 0
        crit_pct = data["crit_time"] / data["total_time"] * 100 if data["total_time"] > 0 else 0
        grip_pct = data["grip_time"] / data["total_time"] * 100 if data["total_time"] > 0 else 0
        
        print(f"{data['name']:<25} {data['total_time']:<12.4f} {data['total_points']:<10} "
              f"{air_pct:<10.1f} {crit_pct:<10.1f} {grip_pct:<10.1f} {speedup:<10.2f}x")
    
    print("\n")
    print("💡 INTERPRETATION:")
    fastest = min(results_data, key=lambda x: x["total_time"])
    slowest = max(results_data, key=lambda x: x["total_time"])
    print(f"  🏃 Schnellste:     {fastest['name']:<25} ({fastest['total_time']:.2f}s)")
    print(f"  🐢 Langsamste:     {slowest['name']:<25} ({slowest['total_time']:.2f}s)")
    print(f"  ⚡ Speedup:        {slowest['total_time'] / fastest['total_time']:.2f}x schneller")
    print(f"  📈 Punkt-Dichte:   {fastest['total_points']} vs {slowest['total_points']} Punkte")
    print(f"  🎯 Best für Daten: {'Optimiert' if 'Optimiert' in fastest['name'] else fastest['name']}")


# ============================================================================
# Visualisierung 3D Plots
# ============================================================================

def plot_two_cube_trajectories_3d():
    """Plottet optimierte 3D Visualisierung für 2-Würfel-Stacking mit 5 key Plots."""
    
    configs = [
        {
            "name": "Standard",
            "params": {"trajectory_resolution": 1.0, "air_speed_multiplier": 1.0, "height_adaptive_speed": False},
            "color": "blue",
        },
        {
            "name": "Schnelle Air (4x)",
            "params": {"trajectory_resolution": 1.0, "air_speed_multiplier": 4.0, "height_adaptive_speed": False},
            "color": "green",
        },
        {
            "name": "Optimiert (1.5x + 4x Air)",
            "params": {"trajectory_resolution": 1.5, "air_speed_multiplier": 4.0, "height_adaptive_speed": True, "critical_height_threshold": 0.05, "critical_speed_factor": 0.8},
            "color": "purple",
        },
    ]
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # Subplot 1: 3D Trajektorien Vergleich (oben links)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    for config in configs:
        sim = SimpleTrajectorySimulator(**config["params"])
        result = sim.simulate_two_cube_stacking()
        
        # Trajektorie kombiniert
        traj_combined = np.vstack([
            result["cycle_1"]["full_trajectory"],
            result["cycle_2"]["full_trajectory"],
        ])
        
        ax1.plot(traj_combined[:, 0], traj_combined[:, 1], traj_combined[:, 2],
                label=config["name"], color=config["color"], linewidth=2, alpha=0.7)
        
        ax1.scatter(*traj_combined[0], color=config["color"], s=150, marker='o', alpha=0.8)  # Start
        ax1.scatter(*traj_combined[-1], color=config["color"], s=150, marker='X', alpha=0.8)  # End
    
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Trajektorien\n(2 Würfel Zyklus)", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Subplot 2: Punkt-Dichte pro Phase (oben mitte)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    phase_names_short = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    width = 0.25
    x_pos = np.arange(10)
    
    for idx, config in enumerate(configs):
        sim = SimpleTrajectorySimulator(**config["params"])
        result = sim.simulate_two_cube_stacking()
        
        # Durchschnittliche Punkte pro Phase (beide Zyklen)
        avg_points = [
            (result["phase_breakdown"][i]["cycle_1_points"] + result["phase_breakdown"][i]["cycle_2_points"]) / 2
            for i in range(10)
        ]
        
        ax2.bar(x_pos + idx * width, avg_points, width, 
               label=config["name"], color=config["color"], alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Durchschn. Punkte")
    ax2.set_title("Punkt-Dichte pro Phase", fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(phase_names_short, fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Subplot 3: Effektive Zeitdauern pro Phase (oben rechts)
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    width = 0.25
    x_pos = np.arange(10)
    
    for idx, config in enumerate(configs):
        sim = SimpleTrajectorySimulator(**config["params"])
        result = sim.simulate_two_cube_stacking()
        
        # Durchschnittliche Zeit pro Phase (beide Zyklen)
        avg_times = [
            (result["phase_breakdown"][i]["cycle_1_time"] + result["phase_breakdown"][i]["cycle_2_time"]) / 2
            for i in range(10)
        ]
        
        ax3.bar(x_pos + idx * width, avg_times, width, 
               label=config["name"], color=config["color"], alpha=0.7, edgecolor='black')
    
    ax3.set_xlabel("Phase")
    ax3.set_ylabel("Zeit (s)")
    ax3.set_title("Effektive Zeitdauern pro Phase", fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels(phase_names_short, fontsize=9)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Subplot 4: Zeit-Anteil pro Phasen-Typ (unten links)
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    width = 0.25
    x_pos = np.arange(len(configs))
    
    air_pcts = []
    crit_pcts = []
    grip_pcts = []
    
    for config in configs:
        sim = SimpleTrajectorySimulator(**config["params"])
        result = sim.simulate_two_cube_stacking()
        
        air_time = sum(
            result["cycle_1"]["point_counts"][i] * sim.events_dt[i] for i in sim.AIR_PHASES
        ) + sum(
            result["cycle_2"]["point_counts"][i] * sim.events_dt[i] for i in sim.AIR_PHASES
        )
        crit_time = sum(
            result["cycle_1"]["point_counts"][i] * sim.events_dt[i] for i in sim.CRITICAL_PHASES
        ) + sum(
            result["cycle_2"]["point_counts"][i] * sim.events_dt[i] for i in sim.CRITICAL_PHASES
        )
        grip_time = sum(
            result["cycle_1"]["point_counts"][i] * sim.events_dt[i] for i in sim.GRIPPER_PHASES
        ) + sum(
            result["cycle_2"]["point_counts"][i] * sim.events_dt[i] for i in sim.GRIPPER_PHASES
        )
        
        total_time = result["total_time"]
        air_pcts.append(air_time / total_time * 100 if total_time > 0 else 0)
        crit_pcts.append(crit_time / total_time * 100 if total_time > 0 else 0)
        grip_pcts.append(grip_time / total_time * 100 if total_time > 0 else 0)
    
    bars1 = ax4.bar(x_pos - width, air_pcts, width, label="AIR", color='lightblue', alpha=0.8, edgecolor='black')
    bars2 = ax4.bar(x_pos, crit_pcts, width, label="CRITICAL", color='lightcoral', alpha=0.8, edgecolor='black')
    bars3 = ax4.bar(x_pos + width, grip_pcts, width, label="GRIP", color='lightgreen', alpha=0.8, edgecolor='black')
    
    # Prozentwerte auf Balken schreiben
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax4.set_ylabel("Prozentanteil (%)")
    ax4.set_title("Zeit-Anteil pro Phasen-Typ", fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 100])
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([c["name"] for c in configs], rotation=15, ha='right', fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Subplot 5: Phase-Zeit Heatmap (unten, groß)
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1:])
    
    phase_times_matrix = []
    
    for config in configs:
        sim = SimpleTrajectorySimulator(**config["params"])
        result = sim.simulate_two_cube_stacking()
        phase_times = []
        for i in range(10):
            pb = result["phase_breakdown"][i]
            phase_time = pb["cycle_1_time"] + pb["cycle_2_time"]
            phase_times.append(phase_time)
        phase_times_matrix.append(phase_times)
    
    phase_times_matrix = np.array(phase_times_matrix).T  # Transpose für richtige Shape
    
    im = ax5.imshow(phase_times_matrix, cmap='YlOrRd', aspect='auto')
    ax5.set_xlabel("Konfiguration", fontsize=11)
    ax5.set_ylabel("Phase", fontsize=11)
    ax5.set_title("Phase-Zeit Heatmap (beide Zyklen kombiniert)", fontsize=12, fontweight='bold')
    ax5.set_xticks(np.arange(len(configs)))
    ax5.set_yticks(np.arange(10))
    ax5.set_xticklabels([c["name"] for c in configs], rotation=15, ha='right', fontsize=10)
    ax5.set_yticklabels([f"Phase {i}" for i in range(10)], fontsize=9)
    
    # Werte in Cells schreiben
    for i in range(10):
        for j in range(len(configs)):
            text = ax5.text(j, i, f'{phase_times_matrix[i, j]:.3f}s',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax5, label="Zeit (s)")
    
    plt.savefig("two_cube_stacking_comparison.png", dpi=150, bbox_inches='tight')
    print("✅ Plot gespeichert: two_cube_stacking_comparison.png")
    plt.show()


def print_parameter_guide():
    """Druckt einen Leitfaden für die Parameter-Auswahl."""
    
    print("\n" + "="*100)
    print("PARAMETER-LEITFADEN")
    print("="*100)
    
    guide = """
🎯 TRAJECTORY_RESOLUTION (Standard: 1.0)
   Skaliert alle Phase-Zeiten gleichmäßig.
   
   Effekt:           Punkt-Dichte    Geschwindigkeit    Verwendung
   ─────────────────────────────────────────────────────────────────
   < 1.0 (z.B 0.5)   HÖHER           LANGSAMER         Maximale Präzision
   = 1.0             MITTEL          NORMAL            Standard
   > 1.0 (z.B 1.5)   NIEDRIGER       SCHNELLER         Schnelle Bewegung
   
   💡 Nutze > 1.0 wenn die Bewegung zu langsam ist

───────────────────────────────────────────────────────────────────────

🎯 AIR_SPEED_MULTIPLIER (Standard: 1.0)
   Zusätzlicher Speed-Boost FÜR AIR-PHASEN (0,4,5,8,9).
   Beeinflusst NICHT die kritischen Phasen (1,6) → Greifgenauigkeit bleibt!
   
   Effekt:           Punkt-Dichte (AIR)  Luftbewegung     Greifen
   ─────────────────────────────────────────────────────────────────
   = 1.0             Normal              Normal           Präzise
   = 2.0             Halb                 2x schneller     Präzise
   = 4.0             Ein Viertel         4x schneller     Präzise
   
   💡 Nutze 3-4 für schnelle Datensammlung (Luft ist schnell, Greifen bleibt präzise)

───────────────────────────────────────────────────────────────────────

🎯 HEIGHT_ADAPTIVE_SPEED (Standard: False)
   DYNAMISCHE Geschwindigkeit basierend auf Z-Höhe!
   Unter critical_height_threshold → wird langsamer/präziser
   
   Mit Height Adaptation:
   • Z > 0.15m:      Normal-Speed (schnell in der Luft)
   • Z < 0.15m:      Langsamer (critical_speed_factor * normal_speed)
   
   Beispiel mit critical_speed_factor=0.25:
   • Z=0.2m:  dt = normal
   • Z=0.1m:  dt = 0.25 × normal (4x langsamer!)
   • Z=0.01m: dt = 0.25 × normal (4x langsamer!)
   
   💡 Nutze True für präzise Bewegungen nahe am Boden

───────────────────────────────────────────────────────────────────────

🎯 CRITICAL_HEIGHT_THRESHOLD (Standard: 0.15 für Height Adaptation)
   Grenze (in Metern) unter der Height Adaptation aktiv wird.
   
   Effekt:
   • Höher (z.B 0.2m):   Mehr Punkte mit langsamer Bewegung
   • Niedriger (z.B 0.05m): Nur direkt über Boden langsam
   
   💡 Nutze 0.05-0.1m um nur Greif/Ablage-Phasen zu verlangsamen

───────────────────────────────────────────────────────────────────────

🎯 CRITICAL_SPEED_FACTOR (Standard: 0.25 für Height Adaptation)
   Multiplikator für Speed in kritischer Zone.
   
   Effekt:            Geschwindigkeit bei Z < threshold
   ──────────────────────────────────────────────────
   = 1.0              NORMAL (keine Verlangsamung)
   = 0.5              2x langsamer
   = 0.25             4x langsamer
   = 0.1              10x langsamer
   
   💡 Niedriger = Präziser aber langsamer (trade-off)

═══════════════════════════════════════════════════════════════════════

🚀 EMPFOHLENE PRESETS
───────────────────────────────────────────────────────────────────────

1️⃣ SCHNELLE DATENSAMMLUNG (Was in deinem Code aktiv ist):
   trajectory_resolution=1.5
   air_speed_multiplier=4.0
   height_adaptive_speed=True
   critical_height_threshold=0.05
   critical_speed_factor=0.8
   
   → Schnelle Luft-Bewegungen (AIR 6x schneller), 
     Greifen bleibt präzise (Critical dt unverändern-ert)
   
───────────────────────────────────────────────────────────────────────

2️⃣ MAXIMALE PRÄZISION:
   trajectory_resolution=0.5
   air_speed_multiplier=0.5
   height_adaptive_speed=True
   critical_height_threshold=0.2
   critical_speed_factor=0.1
   
   → Sehr feine Auflösung überall, extrem langsam nahe Boden
   
───────────────────────────────────────────────────────────────────────

3️⃣ BALANCIERT:
   trajectory_resolution=1.0
   air_speed_multiplier=2.0
   height_adaptive_speed=True
   critical_height_threshold=0.1
   critical_speed_factor=0.3
   
   → Mittel-Geschwindigkeit, adaptiv nahe Boden
   
═══════════════════════════════════════════════════════════════════════
"""
    print(guide)


# ============================================================================
# Hauptfunktion
# ============================================================================

def main():
    print("\n" + "="*100)
    print("🎯 FRANKA CONTROLLER - TRAJEKTORIE-VISUALISIERUNG")
    print("="*100)
    
    print("\nDieses Skript zeigt die Auswirkungen der Controller-Parameter")
    print("auf die resultierende Trajektorie und deren Auflösung (Punkt-Dichte).\n")
    
    # Drucke Parameter-Leitfaden
    print_parameter_guide()
    
    # Erstelle Analyse-Tabelle
    create_detailed_analysis_table()
    
    # Erstelle 3D Plots
    print("\n\nErstelle 3D-Visualisierung...")
    plot_trajectories_3d()
    
    # ====================================================================
    # 2-WÜRFEL STACKING ANALYSE
    # ====================================================================
    print("\n\n" + "="*120)
    print("🚀 WECHSEL ZU 2-WÜRFEL STACKING ANALYSE")
    print("="*120)
    
    create_two_cube_comparison()
    
    print("\n\nErstelle 3D-Visualisierung für 2-Würfel Stacking...")
    plot_two_cube_trajectories_3d()
    
    print("\n" + "="*100)
    print("✅ VISUALISIERUNG ABGESCHLOSSEN!")
    print("="*100)
    print("""
Erkenntnisse:

1. TRAJECTORY_RESOLUTION:
   • Skaliert alle Phasen → proportionale Punkt-Dichte Änderung
   • > 1.0: Schneller aber grober
   • < 1.0: Langsamer aber feinere Auflösung

2. AIR_SPEED_MULTIPLIER:
   • NUR AIR-Phasen (0,4,5,8,9) beeinflusst
   • CRITICAL-Phasen (1,6) bleiben gleich → Greifgenauigkeit!
   • 4.0 = Luftbewegung 4x schneller, Greifen unverändert

3. HEIGHT_ADAPTIVE_SPEED:
   • Dynamische Anpassung basierend auf Z-Höhe
   • Unter Schwelle: Automatisch langsamer/präziser
   • Ideal für: Exakte Griff- und Ablage-Operationen

4. KOMBINATION (OPTIMAL):
   • trajectory_resolution=1.5: Base 1.5x schneller
   • air_speed_multiplier=4.0: Luft 6x schneller (1.5×4)
   • height_adaptive_speed=True: Nahe Boden adaptiv langsam
   • critical_speed_factor=0.8: Nur leicht verlangsamt (80% statt 25%)
   
   → Schnelle Episode, präzise Greifer-Operationen ✓

5. 2-WÜRFEL STACKING:
   • Vergleiche die Gesamtzeit für komplette Episoden
   • Speedup zeigt relative Unterschiede zwischen Konfigurationen
   • Optimization hat großen Effekt auf Datensammlung-Geschwindigkeit
""")


if __name__ == "__main__":
    main()
