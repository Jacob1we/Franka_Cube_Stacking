"""
H5 Dataset Debug Tutorial
=========================

Dieses Skript zeigt, wie man H5-Datensätze im DINO World Model Format
analysiert und debuggt.

Verwendung:
    Aktiviere zuerst die PyTorch-Umgebung:
    > D:\00_Coding\pytorch\env_pytorch\Scripts\activate
    
    Dann führe dieses Skript aus:
    > python h5_debug_tutorial.py

Oder nutze die einzelnen Funktionen interaktiv.
"""

import h5py
import torch
import numpy as np
import pickle
from pathlib import Path


# =============================================================================
# KONFIGURATION - Passe diese Pfade an deine Datensätze an
# =============================================================================

# Referenz-Datensatz (Original DINO WM Format)
REFERENCE_DATASET = Path(r"e:\00_Coding_SSD\fcs_datasets\deformable_rop_sample")

# Dein eigener Datensatz (zum Vergleichen)
MY_DATASET = Path(r"e:\00_Coding_SSD\fcs_datasets\2026_01_18_1239_fcs_dset")


# =============================================================================
# 1. H5-DATEI STRUKTUR ANZEIGEN
# =============================================================================

def show_h5_structure(h5_path: str):
    """
    Zeigt die komplette Struktur einer H5-Datei an.
    
    Verwendung:
        show_h5_structure("pfad/zu/datei.h5")
    """
    def print_attrs(name, obj):
        """Rekursive Hilfsfunktion für h5py.visititems()"""
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}📊 {name}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}📁 {name}/")
    
    print(f"\n{'='*60}")
    print(f"H5-Struktur: {h5_path}")
    print('='*60)
    
    with h5py.File(h5_path, 'r') as f:
        print("📁 / (root)")
        f.visititems(print_attrs)


# =============================================================================
# 2. EINZELNE H5-DATEI DETAILLIERT ANALYSIEREN
# =============================================================================

def analyze_h5_timestep(h5_path: str):
    """
    Analysiert eine einzelne H5-Datei (einen Timestep) im Detail.
    
    Zeigt:
    - Action: Format und Werte
    - EEF States: Position und Quaternion des End-Effektors
    - Positions: Partikel/Objekt-Positionen
    - Info: Metadaten
    - Observations: Bild-Daten
    
    Verwendung:
        analyze_h5_timestep("000000/00.h5")
    """
    print(f"\n{'='*60}")
    print(f"Detailanalyse: {h5_path}")
    print('='*60)
    
    with h5py.File(h5_path, 'r') as f:
        
        # --- ACTION ---
        if 'action' in f:
            action = f['action'][:]
            print("\n📌 ACTION")
            print("-" * 40)
            print(f"  Shape: {action.shape}")
            print(f"  Dtype: {action.dtype}")
            print(f"  Values: {action}")
            
            # Interpretation je nach Dimension
            if action.shape == (4,):
                print("\n  Interpretation (4D - 2D Bewegung):")
                print(f"    start_x, start_z: [{action[0]:.4f}, {action[1]:.4f}]")
                print(f"    end_x, end_z:     [{action[2]:.4f}, {action[3]:.4f}]")
            elif action.shape == (6,):
                print("\n  Interpretation (6D - 3D Bewegung):")
                print(f"    prev_ee_pos: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]")
                print(f"    curr_ee_pos: [{action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}]")
        
        # --- EEF STATES ---
        if 'eef_states' in f:
            eef = f['eef_states'][:]
            print("\n📌 EEF_STATES (End-Effector States)")
            print("-" * 40)
            print(f"  Shape: {eef.shape}")
            print(f"  Dtype: {eef.dtype}")
            
            # Flatten für einfachere Analyse
            flat = eef.flatten()
            print(f"\n  Interpretation (14 Werte = 2x Position + 2x Quaternion):")
            print(f"    Position 1:   [{flat[0]:.4f}, {flat[1]:.4f}, {flat[2]:.4f}]")
            print(f"    Position 2:   [{flat[3]:.4f}, {flat[4]:.4f}, {flat[5]:.4f}]")
            print(f"    Quaternion 1: [{flat[6]:.4f}, {flat[7]:.4f}, {flat[8]:.4f}, {flat[9]:.4f}]")
            print(f"    Quaternion 2: [{flat[10]:.4f}, {flat[11]:.4f}, {flat[12]:.4f}, {flat[13]:.4f}]")
            
            # Prüfe ob Position 1 == Position 2
            pos_equal = np.allclose(flat[0:3], flat[3:6])
            quat_equal = np.allclose(flat[6:10], flat[10:14])
            print(f"\n  ⚠️  Pos1 == Pos2: {pos_equal}")
            print(f"  ⚠️  Quat1 == Quat2: {quat_equal}")
            print("  (Im DINO WM Format sind diese normalerweise identisch)")
        
        # --- POSITIONS ---
        if 'positions' in f:
            pos = f['positions'][:]
            print("\n📌 POSITIONS (Partikel/Objekt-Positionen)")
            print("-" * 40)
            print(f"  Shape: {pos.shape}")
            print(f"  Dtype: {pos.dtype}")
            
            if len(pos.shape) == 3:
                n_particles = pos.shape[1]
                values_per_particle = pos.shape[2]
                print(f"  n_particles: {n_particles}")
                print(f"  values_per_particle: {values_per_particle}")
                
                print(f"\n  Erste 3 Partikel/Objekte:")
                for i in range(min(3, n_particles)):
                    vals = pos[0, i, :]
                    if values_per_particle == 3:
                        print(f"    [{i}]: x={vals[0]:.4f}, y={vals[1]:.4f}, z={vals[2]:.4f}")
                    elif values_per_particle == 4:
                        print(f"    [{i}]: x={vals[0]:.4f}, y={vals[1]:.4f}, z={vals[2]:.4f}, w={vals[3]:.4f}")
        
        # --- INFO ---
        if 'info' in f:
            print("\n📌 INFO (Metadaten)")
            print("-" * 40)
            for key in f['info'].keys():
                val = f['info'][key][()]
                print(f"  {key}: {val}")
        
        # --- OBSERVATIONS ---
        if 'observations' in f:
            print("\n📌 OBSERVATIONS (Bilder)")
            print("-" * 40)
            obs = f['observations']
            
            if 'color' in obs:
                for cam in obs['color'].keys():
                    data = obs['color'][cam][:]
                    print(f"  color/{cam}: shape={data.shape}, dtype={data.dtype}, range=[{data.min():.1f}, {data.max():.1f}]")
            
            if 'depth' in obs:
                for cam in obs['depth'].keys():
                    data = obs['depth'][cam][:]
                    print(f"  depth/{cam}: shape={data.shape}, dtype={data.dtype}, range=[{data.min()}, {data.max()}]")


# =============================================================================
# 3. GLOBALE DATEIEN ANALYSIEREN (actions.pth, states.pth)
# =============================================================================

def analyze_global_files(dataset_path: str):
    """
    Analysiert die globalen Dateien eines Datensatzes:
    - actions.pth
    - states.pth
    - cameras/intrinsic.npy
    - cameras/extrinsic.npy
    
    Verwendung:
        analyze_global_files("pfad/zum/datensatz")
    """
    dataset_path = Path(dataset_path)
    
    print(f"\n{'='*60}")
    print(f"Globale Dateien: {dataset_path}")
    print('='*60)
    
    # --- ACTIONS.PTH ---
    actions_path = dataset_path / "actions.pth"
    if actions_path.exists():
        actions = torch.load(actions_path, weights_only=False)
        print("\n📌 ACTIONS.PTH")
        print("-" * 40)
        print(f"  Shape: {actions.shape}")
        print(f"  Dtype: {actions.dtype}")
        
        if len(actions.shape) == 3:
            print(f"  Interpretation: (n_episodes={actions.shape[0]}, n_timesteps={actions.shape[1]}, action_dim={actions.shape[2]})")
            print(f"\n  Erste Episode, erste 3 Actions:")
            for i in range(min(3, actions.shape[1])):
                print(f"    Timestep {i}: {actions[0, i, :].numpy()}")
    else:
        print(f"\n⚠️  actions.pth nicht gefunden: {actions_path}")
    
    # --- STATES.PTH ---
    states_path = dataset_path / "states.pth"
    if states_path.exists():
        states = torch.load(states_path, weights_only=False)
        print("\n📌 STATES.PTH")
        print("-" * 40)
        print(f"  Shape: {states.shape}")
        print(f"  Dtype: {states.dtype}")
        
        if len(states.shape) == 4:
            print(f"  Interpretation: (n_episodes={states.shape[0]}, n_timesteps={states.shape[1]}, n_particles={states.shape[2]}, values={states.shape[3]})")
        elif len(states.shape) == 3:
            print(f"  Interpretation: (n_episodes={states.shape[0]}, n_timesteps={states.shape[1]}, state_dim={states.shape[2]})")
    else:
        print(f"\n⚠️  states.pth nicht gefunden: {states_path}")
    
    # --- CAMERAS ---
    intrinsic_path = dataset_path / "cameras" / "intrinsic.npy"
    extrinsic_path = dataset_path / "cameras" / "extrinsic.npy"
    
    print("\n📌 CAMERA CALIBRATION")
    print("-" * 40)
    
    if intrinsic_path.exists():
        intrinsic = np.load(intrinsic_path)
        print(f"  intrinsic.npy: shape={intrinsic.shape}, dtype={intrinsic.dtype}")
    else:
        print(f"  ⚠️  intrinsic.npy nicht gefunden")
    
    if extrinsic_path.exists():
        extrinsic = np.load(extrinsic_path)
        print(f"  extrinsic.npy: shape={extrinsic.shape}, dtype={extrinsic.dtype}")
    else:
        print(f"  ⚠️  extrinsic.npy nicht gefunden")


# =============================================================================
# 4. EPISODEN-ORDNER ANALYSIEREN
# =============================================================================

def analyze_episode(episode_path: str):
    """
    Analysiert einen kompletten Episoden-Ordner.
    
    Zeigt:
    - Anzahl H5-Dateien (Timesteps)
    - obses.pth Format
    - property_params.pkl Inhalt
    
    Verwendung:
        analyze_episode("datensatz/000000")
    """
    episode_path = Path(episode_path)
    
    print(f"\n{'='*60}")
    print(f"Episode: {episode_path}")
    print('='*60)
    
    # H5-Dateien zählen
    h5_files = sorted(episode_path.glob("*.h5"))
    print(f"\n📌 H5-DATEIEN (Timesteps)")
    print("-" * 40)
    print(f"  Anzahl: {len(h5_files)}")
    if h5_files:
        print(f"  Erste: {h5_files[0].name}")
        print(f"  Letzte: {h5_files[-1].name}")
    
    # obses.pth
    obses_path = episode_path / "obses.pth"
    if obses_path.exists():
        obses = torch.load(obses_path, weights_only=False)
        print(f"\n📌 OBSES.PTH")
        print("-" * 40)
        print(f"  Shape: {obses.shape}")
        print(f"  Dtype: {obses.dtype}")
        if len(obses.shape) == 4:
            print(f"  Interpretation: (T={obses.shape[0]}, H={obses.shape[1]}, W={obses.shape[2]}, C={obses.shape[3]})")
        print(f"  Value range: [{obses.min():.1f}, {obses.max():.1f}]")
    else:
        print(f"\n⚠️  obses.pth nicht gefunden")
    
    # property_params.pkl
    params_path = episode_path / "property_params.pkl"
    if params_path.exists():
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
        print(f"\n📌 PROPERTY_PARAMS.PKL")
        print("-" * 40)
        for key, value in params.items():
            print(f"  {key}: {value}")
    else:
        print(f"\n⚠️  property_params.pkl nicht gefunden")


# =============================================================================
# 5. ZWEI DATENSÄTZE VERGLEICHEN
# =============================================================================

def compare_datasets(dataset1: str, dataset2: str):
    """
    Vergleicht zwei Datensätze und zeigt Unterschiede im Format.
    
    Verwendung:
        compare_datasets("referenz_datensatz", "mein_datensatz")
    """
    d1 = Path(dataset1)
    d2 = Path(dataset2)
    
    print(f"\n{'='*60}")
    print("DATENSATZ-VERGLEICH")
    print('='*60)
    print(f"  Dataset 1: {d1.name}")
    print(f"  Dataset 2: {d2.name}")
    
    # Finde erste Episode und erste H5 in beiden
    ep1 = sorted(d1.glob("0*"))[0] if list(d1.glob("0*")) else None
    ep2 = sorted(d2.glob("0*"))[0] if list(d2.glob("0*")) else None
    
    if not ep1 or not ep2:
        print("\n⚠️  Konnte keine Episoden finden")
        return
    
    h5_1 = sorted(ep1.glob("*.h5"))[0] if list(ep1.glob("*.h5")) else None
    h5_2 = sorted(ep2.glob("*.h5"))[0] if list(ep2.glob("*.h5")) else None
    
    if not h5_1 or not h5_2:
        print("\n⚠️  Konnte keine H5-Dateien finden")
        return
    
    print(f"\n📊 H5-FORMAT VERGLEICH")
    print("-" * 60)
    print(f"{'Feld':<20} {'Dataset 1':<20} {'Dataset 2':<20}")
    print("-" * 60)
    
    with h5py.File(h5_1, 'r') as f1, h5py.File(h5_2, 'r') as f2:
        # Vergleiche Felder
        fields = ['action', 'eef_states', 'positions']
        for field in fields:
            shape1 = f1[field].shape if field in f1 else "N/A"
            shape2 = f2[field].shape if field in f2 else "N/A"
            match = "✅" if str(shape1) == str(shape2) else "❌"
            print(f"{field:<20} {str(shape1):<20} {str(shape2):<20} {match}")


# =============================================================================
# 6. SCHNELL-DEBUG: Einzelne Werte extrahieren
# =============================================================================

def quick_debug(h5_path: str):
    """
    Schnelles Debugging: Extrahiert die wichtigsten Werte.
    
    Verwendung:
        quick_debug("000000/00.h5")
    """
    print(f"\n🔍 Quick Debug: {h5_path}")
    print("-" * 50)
    
    with h5py.File(h5_path, 'r') as f:
        # Shapes auf einen Blick
        print("Shapes:")
        for key in ['action', 'eef_states', 'positions']:
            if key in f:
                print(f"  {key}: {f[key].shape}")
        
        # Info
        if 'info' in f:
            print("\nInfo:")
            for key in f['info'].keys():
                print(f"  {key}: {f['info'][key][()]}")


# =============================================================================
# MAIN - Führt alle Analysen aus
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  H5 DATASET DEBUG TUTORIAL")
    print("="*70)
    
    # 1. Referenz-Datensatz analysieren
    print("\n\n" + "🔹"*30)
    print("  REFERENZ-DATENSATZ (deformable_rop_sample)")
    print("🔹"*30)
    
    ref_episode = REFERENCE_DATASET / "000000"
    ref_h5 = ref_episode / "00.h5"
    
    if ref_h5.exists():
        show_h5_structure(str(ref_h5))
        analyze_h5_timestep(str(ref_h5))
        analyze_episode(str(ref_episode))
        analyze_global_files(str(REFERENCE_DATASET))
    else:
        print(f"\n⚠️  Referenz-Datensatz nicht gefunden: {REFERENCE_DATASET}")
    
    # 2. Eigenen Datensatz analysieren (falls vorhanden)
    if MY_DATASET.exists():
        print("\n\n" + "🔹"*30)
        print("  EIGENER DATENSATZ")
        print("🔹"*30)
        
        my_episode = sorted(MY_DATASET.glob("0*"))[0] if list(MY_DATASET.glob("0*")) else None
        if my_episode:
            my_h5 = sorted(my_episode.glob("*.h5"))[0] if list(my_episode.glob("*.h5")) else None
            if my_h5:
                analyze_h5_timestep(str(my_h5))
    
    # 3. Vergleich
    if REFERENCE_DATASET.exists() and MY_DATASET.exists():
        compare_datasets(str(REFERENCE_DATASET), str(MY_DATASET))
    
    print("\n\n" + "="*70)
    print("  FERTIG! Nutze die einzelnen Funktionen für detaillierte Analysen.")
    print("="*70)
    print("""
Verfügbare Funktionen:
    show_h5_structure(h5_path)      - Zeigt H5-Dateistruktur
    analyze_h5_timestep(h5_path)    - Analysiert einen Timestep
    analyze_episode(episode_path)   - Analysiert eine Episode
    analyze_global_files(dataset)   - Analysiert actions.pth, states.pth
    compare_datasets(d1, d2)        - Vergleicht zwei Datensätze
    quick_debug(h5_path)            - Schnelle Übersicht
""")
