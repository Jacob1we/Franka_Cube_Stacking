#!/usr/bin/env python3
"""
Dataset Decoder - Dekodiert H5, PTH und PKL Dateien aus Robotik-Datensätzen.
Funktioniert mit:
- FCS (Franka Cube Stacking) Datensätzen
- Deformable/Rope Datensätzen

Ausgabe:
- Klartext-Report (Konsole + TXT)
- Extrahierte Bilder (PNG)
- Strukturierte JSON-Zusammenfassung

Verwendung:
    python decode_dataset.py <dataset_path> [--episode <id>] [--output <dir>] [--all-frames]

Beispiele:
    python decode_dataset.py D:/00_Coding/fcs_datasets/2026_01_17_2042_fcs_dset
    python decode_dataset.py D:/00_Coding/deformable/deformable/rope --episode 0
    python decode_dataset.py ./my_dataset --output ./decoded --all-frames
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np

# Optional imports
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("⚠️  h5py nicht installiert - H5-Dateien können nicht gelesen werden")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  torch nicht installiert - PTH-Dateien können nicht gelesen werden")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️  PIL nicht installiert - Bilder können nicht gespeichert werden")


def format_array(arr: np.ndarray, max_elements: int = 20) -> str:
    """Formatiert ein Array für die Textausgabe."""
    if arr.size <= max_elements:
        return f"{arr.tolist()}"
    else:
        flat = arr.flatten()
        return f"[{flat[0]:.4f}, {flat[1]:.4f}, ... ({arr.size} elements) ..., {flat[-2]:.4f}, {flat[-1]:.4f}]"


def decode_h5_file(h5_path: Path, output_dir: Optional[Path] = None, 
                   save_images: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """
    Dekodiert eine H5-Datei und extrahiert alle Daten.
    
    Returns:
        Dictionary mit allen extrahierten Daten und Metadaten
    """
    if not HAS_H5PY:
        return {"error": "h5py nicht installiert"}
    
    result = {
        "file": str(h5_path),
        "datasets": {},
        "groups": [],
        "attributes": {},
    }
    
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"📦 H5-DATEI: {h5_path.name}")
    lines.append(f"{'='*80}")
    
    with h5py.File(h5_path, "r") as f:
        # Rekursiv alle Datasets und Groups durchgehen
        def visit_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                dtype_str = str(obj.dtype)
                shape = obj.shape
                
                # Metadaten speichern
                result["datasets"][name] = {
                    "shape": list(shape),
                    "dtype": dtype_str,
                    "size": obj.size,
                }
                
                lines.append(f"\n  📄 {name}")
                lines.append(f"     Shape: {shape}")
                lines.append(f"     Dtype: {dtype_str}")
                
                # Werte anzeigen
                if shape == ():  # Scalar
                    lines.append(f"     Wert: {data}")
                    result["datasets"][name]["value"] = data.item() if hasattr(data, 'item') else data
                elif obj.size <= 50:
                    lines.append(f"     Werte: {format_array(np.asarray(data))}")
                    result["datasets"][name]["values_preview"] = np.asarray(data).tolist()
                else:
                    lines.append(f"     Werte (Vorschau): {format_array(np.asarray(data))}")
                    lines.append(f"     Min: {np.min(data):.6f}, Max: {np.max(data):.6f}, Mean: {np.mean(data):.6f}")
                    result["datasets"][name]["stats"] = {
                        "min": float(np.min(data)),
                        "max": float(np.max(data)),
                        "mean": float(np.mean(data)),
                    }
                
                # Bilder speichern
                if save_images and output_dir and "color" in name and len(shape) >= 3:
                    img_data = np.asarray(data)
                    if img_data.ndim == 4:  # (batch, H, W, C)
                        img_data = img_data[0]
                    if img_data.max() > 1.0:
                        img_data = img_data.astype(np.uint8)
                    else:
                        img_data = (img_data * 255).astype(np.uint8)
                    
                    if HAS_PIL:
                        img_name = name.replace("/", "_") + ".png"
                        img_path = output_dir / "images" / h5_path.stem / img_name
                        img_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(img_data).save(img_path)
                        lines.append(f"     → Bild gespeichert: {img_path}")
                
                # Depth-Bilder speichern (als 16-bit PNG)
                if save_images and output_dir and "depth" in name and len(shape) >= 2:
                    depth_data = np.asarray(data)
                    if depth_data.ndim == 3:  # (batch, H, W)
                        depth_data = depth_data[0]
                    
                    if HAS_PIL:
                        # Normalisiere für Visualisierung
                        depth_vis = depth_data.astype(np.float32)
                        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
                        depth_vis = (depth_vis * 255).astype(np.uint8)
                        
                        img_name = name.replace("/", "_") + ".png"
                        img_path = output_dir / "images" / h5_path.stem / img_name
                        img_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(depth_vis, mode='L').save(img_path)
                        lines.append(f"     → Depth-Bild gespeichert: {img_path}")
                
                # Attribute
                if obj.attrs:
                    lines.append(f"     Attribute:")
                    for attr_name, attr_value in obj.attrs.items():
                        lines.append(f"       - {attr_name}: {attr_value}")
                        result["datasets"][name][f"attr_{attr_name}"] = str(attr_value)
                        
            elif isinstance(obj, h5py.Group):
                result["groups"].append(name)
                if obj.attrs:
                    for attr_name, attr_value in obj.attrs.items():
                        result["attributes"][f"{name}/{attr_name}"] = str(attr_value)
        
        f.visititems(visit_item)
        
        # Root-Attribute
        for attr_name, attr_value in f.attrs.items():
            result["attributes"][attr_name] = str(attr_value)
    
    if verbose:
        print("\n".join(lines))
    
    return result


def decode_pth_file(pth_path: Path, output_dir: Optional[Path] = None,
                    save_images: bool = True, all_frames: bool = False,
                    verbose: bool = True) -> Dict[str, Any]:
    """
    Dekodiert eine PTH-Datei (PyTorch Tensor).
    """
    if not HAS_TORCH:
        return {"error": "torch nicht installiert"}
    
    result = {
        "file": str(pth_path),
        "type": None,
        "shape": None,
        "dtype": None,
    }
    
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"🔥 PTH-DATEI: {pth_path.name}")
    lines.append(f"{'='*80}")
    
    data = torch.load(pth_path, map_location='cpu')
    
    if isinstance(data, torch.Tensor):
        result["type"] = "Tensor"
        result["shape"] = list(data.shape)
        result["dtype"] = str(data.dtype)
        result["min"] = float(data.min().item())
        result["max"] = float(data.max().item())
        result["mean"] = float(data.float().mean().item())
        
        lines.append(f"\n  Type: torch.Tensor")
        lines.append(f"  Shape: {data.shape}")
        lines.append(f"  Dtype: {data.dtype}")
        lines.append(f"  Min: {result['min']:.4f}, Max: {result['max']:.4f}, Mean: {result['mean']:.4f}")
        
        # Bilder speichern (wenn es Bilddaten sind)
        if save_images and output_dir and len(data.shape) >= 3:
            img_dir = output_dir / "images" / pth_path.stem
            img_dir.mkdir(parents=True, exist_ok=True)
            
            # Bestimme welche Frames gespeichert werden
            if data.ndim == 4:  # (T, H, W, C) oder (T, C, H, W)
                T = data.shape[0]
                if all_frames:
                    frames_to_save = list(range(T))
                else:
                    frames_to_save = [0, T//4, T//2, 3*T//4, T-1]  # 5 Frames
                    frames_to_save = list(set(frames_to_save))  # Duplikate entfernen
                
                for i in sorted(frames_to_save):
                    if i >= T:
                        continue
                    frame = data[i].numpy()
                    
                    # Format erkennen: (H, W, C) oder (C, H, W)
                    if frame.shape[-1] in [1, 3, 4]:  # (H, W, C)
                        pass
                    elif frame.shape[0] in [1, 3, 4]:  # (C, H, W)
                        frame = np.transpose(frame, (1, 2, 0))
                    
                    if frame.max() > 1.0:
                        frame = frame.astype(np.uint8)
                    else:
                        frame = (frame * 255).astype(np.uint8)
                    
                    if frame.shape[-1] == 1:
                        frame = frame.squeeze(-1)
                    
                    if HAS_PIL:
                        img_path = img_dir / f"frame_{i:04d}.png"
                        if frame.ndim == 2:
                            Image.fromarray(frame, mode='L').save(img_path)
                        else:
                            Image.fromarray(frame).save(img_path)
                        lines.append(f"  → Frame {i} gespeichert: {img_path}")
                
                lines.append(f"  → {len(frames_to_save)} Frames gespeichert in: {img_dir}")
    
    elif isinstance(data, dict):
        result["type"] = "dict"
        result["keys"] = list(data.keys())
        lines.append(f"\n  Type: dict")
        lines.append(f"  Keys: {list(data.keys())}")
        
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                lines.append(f"\n    {key}:")
                lines.append(f"      Shape: {value.shape}")
                lines.append(f"      Dtype: {value.dtype}")
            else:
                lines.append(f"\n    {key}: {type(value).__name__}")
    
    else:
        result["type"] = type(data).__name__
        lines.append(f"\n  Type: {type(data).__name__}")
        lines.append(f"  Content: {str(data)[:200]}...")
    
    if verbose:
        print("\n".join(lines))
    
    return result


def decode_pkl_file(pkl_path: Path, verbose: bool = True) -> Dict[str, Any]:
    """
    Dekodiert eine PKL-Datei (Pickle).
    """
    result = {
        "file": str(pkl_path),
        "type": None,
        "content": None,
    }
    
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"🥒 PKL-DATEI: {pkl_path.name}")
    lines.append(f"{'='*80}")
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    result["type"] = type(data).__name__
    lines.append(f"\n  Type: {type(data).__name__}")
    
    if isinstance(data, dict):
        result["keys"] = list(data.keys())
        result["content"] = {}
        lines.append(f"  Keys: {list(data.keys())}")
        lines.append(f"\n  Inhalt:")
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                lines.append(f"    {key}:")
                lines.append(f"      Type: numpy.ndarray")
                lines.append(f"      Shape: {value.shape}")
                lines.append(f"      Dtype: {value.dtype}")
                if value.size <= 20:
                    lines.append(f"      Werte: {value.tolist()}")
                result["content"][key] = {
                    "type": "ndarray",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            elif isinstance(value, (int, float, str, bool)):
                lines.append(f"    {key}: {value}")
                result["content"][key] = value
            elif isinstance(value, (list, tuple)):
                lines.append(f"    {key}: {type(value).__name__} mit {len(value)} Elementen")
                if len(value) <= 10:
                    lines.append(f"      Werte: {value}")
                result["content"][key] = {"type": type(value).__name__, "length": len(value)}
            else:
                lines.append(f"    {key}: {type(value).__name__}")
                result["content"][key] = {"type": type(value).__name__}
    else:
        result["content"] = str(data)[:500]
        lines.append(f"  Content: {str(data)[:500]}")
    
    if verbose:
        print("\n".join(lines))
    
    return result


def decode_npy_file(npy_path: Path, verbose: bool = True) -> Dict[str, Any]:
    """
    Dekodiert eine NPY-Datei (NumPy Array).
    """
    result = {
        "file": str(npy_path),
    }
    
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"📊 NPY-DATEI: {npy_path.name}")
    lines.append(f"{'='*80}")
    
    data = np.load(npy_path)
    
    result["shape"] = list(data.shape)
    result["dtype"] = str(data.dtype)
    result["min"] = float(np.min(data))
    result["max"] = float(np.max(data))
    
    lines.append(f"\n  Shape: {data.shape}")
    lines.append(f"  Dtype: {data.dtype}")
    lines.append(f"  Min: {result['min']:.6f}, Max: {result['max']:.6f}")
    
    if data.size <= 50:
        lines.append(f"  Werte:\n{data}")
        result["values"] = data.tolist()
    else:
        lines.append(f"  Werte (Vorschau): {format_array(data)}")
    
    if verbose:
        print("\n".join(lines))
    
    return result


def decode_episode(episode_dir: Path, output_dir: Optional[Path] = None,
                   all_frames: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """
    Dekodiert eine komplette Episode.
    """
    result = {
        "episode_path": str(episode_dir),
        "files": {},
    }
    
    print(f"\n{'#'*80}")
    print(f"# EPISODE: {episode_dir.name}")
    print(f"{'#'*80}")
    
    # PTH-Dateien
    for pth_file in sorted(episode_dir.glob("*.pth")):
        result["files"][pth_file.name] = decode_pth_file(
            pth_file, output_dir, save_images=True, all_frames=all_frames, verbose=verbose
        )
    
    # PKL-Dateien
    for pkl_file in sorted(episode_dir.glob("*.pkl")):
        result["files"][pkl_file.name] = decode_pkl_file(pkl_file, verbose=verbose)
    
    # H5-Dateien
    h5_files = sorted(episode_dir.glob("*.h5"))
    if h5_files:
        # Zeige Struktur der ersten H5-Datei ausführlich
        result["files"][h5_files[0].name] = decode_h5_file(
            h5_files[0], output_dir, save_images=True, verbose=verbose
        )
        
        # Für weitere H5-Dateien nur Zusammenfassung
        if len(h5_files) > 1:
            print(f"\n  ... und {len(h5_files)-1} weitere H5-Dateien mit gleicher Struktur")
            result["h5_count"] = len(h5_files)
    
    return result


def decode_dataset(dataset_path: Path, output_dir: Optional[Path] = None,
                   episode_id: Optional[int] = None, all_frames: bool = False,
                   verbose: bool = True) -> Dict[str, Any]:
    """
    Dekodiert einen kompletten Datensatz oder eine einzelne Episode.
    """
    result = {
        "dataset_path": str(dataset_path),
        "timestamp": datetime.now().isoformat(),
        "cameras": {},
        "episodes": {},
    }
    
    print(f"\n{'#'*80}")
    print(f"# DATENSATZ-DECODER")
    print(f"# Pfad: {dataset_path}")
    print(f"# Zeitstempel: {result['timestamp']}")
    print(f"{'#'*80}")
    
    # Output-Verzeichnis erstellen
    if output_dir is None:
        output_dir = dataset_path / "decoded_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Kamera-Kalibrierung
    cameras_dir = dataset_path / "cameras"
    if cameras_dir.exists():
        print(f"\n📷 KAMERA-KALIBRIERUNG:")
        for npy_file in sorted(cameras_dir.glob("*.npy")):
            result["cameras"][npy_file.name] = decode_npy_file(npy_file, verbose=verbose)
    
    # Episoden finden
    episode_dirs = sorted([
        d for d in dataset_path.iterdir() 
        if d.is_dir() and d.name.isdigit()
    ])
    
    print(f"\n📁 EPISODEN GEFUNDEN: {len(episode_dirs)}")
    
    if episode_id is not None:
        # Nur bestimmte Episode dekodieren
        episode_dir = dataset_path / f"{episode_id:06d}"
        if episode_dir.exists():
            result["episodes"][episode_dir.name] = decode_episode(
                episode_dir, output_dir, all_frames=all_frames, verbose=verbose
            )
        else:
            print(f"❌ Episode {episode_id} nicht gefunden!")
    else:
        # Erste und letzte Episode dekodieren (als Beispiel)
        if episode_dirs:
            print(f"\n  Dekodiere erste Episode als Beispiel...")
            result["episodes"][episode_dirs[0].name] = decode_episode(
                episode_dirs[0], output_dir, all_frames=all_frames, verbose=verbose
            )
            
            if len(episode_dirs) > 1:
                print(f"\n  ... {len(episode_dirs) - 1} weitere Episoden vorhanden")
    
    # Zusammenfassung speichern
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ Zusammenfassung gespeichert: {summary_path}")
    
    # Textreport speichern
    report_path = output_dir / "dataset_report.txt"
    # (Der Report wurde bereits auf stdout ausgegeben, hier nur Hinweis)
    print(f"✅ Bilder gespeichert in: {output_dir / 'images'}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Dekodiert Robotik-Datensätze (H5, PTH, PKL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python decode_dataset.py D:/00_Coding/fcs_datasets/2026_01_17_2042_fcs_dset
  python decode_dataset.py D:/00_Coding/deformable/deformable/rope --episode 0
  python decode_dataset.py ./my_dataset --output ./decoded --all-frames
        """
    )
    
    parser.add_argument("dataset_path", type=Path, 
                        help="Pfad zum Datensatz-Verzeichnis")
    parser.add_argument("--episode", "-e", type=int, default=None,
                        help="Nur bestimmte Episode dekodieren (z.B. 0, 1, 42)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Ausgabe-Verzeichnis (Default: <dataset>/decoded_output)")
    parser.add_argument("--all-frames", "-a", action="store_true",
                        help="Alle Frames als Bilder speichern (nicht nur Auswahl)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Weniger Ausgabe auf der Konsole")
    
    args = parser.parse_args()
    
    if not args.dataset_path.exists():
        print(f"❌ Pfad existiert nicht: {args.dataset_path}")
        sys.exit(1)
    
    decode_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output,
        episode_id=args.episode,
        all_frames=args.all_frames,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
