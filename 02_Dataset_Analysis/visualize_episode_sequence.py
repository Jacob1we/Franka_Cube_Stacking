#!/usr/bin/env python3
"""
visualize_episode_sequence.py

Visualisiert eine Episode als Bildsequenz oder GIF, um die zeitliche
Progression zu überprüfen.

Nutzt entweder:
1. dset_extracted/episodes/<ep>/timesteps/<ts>/<camera>.png
2. Oder direkt die obses.pth Datei aus dem Rohdatensatz

Aufruf:
  python visualize_episode_sequence.py --source <pfad> --episode 0 --camera rgb_cam_0.png
  python visualize_episode_sequence.py --source <pfad> --episode 0 --output episode_0.gif
"""


'''
# Kameras auflisten
python visualize_episode_sequence.py --source "pfad/dset_extracted" --episode 0 --list-cameras

# GIF erstellen
python visualize_episode_sequence.py --source "pfad/dset_extracted" --episode 0 --camera color_cam_0.png --output episode_0.gif

# Montage erstellen
python visualize_episode_sequence.py --source "pfad/dset_extracted" --episode 0 --camera color_cam_0.png --montage --output episode_0.png

# Frame-Differenzen analysieren (direkt aus obses.pth)
python visualize_episode_sequence.py --source "pfad/zum/datensatz" --episode 0 --analyze
'''

import argparse
from pathlib import Path
import sys

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def load_from_extracted(episodes_root: Path, episode_idx: int, camera_key: str):
    """Lade Bilder aus dset_extracted/episodes/<ep>/timesteps/<ts>/<camera>.png"""
    ep_name = f"{episode_idx:06d}"
    ep_dir = episodes_root / ep_name / "timesteps"
    
    if not ep_dir.exists():
        print(f"Episode-Ordner nicht gefunden: {ep_dir}")
        return []
    
    timesteps = sorted([t for t in ep_dir.iterdir() if t.is_dir()], 
                       key=lambda x: int(x.name) if x.name.isdigit() else x.name)
    
    images = []
    for ts_dir in timesteps:
        img_path = ts_dir / camera_key
        if img_path.exists():
            images.append((ts_dir.name, img_path))
        else:
            # Versuche ohne Pfad, nur Dateiname
            for f in ts_dir.iterdir():
                if f.name == camera_key or camera_key in f.name:
                    images.append((ts_dir.name, f))
                    break
    
    return images


def load_from_obses_pth(dataset_root: Path, episode_idx: int):
    """Lade Bilder direkt aus obses.pth"""
    if not HAS_TORCH:
        print("PyTorch nicht installiert - kann obses.pth nicht laden")
        return []
    
    ep_name = f"{episode_idx:06d}"
    obses_path = dataset_root / ep_name / "obses.pth"
    
    if not obses_path.exists():
        print(f"obses.pth nicht gefunden: {obses_path}")
        return []
    
    print(f"Lade {obses_path}...")
    obses = torch.load(obses_path, map_location='cpu')
    
    print(f"  Shape: {obses.shape}")
    print(f"  Dtype: {obses.dtype}")
    print(f"  Min/Max: {obses.min()}/{obses.max()}")
    
    return obses


def create_montage(images, cols=5, resize_to=(256, 256)):
    """Erstelle eine Montage aus mehreren Bildern"""
    if not HAS_PIL:
        print("PIL nicht installiert")
        return None
    
    if not images:
        return None
    
    # Lade alle Bilder
    pil_images = []
    for ts_name, img_path in images:
        img = Image.open(img_path)
        if resize_to:
            img = img.resize(resize_to, Image.Resampling.LANCZOS)
        pil_images.append((ts_name, img))
    
    if not pil_images:
        return None
    
    w, h = pil_images[0][1].size
    rows = (len(pil_images) + cols - 1) // cols
    
    montage = Image.new('RGB', (cols * w, rows * h), color=(30, 30, 30))
    
    for i, (ts_name, img) in enumerate(pil_images):
        row = i // cols
        col = i % cols
        montage.paste(img, (col * w, row * h))
    
    return montage


def create_gif(images, output_path: Path, duration_ms=200, resize_to=(256, 256)):
    """Erstelle ein animiertes GIF"""
    if not HAS_PIL:
        print("PIL nicht installiert")
        return False
    
    if not images:
        return False
    
    pil_images = []
    for ts_name, img_path in images:
        img = Image.open(img_path).convert('RGB')
        if resize_to:
            img = img.resize(resize_to, Image.Resampling.LANCZOS)
        pil_images.append(img)
    
    if not pil_images:
        return False
    
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"GIF gespeichert: {output_path}")
    return True


def analyze_obses(dataset_root: Path, episode_idx: int):
    """Analysiere die obses.pth Datei einer Episode"""
    if not HAS_TORCH or not HAS_NUMPY:
        print("PyTorch/NumPy nicht installiert")
        return
    
    obses = load_from_obses_pth(dataset_root, episode_idx)
    if obses is None or len(obses) == 0:
        return
    
    # Analysiere Unterschiede zwischen aufeinanderfolgenden Frames
    print("\n=== Frame-Differenz-Analyse ===")
    obses_np = obses.numpy().astype(np.float32)
    
    diffs = []
    for i in range(1, len(obses_np)):
        diff = np.abs(obses_np[i] - obses_np[i-1]).mean()
        diffs.append(diff)
    
    print(f"Durchschnittliche Differenz zwischen Frames: {np.mean(diffs):.4f}")
    print(f"Min Differenz: {np.min(diffs):.4f} (Frame {np.argmin(diffs)+1})")
    print(f"Max Differenz: {np.max(diffs):.4f} (Frame {np.argmax(diffs)+1})")
    
    # Zeige Differenzen
    print("\nDifferenzen zwischen aufeinanderfolgenden Frames:")
    for i, d in enumerate(diffs[:20]):  # Erste 20
        bar = '█' * int(d * 10)
        print(f"  Frame {i:3d} -> {i+1:3d}: {d:6.2f} {bar}")
    if len(diffs) > 20:
        print(f"  ... ({len(diffs)-20} weitere Frames)")


def list_available_cameras(episodes_root: Path, episode_idx: int):
    """Liste verfügbare Kamera-Keys"""
    ep_name = f"{episode_idx:06d}"
    ts_root = episodes_root / ep_name / "timesteps"
    
    if not ts_root.exists():
        return []
    
    # Nimm ersten Timestep
    first_ts = sorted(ts_root.iterdir())[0] if list(ts_root.iterdir()) else None
    if not first_ts:
        return []
    
    cameras = [f.name for f in first_ts.iterdir() if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    return sorted(cameras)


def main():
    parser = argparse.ArgumentParser(description='Visualisiere Episode-Sequenz')
    parser.add_argument('--source', '-s', required=True, help='Pfad zum Datensatz (dset_extracted oder Roh-Datensatz)')
    parser.add_argument('--episode', '-e', type=int, default=0, help='Episode-Index (default: 0)')
    parser.add_argument('--camera', '-c', default=None, help='Kamera-Key (z.B. rgb_cam_0.png)')
    parser.add_argument('--output', '-o', default=None, help='Output-Pfad für GIF oder Montage')
    parser.add_argument('--montage', action='store_true', help='Erstelle Montage statt GIF')
    parser.add_argument('--analyze', action='store_true', help='Analysiere obses.pth (Frame-Differenzen)')
    parser.add_argument('--list-cameras', action='store_true', help='Liste verfügbare Kameras')
    args = parser.parse_args()
    
    src = Path(args.source).expanduser().resolve()
    
    # Bestimme ob dset_extracted oder Roh-Datensatz
    if (src / 'episodes').exists():
        episodes_root = src / 'episodes'
        is_extracted = True
    elif (src / 'dset_extracted' / 'episodes').exists():
        episodes_root = src / 'dset_extracted' / 'episodes'
        is_extracted = True
    else:
        episodes_root = src
        is_extracted = False
    
    print(f"Source: {src}")
    print(f"Episodes root: {episodes_root}")
    print(f"Extracted format: {is_extracted}")
    print(f"Episode: {args.episode}")
    
    # Liste Kameras
    if args.list_cameras or args.camera is None:
        cameras = list_available_cameras(episodes_root, args.episode)
        print(f"\nVerfügbare Kameras in Episode {args.episode}:")
        for c in cameras:
            print(f"  - {c}")
        if args.list_cameras:
            return
        if not args.camera and cameras:
            # Wähle erste RGB-Kamera als Default
            rgb_cams = [c for c in cameras if 'rgb' in c.lower()]
            args.camera = rgb_cams[0] if rgb_cams else cameras[0]
            print(f"\nAuto-selected camera: {args.camera}")
    
    # Analysiere obses.pth
    if args.analyze:
        # Finde Roh-Datensatz-Root
        raw_root = src
        if is_extracted and 'dset_extracted' in str(src):
            raw_root = src.parent
        analyze_obses(raw_root, args.episode)
        return
    
    # Lade Bilder
    if is_extracted and args.camera:
        images = load_from_extracted(episodes_root, args.episode, args.camera)
        print(f"\nGefunden: {len(images)} Bilder für Kamera '{args.camera}'")
        
        if not images:
            print("Keine Bilder gefunden!")
            return
        
        # Zeige Timesteps
        print("Timesteps:", [ts for ts, _ in images[:10]], "..." if len(images) > 10 else "")
        
        # Output
        if args.output:
            out_path = Path(args.output)
            if args.montage or out_path.suffix.lower() in {'.png', '.jpg'}:
                montage = create_montage(images)
                if montage:
                    montage.save(out_path)
                    print(f"Montage gespeichert: {out_path}")
            else:
                create_gif(images, out_path)
        else:
            # Default: Erstelle GIF
            out_path = src / f"episode_{args.episode:06d}_{args.camera.replace('.png', '')}.gif"
            create_gif(images, out_path)
    else:
        print("Kein Kamera-Key angegeben und keine Bilder im extracted Format gefunden.")
        print("Verwende --analyze um obses.pth zu analysieren.")


if __name__ == '__main__':
    main()
