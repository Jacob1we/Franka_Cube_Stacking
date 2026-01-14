"""
Kleines Skript um obses.pth zu analysieren - versucht verschiedene Methoden
"""
import sys
from pathlib import Path

episode_path = Path("/media/tsp_jw/fc8bca1b-cab8-4522-81d0-06172d2beae8/deformable/rope/000001")
obses_path = episode_path / "obses.pth"

print("Versuche obses.pth zu laden...")

# Methode 1: PyTorch
try:
    import torch
    obses = torch.load(obses_path, map_location='cpu')
    print(f"✅ PyTorch erfolgreich!")
    print(f"   Type: {type(obses)}")
    print(f"   Shape: {obses.shape}")
    print(f"   Dtype: {obses.dtype}")
    print(f"   Min: {obses.min().item()}, Max: {obses.max().item()}")
    sys.exit(0)
except ImportError:
    print("❌ PyTorch nicht verfügbar")
except Exception as e:
    print(f"❌ Fehler beim Laden mit PyTorch: {e}")

# Methode 2: Direktes Lesen der Datei (nur Größe)
print(f"\n📊 Datei-Informationen:")
print(f"   Größe: {obses_path.stat().st_size / (1024*1024):.2f} MB")
print(f"   Existiert: {obses_path.exists()}")

# Basierend auf H5-Daten können wir schätzen:
# 21 Timesteps × 4 Kameras × 224×224×3 (RGB) × 1 byte = ~12.6 MB
# Das passt zur Dateigröße von 12.06 MB
print(f"\n💡 Schätzung basierend auf H5-Daten:")
print(f"   21 Timesteps × 224×224×3 (RGB) = (21, 224, 224, 3)")
print(f"   Oder: 21 Timesteps × 4 Kameras × 224×224×3 = (21, 4, 224, 224, 3)")

