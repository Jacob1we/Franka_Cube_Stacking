#!/usr/bin/env python3
"""
extract_frames_by_timestep.py

Kopiert für jede Kamera und für jeden Timestep die Bilder aus allen Episoden
in Episodenreihenfolge in eine neue Ordnerstruktur:

DEST_ROOT/
  camera_<key>/
    timestep_<t>/
      000000.png
      000001.png
      ...

Aufruf:
  python extract_frames_by_timestep.py --source /path/to/dset_extracted --dest /path/to/out

Wenn --dest fehlt, wird neben `source` ein Ordner `by_camera_timestep` erzeugt.

Die Logik ist robust gegenüber unterschiedlichen Unterordnerstrukturen: der Dateiname
relativ zum Timestep-Ordner wird als camera-key verwendet (z.B. "camera_0.png" oder
"cam0/rgb.png").

"""
from pathlib import Path
import argparse
import shutil
import sys
import logging
from typing import Dict

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.exr'}

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('extract_frames')


def list_images_in_timestep(timestep_dir: Path) -> Dict[str, Path]:
    """Return mapping camera_key -> file path for files found under timestep_dir.
    camera_key is the relative path from timestep_dir to the file, using posix separators.
    If multiple files share same key (shouldn't happen), last one wins.
    """
    mapping = {}
    if not timestep_dir.exists():
        return mapping
    for p in timestep_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            try:
                rel = p.relative_to(timestep_dir)
            except Exception:
                rel = p.name
            # use posix style key so it's consistent across OS
            key = str(rel.as_posix())
            mapping[key] = p
    return mapping


def find_episodes(episodes_root: Path):
    if not episodes_root.exists():
        raise FileNotFoundError(f'Episodes root not found: {episodes_root}')
    eps = [p for p in sorted(episodes_root.iterdir()) if p.is_dir()]
    return eps


def gather_timesteps_for_episode(ep: Path):
    ts_root = ep / 'timesteps'
    if not ts_root.exists():
        # fallback: maybe timestep folders are directly in episode
        return [p for p in sorted(ep.iterdir()) if p.is_dir()]
    return [p for p in sorted(ts_root.iterdir()) if p.is_dir()]


def main():
    parser = argparse.ArgumentParser(description='Extract frames by timestep and camera from extracted dataset')
    parser.add_argument('--source', '-s', required=True, help='Path to dset_extracted (contains episodes/ )')
    parser.add_argument('--dest', '-d', default=None, help='Destination root directory')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files in destination')
    parser.add_argument('--dry-run', action='store_true', help='Do not copy, just print planned operations')
    args = parser.parse_args()

    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        logger.error('Source path does not exist: %s', src)
        sys.exit(1)

    # detect episodes folder
    if (src / 'episodes').exists():
        episodes_root = src / 'episodes'
    else:
        # maybe user passed path that *is* the episodes directory
        # check if it contains episode dirs like 000000
        maybe = src
        if any(child.name.isdigit() or child.name.startswith('000') for child in maybe.iterdir() if child.is_dir()):
            episodes_root = maybe
        else:
            logger.error('Could not find episodes/ under source. Found: %s', list(src.iterdir())[:10])
            sys.exit(1)

    episodes = find_episodes(episodes_root)
    if not episodes:
        logger.error('No episodes found under %s', episodes_root)
        sys.exit(1)

    logger.info('Found %d episodes (using %s)', len(episodes), episodes_root)

    # build set of all timestep folder names across episodes (use basename of timestep dirs)
    timestep_names = set()
    ep_to_timesteps = {}
    for ep in episodes:
        tdirs = gather_timesteps_for_episode(ep)
        names = [t.name for t in tdirs]
        ep_to_timesteps[ep.name] = {t.name: t for t in tdirs}
        timestep_names.update(names)

    timestep_names = sorted(timestep_names, key=lambda x: int(x) if x.isdigit() else x)
    logger.info('Discovered %d distinct timestep names (e.g. %s...)', len(timestep_names), timestep_names[:8])

    # gather camera keys by scanning available files in all episodes/timesteps
    camera_keys = set()
    # to limit expensive recursion, sample first n episodes (but we can scan all)
    for ep in episodes:
        tdirs = ep_to_timesteps.get(ep.name, {})
        for tname, tpath in tdirs.items():
            mapping = list_images_in_timestep(tpath)
            camera_keys.update(mapping.keys())
    if not camera_keys:
        logger.error('No image files found in any timestep folders under episodes')
        sys.exit(1)

    camera_keys = sorted(camera_keys)
    logger.info('Discovered %d camera keys (examples): %s', len(camera_keys), camera_keys[:8])

    # Destination
    if args.dest:
        dest_root = Path(args.dest).expanduser().resolve()
    else:
        dest_root = src.parent / (src.name + '_by_camera_timestep')
    dest_root.mkdir(parents=True, exist_ok=True)

    planned = 0
    copied = 0
    skipped = 0

    # For each camera_key and each timestep, gather episode images in episode order
    for cam_key in camera_keys:
        cam_dir = dest_root / cam_key.replace('/', '_')
        # sanitize cam_dir name
        cam_dir.mkdir(parents=True, exist_ok=True)
        for tname in timestep_names:
            out_t_dir = cam_dir / f'timestep_{tname}'
            out_t_dir.mkdir(parents=True, exist_ok=True)
            # for each episode in order
            for ep in episodes:
                ep_timesteps = ep_to_timesteps.get(ep.name, {})
                tpath = ep_timesteps.get(tname)
                if not tpath:
                    # episode doesn't have this timestep -> skip
                    continue
                # source path is tpath / cam_key
                src_file = tpath / cam_key
                if not src_file.exists():
                    # maybe cam_key was a nested path with directories; try to find file with same name in tpath
                    # fallback: look for a file with same basename
                    cand = None
                    for p in tpath.iterdir():
                        if p.is_file() and p.name == Path(cam_key).name:
                            cand = p
                            break
                    if cand is None:
                        # not found
                        continue
                    src_file = cand
                # target name: episode name + ext
                ext = src_file.suffix
                ep_out_name = f'{ep.name}{ext}'
                dest_file = out_t_dir / ep_out_name
                planned += 1
                if dest_file.exists() and not args.overwrite:
                    skipped += 1
                    continue
                if args.dry_run:
                    logger.info('DRY: would copy %s -> %s', src_file, dest_file)
                    continue
                try:
                    shutil.copy2(src_file, dest_file)
                    copied += 1
                except Exception as e:
                    logger.error('Failed to copy %s -> %s: %s', src_file, dest_file, e)

    logger.info('Planned copies: %d, copied: %d, skipped(existing): %d', planned, copied, skipped)
    logger.info('Done. Output root: %s', dest_root)


if __name__ == '__main__':
    main()
