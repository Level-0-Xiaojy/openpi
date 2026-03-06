"""Download pretrained checkpoints and assets from GCS to local cache.

Usage:
    # Download base checkpoints needed for x2robot fine-tuning:
    python scripts/download_checkpoints.py --checkpoints pi0_base pi05_base

    # Download everything:
    python scripts/download_checkpoints.py --all

    # List available checkpoints:
    python scripts/download_checkpoints.py --list

    # Download to a custom directory:
    python scripts/download_checkpoints.py --checkpoints pi0_base --cache-dir /path/to/cache
"""

import argparse
import concurrent.futures
import os
import pathlib
import shutil
import stat
import sys
import time
import urllib.parse

CHECKPOINTS = {
    # Base models for fine-tuning
    "pi0_base": "gs://openpi-assets/checkpoints/pi0_base",
    # "pi0_fast_base": "gs://openpi-assets/checkpoints/pi0_fast_base",
    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base",
    # Fine-tuned models
    # "pi0_fast_droid": "gs://openpi-assets/checkpoints/pi0_fast_droid",
    # "pi0_droid": "gs://openpi-assets/checkpoints/pi0_droid",
    # "pi0_aloha_towel": "gs://openpi-assets/checkpoints/pi0_aloha_towel",
    # "pi0_aloha_tupperware": "gs://openpi-assets/checkpoints/pi0_aloha_tupperware",
    # "pi0_aloha_pen_uncap": "gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap",
    # "pi0_aloha_sim": "gs://openpi-assets/checkpoints/pi0_aloha_sim",
    # "pi0_libero": "gs://openpi-assets/checkpoints/pi0_libero",
    # "pi05_libero": "gs://openpi-assets/checkpoints/pi05_libero",
    # "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid",
}

ASSETS = {
    "paligemma_tokenizer": "gs://big_vision/paligemma_tokenizer.model",
    "paligemma_weights": "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz",
}

DEFAULT_CACHE_DIR = os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")


def url_to_local_path(url: str, cache_dir: pathlib.Path) -> pathlib.Path:
    parsed = urllib.parse.urlparse(url)
    return (cache_dir / parsed.netloc / parsed.path.strip("/")).resolve()


def set_permissions(path: pathlib.Path) -> None:
    dir_perm = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO
    file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for d in dirs:
            (root_path / d).chmod(dir_perm)
        for f in files:
            fp = root_path / f
            perm = file_rw | (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH if fp.stat().st_mode & 0o100 else 0)
            fp.chmod(perm)


def download_one(url: str, cache_dir: pathlib.Path, *, force: bool = False) -> pathlib.Path:
    import fsspec
    import tqdm

    local_path = url_to_local_path(url, cache_dir)
    if local_path.exists() and not force:
        print(f"  Already cached: {local_path}")
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_path = local_path.with_suffix(local_path.suffix + ".partial")

    gs_kwargs = {"token": "anon"} if url.startswith("gs://") else {}
    fs, _ = fsspec.core.url_to_fs(url, **gs_kwargs)
    info = fs.info(url)
    is_dir = info["type"] == "directory" or (info["size"] == 0 and info["name"].endswith("/"))
    total_size = fs.du(url) if is_dir else info["size"]

    print(f"  Downloading {url}")
    print(f"    -> {local_path}  ({total_size / 1024**3:.2f} GiB)")

    with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fs.get, url, str(scratch_path), recursive=is_dir)
        while not future.done():
            current_size = sum(f.stat().st_size for f in [*scratch_path.rglob("*"), scratch_path] if f.is_file())
            pbar.update(current_size - pbar.n)
            time.sleep(1)
        future.result()
        pbar.update(total_size - pbar.n)

    shutil.move(str(scratch_path), str(local_path))
    set_permissions(local_path) if local_path.is_dir() else None
    return local_path


def main():
    parser = argparse.ArgumentParser(description="Download openpi pretrained checkpoints from GCS.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoints", nargs="+", metavar="NAME", choices=list(CHECKPOINTS.keys()),
                       help=f"Checkpoint names to download. Choices: {', '.join(CHECKPOINTS.keys())}")
    group.add_argument("--all", action="store_true", help="Download all checkpoints and assets.")
    group.add_argument("--list", action="store_true", help="List available checkpoints and exit.")
    parser.add_argument("--assets-only", action="store_true",
                        help="Only download shared assets (tokenizer, PaliGemma weights).")
    parser.add_argument("--skip-assets", action="store_true",
                        help="Skip downloading shared assets (tokenizer, PaliGemma weights).")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
                        help=f"Local cache directory (default: {DEFAULT_CACHE_DIR}).")
    parser.add_argument("--force", action="store_true", help="Re-download even if already cached.")
    args = parser.parse_args()

    if args.list:
        print("Available checkpoints:")
        for name, url in CHECKPOINTS.items():
            print(f"  {name:30s} {url}")
        print("\nShared assets (downloaded automatically unless --skip-assets):")
        for name, url in ASSETS.items():
            print(f"  {name:30s} {url}")
        return

    cache_dir = pathlib.Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    urls: list[tuple[str, str]] = []

    if not args.assets_only:
        names = list(CHECKPOINTS.keys()) if args.all else (args.checkpoints or [])
        for name in names:
            urls.append((name, CHECKPOINTS[name]))

    if not args.skip_assets:
        for name, url in ASSETS.items():
            urls.append((name, url))

    if not urls:
        print("Nothing to download.")
        return

    print(f"Cache directory: {cache_dir}")
    print(f"Downloading {len(urls)} item(s):\n")

    failed = []
    for name, url in urls:
        print(f"[{name}]")
        try:
            download_one(url, cache_dir, force=args.force)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed.append(name)
        print()

    if failed:
        print(f"\nFailed downloads: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
