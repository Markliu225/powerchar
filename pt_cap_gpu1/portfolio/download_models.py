"""Populate the local HF cache with every portfolio model -- run on an ONLINE machine, then copy
the cache to the offline measurement host (see OFFLINE_H200.zh.md).

  python3 download_models.py            # downloads all portfolio models (~44 GB)
  python3 download_models.py --list     # only print what would be downloaded + cache paths
"""
from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio import PORTFOLIO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    models = sorted({w["model_id"] for w in PORTFOLIO})
    print(f"portfolio needs {len(models)} models:")
    done = []
    for mid in models:
        if args.list:
            print(f"  {mid}")
            continue
        print(f"=== {mid} ===", flush=True)
        p = snapshot_download(mid, ignore_patterns=["*.pth", "*.onnx", "original/*", "*.gguf"])
        done.append(p)
        print(f"  -> {p}", flush=True)
    if done:
        # models--Org--Name dirs to copy (the snapshot path sits inside them)
        roots = sorted({os.path.dirname(os.path.dirname(p)) for p in done})
        print("\nALL CACHED. Copy these directories to the SAME location on the offline host")
        print("(default: ~/.cache/huggingface/hub/):")
        for r in roots:
            print(f"  {r}")
        print("\ne.g.  rsync -a " + os.path.commonpath(roots) + "/models--* offline-host:~/.cache/huggingface/hub/")


if __name__ == "__main__":
    main()
