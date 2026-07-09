"""Pre-flight gate for an (offline) measurement host -- run BEFORE burning GPU hours.

Verifies, without downloading anything (HF_HUB_OFFLINE-safe):
  1. python deps + versions (torch/CUDA build, transformers, pynvml, numpy; matplotlib optional)
  2. the target GPU is visible to torch AND NVML, and reports its power-limit constraints
  3. `nvidia-smi -pl` privilege actually works (benign: re-sets the current enforced limit)
  4. every portfolio model is FULLY present in the local HF cache (snapshot with weights)
  5. capability probes: energy counter (window-power method), throttle-reason telemetry
Exit code 0 = ready to run; 1 = something is missing (each failure printed with the fix).

  CUDA_VISIBLE_DEVICES=0 [SUDO_PASS=...] python3 preflight.py
"""
from __future__ import annotations
import os, subprocess, sys

GPU = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")
HERE = os.path.dirname(os.path.abspath(__file__))
FAIL = []


def check(name, ok, detail="", fix=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)
        if fix:
            print(f"         fix: {fix}")
    return ok


def main():
    print("== preflight: deps ==")
    try:
        import torch
        check("torch", True, f"{torch.__version__} cuda={torch.version.cuda}")
        check("torch.cuda", torch.cuda.is_available(),
              f"devices={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
              "install a CUDA build of torch matching the driver")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(int(GPU) if int(GPU) < torch.cuda.device_count() else 0)
            cap = torch.cuda.get_device_capability(0)
            check("gpu", True, f"{name} sm_{cap[0]}{cap[1]}")
            try:                       # a wheel without this arch's kernels passes import but dies here
                (torch.zeros(8, device="cuda") + 1).sum().item()
                check("cuda kernel executes", True)
            except Exception as e:
                check("cuda kernel executes", False, str(e)[:80],
                      "torch wheel lacks this GPU's sm_XX kernels -- install a matching CUDA build")
    except Exception as e:
        check("torch", False, str(e), "pip install torch (CUDA build)")
    try:
        import transformers
        ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        check("transformers", ver >= (4, 51), transformers.__version__, ">=4.51 needed for Qwen3")
        if (4, 51) <= ver < (4, 56):
            print("  [WARN] transformers 4.51-4.55: from_pretrained(dtype=) may be ignored/crash --"
                  " the loaders carry a fallback shim, but >=4.56 is safer")
    except Exception as e:
        check("transformers", False, str(e), "pip install 'transformers>=4.51'")
    try:
        import numpy
        check("numpy", True, numpy.__version__)
    except Exception as e:
        check("numpy", False, str(e), "pip install numpy")
    try:
        import matplotlib
        check("matplotlib (optional, for plots)", True, matplotlib.__version__)
    except Exception:
        print("  [WARN] matplotlib missing -- sweeps will run, plots will not")

    print("== preflight: NVML / power-cap privilege ==")
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(int(GPU))
        nm = pynvml.nvmlDeviceGetName(h)
        nm = nm.decode() if isinstance(nm, bytes) else nm
        mn, mx = [x / 1000 for x in pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)]
        cur = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000
        check("nvml", True, f"{nm} | -pl range [{mn:.0f},{mx:.0f}]W, enforced {cur:.0f}W")
        try:
            if pynvml.nvmlDeviceGetPersistenceMode(h) != 1:
                print("  [WARN] persistence mode OFF -- run `sudo nvidia-smi -pm 1` for stable"
                      " clocks and a monotonic energy counter")
        except pynvml.NVMLError:
            pass
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            check("energy counter (window-power method)", True)
        except Exception:
            print("  [WARN] no energy counter -- falls back to sampled power average")
        # benign privilege test: set the cap to its current value
        pw = os.environ.get("SUDO_PASS", "")
        cmd = ["nvidia-smi", "-i", GPU, "-pl", str(int(round(cur)))]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            r = subprocess.run(cmd, text=True, capture_output=True)
        elif pw:
            r = subprocess.run(["sudo", "-S", "-p", ""] + cmd, input=pw + "\n",
                               text=True, capture_output=True)
        else:
            r = subprocess.run(["sudo", "-n"] + cmd, text=True, capture_output=True)
        check("nvidia-smi -pl privilege", r.returncode == 0,
              (r.stderr or r.stdout).strip().splitlines()[-1][:70] if (r.stderr or r.stdout) else "",
              "run as root, or add a NOPASSWD sudoers rule for nvidia-smi, or export SUDO_PASS")
    except Exception as e:
        check("nvml", False, str(e), "driver/NVML mismatch? reboot after driver update")

    print("== preflight: models in local HF cache ==")
    sys.path.insert(0, HERE)
    from portfolio import PORTFOLIO
    os.environ.setdefault("HF_HUB_OFFLINE", "1")     # NEVER hit the network from here
    try:
        from huggingface_hub import snapshot_download
        for mid in sorted({w["model_id"] for w in PORTFOLIO}):
            try:
                p = snapshot_download(mid, local_files_only=True,
                                      ignore_patterns=["*.pth", "*.onnx", "original/*", "*.gguf"])
                import glob
                wf = glob.glob(os.path.join(p, "*.safetensors")) or glob.glob(os.path.join(p, "*.bin"))
                # snapshots are symlinks into blobs/ -- a partial rsync leaves them DANGLING,
                # so resolve and stat every weight file instead of trusting the listing
                ok_w, bad = bool(wf), ""
                for x in wf:
                    try:
                        if os.path.getsize(os.path.realpath(x)) <= 0:
                            ok_w, bad = False, f"empty: {os.path.basename(x)}"; break
                    except OSError:
                        ok_w, bad = False, f"dangling symlink: {os.path.basename(x)}"; break
                check(mid, ok_w, p if ok_w else (bad or "snapshot has no weight files"),
                      "re-copy the FULL model dir (snapshots/ AND blobs/) from the online machine")
            except Exception as e:
                check(mid, False, str(e)[:80],
                      "copy ~/.cache/huggingface/hub/models--... from the online machine")
    except Exception as e:
        check("huggingface_hub", False, str(e), "pip install huggingface_hub")

    print(f"\npreflight: {'READY' if not FAIL else f'{len(FAIL)} FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
