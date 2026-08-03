"""Per-use-case-class power curves on H200 — thin wrapper over ../curves_lib.py.

Retargets the shared builder at the H200 dataset; the per-class co-plotted LOG layout, the
tok/J-on-measured-draw definition, the taxonomy/mapping/caveats, and the CSV are all reused from
the V100 module (nothing re-implemented, so the two hardwares cannot drift).

H200 data notes: PREFILL is clock-swept at a fixed 700 W cap, so its power axis is the MEASURED
draw power_avg_w (~300-710 W); DECODE is cap-swept (200-700 W).

python3 plot_power_curves.py -> fig_workload_power_throughput.png · fig_workload_power_tokj.png
                                workload_power_curves.csv     (all in this h200/ folder)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)                  # workload_analysis/
ROOT = os.path.dirname(PARENT)
sys.path.insert(0, PARENT)
import curves_lib as V                   # noqa: E402  the shared builder + taxonomy

# retarget the shared module's data/power-range globals at H200 (build_power_figs reads them)
V.DATA = os.path.join(ROOT, "data_h200")
V.CAP_LO, V.CAP_HI = 200.0, 700.0
V.F_MAX = V.fitlib.resolve_f_max(V.DATA)
V.CLK_FLOOR = 345.0                             # H200 lowest SM clock, for the eq. (3) calibration


def main():
    V.build_power_figs(os.path.join(PARENT, "workload_classes.csv"), HERE, "H200", "GPU power (W)")


if __name__ == "__main__":
    main()
