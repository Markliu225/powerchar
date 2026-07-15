"""Per-use-case-class power curves on V100 — thin wrapper over ../curves_lib.py.

Symmetric with h200/plot_power_curves.py. The split-phase / co-plotted layout, the updated
first-principles fits (fitlib.fit_*_theory), the taxonomy/mapping/caveats and the CSV are all in
the shared library; this wrapper just runs it on the default V100 dataset (pt_cap_gpu1/portfolio/
data, caps [100,250] W) and writes into this v100/ folder.

python3 plot_power_curves.py -> fig_workload_power_throughput.png · fig_workload_power_tokj.png
                                workload_power_curves.csv     (all in this v100/ folder)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)                  # workload_analysis/
sys.path.insert(0, PARENT)
import curves_lib as V                          # noqa: E402  shared builder + taxonomy (V100 defaults)


def main():
    V.build_power_figs(os.path.join(PARENT, "workload_ratios.csv"), HERE, "V100", "power cap (W)")


if __name__ == "__main__":
    main()
