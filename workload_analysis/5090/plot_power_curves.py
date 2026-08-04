"""Per-use-case-class power curves on RTX 5090 — ⚠ MOCK DATA, thin wrapper over ../curves_lib.py.

The dataset (../../data_5090/) is SYNTHESIZED from the H200 fitted curves scaled by 5090 spec
ratios — no measurement was taken on a 5090 (see data_5090/make_mock_5090.py for the recipe).
Every figure title carries the MOCK tag. Layout/definitions identical to v100/ and h200/.

5090 data notes (mirrors H200 v4): PREFILL is clock-swept at a fixed 575 W cap, so its power axis
is the (synthesized) draw power_avg_w (~170-575 W); DECODE is cap-swept over 200-575 W on the
same geometric grid density as the H200 sweep.

python3 plot_power_curves.py -> fig_workload_power_throughput.png · fig_workload_power_tokj.png
                                workload_power_curves.csv     (all in this 5090/ folder)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)                  # workload_analysis/
ROOT = os.path.dirname(PARENT)
sys.path.insert(0, PARENT)
import curves_lib as V                   # noqa: E402  the shared builder + taxonomy

# retarget the shared module's data/power-range globals at the MOCK 5090 dataset
V.DATA = os.path.join(ROOT, "data_5090")
V.CAP_LO, V.CAP_HI = 160.0, 575.0               # prefill draws reach ~170 W; decode caps 200-575
V.CLK_FLOOR = 210.0                             # Blackwell lowest SM clock
V.F_MAX = V.fitlib.resolve_f_max(V.DATA)


def main():
    V.build_power_figs(os.path.join(PARENT, "workload_classes.csv"), HERE,
                       "RTX 5090 [MOCK DATA — synthesized, no measurement]", "GPU power (W)")


if __name__ == "__main__":
    main()
