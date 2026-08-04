"""MOCK RTX 5090 dataset — SYNTHESIZED, NO MEASUREMENT WAS TAKEN ON A 5090.

Generates data_5090/*.csv + meta.json in exactly the H200 v4 format (same columns, same sweep
methodology: decode = power-cap sweep, prefill = locked-clock sweep at the max cap), so every
downstream tool (fitlib calibration, curves_lib figures, the rack solver) runs unchanged.

HOW THE NUMBERS ARE MADE (mock recipe, deliberately simple and fully disclosed):

1. Shape source: for every workload, the H200 sweep is fitted with the repo's own analytical model
   (fitlib.fit_*_theory on the calibrated H200 power side) — prefill X = a*phi, decode
   X = 1/(max(a/x,b) + max(c/x,d) + e).

2. Spec scaling to a 5090 (per RTX 5090 datasheet vs H200, applied to EFFECTIVE, achieved rates —
   both cards run the same small-model FP16 eager-mode stack, so achieved fractions carry over):
      COMPUTE  x0.72  effective matmul vs H200-achieved  (datasheet ~210 vs ~990 TFLOPS FP16, but
                       the small kernels that cap H200's MFU here favour the higher-clocked
                       consumer part, so the achieved ratio lands far above 210/990)
      BW       x0.95  effective bandwidth vs H200-achieved (GDDR7 1.79 TB/s vs HBM3e 4.8 TB/s;
                       H200 realizes only ~2x V100 on these weight-bound decodes, a level GDDR7
                       nearly matches)
      OVERHEAD x0.85  per-step fixed cost (higher clocks, monolithic die)
   Compute-bound times scale by 1/COMPUTE, bandwidth-bound times by 1/BW, e by OVERHEAD.

3. Power side, eq. (3): P(x) = P_STAT + chi * x^GAMMA with consumer-part constants
   P_STAT = 150 W, GAMMA = 2.7, x = f/F_MAX (F_MAX = 3090 MHz, the top of the Blackwell clock
   table; sustained boost is capped at XBOOST = 0.935 ~ 2890 MHz). chi is per workload x phase:
   scaled off the H200 series' full-power draw fraction so heavy prefills peg the 575 W limit and
   light decodes idle far below it, exactly as on the real cards.

4. Sweeps simulated on that model + measurement noise (thr 3%, draw 0.6%), clocks
   quantized to the 15 MHz Blackwell step. Decode cap grid: geometric [200..575] W, the same
   density as the H200 sweep. Prefill clock grid 810..3090 MHz at a fixed 575 W cap, with the
   cap-droop (power_limited=True) reproduced at the top of the range.

python3 data_5090/make_mock_5090.py   -> 8 x {prefill,decode} CSVs + meta.json  (seeded, reproducible)
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pt_cap_gpu1", "portfolio"))
import fitlib                                                        # noqa: E402

H200 = os.path.join(ROOT, "data_h200")
F_MAX_H200, FLOOR_H200 = 1980.0, 345.0

# ---- RTX 5090 constants (datasheet / driver table; the mock's knobs) ----------------------------
F_MAX = 3090.0            # top of the Blackwell supported SM-clock table (MHz)
CLK_MIN, CLK_STEP = 210, 15
P_STAT, GAMMA = 150.0, 2.7   # consumer power side: GDDR7+board floor, aggressive V/f curve
TDP, PL_MIN = 575.0, 200.0   # mock sweeps the full 200-575 W range (H200-style grid density)
XBOOST = 0.935            # sustained boost ~2890 MHz when power allows
COMPUTE, BW, OVERHEAD = 0.72, 0.95, 0.85     # effective rates vs H200-achieved (see docstring)
# decode cap grid: geometric 200 -> 575 W, same density as the H200 sweep
CAP_GRID = [200, 225, 253, 284, 320, 359, 404, 454, 511, 575]
# prefill locked-clock grid: starts at 810 MHz — below that the dynamic power (~<10 W over P_stat)
# is inside measurement noise and the P->phi inversion is ill-conditioned, which only produces a
# stray off-curve point at the bottom of the figure
CLK_GRID = [810, 1110, 1410, 1710, 1950, 2190, 2430, 2670, 2880, 3090]

rng = np.random.default_rng(5090)
qclk = lambda x: int(round(x * F_MAX / CLK_STEP)) * CLK_STEP
noise = lambda s: 1.0 + rng.normal(0.0, s)


def read_h200(path):
    rows = [r for r in csv.DictReader(open(path)) if float(r["throughput_tok_s"]) > 0]
    cap = np.array([float(r["cap_w"]) for r in rows])
    thr = np.array([float(r["throughput_tok_s"]) for r in rows])
    clk = np.array([float(r["sm_clk_avg"]) for r in rows])
    pwr = np.array([float(r["power_avg_w"]) for r in rows])
    cap_swept = np.ptp(cap) > 1e-6
    if cap_swept:                                # same clock-floor rule as the rest of the repo
        k = clk > FLOOR_H200 * 1.02
        cap, thr, clk, pwr = cap[k], thr[k], clk[k], pwr[k]
        rows = [r for r, kk in zip(rows, k) if kk]
    return (cap if cap_swept else pwr), thr, clk, pwr, rows


def chi_for(draw_frac_h200, headroom=1.04):
    """Per-series dynamic-power span: a series that pegged the H200 cap pegs the 5090 cap too;
    a light one stays proportionally light. chi set so the sustained-boost UNBOUND draw hits the
    target; `headroom` > 1 lets that target exceed the TDP, which is what makes the cap actually
    droop the clock (a heavy tensor load on an uncapped 5090 transients well past 575 W — capping
    it must cost real clocks, not the 1-2 bins a pegged-at-TDP target would give). A deterministic
    per-series ±6% keeps the workloads from all pinching the cap at identical clocks."""
    target = max(P_STAT + 60.0, TDP * draw_frac_h200 * headroom) * rng.uniform(0.94, 1.06)
    return (target - P_STAT) / XBOOST ** GAMMA


def power_of(x, chi):
    return P_STAT + chi * np.asarray(x, float) ** GAMMA


def x_at_cap(cap, chi):
    """Governor clock under a cap: boost until either the cap or the boost ceiling binds."""
    x = ((cap - P_STAT) / chi) ** (1.0 / GAMMA)
    return float(min(XBOOST, x)), bool(x < XBOOST)


def util_temp(draw, phase):
    ug = int(np.clip((96 if phase == "prefill" else 98) + rng.normal(0, 1.2), 90, 100))
    um = int(np.clip((14 if phase == "prefill" else 52) + rng.normal(0, 3), 5, 70))
    t = round(40 + 34 * draw / TDP + rng.normal(0, 1.2), 1)          # consumer cooler: 54-75 C
    return ug, um, t


def main():
    cal_pre = fitlib.calibrate_power_side(H200, F_MAX_H200, FLOOR_H200, "prefill")
    cal_dec = fitlib.calibrate_power_side(H200, F_MAX_H200, FLOOR_H200, "decode")
    wids = sorted({f.rsplit("_", 1)[0] for f in os.listdir(H200) if f.endswith("_prefill.csv")})
    for wid in wids:
        Pp, Tp, Fp, Wp, prow = read_h200(os.path.join(H200, f"{wid}_prefill.csv"))
        Pd, Td, Fd, Wd, drow = read_h200(os.path.join(H200, f"{wid}_decode.csv"))
        B = float(drow[0]["batch"])
        _, pre = fitlib.fit_prefill_theory(Pp, Tp, Fp, F_MAX_H200, cal_pre)
        _, dec = fitlib.fit_decode_theory(Pd, Td, Fd, B, F_MAX_H200, cal_dec)

        # -- 5090 curve parameters: H200 shape x spec ratios (all rates at each card's own f_max) --
        # prefill: linear in the 5090's own x, with the RATIO enforced at the top of both ranges —
        # X_5090(top) = COMPUTE x X_h200(measured top). Anchoring on any H200 slope instead would
        # extrapolate its strongly sublinear thr-vs-clock curve (and its power-limited droop rows)
        # up to x~0.93 and make the mock overtake the card it was scaled DOWN from.
        chi_pre = chi_for(float(Wp.max()) / 700.0, headroom=1.35)   # unbound prefill draw ~750 W
        x_top, _ = x_at_cap(TDP, chi_pre)
        a_pre = COMPUTE * float(Tp.max()) / x_top
        a, b = dec["a"] / COMPUTE, dec["b"] / BW
        c, d = dec["c"] / COMPUTE, dec["d"] / BW
        e = dec["e"] * OVERHEAD
        chi_dec = chi_for(float(Wd.max()) / 700.0)
        Xdec = lambda x: 1.0 / (max(a / x, b) + max(c / x, d) + e)

        # ---------------- prefill: locked-clock sweep at the fixed 575 W cap ----------------
        out = []
        for cset in CLK_GRID:
            x_set = cset / F_MAX
            if power_of(x_set, chi_pre) <= TDP and x_set <= XBOOST:
                x, limited = x_set, False
                clk = cset                                     # clock holds as set
            else:                                              # cap (or boost ceiling) droops it
                x, _ = x_at_cap(TDP, chi_pre)
                limited, clk = True, qclk(x * noise(0.004))
            draw = min(TDP * 1.01, power_of(x, chi_pre)) * noise(0.006)
            thr = a_pre * x * noise(0.03)
            ug, um, t = util_temp(draw, "prefill")
            out.append(dict(phase="prefill", cap_w=int(TDP), throughput_tok_s=round(thr, 1),
                            power_avg_w=round(draw, 1), power_sample_avg_w=round(draw * noise(0.004), 1),
                            sm_clk_avg=clk, mem_clk_avg=14001, util_gpu_avg=ug, util_mem_avg=um,
                            tok_per_joule=round(thr / draw, 2), temp_avg=t,
                            throttle_mask="0x4" if limited else "0x0", thermal_frac=0.0,
                            mem_temp_avg=round(t + 4 + rng.normal(0, 0.6), 1),
                            seq_len=prow[0]["seq_len"], batch=prow[0]["batch"],
                            clk_set_mhz=cset, power_limited=limited))
        _write(wid, "prefill", out)

        # ---------------- decode: cap sweep over the 5090's whole legal range ----------------
        out = []
        for cap in CAP_GRID:
            x, limited = x_at_cap(cap, chi_dec)
            draw = min(power_of(x, chi_dec), cap * 1.01) * noise(0.006)
            thr = Xdec(x) * noise(0.03)
            ug, um, t = util_temp(draw, "decode")
            ctx = int(drow[0]["ctx"]); steps = 32
            out.append(dict(phase="decode", cap_w=cap, throughput_tok_s=round(thr, 1),
                            power_avg_w=round(draw, 1), power_sample_avg_w=round(draw * noise(0.004), 1),
                            sm_clk_avg=qclk(x * noise(0.004)), mem_clk_avg=14001,
                            util_gpu_avg=ug, util_mem_avg=um,
                            tok_per_joule=round(thr / draw, 2), temp_avg=t,
                            throttle_mask="0x4" if limited else "0x0", thermal_frac=0.0,
                            mem_temp_avg=round(t + 4 + rng.normal(0, 0.6), 1),
                            ctx=ctx, batch=int(B), ctx_eff=ctx + steps / 2, steps=steps,
                            window_s=round(max(2.5, steps * B / max(thr, 1e-9)), 2),
                            spread_pct=round(abs(rng.normal(0.4, 0.3)), 2), n_runs=2))
        _write(wid, "decode", out)
        print(f"{wid:18s} pre a={a_pre:9.1f}  dec plateau={1/(b+d+e):8.1f}  "
              f"chi_pre={chi_pre:5.0f} chi_dec={chi_dec:5.0f}")

    meta = dict(gpu_name="NVIDIA GeForce RTX 5090 (MOCK)", MOCK=True,
                driver="mock", torch="mock", cuda="mock",
                pl_min_w=PL_MIN, pl_max_w=TDP, pl_default_w=TDP, pl_enforced_at_start_w=TDP,
                methodology=("MOCK v1: SYNTHESIZED from the H200 fitted curves x (0.72 compute / "
                             "0.95 bandwidth / 0.85 overhead) with a consumer power side "
                             "(P_stat=150 W, gamma=2.7, boost ceiling 2890 MHz); NO MEASUREMENT. "
                             "Sweep layout mirrors H200 v4: decode cap sweep, prefill locked-clock "
                             "sweep at the 575 W cap. Generator: make_mock_5090.py, seed 5090."),
                supported_mem_clocks_mhz=[14001], sm_clk_min_mhz=CLK_MIN, f_max_mhz=int(F_MAX),
                supported_sm_clocks_mhz=list(range(CLK_MIN, int(F_MAX) + 1, CLK_STEP)),
                temp_slowdown_c=90, temp_shutdown_c=95, temp_gpu_max_c=77,
                cap_grid_w=CAP_GRID, prefill_clk_grid_mhz=CLK_GRID, prefill_cap_w=int(TDP))
    with open(os.path.join(HERE, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote meta.json")


def _write(wid, phase, rows):
    path = os.path.join(HERE, f"{wid}_{phase}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in rows]


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
