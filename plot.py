"""Plot throughput-vs-power curves for prefill and decode phases."""
import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def plot_phase(ax_tp, ax_eff, rows, color, label, xkey, xlabel):
    rows = sorted(rows, key=lambda r: r["throughput_tok_s"])
    tp = [r["throughput_tok_s"] for r in rows]
    pw = [r["power_avg_w"] for r in rows]
    # throughput vs power
    ax_tp.plot(tp, pw, "o-", color=color, label=label)
    for r in rows:
        ax_tp.annotate(f"{int(r[xkey])}", (r["throughput_tok_s"], r["power_avg_w"]),
                       fontsize=6, alpha=0.6, xytext=(2, 2), textcoords="offset points")
    # energy efficiency: tokens per joule = throughput / power
    eff = [t / p for t, p in zip(tp, pw)]
    ax_eff.plot(tp, eff, "s-", color=color, label=label)


def main():
    prefill = load("results_prefill.csv")
    decode = load("results_decode.csv")

    # Figure 1: the two requested curves (throughput vs power), separate panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    rows = sorted(prefill, key=lambda r: r["throughput_tok_s"])
    ax.plot([r["throughput_tok_s"] for r in rows], [r["power_avg_w"] for r in rows],
            "o-", color="tab:red")
    for r in rows:
        tag = f"b{int(r['batch'])}x{int(r['seq_len'])}"
        ax.annotate(tag, (r["throughput_tok_s"], r["power_avg_w"]),
                    fontsize=6, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(145, ls="--", color="gray", lw=0.8, label="145W cap")
    ax.set_title("PREFILL: throughput vs power")
    ax.set_xlabel("prefill throughput (tokens/s)")
    ax.set_ylabel("GPU power (W)")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    rows = sorted(decode, key=lambda r: r["throughput_tok_s"])
    ax.plot([r["throughput_tok_s"] for r in rows], [r["power_avg_w"] for r in rows],
            "o-", color="tab:blue")
    for r in rows:
        ax.annotate(f"b{int(r['batch'])}", (r["throughput_tok_s"], r["power_avg_w"]),
                    fontsize=6, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(145, ls="--", color="gray", lw=0.8, label="145W cap")
    ax.set_title("DECODE: throughput vs power")
    ax.set_xlabel("decode throughput (tokens/s)")
    ax.set_ylabel("GPU power (W)")
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle("RTX 5060 / Qwen2.5-1.5B-Instruct fp16 — throughput vs power")
    fig.tight_layout()
    fig.savefig("curves_throughput_vs_power.png", dpi=130)
    print("wrote curves_throughput_vs_power.png")

    # Figure 2: combined comparison + energy efficiency
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
    p = sorted(prefill, key=lambda r: r["throughput_tok_s"])
    d = sorted(decode, key=lambda r: r["throughput_tok_s"])
    axes2[0].plot([r["throughput_tok_s"] for r in p], [r["power_avg_w"] for r in p],
                  "o-", color="tab:red", label="prefill")
    axes2[0].plot([r["throughput_tok_s"] for r in d], [r["power_avg_w"] for r in d],
                  "o-", color="tab:blue", label="decode")
    axes2[0].set_xscale("log")
    axes2[0].axhline(145, ls="--", color="gray", lw=0.8, label="145W cap")
    axes2[0].set_title("throughput vs power (log-x)")
    axes2[0].set_xlabel("throughput (tokens/s, log)")
    axes2[0].set_ylabel("GPU power (W)")
    axes2[0].grid(alpha=0.3, which="both"); axes2[0].legend()

    axes2[1].plot([r["throughput_tok_s"] for r in p],
                  [r["throughput_tok_s"] / r["power_avg_w"] for r in p],
                  "o-", color="tab:red", label="prefill")
    axes2[1].plot([r["throughput_tok_s"] for r in d],
                  [r["throughput_tok_s"] / r["power_avg_w"] for r in d],
                  "o-", color="tab:blue", label="decode")
    axes2[1].set_xscale("log")
    axes2[1].set_title("energy efficiency: tokens per joule")
    axes2[1].set_xlabel("throughput (tokens/s, log)")
    axes2[1].set_ylabel("tokens / joule (tok/s / W)")
    axes2[1].grid(alpha=0.3, which="both"); axes2[1].legend()

    fig2.suptitle("RTX 5060 / Qwen2.5-1.5B-Instruct fp16 — efficiency")
    fig2.tight_layout()
    fig2.savefig("curves_efficiency.png", dpi=130)
    print("wrote curves_efficiency.png")

    # Figure 3: throughput AND power vs the offered-load sweep variable.
    # This disentangles the prefill curve: power saturates ~flat (compute-bound)
    # while throughput varies non-monotonically with sequence length.
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5))

    # prefill x-axis = total tokens per forward (batch*seq_len)
    p = sorted(prefill, key=lambda r: r["total_tokens"] / max(r["iters"], 1))
    px = [r["batch"] * r["seq_len"] for r in p]
    order = sorted(range(len(px)), key=lambda i: px[i])
    px = [px[i] for i in order]
    p = [p[i] for i in order]
    ax = axes3[0]
    ax.plot(px, [r["throughput_tok_s"] for r in p], "o-", color="tab:green",
            label="throughput")
    ax.set_xscale("log")
    ax.set_xlabel("offered load: tokens per prefill forward (batch x seq_len, log)")
    ax.set_ylabel("prefill throughput (tokens/s)", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.grid(alpha=0.3, which="both")
    axp = ax.twinx()
    axp.plot(px, [r["power_avg_w"] for r in p], "s--", color="tab:red",
             label="power")
    axp.axhline(145, ls=":", color="gray", lw=0.8)
    axp.set_ylabel("GPU power (W)", color="tab:red")
    axp.tick_params(axis="y", labelcolor="tab:red")
    axp.set_ylim(0, 155)
    ax.set_title("PREFILL vs offered load")

    # decode x-axis = batch size
    d = sorted(decode, key=lambda r: r["batch"])
    dx = [r["batch"] for r in d]
    ax = axes3[1]
    ax.plot(dx, [r["throughput_tok_s"] for r in d], "o-", color="tab:green",
            label="throughput")
    ax.set_xscale("log")
    ax.set_xlabel("offered load: decode batch size (concurrent seqs, log)")
    ax.set_ylabel("decode throughput (tokens/s)", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.grid(alpha=0.3, which="both")
    axp = ax.twinx()
    axp.plot(dx, [r["power_avg_w"] for r in d], "s--", color="tab:blue",
             label="power")
    axp.axhline(145, ls=":", color="gray", lw=0.8)
    axp.set_ylabel("GPU power (W)", color="tab:blue")
    axp.tick_params(axis="y", labelcolor="tab:blue")
    axp.set_ylim(0, 155)
    ax.set_title("DECODE vs offered load")

    fig3.suptitle("RTX 5060 / Qwen2.5-1.5B-Instruct fp16 — throughput & power vs load")
    fig3.tight_layout()
    fig3.savefig("curves_vs_load.png", dpi=130)
    print("wrote curves_vs_load.png")


if __name__ == "__main__":
    main()
