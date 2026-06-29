"""Extract the decode power<->throughput FRONTIER from the cap×batch sweep.

frontier = best throughput achievable within each power budget (Pareto upper-left envelope):
sort the cap×batch cloud by power, keep every point that sets a new throughput record.

  decode_pt.csv  ->  decode_frontier.csv   (columns: power_w, throughput_tok_s, batch, cap_w)
"""
from __future__ import annotations
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "decode_pt.csv"))))
P = np.array([float(r["power_avg_w"]) for r in rows])
T = np.array([float(r["throughput_tok_s"]) for r in rows])
B = np.array([int(float(r["batch"])) for r in rows])
C = np.array([int(float(r["cap_w"])) for r in rows])

order = np.argsort(P)
keep, mx = [], -1.0
for i in order:
    if T[i] > mx:
        keep.append(i); mx = T[i]

out = os.path.join(HERE, "decode_frontier.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["power_w", "throughput_tok_s", "batch", "cap_w"])
    for i in keep:
        w.writerow([round(P[i], 1), round(T[i], 1), B[i], C[i]])
print(f"wrote {out} ({len(keep)} frontier points of {len(rows)})")
