# polymarket-nvidia-ceo-forecast-01: per-market realized-volatility quantile forecaster
# (Python 3.13, standard library only, no network, deterministic).
#
# Model, calibrated on the practice cases p00..p04 (ground truth taken from each
# case's successor case; p05 has no successor in the kit, so it is not scored):
#
#   * Centre: the market's last price x (persistence). No drift, no mean
#     reversion, no centre smoothing: every variant of those tested made the
#     practice pinball loss worse (drift k=0.2 -> 0.313 vs 0.297; smoothing
#     toward the last-15-min mean -> 0.318).
#   * Spread: q_tau(h) = x + z_tau * sigma_m * h**0.40, clamped to [0, 1] and
#     rounded to 6 decimals, where z_tau is the standard normal quantile.
#     - sigma_m is the market's own realized volatility: the winsorized RMS of
#       its 1-minute returns over the last 120 minutes (returns winsorized at
#       +/-0.05 so one jump does not dominate), per square-root minute.
#     - The 120-minute window matters: longer windows (720/1440/4320 min) are
#       poisoned by the launch-regime repricing (prices that went 0.01 -> 0.83
#       days ago) and their correlation with future |moves| is ~0 or negative,
#       while the last-120-min RMS correlates ~0.55-0.60 with the next 240-min
#       |move|.
#     - Spread grows as h**0.40, not sqrt(h): alpha=0.40 beats 0.50 clearly
#       (0.297 vs 0.334 at the same sigma), i.e. these prices mean-revert /
#       the 1-minute noise partly washes out over hours.
#     - sigma_m is floored at max(0.0005, 0.01 * x) (a market that has not
#       moved in 2 hours can still jump; the floor scales with price level)
#       and capped at 0.008.
#     - Mild asymmetry: upper quantiles (z > 0) are scaled by 0.9, lower
#       quantiles (z < 0) by 1.1. On the practice the empirical hit rates were
#       P(y < q75) = 0.83 and P(y < q95) = 0.91 (upper band too wide) versus
#       P(y < q25) = 0.18 (lower band too narrow); the 0.9/1.1 scaling is the
#       flat optimum of that grid (0.2964 vs 0.2973 symmetric).
#
# Practice pinball loss (p00..p04, mean over every case/market/minute/level):
#   this model  0.296418
#   house baseline (sigma 0.0025, sqrt(h))  0.392518  (official all-6-case
#   baseline is 0.363589; p05, the calmest case, has no ground truth in the kit)
# The model beats the baseline on every single practice case (p00 0.459 vs
# 0.536, p01 0.299 vs 0.380, p02 0.335 vs 0.408, p03 0.240 vs 0.369,
# p04 0.194 vs 0.299) and in every leave-one-case-out split.
#
# Usage: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# Reads every <id>.in in the input directory, writes <id>.out per case.

import json
import os
from math import sqrt

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles at the five levels, to 6 decimals.
Z = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
# Asymmetry factors: upper quantiles slightly narrower, lower slightly wider.
Z_SCALE = (1.1, 1.1, 1.0, 0.9, 0.9)

SIGMA_WINDOW = 120       # minutes of history used for realized volatility
SIGMA_WINSOR = 0.05      # winsorize 1-minute returns at +/- this
SIGMA_FLOOR_ABS = 0.0005
SIGMA_FLOOR_REL = 0.01   # floor also scales with the price level
SIGMA_CAP = 0.008
HORIZON_EXPONENT = 0.40


def realized_sigma(prices):
    """Winsorized RMS of 1-minute returns over the last SIGMA_WINDOW minutes,
    in price units per square-root minute."""
    w = prices[-SIGMA_WINDOW:] if len(prices) > SIGMA_WINDOW else prices
    n = len(w) - 1
    if n < 1:
        return 0.0
    cap = SIGMA_WINSOR
    s = 0.0
    for i in range(n):
        d = w[i + 1] - w[i]
        if d > cap:
            d = cap
        elif d < -cap:
            d = -cap
        s += d * d
    return sqrt(s / n)


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    return round(x, 6) + 0.0


def forecast_market(prices, horizon, horizon_pow):
    """240 (horizon) rows of five quantiles for one market."""
    x = prices[-1] if prices else 0.5
    sig = realized_sigma(prices)
    floor = SIGMA_FLOOR_ABS
    rel = SIGMA_FLOOR_REL * x
    if rel > floor:
        floor = rel
    if sig < floor:
        sig = floor
    elif sig > SIGMA_CAP:
        sig = SIGMA_CAP
    rows = []
    for h in range(1, horizon + 1):
        s = sig * horizon_pow[h]
        row = [price6(x + Z[j] * Z_SCALE[j] * s) for j in range(5)]
        # Defensive: rounding/clamping is monotone, so rows are already
        # non-decreasing; enforce it explicitly so a row can never violate
        # the contest's monotonicity rule through any floating-point edge.
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case["horizon_minutes"])
        horizon_pow = [0.0] + [h ** HORIZON_EXPONENT for h in range(1, horizon + 1)]
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon, horizon_pow)}
            for m in case["markets"]
        ]
        out = json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n"
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(out)


if __name__ == "__main__":
    main()
