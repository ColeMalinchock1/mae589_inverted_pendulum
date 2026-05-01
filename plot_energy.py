"""
Plot kinetic, potential, and total mechanical energy of the cart-double-pendulum
over time. Reads a sim_log.csv with columns:
    t, phase, target_x, x, x_dot, theta1_deg, theta2_deg,
    theta1_dot, theta2_dot, u, ke, pe

Usage:
    python plot_energy.py [csv_path] [out_path]

Defaults:
    csv_path = sim_log.csv
    out_path = figs/single_energy.pdf
"""

import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_log(csv_path):
    times, ke, pe, phases = [], [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["t"]))
            ke.append(float(row["ke"]))
            pe.append(float(row["pe"]))
            phases.append(row["phase"].strip().lower())
    return (
        np.asarray(times),
        np.asarray(ke),
        np.asarray(pe),
        np.asarray(phases),
    )


def find_swingup_to_lqr(times, phases):
    """Return time at which the controller first transitions to LQR, or None."""
    for i in range(1, len(phases)):
        if phases[i] == "lqr" and phases[i - 1] != "lqr":
            return times[i]
    return None


def plot_energy(csv_path, out_path):
    times, ke, pe, phases = load_log(csv_path)
    E = ke + pe

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(times, ke, label=r"Kinetic energy $T$", linewidth=1.4, color="#1f77b4")
    ax.plot(times, pe, label=r"Potential energy $V$", linewidth=1.4, color="#d62728")
    ax.plot(times, E, label=r"Total energy $E = T + V$",
            linewidth=2.2, color="black")

    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.7)

    t_switch = find_swingup_to_lqr(times, phases)
    if t_switch is not None:
        ax.axvline(t_switch, color="#2ca02c", linewidth=1.0, linestyle=":",
                   label=fr"Swing-up$\to$LQR  ($t={t_switch:.2f}$\,s)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy [J]")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.margins(x=0.01)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    # Also save a PNG copy alongside for quick viewing.
    png_path = os.path.splitext(out_path)[0] + ".png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sim_log.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "figs/single_energy.pdf"
    plot_energy(csv_path, out_path)
