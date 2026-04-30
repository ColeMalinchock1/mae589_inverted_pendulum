"""

Generates result figures from sim_log.csv.

Produces 4 independent matplotlib figures:
    Figure 1 - Arm angles over time (success)
    Figure 2 - Cart position vs target over time (accuracy + speed)
    Figure 3 - Control input over time (effort)
    Figure 4 - Phase portrait: theta1 vs theta1_dot (swing-up trajectory)

Run from the src/ directory:
    python plot_results.py

@author Cole Malinchock (malinchc@gmail.com)
@date 4/30/2026

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────────

df = pd.read_csv("logs/sim_log(1.5,-1.5).csv")

# Find the time of first phase transition (swingup → lqr)
lqr_rows = df[df["phase"] == "lqr"]
t_switch = lqr_rows["t"].iloc[0] if not lqr_rows.empty else None

# Separate phases for coloring
swingup = df[df["phase"] == "swingup"]
lqr     = df[df["phase"] == "lqr"]

# Shared style
SWINGUP_COLOR = "#E07B39"
LQR_COLOR     = "#3A86FF"
GRID_ALPHA    = 0.3
PHASE_LINE_STYLE = dict(color="gray", linestyle="--", linewidth=1.2, alpha=0.8)

def add_phase_line(ax):
    """Draw vertical dashed line at swing-up → LQR transition."""
    if t_switch is not None:
        ax.axvline(t_switch, **PHASE_LINE_STYLE)
        ax.text(t_switch + 0.05, ax.get_ylim()[1] * 0.95,
                "LQR\nhandoff", fontsize=8, color="gray", va="top")

def shade_phases(ax):
    """Shade swingup and LQR regions as background bands."""
    ylim = ax.get_ylim()
    if t_switch is not None:
        ax.axvspan(df["t"].iloc[0], t_switch,
                   alpha=0.06, color=SWINGUP_COLOR, label="_swingup_bg")
        ax.axvspan(t_switch, df["t"].iloc[-1],
                   alpha=0.06, color=LQR_COLOR, label="_lqr_bg")
    ax.set_ylim(ylim)


# ── Figure 1: Arm angles over time ────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(df["t"], df["theta1_deg"], color="#C0392B", linewidth=1.5, label="θ₁ (link 1)")
ax1.plot(df["t"], df["theta2_deg"], color="#8E44AD", linewidth=1.5, label="θ₂ (link 2)")
ax1.axhline(0, color="black", linewidth=0.8, linestyle=":")

shade_phases(ax1)
add_phase_line(ax1)

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Angle (°)")
ax1.set_title("Arm Angles Over Time")
ax1.legend(loc="upper right")
ax1.grid(alpha=GRID_ALPHA)

# Add phase legend patches
ax1.legend(handles=[
    mpatches.Patch(color=SWINGUP_COLOR, alpha=0.4, label="Swing-up"),
    mpatches.Patch(color=LQR_COLOR,     alpha=0.4, label="LQR"),
    plt.Line2D([0], [0], color="#C0392B", linewidth=1.5, label="θ₁ (link 1)"),
    plt.Line2D([0], [0], color="#8E44AD", linewidth=1.5, label="θ₂ (link 2)"),
], loc="upper right", fontsize=9)

fig1.tight_layout()


# ── Figure 2: Cart position vs target ─────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(df["t"], df["x"], color="#2C3E50", linewidth=1.5, label="Cart position x")
ax2.plot(df["t"], df["target_x"], color="#27AE60", linewidth=1.2,
         linestyle="--", label="Target x")

shade_phases(ax2)
add_phase_line(ax2)

# Position error fill
ax2.fill_between(df["t"], df["x"], df["target_x"],
                 alpha=0.12, color="#E74C3C", label="Position error")

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Position (m)")
ax2.set_title("Cart Position vs Target")
ax2.legend(handles=[
    mpatches.Patch(color=SWINGUP_COLOR, alpha=0.4, label="Swing-up"),
    mpatches.Patch(color=LQR_COLOR,     alpha=0.4, label="LQR"),
    plt.Line2D([0], [0], color="#2C3E50", linewidth=1.5, label="Cart position x"),
    plt.Line2D([0], [0], color="#27AE60", linewidth=1.2, linestyle="--", label="Target x"),
    mpatches.Patch(color="#E74C3C",     alpha=0.25, label="Position error"),
], loc="upper right", fontsize=9)
ax2.grid(alpha=GRID_ALPHA)

# Annotate steady-state error if LQR phase exists
if not lqr_rows.empty and len(lqr_rows) > 10:
    steady = lqr_rows.tail(int(len(lqr_rows) * 0.1))
    ss_error = (steady["x"] - steady["target_x"]).abs().mean()
    ax2.annotate(f"Steady-state error:\n{ss_error:.4f} m",
                 xy=(lqr_rows["t"].iloc[-1], lqr_rows["x"].iloc[-1]),
                 xytext=(-80, 30), textcoords="offset points",
                 fontsize=8, color="#E74C3C",
                 arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1))

fig2.tight_layout()


# ── Figure 3: Control input over time ─────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(10, 5))

ax3.plot(swingup["t"], swingup["u"], color=SWINGUP_COLOR, linewidth=1.2, label="Swing-up")
ax3.plot(lqr["t"],     lqr["u"],     color=LQR_COLOR,     linewidth=1.2, label="LQR")

add_phase_line(ax3)
ax3.axhline(0, color="black", linewidth=0.6, linestyle=":")

ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Control force u (N)")
ax3.set_title("Control Input Over Time")
ax3.legend(fontsize=9)
ax3.grid(alpha=GRID_ALPHA)

fig3.tight_layout()


# ── Figure 4: Phase portrait — theta1 vs theta1_dot ───────────────────────────

fig4, ax4 = plt.subplots(figsize=(7, 7))

# Draw swing-up trajectory as a gradient from start to end
points_su = np.array([swingup["theta1_deg"], swingup["theta1_dot"]]).T.reshape(-1, 1, 2)
segments_su = np.concatenate([points_su[:-1], points_su[1:]], axis=1)

from matplotlib.collections import LineCollection
lc_su = LineCollection(segments_su, cmap="Oranges",
                        norm=plt.Normalize(0, len(segments_su)),
                        linewidth=1.5, alpha=0.9)
lc_su.set_array(np.arange(len(segments_su)))
ax4.add_collection(lc_su)

# Draw LQR trajectory
if not lqr.empty:
    points_lqr = np.array([lqr["theta1_deg"], lqr["theta1_dot"]]).T.reshape(-1, 1, 2)
    segments_lqr = np.concatenate([points_lqr[:-1], points_lqr[1:]], axis=1)
    lc_lqr = LineCollection(segments_lqr, cmap="Blues",
                             norm=plt.Normalize(0, len(segments_lqr)),
                             linewidth=1.5, alpha=0.9)
    lc_lqr.set_array(np.arange(len(segments_lqr)))
    ax4.add_collection(lc_lqr)

# Mark start, handoff, and end
ax4.scatter(swingup["theta1_deg"].iloc[0],  swingup["theta1_dot"].iloc[0],
            color="green", s=60, zorder=5, label="Start")
if t_switch is not None:
    ax4.scatter(swingup["theta1_deg"].iloc[-1], swingup["theta1_dot"].iloc[-1],
                color="gray",  s=60, zorder=5, marker="D", label="Handoff")
if not lqr.empty:
    ax4.scatter(lqr["theta1_deg"].iloc[-1],  lqr["theta1_dot"].iloc[-1],
                color="red",   s=60, zorder=5, marker="*", label="End")

# Mark the upright equilibrium
ax4.axvline(0, color="black", linewidth=0.8, linestyle=":")
ax4.axhline(0, color="black", linewidth=0.8, linestyle=":")
ax4.scatter(0, 0, color="black", s=80, zorder=6, marker="+", linewidths=2, label="Upright (0°, 0)")

ax4.autoscale()
ax4.set_xlabel("θ₁ (°)")
ax4.set_ylabel("θ̇₁ (rad/s)")
ax4.set_title("Phase Portrait — Link 1")
ax4.legend(fontsize=9)
ax4.grid(alpha=GRID_ALPHA)

# Colorbars
cb_su = fig4.colorbar(lc_su, ax=ax4, fraction=0.03, pad=0.04)
cb_su.set_label("Swing-up progression →", fontsize=8)

fig4.tight_layout()


# ── Show all ───────────────────────────────────────────────────────────────────

plt.show()