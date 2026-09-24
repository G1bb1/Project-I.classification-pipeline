'''
File:
kde_total_surface.py

Description:


Note:

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 23th, 2026     rev 1
'''
"""Part 2b -- the continuous decision surface: at
every point, the largest
of the per-class normalised densities.

Uses a FINE display grid, not the stored one: this
is the surface you
would have if memory were free. Part 3 samples it
onto GRID_N and stores
it as bytes -- comparing the two figures is the
whole point.
"""
import matplotlib.pyplot as plt
import numpy as np
from pipeline_common import (load_split_scaled,
                             fit_density, normalise_peak,
                             VIEW_LIM, REJECT_TAU)

BANDWIDTH = 0.05
DISPLAY_N = 160

X = load_split_scaled()["train"]  # surface is built from training data
labels = np.loadtxt("labels.csv", dtype=int)
clustered = X[labels != -1]
noise = X[labels == -1]  # unclassified (DBSCAN label -1)
classes = sorted(set(labels) - {-1})

g = np.linspace(*VIEW_LIM, DISPLAY_N)  # padded 10% past [0,1], like the 2-D plots
xx, yy = np.meshgrid(g, g)
fine = np.column_stack([xx.ravel(), yy.ravel()])
maps = [normalise_peak(np.exp(fit_density(X[labels == c], BANDWIDTH)
                              .score_samples(fine))
                       .reshape(xx.shape))
        for c in classes]
winning = np.stack(maps).max(axis=0)

fig = plt.figure(figsize=(5.6, 4.0))
ax = fig.add_subplot(projection="3d")

# The FULL probability surface, uncut, so it genuinely passes through the
# tau plane rather than stopping at it. Drawn first and translucent so the
# tau plane (drawn second) shows through where the surface dips below it,
# and the surface’s own peaks show through the plane where it rises above.
ax.plot_surface(xx, yy, winning, color="0.5", alpha=0.55,
                rcount=DISPLAY_N, ccount=DISPLAY_N,
                linewidth=0, antialiased=False)

pxx, pyy = np.meshgrid(VIEW_LIM, VIEW_LIM)
ax.plot_surface(pxx, pyy, np.full_like(pxx, REJECT_TAU), color="0.92",
                alpha=0.25, edgecolor="0.3",
                linewidth=0.8, shade=False)

# Where the surface meets the plane (winning == tau) is the decision
# boundary: a thin dotted line at the intersection, and the same boundary
# projected onto the feature plane below as a thin solid line.
ax.contour(xx, yy, winning, levels=[REJECT_TAU], colors="black",
           linewidths=0.8, linestyles=":")
ax.contour(xx, yy, winning, levels=[REJECT_TAU], colors="black",
           linewidths=0.8, linestyles="solid", offset=0)

# Mark tau explicitly: a dashed guide line up the front-left corner of the
# box to z = tau, with a label -- this corner is clear of the z-axis ticks
# so the plane’s height reads unambiguously.
corner = VIEW_LIM[0]
ax.plot([corner, corner], [corner, corner], [0, REJECT_TAU],
        color="black", lw=1.0, ls="--")
ax.text(corner, corner, REJECT_TAU, f"$\\tau={REJECT_TAU}$ ", fontsize=8,
        ha="right", va="bottom")

ax.scatter(clustered[:, 0], clustered[:, 1], zs=0, zdir="z",
           s=3, c="0.1", depthshade=False, label="classified")
ax.scatter(noise[:, 0], noise[:, 1], zs=0, zdir="z", s=12, facecolors="none",
           edgecolors="0.1", linewidths=0.6, depthshade=False,
           label="unclassified")

ax.set_xlim(*VIEW_LIM)
ax.set_ylim(*VIEW_LIM)
ax.set_zlim(0, 1.05)
ax.set_box_aspect((1, 1, 0.33))

ax.set_xlabel("scaled feature $x_1$")
ax.set_ylabel("scaled feature $x_2$")
ax.set_zlabel("winning density")
ax.set_title(f"Continuous decision surface ($\\tau={REJECT_TAU}$, $h={BANDWIDTH}$)")
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
ax.view_init(elev=30, azim=-58)

fig.savefig("../figures/generated/total_distribution.pdf",
            bbox_inches="tight", transparent=True)