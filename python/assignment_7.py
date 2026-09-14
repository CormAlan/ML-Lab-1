import random
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dtree import allPruned, buildTree, check
from monkdata import attributes, monk1, monk1test, monk3, monk3test

FRACTIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RUNS = 100
DATASETS = [
    ("MONK-1", monk1, monk1test, "#2a78d6"),
    ("MONK-3", monk3, monk3test, "#eb6834"),
]


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def prune(tree, validation):
    accuracy = check(tree, validation)
    while candidates := allPruned(tree):
        bestAccuracy, best = max(((check(t, validation), t) for t in candidates), key=lambda s: s[0])
        # ties are accepted: at equal validation accuracy the smaller tree wins
        if bestAccuracy < accuracy:
            break
        tree, accuracy = best, bestAccuracy
    return tree


random.seed(2421)
results = {}
for name, train, test, _ in DATASETS:
    fullError = 1 - check(buildTree(train, attributes), test)
    rows = []
    for fraction in FRACTIONS:
        pruned, unpruned = [], []
        for _ in range(RUNS):
            t, v = partition(train, fraction)
            tree = buildTree(t, attributes)
            unpruned.append(1 - check(tree, test))
            pruned.append(1 - check(prune(tree, v), test))
        rows.append((fraction, mean(pruned), stdev(pruned), mean(unpruned)))
    results[name] = (fullError, rows)

    print(f"\n======== {name} (unpruned, full training set: {fullError:.4f}) ========")
    print("fraction  pruned mean  pruned std  unpruned mean (same partition)")
    for fraction, m, s, u in rows:
        print(f"{fraction:8}  {m:11.4f}  {s:10.4f}  {u:10.4f}")
    best = min(rows, key=lambda r: r[1])
    print(f"best fraction: {best[0]} (mean test error {best[1]:.4f})")

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for name, _, _, color in DATASETS:
    fullError, rows = results[name]
    xs = [r[0] for r in rows]
    ax.axhline(fullError, color=color, linewidth=1, alpha=0.45, zorder=1,
               label=f"{name} obeskuret, hela träningsmängden")
    ax.plot(xs, [r[1] for r in rows], color=color, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
            label=f"{name} beskuret (medel över {RUNS} körningar)")
    ax.annotate(name, (xs[-1], rows[-1][1]), xytext=(10, 0), textcoords="offset points",
                va="center", color=INK_2, fontsize=10)

ax.set_xticks(FRACTIONS)
ax.set_xlim(0.26, 0.86)
ax.set_ylim(bottom=0)
ax.set_xlabel("fraction (andel av träningsdatan som bygger trädet)", color=INK_2)
ax.set_ylabel("testfel", color=INK_2)
ax.set_title(f"Testfel efter reduced error pruning, {RUNS} slumpade uppdelningar per punkt",
             color=INK, fontsize=11, loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
ax.tick_params(colors=MUTED, labelcolor=INK_2)
ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=2,
          loc="upper center", bbox_to_anchor=(0.5, -0.16))
fig.tight_layout()
fig.savefig("assignment_7.png", dpi=150, facecolor=SURFACE)
