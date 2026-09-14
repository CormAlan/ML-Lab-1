from math import log2

# dtree.entropy works on datasets, so the distribution version is written out here
def entropy(p):
    return 0.0 - sum(x * log2(x) for x in p if x > 0)

DISTRIBUTIONS = [
    ("Two-headed coin", (0, 1)),
    ("Biased coin", (0.9, 0.1)),
    ("Fair coin", (0.5, 0.5)),
    ("Fair die", (1/6,) * 6),
    ("Loaded die", (0.5, 0.1, 0.1, 0.1, 0.1, 0.1)),
    ("Very loaded die", (0.95, 0.01, 0.01, 0.01, 0.01, 0.01)),
]

for name, p in DISTRIBUTIONS:
    print(f"{name:16} {entropy(p):.3f}")
