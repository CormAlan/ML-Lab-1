from dtree import averageGain, bestAttribute, buildTree, mostCommon, select
from monkdata import attributes, monk1

root = bestAttribute(monk1, attributes)
rest = [a for a in attributes if a != root]
print(f"Root: {root}\n")

for v in root.values:
    subset = select(monk1, root, v)
    print(f"======== {root} = {v} ({len(subset)} samples) ========")
    print("  ".join(f"{a} = {averageGain(subset, a):.4f}" for a in rest))
    if all(x.positive for x in subset) or not any(x.positive for x in subset):
        print(f"pure -> leaf {'+' if mostCommon(subset) else '-'}\n")
        continue
    second = bestAttribute(subset, rest)
    # an empty branch inherits the parent's majority class, same as buildTree does
    leaves = [mostCommon(s) if (s := select(subset, second, w)) else mostCommon(subset)
              for w in second.values]
    print(f"split on {second} -> leaves {''.join('+' if c else '-' for c in leaves)}\n")

print(f"buildTree with depth 2: {buildTree(monk1, attributes, 2)}")
