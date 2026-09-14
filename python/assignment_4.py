from dtree import entropy, averageGain, select
from monkdata import attributes, monk1

for attribute in (attributes[4], attributes[5]):
    print(f"======== MONK 1 split on {attribute.name}, gain = {averageGain(monk1, attribute):.4f} ========")
    for v in attribute.values:
        subset = select(monk1, attribute, v)
        nPos = sum(x.positive for x in subset)
        print(f"{attribute.name} = {v}: |S_k| = {len(subset):3}, {nPos:2} positive, entropy = {entropy(subset):.4f}")
