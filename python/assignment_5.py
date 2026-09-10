from dtree import buildTree, check
from monkdata import attributes, monk1, monk1test, monk2, monk2test, monk3, monk3test

DATASETS = [
    ("Monk 1", monk1, monk1test), 
    ("Monk 2", monk2, monk2test), 
    ("Monk 3",  monk3, monk3test)
]

for (name, ds, test_ds) in DATASETS:
    print(f"\n======== {name} ========\n")
    print(tree := buildTree(ds, attributes))
    print(f"\n======== TRAIN")
    print(check(tree, ds))
    print(f"\n======== TEST")
    print(check(tree, test_ds))
