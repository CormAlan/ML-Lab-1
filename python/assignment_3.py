from dtree import entropy, averageGain
from monkdata import attributes, monk1, monk2, monk3

monks = (monk1, monk2, monk3)

for name, monk in zip(("MONK 1", "MONK 2", "MONK 3"), monks):
    print(f"======== {name} ========")
    for attribute in attributes:
        print(f"{attribute.name} = {averageGain(monk, attribute)}")
