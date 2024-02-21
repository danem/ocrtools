from typing import List, Dict

# Turns {dog: ["puppy", "canine"]} into {canine: dog, puppy: dog}
# Useful for creating normalizing mappings
def invert_mapping (mapping: Dict):
    res = {}
    for k, v in mapping.items():
        if isinstance(v, list):
            for vx in v:
                res[vx] = k
        else:
            res[v] = k
    return res

