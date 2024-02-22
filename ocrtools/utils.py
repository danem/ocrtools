from typing import List, Dict
from collections import OrderedDict


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


class CacheDict(OrderedDict):
    """Dict with a limited length, ejecting LRUs as needed."""

    def __init__(self, *args, cache_len: int = 10, **kwargs):
        assert cache_len > 0
        self.cache_len = cache_len

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val