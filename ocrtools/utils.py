from typing import Dict
from collections import OrderedDict
import numpy as np
import ocrtools.types as otypes


# Turns {dog: ["puppy", "canine"]} into {canine: dog, puppy: dog}
# Useful for creating normalizing mappings
def invert_mapping(mapping: Dict):
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


# TODO: Might be better to place these elsewhere...
def page_space_to_image_space(pw, ph, dpi):
    dpi_scale = dpi / 72
    return np.array([[pw * dpi_scale, 0, 0], [0, ph * dpi_scale, 0], [0, 0, 1]])


def clip_space_to_page_space(clip: otypes.BBox):
    return np.array([[clip.width, 0, clip.x], [0, clip.height, clip.y], [0, 0, 1]])


# TODO: Not sure this is right
def page_space_to_clip_space(clip: otypes.BBox):
    return np.array([[1, 0, clip.x], [0, 1, clip.y], [0, 0, 1]])
