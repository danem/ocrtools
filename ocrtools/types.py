from typing import Tuple, List, Any, Callable
import numpy as np

# Boxes represent positions in relative space (ie as a percentage)
# Internal and default representation is XYXY
class BBox:
    def __init__(self,x1,y1,x2,y2) -> None:
        self._points = (x1,y1,x2,y2)
    
    def __getitem__ (self, key):
        return self._points[key]

    @property
    def x (self):
        return self._points[0]

    @property
    def y (self):
        return self._points[1]

    @property
    def width (self):
        return self._points[2] - self._points[0]

    @property
    def height (self):
        return self._points[3] - self._points[1]
    
    @staticmethod
    def from_xywh (x: float, y: float, w: float, h: float):
        return BBox(x,y, x+w, y+h)
    
    @staticmethod
    def from_xyxy (x1: float, y1: float, x2: float, y2: float):
        return BBox(x1, y1, x2, y2)
    
    def transform (self, mat: np.array):
        tl = mat @ np.array([self._points[0], self._points[1], 1])
        br = mat @ np.array([self._points[2], self._points[3], 1])
        return BBox.from_xyxy(
            tl[0], tl[1],
            br[0], br[1]
        )
    
    def translate (self, x, y):
        return BBox.from_xyxy(
            self._points[0] + x,
            self._points[1] + y,
            self._points[2] + x,
            self._points[3] + y,
        )

    def scale (self, x, y):
        return BBox.from_xyxy(
            self._points[0] * x,
            self._points[1] * y,
            self._points[2] * x,
            self._points[3] * y,
        )
        
    
    def as_tuple (self, format = "xyxy", width = 1, height = 1):
        format = format.lower()
        if format == "xyxy":
            return (self._points[0] * width, self._points[1] * height, self._points[2] * width, self._points[3] * height)
        elif format == "xywh":
            return (self._points[0] * width, self._points[1] * height, self.width * width, self.height * height)

def calc_box_vector (box1: BBox, box2: BBox):
    # Unpack the bounding boxes
    left1, top1, right1, bottom1 = box1.as_tuple()
    left2, top2, right2, bottom2 = box2.as_tuple()

    # Calculate the shortest distance along the x axis
    if left1 > right2: dx = left1 - right2
    elif left2 > right1: dx = left2 - right1
    else: dx = 0

    # Calculate the shortest distance along the y axis
    if top1 > bottom2: dy = top1 - bottom2
    elif top2 > bottom1: dy = top2 - bottom1
    else: dy = 0

    return np.array([dx, dy])

def bbox_overlaps (box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1.as_tuple()
    x1_2, y1_2, x2_2, y2_2 = box2.as_tuple()

    # Check if one box is to the left of the other
    if x1_1 > x2_2 or x1_2 > x2_1:
        return False

    # Check if one box is above the other
    if y1_1 > y2_2 or y1_2 > y2_1:
        return False

    # Boxes overlap
    return True

def bbox_contains (box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1.as_tuple()
    x1_2, y1_2, x2_2, y2_2 = box2.as_tuple()
    return x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2

def bboxes_extents (boxes: List[BBox]) -> BBox:
    min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")
    for box in boxes:
        x1, y1, x2, y2 = box.as_tuple()
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
    return BBox.from_xyxy(x1,y1,x2,y2)

def merge_boxes (box1: BBox, box2: BBox) -> BBox:
    box1 = box1.as_tuple()
    box2 = box2.as_tuple()
    return BBox(
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3]),
    )

def expand_box (box: BBox, amt: float):
    x0,y0,x1,y1 = box
    return BBox.from_xyxy(x0-amt, y0-amt, x1+amt, y1+amt)

def merger (
    items: List[Any],
    comparator: Callable[[Any, Any], bool],
    merger: Callable[[Any, Any], Any],
) -> List[Any]:
    item_stack = list(items)
    result = []

    while len(item_stack):
        item = item_stack.pop()
        merged = []

        for i, other_item in enumerate(item_stack):
            if comparator(item, other_item):
                item = merger(item, other_item)
                merged.append(i)

        if not len(merged):
            result.append(item)
        else:
            for i in reversed(merged):
                item_stack.pop(i)

            item_stack.append(item)
    return result

    
