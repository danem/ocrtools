from typing import Tuple
import dataclasses
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
