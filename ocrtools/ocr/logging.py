from typing import List, Tuple, Union
from PIL import Image, ImageDraw
import pathlib
import shutil
import os
from collections import defaultdict

import ocrtools.ocr.ocr as ocr
import ocrtools.pdf as opdf
import ocrtools.types as otypes


def draw_points (img: Image.Image, points: List[Tuple[float,float]]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.width, img.height
    for pt in points:
        x, y = pt[0] * w, pt[1] * h
        draw.point((x,y), "red")
    return img
    
def draw_boxes (
    img: Image.Image, 
    boxes: List[Union[ocr.OCRBox, otypes.BBox]],
    color: str = "red"
) -> Image.Image:
    if not isinstance(boxes, list):
        boxes = [boxes]

    _boxes = []
    for b in boxes:
        if isinstance(b, ocr.OCRBox):
            _boxes.append(b.box.as_tuple("xyxy"))
        elif isinstance(b, otypes.BBox):
            _boxes.append(b.as_tuple("xyxy"))
        else:
            _boxes.append(b)

    draw = ImageDraw.Draw(img)
    w,h = img.width, img.height
    for box in _boxes:
        x1,y1,x2,y2 = box
        draw.rectangle(((x1*w,y1*h),(x2*w,y2*h)), outline=color)
    return img

def draw_ocr_boxes (img: Image.Image, boxes: List[ocr.OCRBox]) -> Image.Image:
    if not isinstance(boxes, list):
        boxes = [boxes]

    draw = ImageDraw.Draw(img)
    w,h = img.width, img.height
    for box in boxes:
        txt = box.to_str()
        rbox = box.box.scale(w,h)
        tx, ty = rbox[0], rbox[3]
        tbox = otypes.BBox.from_xyxy(*draw.textbbox((tx,ty), txt))
        tbox = otypes.expand_box(tbox, 2)
        draw.rectangle(rbox.as_tuple(), outline="red")
        draw.rectangle(tbox.as_tuple(), fill="black", outline="red")
        draw.text((tx, ty), txt, fill="white")
    return img

def draw_ocr_result (
    img: Image.Image, 
    result: ocr.OCRResult
) -> Image.Image:
    boxes = ocr.df_to_ocrbox(result.reads)
    img = draw_ocr_boxes(img, boxes)
    return draw_boxes(img, result.table_boxes, "green")



# TODO: Improve this
__LOGGERS_ENABLED__ = False
__LOGGER_ROOT__ = None
__LOGGERS__ = {}

class OCRLogger:
    def __init__(
        self,
        out_dir: str
    ) -> None:
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._counts = defaultdict(int)
    
    def _log_ocr_results (self, img: Image.Image, boxes: List[ocr.OCRBox], prefix: str = ""):
        prefix = prefix if prefix else "img"
        count = self._counts[prefix]
        self._counts[prefix] += 1

        if not isinstance(img, Image.Image):
            img = opdf.page_image_to_pil(img)

        path = os.path.join(self._out_dir, f"{prefix}-{count}.png")
        img = draw_ocr_boxes(img, boxes)
        img.save(path)

    def __call__ (self, img: Image.Image, boxes: List[ocr.OCRBox], prefix: str = ""):
        self._log_ocr_results(img, boxes, prefix)

def configure_loggers (logger_root: str):
    global __LOGGERS_ENABLED__
    global __LOGGER_ROOT__
    global __LOGGERS__
    __LOGGER_ROOT__ = logger_root
    __LOGGERS_ENABLED__ = True
    __LOGGERS__ = {}
    shutil.rmtree(logger_root, ignore_errors=True)
    
def get_logger (name: str) -> OCRLogger:
    if not __LOGGERS_ENABLED__:
        return lambda a, b, c: None

    if not __LOGGERS__.get(name):
        log_path = os.path.join(__LOGGER_ROOT__, name)
        __LOGGERS__[name] = OCRLogger(log_path)
    return __LOGGERS__[name]