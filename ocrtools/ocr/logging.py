from typing import List, Tuple, Union
from PIL import Image, ImageDraw

import ocrtools.ocr.ocr as ocr
import ocrtools.types as otypes


def draw_points(img: Image.Image, points: List[Tuple[float, float]]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.width, img.height
    for pt in points:
        x, y = pt[0] * w, pt[1] * h
        draw.point((x, y), "red")
    return img


def draw_boxes(
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
    w, h = img.width, img.height
    for box in _boxes:
        x1, y1, x2, y2 = box
        draw.rectangle(((x1 * w, y1 * h), (x2 * w, y2 * h)), outline=color)
    return img


def draw_ocr_boxes(img: Image.Image, boxes: List[ocr.OCRBox]) -> Image.Image:
    if not isinstance(boxes, list):
        boxes = [boxes]

    draw = ImageDraw.Draw(img)
    w, h = img.width, img.height
    for box in boxes:
        txt = box.to_str()
        rbox = box.box.scale(w, h)
        tx, ty = rbox[0], rbox[3]
        tbox = otypes.BBox.from_xyxy(*draw.textbbox((tx, ty), txt))
        tbox = otypes.expand_box(tbox, 2)
        draw.rectangle(rbox.as_tuple(), outline="red")
        draw.rectangle(tbox.as_tuple(), fill="black", outline="red")
        draw.text((tx, ty), txt, fill="white")
    return img


def draw_ocr_result(img: Image.Image, result: ocr.OCRResult) -> Image.Image:
    boxes = result.reads
    img = draw_ocr_boxes(img, boxes)
    return draw_boxes(img, result.table_boxes, "green")


