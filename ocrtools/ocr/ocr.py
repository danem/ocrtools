import sys
from PIL import Image, ImageDraw
from typing import Any, List, Union, Tuple, Callable
import dataclasses
import io
import numpy as np
import os
import pandas as pd
import tempfile
import pathlib
from collections import defaultdict

import ocrtools.types as otypes
import ocrtools.utils as outils

import tesserocr
# This needs to be imported after tesserocr
# see: https://github.com/sirfz/tesserocr/issues/335
import ocrtools.pdf as opdf

@dataclasses.dataclass
class OCRResult:
    reads: pd.DataFrame # columns: [text, id, table_num, table_idx, confidence, left, right, top, bottom]
    tables: List[pd.DataFrame] # raw table data 
    table_confidences: List[pd.DataFrame] # floats in shape of corresponding table
    table_boxes: List[otypes.BBox]

OCRResource = Union[Image.Image, opdf.PageImage, opdf.Page]
OCREngine = Callable[[List[OCRResource]], List[OCRResult]]


def _ocr_resource_to_image (resource: OCRResource) -> Image.Image:
    if isinstance(resource, opdf.PageImage):
        # Fastest way to go from PageImage to PIL image
        return opdf.page_image_to_pil(resource)
    elif isinstance(resource, opdf.Page):
        return opdf.page_image_to_pil(opdf.pdf_page_to_img(resource))
    elif isinstance(resource, Image.Image):
        return resource
    else:
        raise Exception(f"Invalid resource type {type(resource)}")


@dataclasses.dataclass
class OCRBox:
    text: List[str]
    box: otypes.BBox
    ids: List[int]
    confidence: List[float]
    table_idx: List[Tuple[int,int]]

    def to_str (self):
        return " ".join(self.text)
    
    def expand (self, amt: float):
        self.box = otypes.expand_box(self.box, amt)
        return self

# Interface for functions sorting and merging word-level OCR boxes
# For instance, you typically want to join word-level boxes into a single line box
OCRBoxMerger = Callable[[List[OCRBox]], List[OCRBox]]


def _merge_ocr_boxes (box1: OCRBox, box2: OCRBox) -> OCRBox:
    nbox = otypes.merge_boxes(box1.box, box2.box)
    return OCRBox(
        box1.text + box2.text,
        nbox,
        box1.ids + box2.ids,
        box1.confidence + box2.confidence,
        []
    )


# Out format: XYXY
def df_to_ocrbox (df: pd.DataFrame, format="XYXY"):
    # Convert dataframe from tesseract to OCRBoxes
    format = format.upper()
    df = df.dropna(subset=["text"])
    if format == "XYXY":
        boxes = list(np.array([
            df.left, df.top, df.right, df.bottom
        ]).T)
    elif format == "XYWH":
        boxes = list(np.array([
            df.left, df.top, df.left + df.width, df.top + df.height
        ]).T)
    else:
        raise Exception(f"Invalid format {format}")
    

    ocr_boxes = []

    for i in range(len(df)):
        row = df.iloc[i]
        text = str(row.text)

        # Get rid of empty reads often caused by lines 
        if not text.strip(" "):
            continue

        ocr_boxes.append(OCRBox(
            [text],
            otypes.BBox(*boxes[i]),
            [df.index[i]],
            [df.confidence.iloc[i]],
            [df.tidx.iloc[i]]
        ))

    return ocr_boxes

def ocrbox_to_df (boxes: List[OCRBox]):
    txts = []
    l, r, t, b = [], [], [], []
    confs = []
    tidxs = []
    for box in boxes:
        txts.append(box.to_str())
        x1,y1,x2,y2 = box.box.as_tuple()
        l.append(x1); r.append(x2)
        t.append(y1); b.append(y2)
        confs.append(np.mean(box.confidence))
        tidxs.append(None)
    data = zip(txts, confs, tidxs, l, r, t, b)
    return pd.DataFrame(data, columns=["text", "confidence", "tidx", "left", "right", "top", "bottom"])

def _merge_extracted_text (
    boxes: List[OCRBox],
    comparator: Callable[[float, float], bool],
    merger: Callable[[OCRBox, OCRBox], OCRBox],
    sort = True
) -> List[OCRBox]:
    box_stack = list(boxes)
    result = []

    while len(box_stack):
        box = box_stack.pop()
        merged = []

        for i, other_box in enumerate(box_stack):
            vec = otypes.calc_box_vector(box.box, other_box.box)
            if comparator(*vec):
                box = merger(box, other_box)
                merged.append(i)

        if not len(merged):
            result.append(box)
        else:
            for i in reversed(merged):
                box_stack.pop(i)

            box_stack.append(box)

    if sort:
        for i in range(len(result)):
            box = result[i]
            box_pairs = sorted(zip(box.ids, box.text, box.confidence))
            ids, text, confs = zip(*box_pairs)
            result[i] = OCRBox(text, box.box, ids, confs, [])

    return result


def merge_horizontal (
    boxes: List[OCRBox],
    x_dist: int = 10,
    scale: float = 0.01
) -> List[OCRBox]:
    comp = lambda x,y: x < (x_dist * scale) and y <= (1 * scale)
    return _merge_extracted_text(boxes, comp, _merge_ocr_boxes)


# TODO: Fix this, it doesn't actually work...
def merge_vertical (
    boxes: List[OCRBox],
    y_dist: int = 1,
    scale: float = 0.01
) -> List[OCRBox]:
    comp = lambda x,y: y < (y_dist * scale)
    return _merge_extracted_text(boxes, comp, _merge_ocr_boxes)

def OCRMerger (x_dist: int = 10, y_dist: int = 1) -> OCRBoxMerger:
    def ret (boxes: List[OCRBox]) -> List[OCRBox]:
        if x_dist > 0:
            boxes = merge_horizontal(boxes=boxes, x_dist=x_dist)
        if y_dist > 0:
            boxes = merge_vertical(boxes=boxes, y_dist=y_dist)
        return boxes
    return ret

# No merging at all
IdentMerger = lambda boxes: boxes

# Merge lines
DefaultMerger = OCRMerger(10,0)

# Merge all boxes into one
def TotalMerger (boxes: List[OCRBox]) -> List[OCRBox]:
    max_x = float("-inf")
    max_y = float("-inf")
    for box in boxes:
        _,_,x,y = box.box.as_tuple()
        max_x = max(x, max_x)
        max_y = max(y, max_y)
    boxes = merge_horizontal(boxes, max_x, 1)
    boxes = merge_vertical(boxes, max_y, 1)
    return boxes

# Caching layer for OCR
class OCRReader:
    def __init__ (
        self,
        engine: OCREngine 
    ):
        self._engine = engine
        self._cache  = outils.CacheDict(cache_len=100)
    
    def _cache_key (
        self, 
        page: opdf.Page, 
        dpi: int = -1, 
        colorspace: opdf.Colorspace = opdf.CS_RGB
    ):
        page_id = f"{page.parent.name}#{page.xref}#{dpi}#{colorspace}"
        return page_id
    
    def _cache_lookup (
        self,
        key: str,
        clip: otypes.BBox
    ) -> Tuple[opdf.PageImage, List[OCRBox]]:
        for entry in self._cache.get(key, []):
            img, extent, boxes = entry
            if otypes.bbox_contains(extent, clip):
                res_boxes = [b for b in boxes if otypes.bbox_overlaps(clip, b.box)]
                return img, res_boxes

    def _cache_add (
        self,
        key: str,
        clip: otypes.BBox,
        img: opdf.PageImage,
        boxes: List[OCRBox]
    ):
        new_entries = [(img, clip, boxes)]
        for entry in self._cache.get(key, []):
            img, extent, eboxes = entry
            if not otypes.bbox_contains(clip, extent):
                new_entries.append((img, extent, eboxes))
        self._cache[key] = new_entries
    
    def ocr_page (
        self, 
        page: opdf.Page, 
        clip: otypes.BBox = None, 
        dpi: int = None, 
        colorspace: opdf.Colorspace = opdf.CS_RGB
    ):
        clip = clip if clip else otypes.BBox.from_xyxy(0,0,1,1)
        key = self._cache_key(page, dpi, colorspace)
        if res := self._cache_lookup(key, clip):
            return res
        else:
            img = opdf.pdf_page_to_img(page, clip=clip, dpi=dpi, colorspace=colorspace)
            ocr_results = self._engine(img)
            result = []
            for page in ocr_results:
                boxes = df_to_ocrbox(page.reads, "XYXY")
                result += boxes
            self._cache_add(key, clip, img, result)
            return img, result





