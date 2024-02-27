from PIL import Image, ImageDraw
from typing import Any, List, Union, Tuple, Callable
import dataclasses
import pandas as pd

import ocrtools.types as otypes
import ocrtools.utils as outils

import tesserocr
# This needs to be imported after tesserocr
# see: https://github.com/sirfz/tesserocr/issues/335
import ocrtools.pdf as opdf

@dataclasses.dataclass
class OCRBox:
    text: List[str]
    box: otypes.BBox # Boxes are always specified in relative terms
    ids: List[int]
    confidence: List[float]

    def to_str (self):
        return " ".join(self.text)

class OCRResult:
    def __init__ (
        self,
        reads: List[OCRBox] = None,
        tables: List[pd.DataFrame] = None,
        table_confidences: List[pd.DataFrame] = None,
        table_boxes: List[otypes.BBox] = None,
        read_to_table_mapping: List[Tuple[int,int]] = None # TODO
    ):
        self.reads = reads if reads else []
        self.tables = tables if tables else []
        self.table_confidences = table_confidences if table_confidences else []
        self.table_boxes = table_boxes if table_boxes else []
        self.read_to_table_mapping = read_to_table_mapping if read_to_table_mapping else []
        self.extent = otypes.bboxes_extents([r.box for r in self.reads])
    
    def subset (self, region: otypes.BBox):
        # TODO
        ids = []
        boxes = []
        for box in self.reads:
            if otypes.bbox_contains(region, box.box):
                boxes.append(box)
                ids += box.ids
        # TODO: Include tables
        return OCRResult(boxes)

IOCRResource = Union[Image.Image, opdf.PageImage, opdf.Page]
IOCREngine = Callable[[List[IOCRResource]], List[OCRResult]] 

def _ocr_resource_to_image (resource: IOCRResource) -> Image.Image:
    if isinstance(resource, opdf.PageImage):
        # Fastest way to go from PageImage to PIL image
        return opdf.page_image_to_pil(resource)
    elif isinstance(resource, opdf.Page):
        return opdf.page_image_to_pil(opdf.pdf_page_to_img(resource))
    elif isinstance(resource, Image.Image):
        return resource
    else:
        raise Exception(f"Invalid resource type {type(resource)}")


# Interface for functions sorting and merging word-level OCR boxes
# For instance, you typically want to join word-level boxes into a single line box
IOCRBoxMerger = Callable[[List[OCRBox]], List[OCRBox]]

def _merge_ocr_boxes (box1: OCRBox, box2: OCRBox) -> OCRBox:
    nbox = otypes.merge_boxes(box1.box, box2.box)
    return OCRBox(
        box1.text + box2.text,
        nbox,
        box1.ids + box2.ids,
        box1.confidence + box2.confidence
    )


def _merge_extracted_text (
    boxes: List[OCRBox],
    comparator: Callable[[float, float], bool],
    merger: Callable[[OCRBox, OCRBox], OCRBox],
    sort = True
) -> List[OCRBox]:
    cmp = lambda a, b: comparator(*otypes.calc_box_vector(a.box, b.box))
    result = otypes.merger(boxes, cmp, merger)
    if sort:
        for i in range(len(result)):
            box = result[i]
            box_pairs = sorted(zip(box.ids, box.text, box.confidence))
            ids, text, confs = zip(*box_pairs)
            result[i] = OCRBox(text, box.box, ids, confs)
    return result


def merge_horizontal (
    boxes: List[OCRBox],
    x_dist: int = 10,
    scale: float = 0.01
) -> List[OCRBox]:
    comp = lambda x,y: x < (x_dist * scale) and y <= (1 * scale)
    return _merge_extracted_text(boxes, comp, _merge_ocr_boxes)


def merge_vertical (
    boxes: List[OCRBox],
    y_dist: int = 1,
    scale: float = 0.01
) -> List[OCRBox]:
    comp = lambda x,y: y < (y_dist * scale)
    return _merge_extracted_text(boxes, comp, _merge_ocr_boxes)

def OCRMerger (x_dist: int = 10, y_dist: int = 1) -> IOCRBoxMerger:
    def ret (boxes: List[OCRBox]) -> List[OCRBox]:
        if x_dist > 0:
            boxes = merge_horizontal(boxes=boxes, x_dist=x_dist)
        if y_dist > 0:
            boxes = merge_vertical(boxes=boxes, y_dist=y_dist)
        return boxes
    return ret

# Merge all boxes into one
def TotalMerger () -> IOCRBoxMerger:
    def ret (boxes: List[OCRBox]):
        _,_,x,y = otypes.bboxes_extents([b.box for b in boxes]).as_tuple()
        boxes = merge_horizontal(boxes, x, 1)
        boxes = merge_vertical(boxes, y, 1)
        return boxes
    return ret

# No merging at all
IdentMerger = lambda boxes: boxes

# Merge lines
DefaultMerger = OCRMerger(10,0)


# Caching layer for OCR
class OCRReader:
    def __init__ (
        self,
        engine: IOCREngine 
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
    ) -> Tuple[opdf.PageImage, OCRResult]:
        for img, extent, result in self._cache.get(key, []):
            if otypes.bbox_contains(extent, clip):
                return img, result.subset(clip)

    def _cache_add (
        self,
        key: str,
        clip: otypes.BBox,
        img: opdf.PageImage,
        ocr_result: OCRResult
    ):
        new_entries = [(img, clip, ocr_result)]
        for img, extent, result in self._cache.get(key, []):
            if not otypes.bbox_contains(clip, extent):
                new_entries.append((img, extent, result))
        self._cache[key] = new_entries
    
    def ocr_page (
        self, 
        page: opdf.Page, 
        clip: otypes.BBox = None, 
        dpi: int = None, 
        colorspace: opdf.Colorspace = opdf.CS_RGB
    ) -> Tuple[opdf.PageImage, OCRResult]:
        """
        Run OCR on the provided page, at the specified DPI, within the specified `clip` region.
        Returns the page image, and an `OCRResult` with `OCRBox` in *page space* (not clip space).
        """
        clip = clip if clip else otypes.BBox.from_xyxy(0,0,1,1)
        key = self._cache_key(page, dpi, colorspace)
        if res := self._cache_lookup(key, clip):
            return res
        else:
            img = opdf.pdf_page_to_img(page, clip=clip, dpi=dpi, colorspace=colorspace)
            ocr_result = self._engine(img)[0]

            # Transform the reads into page space
            clip2page = outils.clip_space_to_page_space(clip)
            for read in ocr_result.reads:
                read.box = read.box.transform(clip2page)

            self._cache_add(key, clip, img, ocr_result)
            return img, ocr_result





