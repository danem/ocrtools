from PIL import Image, ImageDraw
from typing import Any, List, Union, Tuple, Callable, Dict
import dataclasses
import pandas as pd
import uuid
import os
import queue
import time
import threading
import concurrent.futures

import ocrtools.types as otypes
import ocrtools.utils as outils

import tesserocr

# This needs to be imported after tesserocr
# see: https://github.com/sirfz/tesserocr/issues/335
import ocrtools.pdf as opdf


@dataclasses.dataclass
class OCRBox:
    text: List[str]
    box: otypes.BBox  # Boxes are always specified in relative terms
    ids: List[int]
    confidence: List[float]

    def to_str(self):
        return " ".join(self.text)


class OCRResult:
    def __init__(
        self,
        reads: List[OCRBox] = None,
        tables: List[pd.DataFrame] = None,
        table_confidences: List[pd.DataFrame] = None,
        table_boxes: List[otypes.BBox] = None,
        read_to_table_mapping: Dict[Any, Tuple[int, int, int]] = None,  # TODO
    ):
        self.reads = reads if reads is not None else []
        self.tables = tables if tables is not None else []
        self.table_confidences = table_confidences if table_confidences else []
        self.table_boxes = table_boxes if table_boxes else []
        self.read_to_table_mapping = (
            read_to_table_mapping if read_to_table_mapping else {}
        )
        self.extent = otypes.bboxes_extents([r.box for r in self.reads])

    def subset(self, region: otypes.BBox):
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
IOCREngine = Callable[[IOCRResource], OCRResult]


def _generate_fname() -> str:
    return str(uuid.uuid4().hex)


def _page_name(page: opdf.Page):
    pdir = os.path.dirname(page.parent.name)
    fname = os.path.basename(page.parent.name)
    fname, ext = os.path.splitext(fname)
    return os.path.join(pdir, f"{fname}[{page.number}]{ext}")


def _ocr_resource_to_image(resource: IOCRResource) -> Tuple[str, Image.Image]:
    if isinstance(resource, opdf.PageImage):
        # Fastest way to go from PageImage to PIL image
        fname = _generate_fname()
        return fname, opdf.page_image_to_pil(resource)
    elif isinstance(resource, opdf.Page):
        fname = _page_name(resource)
        return fname, opdf.page_image_to_pil(opdf.pdf_page_to_img(resource))
    elif isinstance(resource, Image.Image):
        fname = _generate_fname()
        return fname, resource
    else:
        raise Exception(f"Invalid resource type {type(resource)}")


# Interface for functions sorting and merging word-level OCR boxes
# For instance, you typically want to join word-level boxes into a single line box
IOCRBoxMerger = Callable[[List[OCRBox]], List[OCRBox]]


def _merge_ocr_boxes(box1: OCRBox, box2: OCRBox) -> OCRBox:
    nbox = otypes.merge_boxes(box1.box, box2.box)
    return OCRBox(
        box1.text + box2.text,
        nbox,
        box1.ids + box2.ids,
        box1.confidence + box2.confidence,
    )


def _merge_extracted_text(
    boxes: List[OCRBox],
    comparator: Callable[[float, float], bool],
    merger: Callable[[OCRBox, OCRBox], OCRBox],
    sort=True,
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


def merge_horizontal(
    boxes: List[OCRBox], x_dist: int = 10, scale: float = 0.01
) -> List[OCRBox]:
    comp = lambda x, y: x < (x_dist * scale) and y == 0  # <= (1 * scale)
    return _merge_extracted_text(boxes, comp, _merge_ocr_boxes)


def merge_vertical(
    boxes: List[OCRBox], y_dist: int = 1, scale: float = 0.01
) -> List[OCRBox]:
    comp = lambda x, y: y < (y_dist * scale)
    return _merge_extracted_text(boxes, comp, _merge_ocr_boxes)


def OCRMerger(x_dist: int = 10, y_dist: int = 1) -> IOCRBoxMerger:
    def ret(boxes: List[OCRBox]) -> List[OCRBox]:
        if x_dist > 0:
            boxes = merge_horizontal(boxes=boxes, x_dist=x_dist)
        if y_dist > 0:
            boxes = merge_vertical(boxes=boxes, y_dist=y_dist)
        return boxes

    return ret


# Merge all boxes into one
def TotalMerger() -> IOCRBoxMerger:
    def ret(boxes: List[OCRBox]):
        _, _, x, y = otypes.bboxes_extents([b.box for b in boxes]).as_tuple()
        boxes = merge_horizontal(boxes, x, 1)
        boxes = merge_vertical(boxes, y, 1)
        return boxes

    return ret


# No merging at all
IdentMerger = lambda boxes: boxes

# Merge lines
DefaultMerger = OCRMerger(10, 0)


# Caching layer for OCR
class OCRReader:
    def __init__(self):
        self._cache = outils.CacheDict(cache_len=100)

    def _perform_ocr(
        self, imgs: List[opdf.PageImage]
    ) -> List[Tuple[opdf.PageImage, OCRResult]]:
        # Overload to support multi-threaded readers
        raise Exception("Not implemented")

    def _cache_key(
        self, page: opdf.Page, dpi: int = -1, colorspace: opdf.Colorspace = opdf.CS_RGB
    ):
        page_id = f"{page.parent.name}#{page.xref}#{dpi}#{colorspace}"
        return page_id

    def _cache_lookup(
        self, key: str, clip: otypes.BBox
    ) -> Tuple[opdf.PageImage, OCRResult]:
        for img, extent, result in self._cache.get(key, []):
            if otypes.bbox_contains(extent, clip):
                return img, result.subset(clip)

    def _cache_add(
        self, key: str, clip: otypes.BBox, img: opdf.PageImage, ocr_result: OCRResult
    ):
        new_entries = [(img, clip, ocr_result)]
        for img, extent, result in self._cache.get(key, []):
            if not otypes.bbox_contains(clip, extent):
                new_entries.append((img, extent, result))
        self._cache[key] = new_entries

    def ocr_pages(
        self,
        pages: List[opdf.Page],
        clip: otypes.BBox = None,
        dpi: int = None,
        colorspace: opdf.Colorspace = opdf.CS_RGB,
    ) -> List[Tuple[opdf.PageImage, OCRResult]]:
        """
        Run OCR on the provided pages, at the specified DPI, within the specified `clip` region.
        Returns the page images, and `OCRResults` with `OCRBox` in *page space* (not clip space).
        """
        results = []
        imgs = []

        clip = clip if clip else otypes.BBox.from_xyxy(0, 0, 1, 1)
        for page in pages:
            key = self._cache_key(page, dpi, colorspace)
            if res := self._cache_lookup(key, clip):
                results.append(res)
            else:
                img = opdf.pdf_page_to_img(
                    page, clip=clip, dpi=dpi, colorspace=colorspace
                )
                imgs.append(img)

        ocr_results = self._perform_ocr(imgs)
        for img, ocr_result in ocr_results:
            # Transform the reads into page space
            clip2page = outils.clip_space_to_page_space(clip)
            for read in ocr_result.reads:
                read.box = read.box.transform(clip2page)

            self._cache_add(key, clip, img, ocr_result)
            results.append((img, ocr_result))

        return results

    def ocr_page(
        self,
        page: opdf.Page,
        clip: otypes.BBox = None,
        dpi: int = None,
        colorspace: opdf.Colorspace = opdf.CS_RGB,
    ) -> Tuple[opdf.PageImage, OCRResult]:
        results = self.ocr_pages([page], clip, dpi, colorspace)
        return results[0]


class SimpleOCRReader(OCRReader):
    def __init__(self, engine: IOCREngine):
        super().__init__()
        self.engine = engine

    def _perform_ocr(
        self, imgs: List[opdf.PageImage]
    ) -> List[Tuple[opdf.PageImage, OCRResult]]:
        results = [self.engine(img) for img in imgs]
        return zip(imgs, results)


class ThreadedOCRReader(OCRReader):
    def __init__(self, engine_ctor, thread_count):
        super().__init__()

        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=thread_count
        )
        self._engine_pool = queue.Queue()
        for _ in range(thread_count):
            self._engine_pool.put(engine_ctor())

    def _ocr_task(self, input: opdf.PageImage):
        engine = self._engine_pool.get()
        result = engine(input)
        self._engine_pool.put(engine)
        return result

    def _perform_ocr(
        self, imgs: List[opdf.PageImage]
    ) -> List[Tuple[opdf.PageImage, OCRResult]]:
        results = self._thread_pool.map(self._ocr_task, imgs)
        return zip(imgs, results)
