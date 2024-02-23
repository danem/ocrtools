from typing import Any, Callable, List, Optional, Tuple
import dataclasses
from collections import defaultdict
import os
import datetime
import re
import numpy as np

import ocrtools.types as otypes
import ocrtools.ocr as oocr
import ocrtools.pdf as opdf
import ocrtools.extraction.tagger as otag
import ocrtools.utils as outils

ExtractionFn = Callable[[List[oocr.OCRBox]], List[Tuple[Any, oocr.OCRBox]]]

@dataclasses.dataclass
class FieldExtractor:
    name: str
    extractor: ExtractionFn
    box: otypes.BBox
    priority: float = 1


@dataclasses.dataclass
class ExtractorGroup:
    box: otypes.BBox
    extractors: List[FieldExtractor]


def _merge_extractors (
    extractors: List[FieldExtractor]
):
    group_stack = []
    result = []
    for fe in extractors:
        group = ExtractorGroup(fe.box, [fe])
        group_stack.append(group)

    while len(group_stack):
        group = group_stack.pop()
        merged = []

        for i, other_group in enumerate(group_stack):
            if otypes.bbox_overlaps(group.box, other_group.box):
                nbox = otypes.merge_boxes(group.box, other_group.box)
                group = ExtractorGroup(nbox, group.extractors + other_group.extractors)
                merged.append(i)

        if not len(merged):
            result.append(group)
        else:
            for i in reversed(merged):
                group_stack.pop(i)

            group_stack.append(group)

    return result

class Extractor:
    def __init__ (
        self,
        extractors: List[FieldExtractor]
    ):
        self._extractors = extractors
        self._groups = _merge_extractors(extractors)
    
    def extract (
        self,
        ocr: oocr.OCRReader,
        page: opdf.Page,
        dpi: int = 100,
        logger: oocr.OCRLogger = None
    ):
        results = defaultdict(list)
        logs = defaultdict(list)
        mappings = []

        for group in self._groups:
            img, result = ocr.ocr_page(page, clip=group.box, dpi=dpi)
            for extractor in group.extractors:
                targets = [box for box in result.reads if otypes.bbox_overlaps(extractor.box, box.box)]
                mappings.append((img, group, extractor, targets))

        for img, group, extractor, boxes in mappings:
            values = extractor.extractor(boxes)
            to_clip = outils.page_space_to_clip_space(group.box)

            for value, box in values:
                results[extractor.name].append(value)
                nbox = (box.box
                    .translate(-group.box.x, -group.box.y)
                )
                lbox = oocr.OCRBox([f"{extractor.name}: {value}"], nbox, [0], [1])
                logs[img].append(lbox)

        if logger:
            dname = os.path.basename(page.parent.name)
            log_prefix = f"{dname}-{page.number}"
            for img, boxes in logs.items():
                logger(img,boxes,log_prefix)

        return results


def make_extractor (merger: oocr.OCRBoxMerger, fn: Callable[[str], Any]) -> ExtractionFn:
    def ret (boxes: List[oocr.OCRBox]):
        boxes = merger(boxes)
        results = []
        for box in boxes:
            res = fn(box.to_str())
            if res:
                results.append((res, box))
        return results
    return ret


# -----------------
# Extractors
# -----------------

def _get_date (tagger: otag.NERTagger, txts: List[str]) -> Optional[datetime.datetime]:
    # _txts = []
    # for t in txts:
    #     _txts.append(re.sub("[\-\_]", " ", t))
    dates = otag.extract_date_strings(tagger, txts)
    if len(dates):
        return dates[0]

    return None

# Extract date using the supplied NERTagger. Should be capable of finding dates within paragraphs
# written in virtually any format
def get_date (tagger: otag.NERTagger, merger: oocr.OCRBoxMerger = oocr.DefaultMerger) -> ExtractionFn:
    return make_extractor(merger, lambda t: _get_date(tagger, t))

# Extract the raw text from the OCR result
def identity (merger: oocr.OCRBoxMerger = oocr.TotalMerger) -> ExtractionFn:
    return make_extractor(merger, lambda t: t)


# Extract text matching the supplied regex
def match (pattern: re.Pattern, merger: oocr.OCRBoxMerger = oocr.DefaultMerger) -> ExtractionFn:
    def ret (boxes: List[oocr.OCRBox]):
        boxes = merger(boxes)
        results = []
        for box in boxes:
            matches = pattern.findall(box.to_str())
            for m in matches:
                if isinstance(m, tuple):
                    for el in m:
                        matches.append(el)
                elif m:
                    results.append((m,box))
        return results
    return ret