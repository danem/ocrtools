from typing import Any, Callable, List, Optional, Tuple
import dataclasses
from collections import defaultdict
import os
import datetime
import re

import ocrtools.types as otypes
import ocrtools.ocr as ocr
import ocrtools.pdf as opdf
import ocrtools.extraction.tagger as otag

ExtractionFn = Callable[[List[ocr.OCRBox]], List[Tuple[Any, ocr.OCRBox]]]

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
        ocr: ocr.OCRReader,
        page: opdf.Page,
        dpi: int = 100,
        logger: ocr.OCRLogger = None
    ):
        # TODO: Should cache the page to image conversion...
        # So it can be used by other extractors

        pw, ph = page.rect.width, page.rect.height
        result = defaultdict(list)
        logs = defaultdict(list)
        mappings = []

        # TODO: The math here is a bit sketchy to say the least...
        dpi_scale = dpi / 72
        idpi_scale = 72 / dpi
        for group in self._groups:
            gbox = group.box.scale(pw*dpi_scale, ph*dpi_scale)
            img, boxes = ocr.ocr_page(ocr, page, clip=group.box, dpi=dpi)
            for extractor in group.extractors:
                targets = []
                for box in boxes:
                    vbox = box.box.scale(img.width * idpi_scale, img.height * idpi_scale)
                    vbox = vbox.translate(gbox.x * idpi_scale, gbox.y * idpi_scale)
                    vbox = vbox.scale(1/pw, 1/ph)
                    if otypes.bbox_overlaps(extractor.box, vbox):
                        targets.append(box)
                mappings.append((img, extractor, targets))

        for img, extractor, boxes in mappings:
            values = extractor.extractor(boxes)
            for value, box in values:
                result[extractor.name].append(value)
                lbox = ocr.OCRBox([f"{extractor.name}: {value}"], box.box, [0], 1, [])
                logs[img].append(lbox)
            # else:
                # logs[img] = []

        if logger:
            dname = os.path.basename(page.parent.name)
            log_prefix = f"{dname}-{page.number}"
            for img, boxes in logs.items():
                logger(img,boxes,log_prefix)

        return result


def make_extractor (merger: ocr.OCRBoxMerger, fn: Callable[[str], Any]) -> ExtractionFn:
    def ret (boxes: List[ocr.OCRBox]):
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
    _txts = []
    for t in txts:
        _txts.append(re.sub("[\-\_]", " ", t))
    dates = otag.extract_date_strings(tagger, _txts)
    if len(dates):
        return dates[0]

    return None

# Extract date using the supplied NERTagger. Should be capable of finding dates within paragraphs
# written in virtually any format
def get_date (tagger: otag.NERTagger, merger: ocr.OCRBoxMerger = ocr.DefaultMerger) -> ExtractionFn:
    return make_extractor(merger, lambda t: _get_date(tagger, t))

# Extract the raw text from the OCR result
def identity (merger: ocr.OCRBoxMerger = ocr.TotalMerger) -> ExtractionFn:
    return make_extractor(merger, lambda t: t)


# Extract text matching the supplied regex
def match (pattern: re.Pattern, merger: ocr.OCRBoxMerger = ocr.DefaultMerger) -> ExtractionFn:
    def ret (boxes: List[ocr.OCRBox]):
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