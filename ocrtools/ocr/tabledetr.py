import ocrtools.ocr.ocr as aocr
import ocrtools.pdf as apdf
import ocrtools.types as stypes

from transformers import pipeline
import os
from typing import List, Union, Any

class TableDetr:
    def __init__ (self, ts):
        self._model = pipeline("object-detection", model="microsoft/table-transformer-detection")
        self._ts = ts
    
    def _locate_tables (self, imgs: List[apdf.Image.Image]):
        results = []
        read_results = self._model(imgs)
        for img, tables in zip(imgs,read_results):
            w,h = img.width, img.height
            boxes = []
            for table in tables:
                box = stypes.BBox.from_xyxy(
                    table["box"]["xmin"] / w,
                    table["box"]["ymin"] / h,
                    table["box"]["xmax"] / w,
                    table["box"]["ymax"] / h
                )
                obox = aocr.OCRBox([], box, [], table["score"], [])
                boxes.append(obox)
            results.append(boxes)

        return results
    
    def locate_tables (self, imgs: Union[apdf.PageImage, List[apdf.PageImage]]):
        if not isinstance(imgs, list):
            imgs = [imgs]

        _imgs = []
        for img in imgs:
            if isinstance(img, apdf.Image.Image):
                img = apdf._thumbnail(img, self._ts[0], self._ts[1])
                _imgs.append(img)
            else:
                img = apdf.page_image_to_pil(img)
                img = apdf._thumbnail(img, self._ts[0], self._ts[1])
                _imgs.append(img)
        
        return self._locate_tables(_imgs)

    
    def __call__ (self, imgs):
        return self.locate_tables(imgs)


