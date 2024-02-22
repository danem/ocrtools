from PIL import Image, ImageDraw
from typing import List, Any
import csv
import io
import os
import pandas as pd
import ocrtools.ocr.ocr as aocr

import tesserocr
# This needs to be imported after tesserocr
# see: https://github.com/sirfz/tesserocr/issues/335
import ocrtools.pdf as apdf

# TODO: make this cross platform
os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata/"


class TesseractEngine:
    def __init__(self) -> None:
        self._config = ""
        self._api = tesserocr.PyTessBaseAPI()

    def _extract_text (self, api, img) -> pd.DataFrame:
        if isinstance(img, Image.Image):
            api.SetImage(img)
        elif isinstance(img, apdf.PageImage):
            data = img.tobytes(output="jpg")
            w,h = img.width, img.height
            ps = apdf._get_page_image_pixel_size(img)
            api.SetImageBytes(data, w, h, ps, ps*w)
        else:
            raise TypeError("Invalid image type supplied")

        result = api.GetTSVText(0)
        kwargs = {'quoting': csv.QUOTE_NONE, 'sep': '\t', "dtype": {"text": str}}
        kwargs['names'] = ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text', "tidx"]

        df = pd.read_csv(io.StringIO(result), **kwargs)

        df.left /= img.width
        df.top  /= img.height
        df.width /= img.width
        df.height /= img.height
        df.width += df.left
        df.height += df.top

        df.rename(columns = {"width": "right", "height": "bottom", "conf": "confidence"}, inplace=True)
        return aocr.OCRResult(
            reads = df,
            tables = [],
            table_confidences = [],
            table_boxes = []
        )

    def __call__(self, imgs: List[aocr.OCRResource]) -> List[aocr.OCRResult]:
        if isinstance(imgs, apdf.PDFDoc):
            imgs = apdf.pdf_doc_to_imgs(imgs)
            imgs = [apdf.page_image_to_pil(p) for p in imgs]

        if not isinstance(imgs, list):
            imgs = [imgs]

        results = []
        for img in imgs:
            img = aocr._ocr_resource_to_image(img)
            res = self._extract_text(self._api, img)
            results.append(res)
        return results

def TesseractReader ():
    return aocr.OCRReader(TesseractEngine())