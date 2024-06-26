from PIL import Image, ImageDraw
from typing import List, Any
import csv
import io
import os
import pandas as pd
import numpy as np
import ocrtools.ocr.ocr as aocr

import tesserocr

# This needs to be imported after tesserocr
# see: https://github.com/sirfz/tesserocr/issues/335
import ocrtools.pdf as apdf

# TODO: make this cross platform
os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata/"


# Out format: XYXY
def df_to_ocrbox(df: pd.DataFrame):
    # Convert dataframe from tesseract to OCRBoxes
    df = df.dropna(subset=["text"])
    boxes = list(np.array([df.left, df.top, df.right, df.bottom]).T)

    ocr_boxes = []

    for i in range(len(df)):
        row = df.iloc[i]
        text = str(row.text)

        # Get rid of empty reads often caused by lines
        if not text.strip(" "):
            continue

        ocr_boxes.append(
            aocr.OCRBox(
                [text],
                aocr.otypes.BBox(*boxes[i]),
                [df.index[i]],
                [df.confidence.iloc[i]],
            )
        )

    return ocr_boxes


class TesseractEngine:
    def __init__(self) -> None:
        self._config = ""
        self._api = tesserocr.PyTessBaseAPI()

    def _extract_text(self, api, img) -> pd.DataFrame:
        if isinstance(img, Image.Image):
            api.SetImage(img)
        elif isinstance(img, apdf.PageImage):
            data = img.tobytes(output="jpg")
            w, h = img.width, img.height
            ps = apdf._get_page_image_pixel_size(img)
            api.SetImageBytes(data, w, h, ps, ps * w)
        else:
            raise TypeError("Invalid image type supplied")

        result = api.GetTSVText(0)
        kwargs = {"quoting": csv.QUOTE_NONE, "sep": "\t", "dtype": {"text": str}}
        kwargs["names"] = [
            "level",
            "page_num",
            "block_num",
            "par_num",
            "line_num",
            "word_num",
            "left",
            "top",
            "width",
            "height",
            "conf",
            "text",
            "tidx",
        ]

        df = pd.read_csv(io.StringIO(result), **kwargs)

        df.left /= img.width
        df.top /= img.height
        df.width /= img.width
        df.height /= img.height
        df.width += df.left
        df.height += df.top
        df.rename(
            columns={"width": "right", "height": "bottom", "conf": "confidence"},
            inplace=True,
        )

        return aocr.OCRResult(
            reads=df_to_ocrbox(df), tables=[], table_confidences=[], table_boxes=[]
        )

    def __call__(self, img: aocr.IOCRResource) -> aocr.OCRResult:
        _, img = aocr._ocr_resource_to_image(img)
        res = self._extract_text(self._api, img)
        return res


def enable_scollview_debugging(engine: TesseractEngine):
    engine._api.SetVariable("interactive_display_mode", "T")
    engine._api.SetVariable("classify_debug_level", "1")
    engine._api.SetVariable("textord_baseline_debug", "1")
    engine._api.SetVariable("textord_debug_images", "1")
    engine._api.SetVariable("textord_tablefind_show_mark", "1")


class TesseractReader(aocr.SimpleOCRReader):
    def __init__(self, engine: TesseractEngine = None):
        super().__init__(engine if engine else TesseractEngine())


class TesseractReaderThreaded(aocr.ThreadedOCRReader):
    def __init__(self, process_count: int = 1):
        super().__init__(TesseractEngine, process_count)
