import ocrtools.ocr.logging as ocr_logging
import ocrtools.extraction.logging as ext_logging
import ocrtools.extraction as oext
import ocrtools.ocr as ocr
import ocrtools.pdf as opdf
import ocrtools.types as otypes

from PIL import Image
import enum
from typing import Union, List
from collections import defaultdict
import os
import pathlib
import shutil

class LogLevel (enum.Enum):
    NONE = -1
    INFO = 1
    DEBUG = 2

__LOGGER_ROOT__ = None
__LOGGER_DIR_NAME__ = "ocrlogs"
__LOGGER_LEVEL__ = LogLevel.NONE
__LOGGERS__ = {}

class OCRLogger:
    LogData = Union[List[ocr.OCRBox], ocr.OCRResult, oext.Extractor]

    def __init__(self, out_dir: str, max_log_level: int) -> None:
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._log_fmt = "{doc_name}-{page_number}"
        self._counts = defaultdict(int)
        self._max_log_level = max_log_level

    def _get_img_path (self, page: opdf.Page):
        dname = os.path.basename(page.parent.name)
        prefix = self._log_fmt.format(doc_name = dname, page_number = page.number)
        count = self._counts[prefix]
        self._counts[prefix] += 1
        path = os.path.join(self._out_dir, f"{prefix}-{count}.png")
        return path
    
    def _log (self, page: opdf.Page, img: Image.Image, data: List[LogData]):
        path = self._get_img_path(page)

        if not isinstance(img, Image.Image):
            img = opdf.page_image_to_pil(img)
        
        for item in data:
            if isinstance(item, ocr.OCRResult):
                img = ocr_logging.draw_ocr_boxes(img, item.reads)
                img = ocr_logging.draw_boxes(img, item.table_boxes)
            elif isinstance(item, oext.Extractor):
                img = ext_logging.draw_extractor(img, item)
            else:
                img = ocr_logging.draw_boxes(img, item)

        img.save(path)

    def info (self, page: opdf.Page, img: Image.Image, data: List[LogData]):
        if self._max_log_level > 0:
            self._log(page, img, data)
    
    def debug (self, page: opdf.Page, img: Image.Image, data: List[LogData]):
        if self._max_log_level > 1:
            self._log(page, img, data)

def configure_loggers(logger_root: str, level: LogLevel = LogLevel.INFO):
    global __LOGGER_LEVEL__
    global __LOGGER_ROOT__
    global __LOGGERS__
    __LOGGER_LEVEL__ = level
    __LOGGERS__ = {}
    logger_root = os.path.abspath(logger_root)
    __LOGGER_ROOT__ = os.path.join(logger_root, __LOGGER_DIR_NAME__)

def get_logger(name: str, clear_logs: bool = True) -> OCRLogger:
    if not __LOGGERS__.get(name):
        log_path = os.path.join(__LOGGER_ROOT__, name)
        if clear_logs and os.path.exists(log_path):
            shutil.rmtree(log_path)

        __LOGGERS__[name] = OCRLogger(log_path, __LOGGER_LEVEL__.value)

    return __LOGGERS__[name]