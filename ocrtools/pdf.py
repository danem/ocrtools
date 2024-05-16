from PIL import Image
from io import BytesIO
from typing import List, Union, IO
import fitz
import os

import ocrtools.types as stypes


class PDFDoc:
    def __init__(self, doc: fitz.Document, page_ids: List[int] = None) -> None:
        self.doc = doc
        if not page_ids:
            page_ids = list(range(self.doc.page_count))
        self.page_ids = sorted(page_ids)
        self._pages = []

    def __getitem__(self, idx):
        if len(self._pages) > idx:
            return self._pages[idx]

        for i, p in enumerate(self.pages()):
            if i == idx:
                return p

    def pages(self):
        if len(self._pages) == self.page_ids:
            return self._pages

        for pid in self.page_ids:
            page = self.doc.load_page(pid)
            self._pages.append(page)
            yield page


PDFResource = Union[str, IO, fitz.Document]
PageImage = fitz.Pixmap
Page = fitz.Page
Colorspace = fitz.Colorspace
EmptyFileError = fitz.EmptyFileError
FileDataError = fitz.FileDataError

CS_GRAY = fitz.csGRAY
CS_RGB = fitz.csRGB


def _get_page_image_mode(pim: PageImage):
    unmultiply = False
    cspace = pim.colorspace
    if cspace is None:
        mode = "L"
    elif cspace.n == 1:
        mode = "L" if pim.alpha == 0 else "LA"
    elif cspace.n == 3:
        mode = "RGB" if pim.alpha == 0 else "RGBA"
        if mode == "RGBA" and unmultiply:
            mode = "RGBa"
    else:
        mode = "CMYK"

    return mode


def _get_page_image_pixel_size(pim: PageImage):
    return pim.colorspace.n


def page_image_to_pil(pim: PageImage) -> Image.Image:
    mode = _get_page_image_mode(pim)
    return Image.frombytes(mode, (pim.width, pim.height), pim.samples)


def get_pdf(doc: PDFResource, pages: List[int] = None) -> PDFDoc:
    if isinstance(doc, PDFDoc):
        return pdf_subset(doc, pages)
    elif isinstance(doc, BytesIO):
        name = doc.name
        doc = fitz.open(stream=doc)
        doc.name = name
        return pdf_subset(PDFDoc(doc), pages)
    elif isinstance(doc, str):
        with open(doc) as fp:
            return get_pdf(fp, pages)
    elif not isinstance(doc, fitz.Document):
        return pdf_subset(PDFDoc(fitz.open(doc)), pages)
    else:
        raise Exception("Invalid type")


def pdf_subset(doc: PDFDoc, pages: List[int] = []) -> PDFDoc:
    if not pages:
        return doc

    ndoc = fitz.Document()

    fname, ext = os.path.splitext(doc.doc.name)
    fname = fname + f"[{','.join([str(p) for p in pages])}]"
    ndoc.name = fname + ext

    for p in sorted(pages):
        ndoc.insert_pdf(doc.doc, p, p)

    res = PDFDoc(ndoc, range(len(pages)))
    return res


def pdf_page_to_img(
    page: Page,
    clip: stypes.BBox = None,
    colorspace: Colorspace = CS_RGB,
    dpi: int = None,
    pil: bool = False,
) -> PageImage:
    clip = clip.as_tuple("xyxy", page.rect.width, page.rect.height) if clip else None
    img = page.get_pixmap(
        alpha=False, annots=False, clip=clip, dpi=dpi, colorspace=colorspace
    )
    if pil:
        img = page_image_to_pil(img)
    return img


def pdf_doc_to_imgs(
    doc: PDFResource,
    pages: List[int] = [],
    clip: stypes.BBox = None,
    colorspace: Colorspace = CS_RGB,
    dpi: int = None,
    pil: bool = False,
) -> List[PageImage]:

    doc = get_pdf(doc)
    # TODO: Broken
    if pages:
        pages = [doc.doc.load_page(i) for i in pages]
    else:
        pages = doc.doc.pages()

    results = []
    for p in pages:
        img = pdf_page_to_img(p, clip=clip, colorspace=colorspace, dpi=dpi, pil=pil)
        results.append(img)

    return results
