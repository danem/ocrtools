import ocrtools.extraction as oex
import ocrtools.ocr as oocr
import ocrtools.types as otypes
import ocrtools.pdf as opdf

from typing import List, Union

def _extract_organizations (tagger: oex.NERTagger, txts: Union[str, List[str]], conf: float = 0.96) -> List[str]:
    return oex.run_tagger(tagger, txts, labels = ["ORG", "PERSON"], confidence = conf)

def org_extractor (org_tagger: oex.NERTagger, merger: oocr.OCRBoxMerger = oocr.DefaultMerger):
    return oex.make_extractor(merger, lambda t: _extract_organizations(org_tagger, t))


# Use BertTagger to extract organization names from region of pdf
org_tagger = oex.BertTagger()
date_tagger = oex.CambertTagger()

extractor = oex.Extractor([
    oex.FieldExtractor("org", org_extractor(org_tagger), otypes.BBox.from_xyxy(0.2, 0.2, 0.8, 0.8)),
    oex.FieldExtractor("date", oex.get_date(date_tagger), otypes.BBox.from_xyxy(0.24,0.24,0.34,0.34))
])

ocr = oocr.TesseractReader()
doc = opdf.get_pdf("doc.pdf")

results = []
for page in doc.pages():
    if res := extractor.extract(ocr, page):
        print(f"Page {page.number}: {res['org']} {res['date']}")





