import ocrtools.extraction as oex
import ocrtools.ocr as oocr
import ocrtools.types as otypes
import ocrtools.pdf as opdf
import ocrtools.extraction.logging as oexl

date_tagger = oex.DateTagger()
extractor = oex.Extractor([
    oex.FieldExtractor("date", oex.get_date(date_tagger), otypes.BBox.from_xyxy(0.05, 0.9, 0.3, 0.99))
])

ocr = oocr.TesseractReader()
doc = opdf.get_pdf("./doc.pdf")

print("Scraping document")
for i,page in enumerate(doc.pages()):
    if res := extractor.extract(ocr, page, dpi=200):
        print(res["date"])


