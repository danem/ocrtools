#### OCRTools

Tools for scraping pdfs with OCR

##### Basic Usage

```python
import ocrtools as ocrt
import re

ocr = ocrt.TesseractReader()

# Find dates within long strings like paragraphs
date_tagger = ocrt.DateTagger()

extractor = ocrt.Extractor([
    ocrt.FieldExtractor("date", ocrt.get_date(date_tagger), ocrt.BBox.from_xyxy(0.12, 0.12, 0.2, 0.2)),
    ocrt.FieldExtractor("address", ocrt.match(re.compile("address: {.*}")), ocrt.BBox.from_xyxy(0.12, 0.12, 0.2, 0.2))
])

doc = ocrt.get_doc("/path/to/doc.pdf")
for page in doc.pages():
    if res := extractor.extract(ocr, page):
        print(f"address: {res['address']}")
        print(f"date: {res['date']}")
```

This searches the PDF and extracts dates and addresses from the specified regions with the fewest possible calls to the OCR engine with only a subset of each page.

##### Logging

Log the OCR results:
```python
ocrt.logging.configure_loggers("/tmp/ocr-logs")
logger = ocrt.logging.get_logger("test")
doc = ocrt.get_doc("/path/to/doc.pdf")
for page in doc.pages():
    if res := extractor.extract(ocr, page, logger=logger):
        print(f"address: {res['address']}")
        print(f"date: {res['date']}")
```
This writes images annotated with OCR and extraction result to `/tmp/ocr-logs/test`.


##### Available OCR Engines

OCRTools comes with `Tesseract` (via Tesserocr) and `Textract` interfaces. To use your own simply provide a function of the following type to `extractor.extract`.

```python
OCRReader = Callable[[List[OCRResource]], List[pandas.DataFrame]]
```

##### Taggers

OCRTools uses `spacy` and huggingface `tranformers` to identify entities such as dates and names within text. To use your own, simply provide a function of the following type:
```python
NERTagger = Callable[[str], List[TokenSpan]]
```



##### Requirements

- fitz (pymupdf)
- dateparse
- pillow
- pandas
- tesserocr
- spacy
- transformers

For AWS Textract
- boto3
- textractcaller
- trp




