from PIL import Image
from textractcaller.t_call import call_textract, Textract_Features, QueriesConfig, Query
from trp.t_pipeline import order_blocks_by_geo
from trp.trp2 import TDocument, TDocumentSchema, TBlockSchema
from typing import List, Any, Tuple, Union
import boto3
import io
import os
import pandas as pd
import numpy as np
import trp
import uuid

import ocrtools.ocr.ocr as aocr
import ocrtools.pdf as apdf
import ocrtools.types as stypes

def df_to_ocrbox (df: pd.DataFrame):
    # Convert dataframe from tesseract to OCRBoxes
    df = df.dropna(subset=["text"])
    boxes = list(np.array([
        df.left, df.top, df.right, df.bottom
    ]).T)
    
    ocr_boxes = []

    for i in range(len(df)):
        row = df.iloc[i]
        text = str(row.text)

        # Get rid of empty reads often caused by lines 
        if not text.strip(" "):
            continue

        ocr_boxes.append(aocr.OCRBox(
            [text],
            aocr.otypes.BBox(*boxes[i]),
            [df.id[i]],
            [df.confidence.iloc[i]]
        ))

    return ocr_boxes


def _trpbox_to_bbox (bbox: trp.BoundingBox):
    return stypes.BBox.from_xywh(bbox.left, bbox.top, bbox.width, bbox.height)

def _get_page_blocks (page: trp.Page):
    res = {}
    for b in page.blocks:
        b = TBlockSchema().load(b)
        if b.block_type != "WORD":
            continue
        res[b.id] = b
    return res

class TextractEngine:
    def __init__ (
        self,
        bucket_name: str,
        aws_profile: str = "default"
    ) -> None:
        self._sess = boto3.Session(profile_name=aws_profile)
        self._s3 = self._sess.client("s3", region_name = self._sess.region_name)
        self._tx = self._sess.client("textract", region_name = self._sess.region_name)
        self._bucket = bucket_name
    

    def _extract_tables (self, doc: trp.Document):
        pages  = []
        for p, page in enumerate(doc.pages):
            blocks = _get_page_blocks(page)
            tables, tconfs, tboxes = [], [], []
            reads, rids, ridxs, rtidx, rconfs, rx1, ry1, rx2, ry2 = [], [], [], [], [], [], [], [], []

            # Add tables
            for t, table in enumerate(page.tables):
                tvs = []
                cvs = []
                for r, row in enumerate(table.rows):
                    row_items = []
                    row_confs = []
                    for c, cell in enumerate(row.cells):
                        x1,y1,x2,y2 = _trpbox_to_bbox(cell.geometry.boundingBox).as_tuple()
                        row_items.append(cell.text)
                        row_confs.append(cell.confidence)
                        ridxs.append((cell.id, t,c,r))
                        reads.append(cell.text)
                        rconfs.append(cell.confidence)
                        rx1.append(x1); rx2.append(x2)
                        ry1.append(y1); ry2.append(y2)
                        rids.append(cell.id)
                        rtidx.append(t)

                        # Exclude block so we can add non-table blocks
                        for word in cell.content:
                            blocks.pop(word.id,None)

                    tvs.append(row_items)
                    cvs.append(row_confs)

                df1 = pd.DataFrame(tvs[1:], columns = tvs[0])
                df2 = pd.DataFrame(cvs)
                tables.append(df1)
                tconfs.append(df2)
                tboxes.append(_trpbox_to_bbox(table.geometry.boundingBox))

            # Add non-table text
            for block in blocks.values():
                x1,y1,x2,y2 = _trpbox_to_bbox(block.geometry.bounding_box).as_tuple()
                reads.append(block.text)
                rids.append(block.id)
                rconfs.append(block.confidence)
                rx1.append(x1); rx2.append(x2)
                ry1.append(y1); ry2.append(y2)

            rdf = pd.DataFrame(zip(reads, rids, rtidx, rconfs, rx1, ry1, rx2, ry2), columns = ["text", "id", "table_num", "confidence", "left", "top", "right", "bottom"])
            rdf = df_to_ocrbox(rdf)

            ridxs = {k: (t,c,r) for k,t,c,r in ridxs}
            pages.append(aocr.OCRResult(
                rdf,
                tables,
                tconfs,
                tboxes,
                ridxs
            ))
        return pages
    
    def _initiate_image_request (self, name: str, resource: Image.Image) -> Any:
        name = os.path.basename(name) + ".jpeg"
        buf = io.BytesIO()
        buf.name = name
        resource.save(buf,"jpeg")
        buf.seek(0)
        self._s3.upload_fileobj(buf, self._bucket, name)
        url = f"s3://{self._bucket}/{name}"

        return call_textract(url, features = [Textract_Features.TABLES], boto3_textract_client=self._tx)

    
    def _initiate_document_request (self, resource: apdf.PDFDoc) -> Any:
        fname = os.path.basename(resource.doc.name)
        buf = io.BytesIO(resource.doc.write())
        self._s3.upload_fileobj(buf, self._bucket, fname)
        url = f"s3://{self._bucket}/{fname}"

        return call_textract(url, features = [Textract_Features.TABLES], boto3_textract_client=self._tx)

    
    def _analyze_document (self, res) -> Any:
        t_doc = TDocumentSchema().load(res)
        ordered_doc = order_blocks_by_geo(t_doc)
        doc = trp.Document(TDocumentSchema().dump(ordered_doc))
        return self._extract_tables(doc)

    def __call__ (self, resources) -> List[aocr.OCRResult]:
        if isinstance(resources, apdf.PDFDoc):
            res = self._initiate_document_request(resources)
            return self._analyze_document(res)
        elif not isinstance(resources, list):
            resources = [resources]

        results = []
        for resource in resources:
            name, resource = aocr._ocr_resource_to_image(resource)
            res = self._initiate_image_request(name, resource)
            res = self._analyze_document(res)
            results.append(res)

        results = [ x for xs in results for x in xs ]
        return results

class TextractReader (aocr.OCRReader):
    def __init__(self, bucket_name, aws_profile):
        super().__init__(TextractEngine(bucket_name, aws_profile))