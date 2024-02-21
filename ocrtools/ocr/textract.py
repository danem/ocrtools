from PIL import Image
from textractcaller.t_call import call_textract, Textract_Features, QueriesConfig, Query
from trp.t_pipeline import order_blocks_by_geo
from trp.trp2 import TDocument, TDocumentSchema, TBlockSchema
from typing import List, Any, Tuple, Union
import boto3
import io
import os
import pandas as pd
import trp
import uuid

import ocrtools.ocr.ocr as aocr
import ocrtools.pdf as apdf
import ocrtools.types as stypes

def generate_fname () -> str:
    return str(uuid.uuid4().bytes)

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

class TextractReader:
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
                        ridxs.append((c,r))
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
                ridxs.append(None)
                rids.append(block.id)
                rconfs.append(block.confidence)
                rx1.append(x1); rx2.append(x2)
                ry1.append(y1); ry2.append(y2)

            rdf = pd.DataFrame(zip(reads, rids, rtidx, ridxs, rconfs, rx1, ry1, rx2, ry2), columns = ["text", "id", "table_num", "table_idx", "confidence", "left", "top", "right", "bottom"])
            pages.append(aocr.OCRResult(
                rdf,
                tables,
                tconfs,
                tboxes
            ))
        return pages
    
    def _initiate_image_request (self, resource: Image.Image) -> Any:
        fname = generate_fname()
        buf = io.BytesIO()
        resource.save(buf,"jpeg")
        self._s3.upload_fileobj(buf, self._bucket, fname)
        url = f"s3://{self._bucket}/{fname}"

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

    def __call__ (self, resources) -> List[pd.DataFrame]:

        if isinstance(resources, apdf.PDFDoc):
            res = self._initiate_document_request(resources)
            return self._analyze_document(res)

        elif not isinstance(resources, list):
            resources = [resources]

        results = []
        for resource in resources:
            resource = aocr._ocr_resource_to_image(resource)
            res = self._initiate_image_request(resource)
            res = self._analyze_document(res)
            results.append(res)

        return results