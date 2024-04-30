import matplotlib.pyplot as plt

import ocrtools as ocrt
import ocrtools.ocr.logging as ocrl

doc = ocrt.get_pdf("notebooks/doc.pdf")
ocr = ocrt.TesseractReaderThreaded(8)
res = ocr.ocr_pages(doc.pages(), dpi = 200)

img = ocrt.page_image_to_pil(res[0][0])
plt.imshow(ocrl.draw_ocr_result(img, res[0][1]))

read = res[0][1]
boxes = ocrt.DefaultMerger(read.reads)
plt.imshow(ocrl.draw_ocr_boxes(img, boxes))