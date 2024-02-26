import ocrtools.pdf as opdf
import argparse

def main ():
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_path")
    args = parser.parse_args()

    doc = opdf.get_pdf(args.doc_path)
    for i, page in enumerate(doc.pages()):
        annos = list(page.annots())
        for m in annos:
            x0 = m.rect.x0 / page.rect.width
            y0 = m.rect.y0 / page.rect.height
            x1 = m.rect.x1 / page.rect.width
            y1 = m.rect.y1 / page.rect.height
            print(i, m.info["content"], f"otypes.BBox.from_xyxy({x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f})")

if __name__ == "__main__":
    main()