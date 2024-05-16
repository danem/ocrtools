from PIL import ImageDraw

def draw_extractor(img, extractor):
    draw = ImageDraw.Draw(img)
    for group in extractor._groups:
        draw.rectangle(
            group.box.as_tuple(width=img.width, height=img.height), outline="red"
        )
        for ex in group.extractors:
            ebox = ex.box.as_tuple(width=img.width, height=img.height)
            draw.rectangle(ebox, outline="green")
            draw.text(ebox, ex.name, fill="green")
    return img
