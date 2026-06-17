
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent

FONT_PATHS = [
    ROOT / "fonts" / "NotoSansCJKsc-Bold.otf",
    ROOT / "fonts" / "NotoSansSC-Bold.ttf",
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"),
    Path("C:/Windows/Fonts/msyhbd.ttc"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
]


def get_font(size):
    """获取可用的中文字体。"""
    for path in FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(path, size)

    raise FileNotFoundError("没有找到可用的中文字体")


def is_vertical_text(text, bbox):
    """根据文字框形状判断是否采用竖排。"""
    _, _, width, height = bbox
    return len(text) > 1 and height > width * 1.4


def calculate_font_size(text, bbox, vertical=False):
    """根据文字框大小自动计算字号。"""
    _, _, width, height = map(int, bbox)
    width = max(width, 1)
    height = max(height, 1)

    for size in range(min(width, height), 9, -1):
        font = get_font(size)
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

        if vertical:
            char_sizes = [
                draw.textbbox((0, 0), char, font=font)
                for char in text
            ]

            text_width = max(
                box[2] - box[0]
                for box in char_sizes
            )

            text_height = sum(
                box[3] - box[1]
                for box in char_sizes
            )

            # 竖排时加入少量字间距
            text_height += max(0, len(text) - 1) * int(size * 0.1)

        else:
            box = draw.textbbox((0, 0), text, font=font)
            text_width = box[2] - box[0]
            text_height = box[3] - box[1]

        if text_width <= width * 0.9 and text_height <= height * 0.9:
            return size

    return 10


def estimate_text_color(original, repaired, bbox):
    """根据原图和修复图的差异估计原文字颜色。"""
    x, y, width, height = map(int, bbox)
    crop_box = (x, y, x + width, y + height)

    original_crop = np.asarray(
        original.crop(crop_box),
        dtype=np.int16
    )

    repaired_crop = np.asarray(
        repaired.crop(crop_box),
        dtype=np.int16
    )

    if original_crop.size == 0 or repaired_crop.size == 0:
        return 0, 0, 0

    difference = np.linalg.norm(
        original_crop - repaired_crop,
        axis=2
    )

    text_pixels = original_crop[difference > 40]

    if len(text_pixels) >= 10:
        color = np.median(text_pixels, axis=0)
        return tuple(
            int(np.clip(value, 0, 255))
            for value in color
        )

    # 无法估计时，根据背景亮度选择黑色或白色
    return (
        (0, 0, 0)
        if repaired_crop.mean() > 128
        else (255, 255, 255)
    )


def get_stroke_color(text_color):
    """生成与文字颜色对比明显的描边颜色。"""
    brightness = (
        0.299 * text_color[0]
        + 0.587 * text_color[1]
        + 0.114 * text_color[2]
    )

    return (0, 0, 0) if brightness > 128 else (255, 255, 255)


def draw_horizontal_text(
    draw,
    text,
    layer_size,
    font,
    fill,
    stroke_width,
    stroke_fill
):
    """居中绘制横排文字。"""
    box = draw.textbbox(
        (0, 0),
        text,
        font=font,
        stroke_width=stroke_width
    )

    left, top, right, bottom = box
    text_width = right - left
    text_height = bottom - top

    x = (layer_size[0] - text_width) / 2 - left
    y = (layer_size[1] - text_height) / 2 - top

    draw.text(
        (x, y),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill
    )


def draw_vertical_text(
    draw,
    text,
    layer_size,
    font,
    fill,
    stroke_width,
    stroke_fill
):
    """逐字居中绘制竖排文字。"""
    boxes = [
        draw.textbbox(
            (0, 0),
            char,
            font=font,
            stroke_width=stroke_width
        )
        for char in text
    ]

    char_heights = [
        box[3] - box[1]
        for box in boxes
    ]

    spacing = max(1, int(font.size * 0.1))
    total_height = sum(char_heights) + spacing * (len(text) - 1)

    current_y = (layer_size[1] - total_height) / 2

    for char, box, char_height in zip(text, boxes, char_heights):
        left, top, right, _ = box
        char_width = right - left

        x = (layer_size[0] - char_width) / 2 - left
        y = current_y - top

        draw.text(
            (x, y),
            char,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill
        )

        current_y += char_height + spacing


def draw_text_high_quality(
    base_image,
    text,
    bbox,
    text_color,
    stroke_color,
    scale=4
):
    """使用高分辨率图层绘制文字，再缩小以减少锯齿。"""
    x, y, width, height = map(int, bbox)

    if width <= 0 or height <= 0:
        return

    vertical = is_vertical_text(text, bbox)
    font_size = calculate_font_size(text, bbox, vertical)
    stroke_width = max(1, int(font_size * 0.06))

    scaled_size = (width * scale, height * scale)

    layer = Image.new(
        "RGBA",
        scaled_size,
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(layer)
    font = get_font(font_size * scale)

    fill = (*text_color, 255)
    stroke_fill = (*stroke_color, 255)
    scaled_stroke = stroke_width * scale

    if vertical:
        draw_vertical_text(
            draw,
            text,
            scaled_size,
            font,
            fill,
            scaled_stroke,
            stroke_fill
        )
    else:
        draw_horizontal_text(
            draw,
            text,
            scaled_size,
            font,
            fill,
            scaled_stroke,
            stroke_fill
        )

    layer = layer.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    base_image.alpha_composite(layer, (x, y))


def render_text(
    original_path,
    repaired_path,
    detections,
    new_texts,
    output_path
):
    """在修复后的图片上绘制新文字并保存。"""
    original = Image.open(original_path).convert("RGB")
    repaired = Image.open(repaired_path).convert("RGB")
    final_image = repaired.convert("RGBA")

    for item, new_text in zip(detections, new_texts):
        text = str(new_text).strip()

        if not text:
            continue

        bbox = item["bbox"]

        text_color = estimate_text_color(
            original,
            repaired,
            bbox
        )

        draw_text_high_quality(
            base_image=final_image,
            text=text,
            bbox=bbox,
            text_color=text_color,
            stroke_color=get_stroke_color(text_color)
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = final_image.convert("RGB")
    result.save(output_path)

    return result

