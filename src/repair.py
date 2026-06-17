from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_image(image_path):
    """读取图片，支持中文路径。"""
    image_data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"图片读取失败：{image_path}")

    return image


def save_image(image, output_path):
    """保存图片，支持中文路径。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix or ".png"

    success, encoded = cv2.imencode(suffix, image)

    if not success:
        raise ValueError("图片编码失败")

    encoded.tofile(str(output_path))


def get_safe_bbox(bbox, image_width, image_height):
    """
    将 bbox 限制在图片范围内。

    bbox 格式：[x, y, width, height]
    """
    x, y, width, height = [int(value) for value in bbox]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_width, x + width)
    y2 = min(image_height, y + height)

    return x1, y1, x2, y2


def create_text_mask_from_bbox(image, detections):
    """
    根据 OCR 检测框生成较精细的文字 mask。
    """
    image_height, image_width = image.shape[:2]

    mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8
    )

    for item in detections:
        bbox = item["bbox"]

        x1, y1, x2, y2 = get_safe_bbox(
            bbox,
            image_width,
            image_height
        )

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        roi_height, roi_width = roi.shape[:2]

        # 使用文字框边缘像素估计背景颜色
        border = min(
            6,
            roi_height // 4,
            roi_width // 4
        )

        if border < 1:
            continue

        top = roi[:border, :, :]
        bottom = roi[-border:, :, :]
        left = roi[:, :border, :]
        right = roi[:, -border:, :]

        border_pixels = np.concatenate(
            [
                top.reshape(-1, 3),
                bottom.reshape(-1, 3),
                left.reshape(-1, 3),
                right.reshape(-1, 3)
            ],
            axis=0
        )

        background_color = np.median(
            border_pixels,
            axis=0
        )

        # 计算各像素与背景颜色之间的差异
        difference = np.linalg.norm(
            roi.astype(np.float32)
            - background_color.astype(np.float32),
            axis=2
        )

        difference_normalized = cv2.normalize(
            difference,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Otsu 自动阈值
        otsu_threshold, _ = cv2.threshold(
            difference_normalized,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        threshold_value = max(
            15,
            int(otsu_threshold * 0.6)
        )

        color_mask = np.zeros_like(
            difference_normalized,
            dtype=np.uint8
        )

        color_mask[
            difference_normalized > threshold_value
        ] = 255

        # 使用边缘检测补充文字轮廓
        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        edges = cv2.dilate(
            edges,
            np.ones((3, 3), np.uint8),
            iterations=1
        )

        text_region = cv2.bitwise_or(
            color_mask,
            edges
        )

        # 闭运算连接断裂区域
        text_region = cv2.morphologyEx(
            text_region,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8)
        )

        # 略微扩张，覆盖文字边缘
        text_region = cv2.dilate(
            text_region,
            np.ones((5, 5), np.uint8),
            iterations=1
        )

        mask[y1:y2, x1:x2] = np.maximum(
            mask[y1:y2, x1:x2],
            text_region
        )

    return mask


def create_mask_preview(image, mask):
    """生成红色文字区域预览图。"""
    red_layer = image.copy()
    red_layer[mask == 255] = [0, 0, 255]

    return cv2.addWeighted(
        image,
        0.7,
        red_layer,
        0.3,
        0
    )


def repair_by_neighbor_interpolation(
    image,
    mask,
    max_iter=300
):
    """
    使用邻域插值填充文字区域。
    """
    repaired = image.astype(np.float32).copy()

    unknown = mask > 0
    known = ~unknown

    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ],
        dtype=np.float32
    )

    for _ in range(max_iter):
        known_float = known.astype(np.float32)

        neighbor_count = cv2.filter2D(
            known_float,
            -1,
            kernel
        )

        fillable = unknown & (neighbor_count > 0)

        if not np.any(fillable):
            break

        for channel_index in range(3):
            channel = repaired[:, :, channel_index]

            neighbor_sum = cv2.filter2D(
                channel * known_float,
                -1,
                kernel
            )

            channel[fillable] = (
                neighbor_sum[fillable]
                / neighbor_count[fillable]
            )

            repaired[:, :, channel_index] = channel

        known[fillable] = True
        unknown[fillable] = False

        if not np.any(unknown):
            break

    repaired = np.clip(
        repaired,
        0,
        255
    ).astype(np.uint8)

    # 对修复边缘轻微平滑
    blurred = cv2.GaussianBlur(
        repaired,
        (5, 5),
        0
    )

    edge_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    dilated = cv2.dilate(
        mask,
        edge_kernel,
        iterations=1
    )

    eroded = cv2.erode(
        mask,
        edge_kernel,
        iterations=1
    )

    edge = dilated - eroded

    repaired[edge == 255] = blurred[edge == 255]

    return repaired


def repair_text(
    image_path,
    detections,
    repaired_output_path,
    mask_output_path=None,
    preview_output_path=None
):
    """
    对单张图片进行旧文字擦除和背景修复。

    参数：
        image_path：原图路径
        detections：OCR 检测结果
        repaired_output_path：修复图保存路径
        mask_output_path：可选，mask 保存路径
        preview_output_path：可选，mask预览图路径

    返回：
        修复后的 PIL.Image
    """
    image = read_image(image_path)

    if not detections:
        raise ValueError("没有检测到文字区域")

    mask = create_text_mask_from_bbox(
        image,
        detections
    )

    if np.count_nonzero(mask) == 0:
        raise ValueError("生成的文字 mask 为空")

    preview = create_mask_preview(
        image,
        mask
    )

    repaired = repair_by_neighbor_interpolation(
        image,
        mask
    )

    save_image(
        repaired,
        repaired_output_path
    )

    if mask_output_path is not None:
        save_image(
            mask,
            mask_output_path
        )

    if preview_output_path is not None:
        save_image(
            preview,
            preview_output_path
        )

    repaired_rgb = cv2.cvtColor(
        repaired,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(repaired_rgb)