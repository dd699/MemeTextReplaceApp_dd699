import os


# 必须写在导入 PaddleOCR 之前
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import cv2
import numpy as np
from paddleocr import PaddleOCR


# 第一次创建模型会下载权重，之后会复用
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    lang="ch",
    enable_mkldnn=False
)


def detect_text(image_path):
    """
    检测单张图片中的文字。

    返回格式：
    [
        {
            "text": "少爷",
            "bbox": [x, y, width, height],
            "confidence": 0.99
        }
    ]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在：{image_path}")

    result = ocr.predict(image_path)

    detections = []

    for page in result:
        data = page.json

        texts = data["res"].get("rec_texts", [])
        scores = data["res"].get("rec_scores", [])
        boxes = data["res"].get("rec_boxes", [])

        for text, score, box in zip(texts, scores, boxes):
            if not str(text).strip():
                continue

            x1, y1, x2, y2 = [int(value) for value in box]

            detections.append({
                "text": str(text),
                "bbox": [
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                ],
                "confidence": float(score)
            })

    return detections


def draw_detection_preview(image_path, detections):
    """在图片上画出文字框，方便网页预览。"""
    image = cv2.imdecode(
        np.fromfile(image_path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("图片读取失败")

    for index, item in enumerate(detections, start=1):
        x, y, width, height = item["bbox"]

        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            2
        )

        cv2.putText(
            image,
            str(index),
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)