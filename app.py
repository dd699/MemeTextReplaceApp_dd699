
from pathlib import Path

import streamlit as st
from PIL import Image

from src.detector import detect_text, draw_detection_preview
from src.repair import repair_text
from src.renderer import render_text


# 页面设置
st.set_page_config(
    page_title="表情包文字替换",
    layout="wide"
)

st.markdown(
    """
    <style>
    div.stButton > button,
    div.stDownloadButton > button {
        background: #111111 !important;
        color: white !important;
        border: none !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #333333 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("表情包文字替换系统")
st.caption("上传图片，检测文字并生成替换结果。")


# 文件目录
ROOT = Path(__file__).parent
WORKSPACE = ROOT / "workspace"

INPUT_DIR = WORKSPACE / "input"
MASK_DIR = WORKSPACE / "repair" / "masks"
PREVIEW_DIR = WORKSPACE / "repair" / "previews"
REPAIRED_DIR = WORKSPACE / "repair" / "repaired"
OUTPUT_DIR = WORKSPACE / "output"

for folder in [
    INPUT_DIR,
    MASK_DIR,
    PREVIEW_DIR,
    REPAIRED_DIR,
    OUTPUT_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


# 状态初始化
for key in [
    "image_path",
    "image_name",
    "detections",
    "repaired_path",
    "final_path"
]:
    if key not in st.session_state:
        st.session_state[key] = None


# 上传图片
uploaded_file = st.file_uploader(
    "选择表情包",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image_path = INPUT_DIR / uploaded_file.name

    if st.session_state.image_name != uploaded_file.name:
        st.session_state.detections = None
        st.session_state.repaired_path = None
        st.session_state.final_path = None

    image_path.write_bytes(uploaded_file.getbuffer())

    st.session_state.image_path = str(image_path)
    st.session_state.image_name = uploaded_file.name


# 检测文字
if st.button(
    "检测文字",
    disabled=st.session_state.image_path is None
):
    try:
        with st.spinner("正在检测文字……"):
            st.session_state.detections = detect_text(
                st.session_state.image_path
            )

        if st.session_state.detections:
            st.success(
                f"检测到 {len(st.session_state.detections)} 个文字区域"
            )
        else:
            st.warning("没有检测到文字")

    except Exception as error:
        st.error(f"检测失败：{error}")


# 四张图同一行
st.subheader("处理流程")

columns = st.columns(4)

titles = [
    "原始图片",
    "检测结果",
    "修复结果",
    "替换结果"
]

for column, title in zip(columns, titles):
    with column:
        st.markdown(f"#### {title}")


with columns[0]:
    if st.session_state.image_path:
        st.image(
            Image.open(st.session_state.image_path),
            use_container_width=True
        )
    else:
        st.info("请上传图片")


with columns[1]:
    if (
        st.session_state.image_path
        and st.session_state.detections
    ):
        preview = draw_detection_preview(
            st.session_state.image_path,
            st.session_state.detections
        )

        st.image(
            preview,
            use_container_width=True
        )
    else:
        st.info("等待检测")


with columns[2]:
    if (
        st.session_state.repaired_path
        and Path(st.session_state.repaired_path).exists()
    ):
        st.image(
            Image.open(st.session_state.repaired_path),
            use_container_width=True
        )
    else:
        st.info("等待修复")


with columns[3]:
    if (
        st.session_state.final_path
        and Path(st.session_state.final_path).exists()
    ):
        st.image(
            Image.open(st.session_state.final_path),
            use_container_width=True
        )
    else:
        st.info("等待生成")


# 输入替换文字
detections = st.session_state.detections

if detections:
    st.subheader("输入替换文字")

    new_texts = []

    for index, item in enumerate(detections):
        new_text = st.text_input(
            f"区域 {index + 1}，原文字：{item['text']}",
            value=item["text"],
            key=f"text_{st.session_state.image_name}_{index}"
        )

        new_texts.append(new_text)


    # 生成结果
    if st.button("生成替换结果"):
        image_path = Path(st.session_state.image_path)
        name = image_path.stem

        repaired_path = REPAIRED_DIR / f"{name}_repaired.png"
        mask_path = MASK_DIR / f"{name}_mask.png"
        preview_path = PREVIEW_DIR / f"{name}_preview.png"
        final_path = OUTPUT_DIR / f"{name}_final.png"

        try:
            with st.spinner("正在生成结果……"):
                repair_text(
                    image_path=image_path,
                    detections=detections,
                    repaired_output_path=repaired_path,
                    mask_output_path=mask_path,
                    preview_output_path=preview_path
                )

                render_text(
                    original_path=image_path,
                    repaired_path=repaired_path,
                    detections=detections,
                    new_texts=new_texts,
                    output_path=final_path
                )

            st.session_state.repaired_path = str(repaired_path)
            st.session_state.final_path = str(final_path)

            st.rerun()

        except Exception as error:
            st.error(f"处理失败：{error}")

# 下载结果
if (
    st.session_state.final_path
    and Path(st.session_state.final_path).exists()
):
    final_path = Path(st.session_state.final_path)

    st.download_button(
        "下载最终图片",
        data=final_path.read_bytes(),
        file_name=final_path.name,
        mime="image/png"
    )

