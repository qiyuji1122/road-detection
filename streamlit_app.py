# -*- coding: utf-8 -*-
"""基于CLIP增强YOLO的道路病害智能检测系统 — Streamlit Web端

运行方式: streamlit run streamlit_app.py
"""

# 修复 Streamlit Cloud 上 opencv-python 缺少 libGL 的问题
import subprocess, sys as _sys
try:
    import cv2  # noqa: F401
except ImportError:
    subprocess.check_call([_sys.executable, "-m", "pip", "install", "-q",
                           "opencv-python-headless", "--force-reinstall"])

import streamlit as st
import cv2
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamlit_utils import (
    apply_theme, auto_load_model, load_clip_classifier,
    run_detection, get_grad_cam, save_record, init_session_state,
    image_to_b64, SEVERITY_EMOJI,
)

# ═══════════════════════════════════════════════════
#  页面配置
# ═══════════════════════════════════════════════════

st.set_page_config(
    page_title="道路病害智能检测系统",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
init_session_state()

# ═══════════════════════════════════════════════════
#  侧边栏
# ═══════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛣️ 智巡路网")
    st.markdown("**CLIP增强YOLOv11 道路病害检测**")
    st.caption("YOLOv11 + CLIP 两阶段架构")

    st.markdown("---")
    st.markdown("### ⚙️ 检测参数")
    col1, col2 = st.columns(2)
    with col1:
        conf_threshold = st.slider("置信度", 0.1, 0.95, 0.25, 0.05)
    with col2:
        iou_threshold = st.slider("IoU", 0.1, 0.95, 0.45, 0.05)

    st.markdown("---")
    enable_clip = st.checkbox("🧠 启用CLIP细粒度分类", value=False,
                               help="对每个检测区域进行裂缝/坑洞子类细分（首次加载约30秒）")
    enable_grad_cam = st.checkbox("🔥 启用Grad-CAM热力图", value=True)

    st.markdown("---")
    st.caption("© 2025 智巡路网 — 道路病害智能检测系统")

# ═══════════════════════════════════════════════════
#  主界面
# ═══════════════════════════════════════════════════

st.markdown("# 🛣️ 道路病害智能检测系统")
st.markdown("*基于CLIP增强YOLOv11 · 道路病害目标检测与可解释性分析*")
st.markdown("---")

# 自动加载模型
model, model_path = auto_load_model()
if model:
    st.success(f"✅ 模型加载成功 — 类别: {list(model.names.values())}")
else:
    st.warning("⚠️ 未找到模型权重文件，请将 best.pt 放在项目根目录")

# ═══════════════════════════════════════════════════
#  图片上传与检测
# ═══════════════════════════════════════════════════

st.markdown("### 📤 上传图片")
uploaded_files = st.file_uploader(
    "选择道路图片（支持多张）",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
    help="支持 JPG/PNG/BMP 格式"
)

if uploaded_files and model:
    for idx, uploaded_file in enumerate(uploaded_files):
        if idx > 0:
            st.markdown("---")

        # 读取图片
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.markdown(f"#### 📷 {uploaded_file.name}")

        if st.button(f"🔍 检测此图片", key=f"detect_{idx}", type="primary"):
            # 加载 CLIP（如需要）
            clip = None
            if enable_clip:
                with st.spinner("正在加载CLIP模型..."):
                    try:
                        clip = load_clip_classifier()
                    except Exception as e:
                        st.warning(f"CLIP加载失败，将跳过细分: {e}")

            # YOLO 检测
            with st.spinner("正在进行目标检测..."):
                annotated, detections, inference_ms = run_detection(
                    model, image, conf=conf_threshold, iou=iou_threshold, clip=clip
                )
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Grad-CAM
            grad_cam_heatmap = None
            grad_cam_overlay = None
            if enable_grad_cam and detections:
                with st.spinner("正在生成Grad-CAM热力图..."):
                    cam_result = get_grad_cam(model, image, conf=conf_threshold, iou=iou_threshold)
                    if cam_result.get("success"):
                        grad_cam_heatmap = cv2.cvtColor(cam_result["heatmap"], cv2.COLOR_BGR2RGB)
                        grad_cam_overlay = cv2.cvtColor(cam_result["overlay"], cv2.COLOR_BGR2RGB)
                    else:
                        st.caption(f"Grad-CAM暂不可用: {cam_result.get('error', '')}")

            # ── 指标卡片 ──
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("检测目标数", len(detections))
            with m2:
                st.metric("推理耗时", f"{inference_ms:.0f} ms")
            with m3:
                crack_count = sum(1 for d in detections if d["类别"] == "crack")
                st.metric("裂缝 crack", crack_count)
            with m4:
                pothole_count = sum(1 for d in detections if d["类别"] == "pothole")
                st.metric("坑洞 pothole", pothole_count)

            # ── 图片展示 ──
            if grad_cam_overlay is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(image_rgb, caption="原始图片", use_container_width=True)
                with c2:
                    st.image(annotated_rgb, caption="检测结果", use_container_width=True)
                c3, c4 = st.columns(2)
                with c3:
                    st.image(grad_cam_heatmap, caption="Grad-CAM 热力图", use_container_width=True)
                with c4:
                    st.image(grad_cam_overlay, caption="热力图叠加", use_container_width=True)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(image_rgb, caption="原始图片", use_container_width=True)
                with c2:
                    st.image(annotated_rgb, caption="检测结果", use_container_width=True)

            # ── 检测结果表格 ──
            if detections:
                severity_map = {"严重": "🔴 严重", "中等": "🟡 中等", "轻微": "🟢 轻微"}
                display_rows = []
                for d in detections:
                    row = {"类别": d["类别"]}
                    if d.get("CLIP细分"):
                        row["CLIP细分"] = d["CLIP细分"]
                    row["置信度"] = f"{d['置信度']:.1%}"
                    row["面积占比"] = f"{d['面积占比']:.2%}"
                    row["严重等级"] = severity_map.get(d["严重等级"], d["严重等级"])
                    display_rows.append(row)

                st.dataframe(display_rows, use_container_width=True, hide_index=True)

                # 保存到会话历史
                img_b64 = image_to_b64(annotated)
                rec_id = save_record("图片检测", detections, img_b64)
                st.caption(f"已保存至会话历史: {rec_id}")

                # 导出按钮
                dl1, dl2, dl3 = st.columns(3)
                with dl1:
                    _, buf = cv2.imencode(".jpg", annotated)
                    st.download_button(
                        "📥 下载检测结果图",
                        data=buf.tobytes(),
                        file_name=f"detection_{uploaded_file.name}",
                        mime="image/jpeg",
                        key=f"dl_det_{idx}",
                        use_container_width=True,
                    )
                if grad_cam_overlay is not None:
                    with dl2:
                        _, buf2 = cv2.imencode(".jpg", cv2.cvtColor(grad_cam_overlay, cv2.COLOR_RGB2BGR))
                        st.download_button(
                            "📥 下载热力图叠加",
                            data=buf2.tobytes(),
                            file_name=f"gradcam_{uploaded_file.name}",
                            mime="image/jpeg",
                            key=f"dl_cam_{idx}",
                            use_container_width=True,
                        )
            else:
                st.info("未检测到道路病害目标")

elif uploaded_files and not model:
    st.warning("请先确保模型加载成功")

# ═══════════════════════════════════════════════════
#  底部说明
# ═══════════════════════════════════════════════════

st.markdown("---")
with st.expander("📖 系统说明"):
    st.markdown("""
    **基于CLIP增强YOLO的道路病害智能检测系统** 采用两阶段检测架构：

    **第一阶段 — YOLOv11目标检测：** 使用在RDD2022数据集上训练的YOLOv11n模型进行道路病害的目标检测与定位，支持裂缝(crack)和坑洞(pothole)两类病害的识别。

    **第二阶段 — CLIP语义增强：** 利用CLIP视觉-语言模型的跨模态语义理解能力，对检测到的病害区域进行细粒度分类（如纵向裂缝、网状裂缝、小型坑洼、大型坑洼等10种子类），并评估严重程度（轻微/中等/严重）。

    **Grad-CAM可解释性分析：** 通过热力图可视化模型的关注区域，帮助理解检测决策过程。

    **数据集：** RDD2022道路病害数据集（16,600张训练图片 + 7,100张验证图片）
    """)
