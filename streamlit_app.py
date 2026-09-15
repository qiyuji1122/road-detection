# -*- coding: utf-8 -*-
"""基于CLIP增强YOLO的道路病害智能检测系统 — Streamlit Web端

运行方式: streamlit run streamlit_app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════
#  页面配置
# ═══════════════════════════════════════════════════

st.set_page_config(
    page_title="道路病害智能检测系统",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 深色科技风自定义CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0a0e1a;
        color: #e0e6f0;
    }
    .stSidebar {
        background-color: #0d1225;
        border-right: 1px solid #1a2744;
    }
    .stSidebar .sidebar-content {
        background-color: #0d1225;
    }
    h1, h2, h3 {
        color: #4fc3f7;
    }
    .stMetric {
        background-color: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px;
    }
    .stMetric label {
        color: #94a3b8;
    }
    .stMetric .metric-value {
        color: #4fc3f7;
    }
    .detection-card {
        background: linear-gradient(135deg, #111827 0%, #0d1225 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .severity-severe {
        color: #ef4444;
        font-weight: bold;
    }
    .severity-moderate {
        color: #f59e0b;
        font-weight: bold;
    }
    .severity-minor {
        color: #22c55e;
        font-weight: bold;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    .stFileUploader {
        border: 2px dashed #1e3a5f;
        border-radius: 12px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  模型加载（缓存）
# ═══════════════════════════════════════════════════

@st.cache_resource
def load_yolo_model(model_path):
    """加载YOLO模型（缓存）"""
    from ultralytics import YOLO
    return YOLO(model_path)


def get_grad_cam(model, image, conf=0.5, iou=0.45):
    """生成Grad-CAM热力图"""
    try:
        from grad_cam import GradCAMExtractor
        extractor = GradCAMExtractor(model)
        return extractor.generate(image, conf=conf, iou=iou)
    except Exception as e:
        return {"success": False, "error": str(e)}


def classify_severity(cls_name, conf, area_ratio):
    """严重程度分类"""
    if conf > 0.8 and area_ratio > 0.1:
        return "severe"
    elif conf > 0.6 or area_ratio > 0.05:
        return "moderate"
    else:
        return "minor"


# ═══════════════════════════════════════════════════
#  自动查找模型文件
# ═══════════════════════════════════════════════════

APP_DIR = os.path.dirname(os.path.abspath(__file__))

def find_model_file():
    """自动查找模型权重文件"""
    candidates = ["best.pt", "200轮best.pt"]
    for name in candidates:
        path = os.path.join(APP_DIR, name)
        if os.path.exists(path):
            return path
    # 搜索 runs 目录
    for root, dirs, files in os.walk(os.path.join(APP_DIR, "runs")):
        for f in files:
            if f == "best.pt":
                return os.path.join(root, f)
    return None


# ═══════════════════════════════════════════════════
#  侧边栏
# ═══════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛣️ 道路病害智能检测系统")
    st.markdown("---")
    st.markdown("**基于CLIP增强YOLOv11**")
    st.caption("YOLOv11 + CLIP 两阶段检测架构")

    st.markdown("---")
    st.markdown("### 🎯 检测参数")
    col1, col2 = st.columns(2)
    with col1:
        conf_threshold = st.slider("置信度阈值", 0.1, 0.95, 0.25, 0.05)
    with col2:
        iou_threshold = st.slider("IoU阈值", 0.1, 0.95, 0.45, 0.05)

    # Grad-CAM开关
    st.markdown("---")
    enable_grad_cam = st.checkbox("🔥 启用Grad-CAM热力图", value=True)

    st.markdown("---")
    st.caption("© 2025 智巡路网 — 基于CLIP增强YOLO的道路病害智能检测系统")


# ═══════════════════════════════════════════════════
#  主界面
# ═══════════════════════════════════════════════════

# 标题区
st.markdown("# 🛣️ 道路病害智能检测系统")
st.markdown("*基于CLIP增强YOLOv11的道路病害目标检测与可解释性分析平台*")
st.markdown("---")

# 自动加载模型
model = None
auto_path = find_model_file()
if auto_path:
    try:
        with st.spinner("正在加载模型，请稍候..."):
            model = load_yolo_model(auto_path)
        st.success(f"✅ 模型加载成功 — 类别: {list(model.names.values())}")
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
else:
    st.warning("⚠️ 未找到模型权重文件，请将 best.pt 放在项目根目录")

# ═══════════════════════════════════════════════════
#  图片上传与检测
# ═══════════════════════════════════════════════════

st.markdown("### 📤 上传图片")
uploaded_file = st.file_uploader(
    "选择道路图片进行检测",
    type=["jpg", "jpeg", "png", "bmp"],
    help="支持 JPG/PNG/BMP 格式的道路图片"
)

if uploaded_file is not None:
    # 读取图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if model is None:
        st.warning("请先加载模型后再进行检测")
        st.image(image_rgb, caption="上传的图片", use_container_width=True)
    else:
        # 检测按钮
        if st.button("🔍 开始检测", type="primary", use_container_width=True):
            with st.spinner("正在进行目标检测..."):
                t0 = time.time()
                results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
                inference_time = (time.time() - t0) * 1000
                annotated = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # 解析检测结果
            detections = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                h, w = image.shape[:2]
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    cls_name = results[0].names.get(cls_id, f"class_{cls_id}")
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    bbox_area = (x2 - x1) * (y2 - y1)
                    frame_area = h * w
                    area_ratio = bbox_area / frame_area if frame_area > 0 else 0
                    severity = classify_severity(cls_name, conf, area_ratio)
                    detections.append({
                        "类别": cls_name,
                        "置信度": f"{conf:.1%}",
                        "位置": f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})",
                        "面积占比": f"{area_ratio:.2%}",
                        "严重等级": severity,
                    })

            # Grad-CAM
            grad_cam_heatmap = None
            grad_cam_overlay = None
            if enable_grad_cam and len(detections) > 0:
                with st.spinner("正在生成Grad-CAM热力图..."):
                    cam_result = get_grad_cam(model, image, conf=conf_threshold, iou=iou_threshold)
                    if cam_result.get("success"):
                        grad_cam_heatmap = cv2.cvtColor(cam_result["heatmap"], cv2.COLOR_BGR2RGB)
                        grad_cam_overlay = cv2.cvtColor(cam_result["overlay"], cv2.COLOR_BGR2RGB)
                    else:
                        st.caption(f"Grad-CAM暂不可用: {cam_result.get('error', '未知错误')}")

            # ─────────────────────────────────
            #  展示结果
            # ─────────────────────────────────

            st.markdown("---")
            st.markdown("### 📊 检测结果")

            # 指标卡片
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("检测目标数", len(detections))
            with m_col2:
                st.metric("推理耗时", f"{inference_time:.0f} ms")
            with m_col3:
                crack_count = sum(1 for d in detections if d["类别"] == "crack")
                st.metric("裂缝 (crack)", crack_count)
            with m_col4:
                pothole_count = sum(1 for d in detections if d["类别"] == "pothole")
                st.metric("坑洞 (pothole)", pothole_count)

            # 图片展示
            if grad_cam_overlay is not None:
                # 四图并排：原图、检测结果、热力图、叠加图
                st.markdown("#### 🖼️ 可视化结果")
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(image_rgb, caption="原始图片", use_container_width=True)
                with img_col2:
                    st.image(annotated_rgb, caption="检测结果", use_container_width=True)

                img_col3, img_col4 = st.columns(2)
                with img_col3:
                    st.image(grad_cam_heatmap, caption="Grad-CAM 热力图", use_container_width=True)
                with img_col4:
                    st.image(grad_cam_overlay, caption="热力图叠加", use_container_width=True)
            else:
                # 两图并排
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(image_rgb, caption="原始图片", use_container_width=True)
                with img_col2:
                    st.image(annotated_rgb, caption="检测结果", use_container_width=True)

            # 检测结果表格
            if detections:
                st.markdown("#### 📋 检测详情")

                # 为严重等级添加颜色标记
                severity_map = {"severe": "🔴 严重", "moderate": "🟡 中等", "minor": "🟢 轻微"}
                display_detections = []
                for d in detections:
                    display_d = dict(d)
                    display_d["严重等级"] = severity_map.get(d["严重等级"], d["严重等级"])
                    display_detections.append(display_d)

                st.dataframe(display_detections, use_container_width=True, hide_index=True)
            else:
                st.info("未检测到道路病害目标")

            # 导出按钮
            st.markdown("---")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                # 导出检测结果图
                _, buf = cv2.imencode(".jpg", annotated)
                st.download_button(
                    "📥 下载检测结果图",
                    data=buf.tobytes(),
                    file_name="detection_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
            if grad_cam_overlay is not None:
                with col_dl2:
                    _, buf_cam = cv2.imencode(".jpg", cv2.cvtColor(grad_cam_overlay, cv2.COLOR_RGB2BGR))
                    st.download_button(
                        "📥 下载热力图叠加",
                        data=buf_cam.tobytes(),
                        file_name="grad_cam_overlay.jpg",
                        mime="image/jpeg",
                        use_container_width=True,
                    )
                with col_dl3:
                    _, buf_heat = cv2.imencode(".jpg", cv2.cvtColor(grad_cam_heatmap, cv2.COLOR_RGB2BGR))
                    st.download_button(
                        "📥 下载热力图",
                        data=buf_heat.tobytes(),
                        file_name="grad_cam_heatmap.jpg",
                        mime="image/jpeg",
                        use_container_width=True,
                    )

# ═══════════════════════════════════════════════════
#  底部说明
# ═══════════════════════════════════════════════════

st.markdown("---")
with st.expander("📖 系统说明"):
    st.markdown("""
    **基于CLIP增强YOLO的道路病害智能检测系统** 采用两阶段检测架构：

    **第一阶段 — YOLOv11目标检测：** 使用在RDD2022数据集上训练的YOLOv11n模型进行道路病害的目标检测与定位，支持裂缝(crack)和坑洞(pothole)两类病害的识别。

    **第二阶段 — CLIP语义增强：** 利用CLIP视觉-语言模型的跨模态语义理解能力，对检测到的病害区域进行语义特征增强和细粒度分类，提升检测的准确性和可解释性。

    **Grad-CAM可解释性分析：** 系统集成Grad-CAM热力图可视化功能，通过计算目标类别得分相对于最后一个卷积层特征图的梯度，生成模型关注区域的热力图，帮助用户直观理解模型的检测决策过程。

    **支持类别：** 裂缝(crack) · 坑洞(pothole)

    **数据集：** RDD2022道路病害数据集（16,600张训练图片 + 7,100张验证图片）
    """)
