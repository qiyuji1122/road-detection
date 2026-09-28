# -*- coding: utf-8 -*-
"""Streamlit 共享工具模块 — 模型缓存、CLIP缓存、会话存储、检测逻辑"""

import os
import sys
import time
import csv
import io
import numpy as np
import streamlit as st
from datetime import datetime

# 项目根目录
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════
#  深色科技风 CSS（所有页面共用）
# ═══════════════════════════════════════════════════

THEME_CSS = """
<style>
    :root {
        --bg: #07111f;
        --panel: rgba(12, 29, 48, .86);
        --panel-2: #0c1b2d;
        --line: rgba(148, 163, 184, .15);
        --muted: #8fa5ba;
        --text: #eaf3fb;
        --brand: #19c2b1;
        --brand-2: #38bdf8;
        --warning: #f3b55b;
    }
    .stApp {
        color: var(--text);
        background:
            linear-gradient(90deg, rgba(3, 10, 20, .97) 0%, rgba(4, 15, 28, .90) 37%, rgba(4, 15, 28, .68) 72%, rgba(4, 13, 25, .76) 100%),
            linear-gradient(180deg, rgba(3, 10, 19, .12) 0%, rgba(5, 15, 27, .50) 54%, #07111f 100%),
            url("/app/static/autonomous-road-inspection.png") center top / cover fixed no-repeat,
            var(--bg);
    }
    [data-testid="stAppViewContainer"] {
        background: transparent;
    }
    [data-testid="stHeader"] {
        background: rgba(4, 13, 24, .30);
        backdrop-filter: blur(12px);
    }
    .stApp, .stApp button, .stApp input, .stApp textarea {
        font-family: Inter, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
    }
    .block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 4rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(9, 24, 39, .96) 0%, rgba(7, 17, 31, .97) 100%);
        backdrop-filter: blur(18px);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #b7c7d6; }

    /* Keep Streamlit's native reopen button available after the sidebar is collapsed. */
    [data-testid="stSidebarCollapsedControl"],
    button[data-testid="stBaseButton-headerNoPadding"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 1000000 !important;
    }
    div[data-testid="stToolbar"] { visibility: visible; }

    h1, h2, h3 { color: #f5fbff; letter-spacing: -.025em; }
    h1 { font-size: clamp(2rem, 3vw, 3.15rem) !important; }
    a { color: var(--brand-2) !important; }
    [data-testid="stMetric"] {
        min-height: 118px;
        background: linear-gradient(145deg, rgba(15, 39, 61, .94), rgba(9, 26, 43, .94));
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, .16);
    }
    [data-testid="stMetric"] label { color: var(--muted); }
    [data-testid="stMetricValue"] { color: #f6fbff; }
    [data-testid="stMetricDelta"] { color: var(--brand); }
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(56, 189, 248, .34);
        border-radius: 16px;
        padding: 4px;
        background: rgba(12, 29, 48, .5);
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px; border-bottom: 1px solid var(--line);
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        height: 46px; border-radius: 10px 10px 0 0; padding-inline: 18px;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(56, 189, 248, .22);
        min-height: 42px;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea795, #0e85ac);
        border: 0;
        color: white;
        box-shadow: 0 8px 24px rgba(14, 167, 149, .22);
    }
    .platform-eyebrow {
        color: #64d8cb; font-size: .76rem; font-weight: 760; letter-spacing: .14em;
        text-transform: uppercase; margin-bottom: .55rem;
    }
    .platform-subtitle { color: var(--muted); font-size: 1.03rem; max-width: 800px; line-height: 1.75; }
    .surface-card {
        height: 100%; padding: 22px; border-radius: 18px;
        border: 1px solid var(--line);
        background: linear-gradient(145deg, rgba(14, 36, 57, .90), rgba(8, 25, 41, .90));
        box-shadow: 0 14px 42px rgba(0,0,0,.15);
    }
    .surface-card h4 { margin: 0 0 8px; color: #edf8ff; font-size: 1.02rem; }
    .surface-card p { margin: 0; color: var(--muted); line-height: 1.65; }
    .status-pill {
        display: inline-flex; align-items: center; gap: 7px; padding: 6px 10px;
        border-radius: 999px; background: rgba(25, 194, 177, .11);
        border: 1px solid rgba(25, 194, 177, .25); color: #73e2d5; font-size: .78rem;
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #23d5b7; box-shadow: 0 0 12px #23d5b7; }
    .hero-copy { padding: 20px 0 12px; }
    .hero-copy {
        min-height: 385px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-shadow: 0 2px 28px rgba(0, 0, 0, .58);
    }
    .hero-copy h1 { margin: .35rem 0 .75rem; }
    .hero-copy .lead { color: #a9bed0; font-size: 1.08rem; line-height: 1.8; max-width: 680px; }
    .hero-points { margin-top: 1.1rem; color: #cbd9e4; font-size: .9rem; }
    .hero-art img { border-radius: 22px; border: 1px solid var(--line); box-shadow: 0 24px 70px rgba(0,0,0,.35); }
    .section-kicker { color: var(--brand); font-weight: 700; font-size: .78rem; letter-spacing: .08em; }
    .empty-panel { text-align:center; padding: 52px 24px; border: 1px dashed rgba(148,163,184,.2); border-radius: 16px; color: var(--muted); }
    .severity-severe { color: #fb7185; font-weight: bold; }
    .severity-moderate { color: #f6bd60; font-weight: bold; }
    .severity-minor { color: #36d399; font-weight: bold; }
    @media (max-width: 900px) {
        .stApp {
            background-position: 64% top;
            background-attachment: scroll;
        }
        .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: .9rem; }
        [data-testid="stHorizontalBlock"] { gap: .8rem; }
        .hero-copy { min-height: 340px; padding-top: 0; }
        [data-testid="stMetric"] { min-height: 100px; padding: 14px; }
    }
</style>
"""


def apply_theme():
    """应用深色科技风主题"""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  模型加载（全局缓存）
# ═══════════════════════════════════════════════════

@st.cache_resource
def load_yolo_model(model_path):
    """加载YOLO模型（缓存）"""
    from ultralytics import YOLO
    return YOLO(model_path)


@st.cache_resource
def load_clip_classifier():
    """加载CLIP分类器（缓存，首次约30秒）"""
    from clip_classifier import CLIPClassifier
    return CLIPClassifier(device="cpu")


def find_model_file():
    """自动查找模型权重文件"""
    candidates = ["best.pt", "200轮best.pt"]
    for name in candidates:
        path = os.path.join(APP_DIR, name)
        if os.path.exists(path):
            return path
    for root, dirs, files in os.walk(os.path.join(APP_DIR, "runs")):
        for f in files:
            if f == "best.pt":
                return os.path.join(root, f)
    return None


def auto_load_model():
    """自动查找并加载模型，返回 (model, path) 或 (None, None)"""
    path = find_model_file()
    if path:
        try:
            model = load_yolo_model(path)
            return model, path
        except Exception:
            return None, None
    return None, None


# ═══════════════════════════════════════════════════
#  严重度分类
# ═══════════════════════════════════════════════════

def classify_severity(conf, area_ratio):
    """基于置信度和面积占比判断严重程度"""
    if conf > 0.8 and area_ratio > 0.1:
        return "严重"
    elif conf > 0.6 or area_ratio > 0.05:
        return "中等"
    else:
        return "轻微"


SEVERITY_EMOJI = {"严重": "🔴", "中等": "🟡", "轻微": "🟢"}


# ═══════════════════════════════════════════════════
#  检测逻辑
# ═══════════════════════════════════════════════════

def run_detection(model, image, conf=0.25, iou=0.45, clip=None):
    """
    执行 YOLO 检测 + 可选 CLIP 细分

    Args:
        model: YOLO 模型实例
        image: BGR numpy 图像
        conf: 置信度阈值
        iou: IoU 阈值
        clip: CLIPClassifier 实例（None 则跳过细分）

    Returns:
        annotated_bgr: 标注后的 BGR 图像
        detections: list of dict
        inference_ms: 推理耗时（毫秒）
    """
    t0 = time.time()
    results = model(image, conf=conf, iou=iou, verbose=False)
    inference_ms = (time.time() - t0) * 1000

    annotated = results[0].plot()
    detections = []

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        h, w = image.shape[:2]
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = results[0].names.get(cls_id, f"class_{cls_id}")
            det_conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox_area = (x2 - x1) * (y2 - y1)
            area_ratio = bbox_area / (h * w) if h * w > 0 else 0

            det = {
                "类别": cls_name,
                "置信度": det_conf,
                "位置": f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})",
                "面积占比": area_ratio,
                "严重等级": classify_severity(det_conf, area_ratio),
            }

            # CLIP 细分
            if clip is not None:
                crop = image[int(y1):int(y2), int(x1):int(x2)]
                if crop.size > 0:
                    try:
                        clip_result = clip.classify(crop, cls_name)
                        det["CLIP细分"] = clip_result.get("fine_class", "")
                        det["CLIP细分置信度"] = clip_result.get("fine_confidence", 0)
                        det["CLIP严重度"] = clip_result.get("clip_severity", "")
                        # 用 CLIP 的严重度覆盖
                        if clip_result.get("clip_severity"):
                            det["严重等级"] = clip_result["clip_severity"]
                    except Exception:
                        pass

            detections.append(det)

    return annotated, detections, inference_ms


def get_grad_cam(model, image, conf=0.25, iou=0.45):
    """生成 Grad-CAM 热力图"""
    try:
        from grad_cam import GradCAMExtractor
        extractor = GradCAMExtractor(model)
        return extractor.generate(image, conf=conf, iou=iou)
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════
#  会话存储（session_state）
# ═══════════════════════════════════════════════════

def init_session_state():
    """初始化 session_state"""
    defaults = {
        "records": [],
        "tasks": [],
        "archives": [],
        "clip_cases": [],
        "system_settings": {
            "confidence": 0.25,
            "iou": 0.45,
            "enable_grad_cam": False,
            "enable_clip": False,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value


def save_record(source, detections, image_b64=None):
    """
    保存一条检测记录到 session_state

    Args:
        source: 检测来源（"图片检测" / "开放词汇检测"）
        detections: 检测结果列表
        image_b64: 可选的 base64 编码检测图片
    """
    init_session_state()
    now = datetime.now()
    today_count = sum(
        1 for r in st.session_state["records"]
        if r.get("timestamp", "").startswith(now.strftime("%Y-%m-%d"))
    )
    record_id = f"REC-{now.strftime('%Y%m%d')}-{today_count + 1:03d}"

    record = {
        "id": record_id,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "detections": detections,
        "image_b64": image_b64,
        "detection_count": len(detections),
    }
    st.session_state["records"].append(record)
    return record_id


def get_records():
    """获取所有会话记录"""
    init_session_state()
    return st.session_state["records"]


def get_statistics():
    """
    计算会话统计数据

    Returns:
        dict with keys: total_records, total_detections, today_count,
                        avg_confidence, class_counts, severity_counts
    """
    records = get_records()
    total_records = len(records)
    total_detections = sum(r.get("detection_count", 0) for r in records)

    today_str = datetime.now().strftime("%Y-%m-%d")
    today_count = sum(
        r.get("detection_count", 0) for r in records
        if r.get("timestamp", "").startswith(today_str)
    )

    all_confs = []
    class_counts = {}
    severity_counts = {"轻微": 0, "中等": 0, "严重": 0}

    for r in records:
        for d in r.get("detections", []):
            if isinstance(d.get("置信度"), (int, float)):
                all_confs.append(d["置信度"])
            cls = d.get("类别", "unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1
            sev = d.get("严重等级", "轻微")
            if sev in severity_counts:
                severity_counts[sev] += 1

    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0

    return {
        "total_records": total_records,
        "total_detections": total_detections,
        "today_count": today_count,
        "avg_confidence": avg_conf,
        "class_counts": class_counts,
        "severity_counts": severity_counts,
    }


# ═══════════════════════════════════════════════════
#  导出功能
# ═══════════════════════════════════════════════════

def export_csv(records):
    """将记录导出为 CSV 字节流"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["记录ID", "时间", "来源", "检测数", "类别", "置信度", "CLIP细分", "严重等级", "位置"])

    for r in records:
        for d in r.get("detections", []):
            writer.writerow([
                r.get("id", ""),
                r.get("timestamp", ""),
                r.get("source", ""),
                r.get("detection_count", 0),
                d.get("类别", ""),
                f"{d.get('置信度', 0):.2f}" if isinstance(d.get("置信度"), (int, float)) else d.get("置信度", ""),
                d.get("CLIP细分", ""),
                d.get("严重等级", ""),
                d.get("位置", ""),
            ])

    return output.getvalue().encode("utf-8-sig")


def generate_report(records, title="道路病害检测报告"):
    """生成文本报告"""
    stats = get_statistics()
    lines = [
        "=" * 60,
        f"  {title}",
        "=" * 60,
        f"  生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  报告范围：本次会话全部记录",
        "-" * 60,
        "",
        "【概况统计】",
        f"  检测次数：{stats['total_records']}",
        f"  检出总数：{stats['total_detections']}",
        f"  平均置信度：{stats['avg_confidence']:.2%}",
        "",
        "【类别分布】",
    ]

    for cls, count in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"  {cls}：{count} 次")

    lines.append("")
    lines.append("【严重程度分布】")
    for sev, count in stats["severity_counts"].items():
        lines.append(f"  {sev}：{count} 次")

    lines.append("")
    lines.append("=" * 60)
    lines.append("【检测详情】（最近20条记录）")
    lines.append("-" * 60)

    for r in records[-20:]:
        lines.append(f"\n  ▸ {r['id']}  |  {r['timestamp']}  |  来源: {r['source']}")
        for i, d in enumerate(r.get("detections", []), 1):
            conf_str = f"{d['置信度']:.1%}" if isinstance(d.get("置信度"), (int, float)) else str(d.get("置信度", ""))
            clip_str = f"  CLIP细分: {d.get('CLIP细分', 'N/A')}" if d.get("CLIP细分") else ""
            lines.append(
                f"    {i}. {d.get('类别', '')}  置信度: {conf_str}  "
                f"严重: {d.get('严重等级', '')}{clip_str}"
            )

    lines.append("")
    lines.append("=" * 60)
    lines.append("  — 报告结束 —")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════
#  图片编解码工具
# ═══════════════════════════════════════════════════

def image_to_b64(image_bgr):
    """BGR numpy → base64 字符串"""
    import base64
    import cv2
    _, buf = cv2.imencode(".jpg", image_bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def b64_to_image(b64_str):
    """base64 字符串 → BGR numpy"""
    import base64
    import cv2
    raw = base64.b64decode(b64_str)
    buf = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
