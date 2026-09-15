# -*- coding: utf-8 -*-
"""数据总览页面 — 道路病害检测系统仪表板"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from streamlit_utils import (
    apply_theme,
    init_session_state,
    get_records,
    get_statistics,
    b64_to_image,
    SEVERITY_EMOJI,
)

# ── 初始化 ──────────────────────────────────────────
apply_theme()
init_session_state()

st.markdown("# 📊 数据总览")
st.markdown("---")

# ── 获取数据 ────────────────────────────────────────
records = get_records()
stats = get_statistics()

# ═══════════════════════════════════════════════════
#  1. 四项核心指标卡片
# ═══════════════════════════════════════════════════

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="总检测次数", value=stats["total_records"])

with col2:
    st.metric(label="病害检出总数", value=stats["total_detections"])

with col3:
    st.metric(label="今日检出", value=stats["today_count"])

with col4:
    avg_conf_str = f"{stats['avg_confidence']:.1%}" if stats["avg_confidence"] > 0 else "N/A"
    st.metric(label="平均置信度", value=avg_conf_str)

st.markdown("---")

# ═══════════════════════════════════════════════════
#  2. 类别分布柱状图
# ═══════════════════════════════════════════════════

st.markdown("### 类别分布")

class_counts = stats["class_counts"]

if not class_counts:
    st.info("暂无检测数据，请先在主页进行图片检测")
else:
    # 构建 DataFrame 供 st.bar_chart 使用
    df_class = pd.DataFrame(
        {"类别": list(class_counts.keys()), "数量": list(class_counts.values())}
    )
    df_class = df_class.set_index("类别")
    st.bar_chart(df_class)

st.markdown("---")

# ═══════════════════════════════════════════════════
#  3. 严重程度分布
# ═══════════════════════════════════════════════════

st.markdown("### 严重程度分布")

severity_counts = stats["severity_counts"]

sev_col1, sev_col2, sev_col3 = st.columns(3)

# 轻微 — 绿色
with sev_col1:
    st.markdown(
        f"<div style='text-align:center;'>"
        f"<span style='color:#22c55e; font-size:1.1rem; font-weight:bold;'>"
        f"{SEVERITY_EMOJI.get('轻微', '🟢')} 轻微</span><br>"
        f"<span style='font-size:2rem; color:#22c55e; font-weight:bold;'>"
        f"{severity_counts.get('轻微', 0)}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    total_sev = sum(severity_counts.values())
    minor_ratio = severity_counts.get("轻微", 0) / total_sev if total_sev > 0 else 0
    st.progress(minor_ratio)

# 中等 — 黄色
with sev_col2:
    st.markdown(
        f"<div style='text-align:center;'>"
        f"<span style='color:#f59e0b; font-size:1.1rem; font-weight:bold;'>"
        f"{SEVERITY_EMOJI.get('中等', '🟡')} 中等</span><br>"
        f"<span style='font-size:2rem; color:#f59e0b; font-weight:bold;'>"
        f"{severity_counts.get('中等', 0)}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    moderate_ratio = severity_counts.get("中等", 0) / total_sev if total_sev > 0 else 0
    st.progress(moderate_ratio)

# 严重 — 红色
with sev_col3:
    st.markdown(
        f"<div style='text-align:center;'>"
        f"<span style='color:#ef4444; font-size:1.1rem; font-weight:bold;'>"
        f"{SEVERITY_EMOJI.get('严重', '🔴')} 严重</span><br>"
        f"<span style='font-size:2rem; color:#ef4444; font-weight:bold;'>"
        f"{severity_counts.get('严重', 0)}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    severe_ratio = severity_counts.get("严重", 0) / total_sev if total_sev > 0 else 0
    st.progress(severe_ratio)

st.markdown("---")

# ═══════════════════════════════════════════════════
#  4. 检测图片画廊
# ═══════════════════════════════════════════════════

st.markdown("### 检测图片画廊")

# 筛选含有 base64 图片的记录，取最近 12 条
image_records = [r for r in records if r.get("image_b64")]
image_records = image_records[-12:]

if not image_records:
    st.info("暂无检测图片，请先在主页进行图片检测")
else:
    # 每行 3 列
    for row_start in range(0, len(image_records), 3):
        cols = st.columns(3)
        for col_idx in range(3):
            idx = row_start + col_idx
            if idx >= len(image_records):
                break
            rec = image_records[idx]
            with cols[col_idx]:
                try:
                    img_bgr = b64_to_image(rec["image_b64"])
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True)
                except Exception:
                    st.image(
                        np.zeros((200, 300, 3), dtype=np.uint8),
                        caption="图片加载失败",
                        use_container_width=True,
                    )
                st.caption(
                    f"**{rec.get('id', 'N/A')}** "
                    f"| {rec.get('timestamp', '')} "
                    f"| 检测: {rec.get('detection_count', 0)} 个目标"
                )
