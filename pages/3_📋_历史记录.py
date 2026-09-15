# -*- coding: utf-8 -*-
"""历史记录页 — 浏览、筛选、导出本次会话的检测记录"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd

from streamlit_utils import (
    apply_theme,
    init_session_state,
    get_records,
    export_csv,
    b64_to_image,
    SEVERITY_EMOJI,
)

# ── 全局初始化 ──────────────────────────────────────
apply_theme()
init_session_state()

# ── 页面标题 ────────────────────────────────────────
st.markdown("# 📋 历史记录")

st.divider()

# ── 1. 筛选栏 ──────────────────────────────────────
col_cls, col_kw = st.columns([1, 2])

with col_cls:
    class_filter = st.selectbox("类别筛选", ["全部", "crack", "pothole"])

with col_kw:
    keyword = st.text_input("关键词搜索（类别 / CLIP细分 / 来源）", "")

# ── 2. 应用筛选逻辑 ─────────────────────────────────
all_records = get_records()
filtered = []

for rec in all_records:
    # 类别筛选
    if class_filter != "全部":
        classes_in_rec = {d.get("类别", "") for d in rec.get("detections", [])}
        if class_filter not in classes_in_rec:
            continue

    # 关键词筛选（搜索类别、CLIP细分、来源）
    if keyword.strip():
        kw = keyword.strip().lower()
        hit = False
        # 搜索来源
        if kw in rec.get("source", "").lower():
            hit = True
        # 搜索每条检测的类别和CLIP细分
        if not hit:
            for d in rec.get("detections", []):
                if kw in d.get("类别", "").lower():
                    hit = True
                    break
                if kw in d.get("CLIP细分", "").lower():
                    hit = True
                    break
        if not hit:
            continue

    filtered.append(rec)

# ── 3. 记录计数 ─────────────────────────────────────
st.markdown(f"**共 {len(filtered)} 条记录**")

st.divider()

# ── 4. 记录列表（最新在前） ──────────────────────────
if not filtered:
    st.info("暂无检测记录。请先在主页或开放词汇检测页进行检测。")
else:
    for rec in reversed(filtered):
        rec_id = rec.get("id", "未知")
        ts = rec.get("timestamp", "")
        source = rec.get("source", "")
        det_count = rec.get("detection_count", 0)

        with st.expander(
            f"{rec_id} | {ts} | {source} | {det_count}个目标",
            expanded=False,
        ):
            # ── 展示检测图片 ──
            b64_img = rec.get("image_b64")
            if b64_img:
                try:
                    img_bgr = b64_to_image(b64_img)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True)
                except Exception:
                    pass

            # ── 检测结果表格 ──
            detections = rec.get("detections", [])
            if detections:
                table_rows = []
                for d in detections:
                    cls_name = d.get("类别", "")
                    clip_fine = d.get("CLIP细分", "")

                    # 置信度格式化为百分比
                    conf_raw = d.get("置信度", 0)
                    if isinstance(conf_raw, (int, float)):
                        conf_str = f"{conf_raw:.1%}"
                    else:
                        conf_str = str(conf_raw)

                    # 面积占比格式化为百分比
                    area_raw = d.get("面积占比", 0)
                    if isinstance(area_raw, (int, float)):
                        area_str = f"{area_raw:.2%}"
                    else:
                        area_str = str(area_raw)

                    # 严重等级 + emoji
                    severity = d.get("严重等级", "轻微")
                    emoji = SEVERITY_EMOJI.get(severity, "")
                    severity_str = f"{emoji} {severity}"

                    row = {
                        "类别": cls_name,
                        "置信度": conf_str,
                        "面积占比": area_str,
                        "严重等级": severity_str,
                    }
                    # CLIP细分列仅在有数据时添加
                    if clip_fine:
                        row["CLIP细分"] = clip_fine

                    table_rows.append(row)

                # 确保列顺序：类别, CLIP细分(若有), 置信度, 面积占比, 严重等级
                has_clip = any("CLIP细分" in r for r in table_rows)
                if has_clip:
                    columns_order = ["类别", "CLIP细分", "置信度", "面积占比", "严重等级"]
                else:
                    columns_order = ["类别", "置信度", "面积占比", "严重等级"]

                # 补齐缺失列
                for r in table_rows:
                    for col in columns_order:
                        r.setdefault(col, "")

                df = pd.DataFrame(table_rows, columns=columns_order)
                st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # ── 5. 导出区 ───────────────────────────────────
    csv_bytes = export_csv(filtered)
    st.download_button(
        label="📥 导出全部记录为CSV",
        data=csv_bytes,
        file_name="detection_records.csv",
        mime="text/csv",
    )
