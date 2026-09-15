# -*- coding: utf-8 -*-
"""报告生成页面 — 生成、预览和导出检测报告"""

import streamlit as st
from streamlit_utils import (
    apply_theme,
    init_session_state,
    get_records,
    generate_report,
    export_csv,
    get_statistics,
)

# ── 主题 & 会话初始化 ──────────────────────────────
apply_theme()
init_session_state()

# ── 页面标题 ────────────────────────────────────────
st.markdown("# 📄 报告生成")
st.markdown("---")

# ── 获取记录 ────────────────────────────────────────
records = get_records()
stats = get_statistics()

# ═══════════════════════════════════════════════════
#  空状态
# ═══════════════════════════════════════════════════
if not records:
    st.info("暂无检测记录，无法生成报告。请先进行检测。")
    st.stop()

# ═══════════════════════════════════════════════════
#  1. 配置卡片
# ═══════════════════════════════════════════════════
st.subheader("⚙️ 报告配置")

report_title = st.text_input(
    "报告标题",
    value="道路病害检测报告",
    placeholder="请输入报告标题",
)

col_q1, col_q2 = st.columns(2)
with col_q1:
    st.metric("检测记录总数", f"{stats['total_records']} 条")
with col_q2:
    st.metric("检出目标总数", f"{stats['total_detections']} 个")

st.markdown("---")

# ═══════════════════════════════════════════════════
#  2. 操作按钮 — 生成报告
# ═══════════════════════════════════════════════════
col_btn1, col_btn2 = st.columns([1, 3])

with col_btn1:
    generate_clicked = st.button("生成报告预览", type="primary", use_container_width=True)

if generate_clicked:
    report_text = generate_report(records, title=report_title)
    st.session_state["report_text"] = report_text

# ═══════════════════════════════════════════════════
#  3. 报告预览
# ═══════════════════════════════════════════════════
st.subheader("📝 报告预览")

report_text = st.session_state.get("report_text", None)

if report_text:
    st.code(report_text, language="text")
else:
    st.markdown(
        "<div style='text-align:center; color:#64748b; padding:32px 0;'>"
        "点击上方 <b>生成报告预览</b> 按钮查看报告内容"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ═══════════════════════════════════════════════════
#  4. 导出按钮
# ═══════════════════════════════════════════════════
st.subheader("📤 导出报告")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    if report_text:
        st.download_button(
            label="📥 导出 TXT 报告",
            data=report_text.encode("utf-8"),
            file_name=f"{report_title}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.button("📥 导出 TXT 报告", disabled=True, use_container_width=True)

with col_exp2:
    st.download_button(
        label="📥 导出 CSV 数据",
        data=export_csv(records),
        file_name=f"{report_title}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")

# ═══════════════════════════════════════════════════
#  5. 统计摘要
# ═══════════════════════════════════════════════════
st.subheader("📊 统计摘要")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.metric("平均置信度", f"{stats['avg_confidence']:.2%}")
with col_s2:
    st.metric("今日检出数", f"{stats['today_count']} 个")
with col_s3:
    severity_summary = " / ".join(
        f"{k}: {v}" for k, v in stats["severity_counts"].items()
    )
    st.metric("严重度分布", severity_summary if stats["total_detections"] else "—")

# 类别分布
if stats["class_counts"]:
    st.markdown("**类别分布：**")
    cols = st.columns(min(len(stats["class_counts"]), 6))
    for idx, (cls_name, count) in enumerate(
        sorted(stats["class_counts"].items(), key=lambda x: -x[1])
    ):
        with cols[idx % len(cols)]:
            st.metric(cls_name, f"{count} 次")
