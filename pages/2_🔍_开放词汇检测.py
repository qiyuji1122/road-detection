# -*- coding: utf-8 -*-
"""开放词汇检测页面 — CLIP多尺度滑动窗口定位匹配区域"""

import streamlit as st
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streamlit_utils import (
    apply_theme, init_session_state, load_clip_classifier,
    save_record, image_to_b64,
)

# ═══════════════════════════════════════════════════
#  页面初始化
# ═══════════════════════════════════════════════════

st.set_page_config(
    page_title="开放词汇检测 - CLIP",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
init_session_state()

# ═══════════════════════════════════════════════════
#  页面标题与说明
# ═══════════════════════════════════════════════════

st.markdown("# 🔍 开放词汇检测（CLIP Open-Vocabulary）")
st.markdown(
    "输入任意文本描述，CLIP将在图片中定位匹配区域。支持中文输入（自动翻译）。处理时间约10-60秒。"
)
st.markdown("---")

# ═══════════════════════════════════════════════════
#  预设模板
# ═══════════════════════════════════════════════════

PRESET_TEMPLATES = {
    "自定义": "",
    "道路病害扩展": "标线磨损, 路面积水, 路面沉陷, 修补痕迹, 井盖周围破损",
    "交通设施": "交通标志, 道路护栏, 路灯, 减速带, 排水沟",
    "路面状况": "路面油污, 碎石散落, 路面结冰, 车辙印, 轮胎痕迹",
}

# ═══════════════════════════════════════════════════
#  侧边栏 — 参数控制
# ═══════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔍 开放词汇检测")
    st.caption("CLIP多尺度滑动窗口 · 零样本定位")

    st.markdown("---")
    st.markdown("### ⚙️ 检测参数")
    threshold = st.slider(
        "匹配阈值", min_value=0.10, max_value=0.90, value=0.40, step=0.05,
        help="对比评分阈值，越高越严格，越低检出越多"
    )

    st.markdown("---")
    st.caption("© 2025 智巡路网 — 道路病害智能检测系统")

# ═══════════════════════════════════════════════════
#  1. 图片上传
# ═══════════════════════════════════════════════════

st.markdown("### 📤 上传图片")
uploaded_file = st.file_uploader(
    "选择一张道路图片",
    type=["jpg", "jpeg", "png", "bmp"],
    help="支持 JPG/JPEG/PNG/BMP 格式的单张图片",
)

image = None
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        st.success(f"✅ 图片加载成功 — {w}×{h} px")
        st.image(image_rgb, caption="原始图片", use_container_width=True)
    else:
        st.error("❌ 图片解码失败，请检查文件是否损坏")

# ═══════════════════════════════════════════════════
#  2. CLIP 模型加载
# ═══════════════════════════════════════════════════

st.markdown("### 🧠 CLIP模型")
col_load, col_status = st.columns([1, 3])

with col_load:
    load_btn = st.button("加载CLIP模型", type="primary", use_container_width=True)

clip = None
with col_status:
    if load_btn:
        with st.spinner("正在加载CLIP模型（首次约30秒）..."):
            try:
                clip = load_clip_classifier()
                st.success("✅ CLIP模型加载成功")
            except Exception as e:
                st.error(f"❌ CLIP模型加载失败: {e}")
    else:
        # 尝试静默加载（若已缓存则几乎无延迟）
        try:
            clip = load_clip_classifier()
            st.success("✅ CLIP模型已就绪")
        except Exception:
            st.info("点击上方按钮加载CLIP模型")

# ═══════════════════════════════════════════════════
#  3. 提示词输入区
# ═══════════════════════════════════════════════════

st.markdown("### ✍️ 检测描述")
st.markdown(
    '<div style="background-color:#111827; border:1px solid #1e3a5f; '
    'border-radius:12px; padding:16px; margin-bottom:8px;">',
    unsafe_allow_html=True,
)

# 预设模板选择
preset_name = st.selectbox(
    "预设模板",
    options=list(PRESET_TEMPLATES.keys()),
    index=0,
    help='选择预设模板可自动填入常用描述，也可选"自定义"手动输入',
)

# 当选择预设时，自动填充文本框（通过 session_state 驱动）
if preset_name != "自定义" and PRESET_TEMPLATES[preset_name]:
    default_text = PRESET_TEMPLATES[preset_name]
    st.session_state["_ovd_prompt_input"] = default_text

prompt_text = st.text_input(
    "自定义描述（多个描述用逗号分隔）",
    placeholder="白色标线磨损, 路面积水, 井盖周围破损, 路面沉陷",
    key="_ovd_prompt_input",
    help="支持英文逗号(,)、中文逗号(，)、顿号(、)分隔多个描述",
)

st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
#  4. 解析提示词
# ═══════════════════════════════════════════════════

def parse_prompts(raw_text: str) -> list:
    """将用户输入的文本按多种分隔符拆分为列表"""
    import re
    # 用正则同时匹配英文逗号、中文逗号、顿号
    parts = re.split(r'[,，、]', raw_text)
    # 去空、去重（保序）
    seen = set()
    prompts = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            prompts.append(p)
    return prompts

prompts = parse_prompts(prompt_text)

if prompts:
    st.caption(f"将检测 {len(prompts)} 个描述: " + " | ".join(prompts))

# ═══════════════════════════════════════════════════
#  5. 执行检测
# ═══════════════════════════════════════════════════

st.markdown("---")
detect_btn = st.button(
    "🚀 开始检测", type="primary", use_container_width=True,
    disabled=(image is None or not prompts),
)

if detect_btn and image is not None and prompts:
    if clip is None:
        st.warning("⚠️ 请先加载CLIP模型")
        st.stop()

    # 执行开放词汇检测
    with st.spinner(f"正在对 {len(prompts)} 个描述进行多尺度扫描（约10-60秒）..."):
        results = clip.open_vocab_detect(
            image, prompts, threshold=threshold,
        )

    # 统计信息
    st.markdown("### 📊 检测结果")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("检出区域数", len(results))
    with m2:
        matched_labels = set(d["label"] for d in results)
        st.metric("匹配描述数", len(matched_labels))
    with m3:
        if results:
            avg_conf = sum(d["confidence"] for d in results) / len(results)
            st.metric("平均置信度", f"{avg_conf:.1%}")
        else:
            st.metric("平均置信度", "N/A")

    # ── 6. 结果展示 ──

    if results:
        # 绘制标注图
        annotated = clip.draw_detections(image, results)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 左右对比：原图 vs 标注图
        c1, c2 = st.columns(2)
        with c1:
            st.image(image_rgb, caption="原始图片", use_container_width=True)
        with c2:
            st.image(annotated_rgb, caption="检测结果标注", use_container_width=True)

        # 结果表格
        st.markdown("#### 📋 检测详情")
        table_rows = []
        for i, det in enumerate(results, 1):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            table_rows.append({
                "序号": i,
                "匹配描述": det["label"],
                "置信度": f"{det['confidence']:.2%}",
                "位置": f"({x1}, {y1}, {x2}, {y2})",
                "窗口大小": f"{det['patch_size']}px",
            })

        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        # 保存到会话历史
        img_b64 = image_to_b64(annotated)

        # 构造 detections 列表（与 save_record 兼容的格式）
        record_detections = []
        for det in results:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            record_detections.append({
                "类别": det["label"],
                "置信度": det["confidence"],
                "位置": f"({x1}, {y1}, {x2}, {y2})",
                "面积占比": (x2 - x1) * (y2 - y1) / (image.shape[0] * image.shape[1]),
                "严重等级": "待评估",
            })

        rec_id = save_record("开放词汇检测", record_detections, img_b64)
        st.caption(f"已保存至会话历史: {rec_id}")

        # 下载按钮
        dl1, dl2, _ = st.columns([1, 1, 2])
        with dl1:
            _, buf = cv2.imencode(".jpg", annotated)
            st.download_button(
                "📥 下载检测结果图",
                data=buf.tobytes(),
                file_name=f"open_vocab_{uploaded_file.name}" if uploaded_file else "open_vocab_result.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
        with dl2:
            _, buf_orig = cv2.imencode(".jpg", image)
            st.download_button(
                "📥 下载原始图片",
                data=buf_orig.tobytes(),
                file_name=uploaded_file.name if uploaded_file else "original.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )
    else:
        st.info("未找到匹配区域，可尝试降低阈值或更换描述")

# ═══════════════════════════════════════════════════
#  底部说明
# ═══════════════════════════════════════════════════

st.markdown("---")
with st.expander("📖 开放词汇检测说明"):
    st.markdown("""
    **CLIP开放词汇检测** 不依赖预定义的类别标签，而是根据用户输入的任意文本描述，
    在图片中搜索并定位语义匹配的区域。

    **工作原理：**
    1. **多尺度滑动窗口：** 使用多种尺寸的窗口（128/192/256/320 px）在图片上滑动，提取局部区域
    2. **提示增强：** 对用户输入的每个描述自动生成多个英文描述变体，增强语义覆盖
    3. **背景对比评分：** 引入背景负样本（正常路面），通过正样本 vs 背景的对比评分抑制误检
    4. **NMS去重：** 对高度重叠的检测框进行非极大值抑制，保留最优结果

    **使用建议：**
    - 描述越具体，定位越精准（如"纵向裂缝"优于"裂缝"）
    - 阈值降低可检出更多区域，但可能增加误检
    - 支持中文输入，系统自动翻译为英文后进行匹配
    - 处理时间取决于图片尺寸，一般10-60秒
    """)
