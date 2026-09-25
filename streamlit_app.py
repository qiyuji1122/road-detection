# -*- coding: utf-8 -*-
"""智巡路网 · 无人车道路病害智能检测平台。"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from platform_ui import NAV_ITEMS, PAGE_RENDERERS
from streamlit_utils import apply_theme, auto_load_model, init_session_state


st.set_page_config(
    page_title="智巡路网 · 道路病害智能检测平台",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "About": "智巡路网 · YOLOv11 + CLIP 无人车道路病害智能检测平台",
    },
)

apply_theme()
init_session_state()

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "数据总览"


with st.sidebar:
    st.markdown(
        """
        <div style="padding:.35rem 0 1.2rem">
          <div style="display:flex;align-items:center;gap:11px">
            <div style="width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#19c2b1,#1685ae);display:grid;place-items:center;font-weight:850;color:white">ZR</div>
            <div>
              <div style="color:#f2f8fc;font-weight:760;font-size:1.05rem">智巡路网</div>
              <div style="color:#70899d;font-size:.72rem;letter-spacing:.08em">ROAD INSPECTION OS</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    labels = [item[0] for item in NAV_ITEMS]
    current = st.session_state.get("nav_page", "数据总览")
    if current not in labels:
        current = "数据总览"
    chosen = st.radio(
        "主导航",
        labels,
        index=labels.index(current),
        label_visibility="collapsed",
        key="sidebar_nav",
    )
    st.session_state["nav_page"] = chosen

    st.markdown("---")
    model, _ = auto_load_model()
    if model:
        st.markdown('<span class="status-pill"><span class="status-dot"></span>推理服务在线</span>', unsafe_allow_html=True)
        st.caption("YOLOv11 · 2 类病害 · CLIP 按需加载")
    else:
        st.warning("检测权重未加载")
    st.markdown("---")
    st.caption("v2.1 · 车路协同视觉智能\n\n© 2026 智巡路网")


renderer = PAGE_RENDERERS.get(st.session_state["nav_page"], PAGE_RENDERERS["数据总览"])
renderer()
