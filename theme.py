# -*- coding: utf-8 -*-
"""全局主题配置 — 颜色常量 + QSS样式表 + 字体"""

# ───────── 颜色常量 ─────────
BG_ROOT     = "#050a18"
BG_CARD     = "#0c1630"
BG_INPUT    = "#0f1d3d"
BG_HOVER    = "#132244"
BORDER      = "#1a2a4a"
BORDER_HI   = "#2a4a7a"
ACCENT      = "#00d4ff"
ACCENT_DIM  = "#0a3a4a"
GREEN       = "#00e676"
YELLOW      = "#ffc107"
ORANGE      = "#ff9100"
RED         = "#ff1744"
TEXT        = "#e0e8f4"
TEXT_DIM    = "#6b7fa3"
TEXT_BRIGHT = "#ffffff"

# 检测框颜色（按类别）
CLASS_COLORS = {
    "crack":   YELLOW,
    "pothole": RED,
    "bump":    ORANGE,
    "loose":   ACCENT,
}
# 通用 fallback 颜色
CLASS_COLORS_LIST = [YELLOW, RED, ORANGE, ACCENT, GREEN, "#e040fb", "#7c4dff", "#ff6e40"]

# 严重等级颜色
SEVERITY_COLORS = {
    "minor":    GREEN,
    "moderate": YELLOW,
    "severe":   RED,
}


def get_global_qss():
    """返回全局QSS样式表字符串"""
    return f"""
    /* ── 全局 ── */
    QMainWindow, QWidget {{
        background-color: {BG_ROOT};
        color: {TEXT};
        font-family: "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif;
        font-size: 13px;
    }}

    /* ── 侧边栏 ── */
    #sidebar {{
        background-color: #08101e;
        border-right: 1px solid {BORDER};
    }}
    .sidebar-btn {{
        background-color: transparent;
        color: {TEXT_DIM};
        border: none;
        border-left: 3px solid transparent;
        padding: 14px 18px;
        text-align: left;
        font-size: 14px;
        font-weight: 500;
    }}
    .sidebar-btn:hover {{
        background-color: {BG_HOVER};
        color: {TEXT};
    }}
    .sidebar-btn[active="true"] {{
        background-color: {ACCENT_DIM};
        color: {ACCENT};
        border-left: 3px solid {ACCENT};
        font-weight: 700;
    }}

    /* ── 卡片面板 ── */
    .card {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 16px;
    }}
    .card-title {{
        color: {ACCENT};
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1px;
        padding-bottom: 8px;
    }}

    /* ── 按钮 ── */
    QPushButton {{
        background-color: {BG_INPUT};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 8px 20px;
        font-size: 13px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {BG_HOVER};
        border-color: {BORDER_HI};
    }}
    QPushButton:pressed {{
        background-color: {ACCENT_DIM};
    }}
    QPushButton[primary="true"] {{
        background-color: {ACCENT};
        color: #001820;
        border: none;
        font-weight: 700;
    }}
    QPushButton[primary="true"]:hover {{
        background-color: #33ddff;
    }}
    QPushButton[danger="true"] {{
        background-color: rgba(255,23,68,0.15);
        color: {RED};
        border: 1px solid rgba(255,23,68,0.3);
    }}
    QPushButton:disabled {{
        background-color: {BG_CARD};
        color: {TEXT_DIM};
        border-color: {BORDER};
    }}

    /* ── 输入框 ── */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {BG_INPUT};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 13px;
        selection-background-color: {ACCENT};
    }}
    QLineEdit:focus, QComboBox:focus {{
        border-color: {ACCENT};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 30px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {BG_CARD};
        color: {TEXT};
        border: 1px solid {BORDER};
        selection-background-color: {ACCENT_DIM};
    }}

    /* ── 滑块 ── */
    QSlider::groove:horizontal {{
        background: {BG_INPUT};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT};
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }}
    QSlider::sub-page:horizontal {{
        background: {ACCENT};
        border-radius: 3px;
    }}

    /* ── 列表 ── */
    QListWidget {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 4px;
        outline: none;
    }}
    QListWidget::item {{
        padding: 10px 12px;
        border-radius: 4px;
        margin: 2px 0;
    }}
    QListWidget::item:hover {{
        background-color: {BG_HOVER};
    }}
    QListWidget::item:selected {{
        background-color: {ACCENT_DIM};
        border-left: 3px solid {ACCENT};
    }}

    /* ── 表格 ── */
    QTableWidget {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 6px;
        gridline-color: {BORDER};
        selection-background-color: {ACCENT_DIM};
    }}
    QTableWidget::item {{
        padding: 8px;
    }}
    QHeaderView::section {{
        background-color: {BG_INPUT};
        color: {TEXT_DIM};
        border: none;
        border-bottom: 1px solid {BORDER};
        border-right: 1px solid {BORDER};
        padding: 8px 12px;
        font-weight: 600;
        font-size: 12px;
    }}

    /* ── 滚动条 ── */
    QScrollBar:vertical {{
        background: transparent;
        width: 6px;
    }}
    QScrollBar::handle:vertical {{
        background: rgba(0,212,255,0.2);
        border-radius: 3px;
        min-height: 30px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 6px;
    }}
    QScrollBar::handle:horizontal {{
        background: rgba(0,212,255,0.2);
        border-radius: 3px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    /* ── 进度条 ── */
    QProgressBar {{
        background-color: {BG_INPUT};
        border: none;
        border-radius: 4px;
        height: 8px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        border-radius: 4px;
    }}

    /* ── 分组框 ── */
    QGroupBox {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 20px;
        font-weight: 600;
        color: {ACCENT};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 16px;
        padding: 0 8px;
    }}

    /* ── 日期选择 ── */
    QDateEdit {{
        background-color: {BG_INPUT};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 8px 12px;
    }}

    /* ── 标签页 ── */
    QLabel[heading="true"] {{
        font-size: 20px;
        font-weight: 700;
        color: {TEXT_BRIGHT};
    }}
    QLabel[subheading="true"] {{
        font-size: 12px;
        color: {TEXT_DIM};
        letter-spacing: 2px;
    }}
    """
