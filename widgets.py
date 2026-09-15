# -*- coding: utf-8 -*-
"""可复用UI组件 — 统计卡片、饼图、柱状图、标题栏等"""

import math
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy, QDialog
)
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QBrush, QLinearGradient, QPainterPath, QImage, QPixmap
import theme as T


class SectionHeader(QFrame):
    """带左侧强调线的区块标题"""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(10)

        bar = QFrame()
        bar.setFixedSize(3, 16)
        bar.setStyleSheet(f"background-color: {T.ACCENT}; border-radius: 2px;")
        layout.addWidget(bar)

        label = QLabel(text)
        label.setStyleSheet(f"color: {T.ACCENT}; font-size: 14px; font-weight: 700; letter-spacing: 1px;")
        layout.addWidget(label)
        layout.addStretch()


class StatCard(QFrame):
    """统计数值卡片"""

    def __init__(self, title: str, value: str = "0", unit: str = "", color: str = T.ACCENT, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.setStyleSheet(f"""
            StatCard {{
                background-color: {T.BG_CARD};
                border: 1px solid {T.BORDER};
                border-radius: 8px;
                padding: 16px;
            }}
            StatCard:hover {{
                border-color: {T.BORDER_HI};
            }}
        """)
        self.setMinimumHeight(90)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        # 数值
        val_layout = QHBoxLayout()
        val_layout.setSpacing(4)
        self._value_label = QLabel(value)
        self._value_label.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: 700;")
        val_layout.addWidget(self._value_label)
        if unit:
            unit_label = QLabel(unit)
            unit_label.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 13px; padding-bottom: 4px;")
            val_layout.addWidget(unit_label)
        val_layout.addStretch()
        layout.addLayout(val_layout)

        # 标题
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 12px; letter-spacing: 1px;")
        layout.addWidget(self._title_label)

    def set_value(self, value: str):
        self._value_label.setText(value)


class PaintedPieChart(QWidget):
    """QPainter手绘饼图/环形图"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 180)
        self._data = []       # [(label, value, color_hex), ...]
        self._donut = True    # 环形图

    def set_data(self, data: list):
        """data: [(label, value, color_hex), ...]"""
        self._data = data
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        # 左侧饼图区域，右侧图例
        pie_size = min(w * 0.55, h - 20)
        cx, cy = pie_size / 2 + 10, h / 2
        rect = QRectF(cx - pie_size / 2, cy - pie_size / 2, pie_size, pie_size)

        total = sum(v for _, v, _ in self._data)
        if total == 0:
            return

        # 画扇形
        start_angle = 90 * 16  # Qt角度*16
        for label, value, color_hex in self._data:
            span = int(value / total * 360 * 16)
            color = QColor(color_hex)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawPie(rect, start_angle, span)
            start_angle += span

        # 环形挖空
        if self._donut:
            inner = pie_size * 0.5
            inner_rect = QRectF(cx - inner / 2, cy - inner / 2, inner, inner)
            painter.setBrush(QColor(T.BG_CARD))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(inner_rect)

        # 图例（右侧）
        legend_x = pie_size + 30
        legend_y = max(10, (h - len(self._data) * 26) / 2)
        painter.setFont(QFont("PingFang SC", 11))
        for i, (label, value, color_hex) in enumerate(self._data):
            y = legend_y + i * 26
            # 色块
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(color_hex))
            painter.drawRoundedRect(QRectF(legend_x, y + 2, 12, 12), 2, 2)
            # 文字
            painter.setPen(QColor(T.TEXT))
            pct = f"{value / total * 100:.0f}%"
            painter.drawText(int(legend_x + 18), int(y + 13), f"{label}  {pct}")

        painter.end()


class PaintedBarChart(QWidget):
    """QPainter手绘柱状图"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 160)
        self._data = []       # [(label, value), ...]
        self._bar_color = T.ACCENT

    def set_data(self, data: list, bar_color: str = None):
        """data: [(label, value), ...]"""
        self._data = data
        if bar_color:
            self._bar_color = bar_color
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        margin_left, margin_bottom, margin_top = 40, 30, 10
        chart_w = w - margin_left - 20
        chart_h = h - margin_bottom - margin_top

        max_val = max(v for _, v in self._data) if self._data else 1
        if max_val == 0:
            max_val = 1

        n = len(self._data)
        bar_gap = 8
        bar_width = max(12, (chart_w - bar_gap * (n + 1)) / n)

        # 网格线
        painter.setPen(QPen(QColor(T.BORDER), 1, Qt.DotLine))
        for i in range(5):
            y = int(margin_top + chart_h * (1 - i / 4))
            painter.drawLine(int(margin_left), y, int(w - 20), y)
            painter.setPen(QColor(T.TEXT_DIM))
            painter.setFont(QFont("Rajdhani", 9))
            painter.drawText(2, y - 2, int(margin_left - 6), 16, Qt.AlignRight | Qt.AlignVCenter,
                             str(int(max_val * i / 4)))
            painter.setPen(QPen(QColor(T.BORDER), 1, Qt.DotLine))

        # 柱子
        for i, (label, value) in enumerate(self._data):
            x = margin_left + bar_gap + i * (bar_width + bar_gap)
            bar_h = (value / max_val) * chart_h
            y = margin_top + chart_h - bar_h

            # 渐变柱
            gradient = QLinearGradient(x, y, x, margin_top + chart_h)
            gradient.setColorAt(0, QColor(self._bar_color))
            c = QColor(self._bar_color)
            c.setAlpha(40)
            gradient.setColorAt(1, c)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(QRectF(x, y, bar_width, bar_h), 3, 3)

            # 数值标签
            painter.setPen(QColor(T.TEXT))
            painter.setFont(QFont("Rajdhani", 9, QFont.Bold))
            painter.drawText(QRectF(x - 2, y - 16, bar_width + 4, 14),
                             Qt.AlignCenter, str(value))

            # X轴标签
            painter.setPen(QColor(T.TEXT_DIM))
            painter.setFont(QFont("Rajdhani", 9))
            painter.drawText(QRectF(x - 4, margin_top + chart_h + 4, bar_width + 8, 20),
                             Qt.AlignCenter, label)

        painter.end()


class SeverityBar(QWidget):
    """严重等级进度条"""

    def __init__(self, label: str, count: int, severity: str, parent=None):
        super().__init__(parent)
        color = T.SEVERITY_COLORS.get(severity, T.TEXT_DIM)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)

        # 色块
        dot = QFrame()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background-color: {color}; border-radius: 2px;")
        layout.addWidget(dot)

        # 名称
        name = QLabel(label)
        name.setFixedWidth(40)
        name.setStyleSheet(f"color: {T.TEXT}; font-size: 13px;")
        layout.addWidget(name)

        # 进度条
        self._bar = QFrame()
        self._bar.setFixedHeight(8)
        self._bar.setStyleSheet(f"background-color: {T.BG_INPUT}; border-radius: 4px;")
        layout.addWidget(self._bar, 1)

        # 数值
        self._count_label = QLabel(str(count))
        self._count_label.setStyleSheet(f"color: {T.TEXT_BRIGHT}; font-size: 16px; font-weight: 700;")
        self._count_label.setMinimumWidth(40)
        self._count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._count_label)

        self._color = color
        self._max_count = 1

    def set_max(self, max_count: int):
        self._max_count = max(max_count, 1)

    def update_value(self, count: int):
        self._count_label.setText(str(count))
        pct = min(count / self._max_count, 1.0)
        self._bar.setStyleSheet(
            f"background-color: {T.BG_INPUT}; border-radius: 4px;"
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {self._color}, stop:{pct} {self._color},"
            f"stop:{pct + 0.01} {T.BG_INPUT});"
        )


# ═══════════════════════════════════════════════════
#  检测结果图片缩略卡片 + 图片查看弹窗 + 历史图库
# ═══════════════════════════════════════════════════

import os
import cv2

SEV_NAMES = {"minor": "轻微", "moderate": "中等", "severe": "严重"}


class DetectionThumbnail(QFrame):
    """单张检测结果缩略卡片 — 显示检测图片、类型、置信度、时间和严重等级"""

    clicked = pyqtSignal(str, str)  # (image_path, record_info)

    def __init__(self, record, base_dir, parent=None):
        super().__init__(parent)
        self._record = record
        self._base_dir = base_dir
        self.setCursor(Qt.PointingHandCursor)

        self.setStyleSheet(f"""
            DetectionThumbnail {{
                background-color: {T.BG_CARD};
                border: 1px solid {T.BORDER};
                border-radius: 8px;
            }}
            DetectionThumbnail:hover {{
                border-color: {T.ACCENT};
                background-color: {T.BG_HOVER};
            }}
        """)
        self.setFixedWidth(210)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 图片区域 ──
        self._img_frame = QFrame()
        self._img_frame.setFixedSize(210, 130)
        self._img_frame.setStyleSheet(
            f"background: #0a1020; border-radius: 8px 8px 0 0;"
        )

        img_path = record.get("image_path", "")
        full_path = os.path.join(base_dir, img_path) if img_path else ""

        if full_path and os.path.exists(full_path):
            img = cv2.imread(full_path)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(qimg).scaled(210, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                img_label = QLabel(self._img_frame)
                img_label.setPixmap(pixmap)
                img_label.setAlignment(Qt.AlignCenter)
                img_label.resize(210, 130)

        # 严重程度角标
        det_list = record.get("detections", [])
        worst_sev = "minor"
        for d in det_list:
            s = d.get("severity", "minor")
            if s == "severe":
                worst_sev = "severe"
                break
            elif s == "moderate":
                worst_sev = "moderate"
        sev_color = T.SEVERITY_COLORS.get(worst_sev, T.TEXT_DIM)
        badge = QLabel(SEV_NAMES.get(worst_sev, ""), self._img_frame)
        badge.setStyleSheet(
            f"background: {sev_color}; color: #fff; font-size: 10px; font-weight: 700; "
            f"padding: 2px 6px; border-radius: 3px;"
        )
        badge.adjustSize()
        badge.move(210 - badge.width() - 6, 6)

        layout.addWidget(self._img_frame)

        # ── 信息栏 ──
        info = QFrame()
        info.setStyleSheet(f"background: transparent; padding: 6px 8px;")
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(8, 6, 8, 6)
        info_layout.setSpacing(3)

        # 类型 + 置信度
        top_row = QHBoxLayout()
        classes = sorted(set(d.get("class", "") for d in det_list))
        avg_conf = sum(d.get("confidence", 0) for d in det_list) / max(len(det_list), 1)
        class_text = ", ".join(classes) if classes else "—"

        lbl_cls = QLabel(class_text)
        lbl_cls.setStyleSheet(f"color: {T.ACCENT}; font-size: 12px; font-weight: 600;")
        top_row.addWidget(lbl_cls)
        top_row.addStretch()
        lbl_conf = QLabel(f"{avg_conf:.0%}")
        lbl_conf.setStyleSheet(f"color: {T.TEXT}; font-size: 12px; font-weight: 600;")
        top_row.addWidget(lbl_conf)
        info_layout.addLayout(top_row)

        # 时间
        ts = record.get("timestamp", "")[:16]
        lbl_ts = QLabel(ts)
        lbl_ts.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 10px;")
        info_layout.addWidget(lbl_ts)

        # 目标数量
        n_det = len(det_list)
        lbl_n = QLabel(f"{n_det} 个目标检出")
        lbl_n.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 10px;")
        info_layout.addWidget(lbl_n)

        layout.addWidget(info)

    def mousePressEvent(self, event):
        img_path = self._record.get("image_path", "")
        if img_path:
            full_path = os.path.join(self._base_dir, img_path)
            rid = self._record.get("id", "")
            ts = self._record.get("timestamp", "")[:19]
            self.clicked.emit(full_path, f"{rid}  |  {ts}")
        super().mousePressEvent(event)


class ViewImageDialog(QDialog):
    """全屏查看检测图片弹窗"""

    def __init__(self, image_path, title="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setStyleSheet(f"background-color: #050a18;")
        self.resize(900, 650)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # 标题栏
        header = QLabel(title)
        header.setStyleSheet(
            f"color: {T.ACCENT}; font-size: 13px; font-weight: 600; "
            f"padding: 4px 0; letter-spacing: 1px;"
        )
        layout.addWidget(header)

        # 图片
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled = pixmap.scaled(860, 560, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label.setPixmap(scaled)
        else:
            img_label.setText("图片不存在")
            img_label.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 16px;")

        layout.addWidget(img_label, 1)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


class DetectionGallery(QWidget):
    """检测结果历史图片网格 — 自适应列数，点击可放大查看"""

    def __init__(self, storage, parent=None):
        super().__init__(parent)
        self._storage = storage
        self._records = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background: transparent;")
        self._grid_layout = QVBoxLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(10)
        layout.addWidget(self._grid_widget, 1)

    def _clear(self):
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def load_records(self, records):
        self._records = [r for r in records if r.get("image_path")]
        self._build_grid()

    def _build_grid(self):
        self._clear()
        if not self._records:
            placeholder = QLabel("暂无历史检测图片 — 检测后点击「保存到历史记录」即可在此查看")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet(
                f"color: {T.TEXT_DIM}; font-size: 13px; padding: 40px 0;"
            )
            self._grid_layout.addWidget(placeholder)
            return

        base_dir = str(self._storage.base_dir)
        card_width = 210
        available = max(self.width(), card_width * 2)
        cols = max(available // (card_width + 12), 2)

        row_widget = None
        row_layout = None
        count = 0

        for record in self._records[:24]:  # 最多显示24张
            if count % cols == 0:
                row_widget = QWidget()
                row_widget.setStyleSheet("background: transparent;")
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(10)
                self._grid_layout.addWidget(row_widget)

            thumb = DetectionThumbnail(record, base_dir)
            thumb.clicked.connect(self._on_thumb_clicked)
            row_layout.addWidget(thumb)
            count += 1

        # 补齐空位保持对齐
        if row_layout and count % cols != 0:
            row_layout.addStretch()

        self._grid_layout.addStretch()

    def _on_thumb_clicked(self, image_path, info):
        dlg = ViewImageDialog(image_path, info, self)
        dlg.exec_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._records:
            self._build_grid()
