#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于无人车平台的道路病害目标检测系统
Road Disease Detection System Dashboard
PyQt5 Dark Sci-Fi Theme
"""

import sys
import math
import random
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QGridLayout, QListWidget, QListWidgetItem,
    QPushButton, QSizePolicy, QSpacerItem, QProgressBar
)
from PyQt5.QtCore import (
    Qt, QTimer, QRectF, QPointF, QPoint, pyqtSignal, QSize
)
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QFontDatabase, QPen, QBrush,
    QLinearGradient, QRadialGradient, QConicalGradient, QPainterPath
)

# Try to import QtChart
try:
    from PyQt5.QtChart import (
        QChart, QChartView, QPieSeries, QPieSlice,
        QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis
    )
    HAS_CHART = True
except ImportError:
    HAS_CHART = False


# ─────────────────── Color Constants ───────────────────
BG_COLOR = "#050a18"
CARD_BG = "rgba(8, 22, 48, 0.85)"
ACCENT = "#00d4ff"
GREEN = "#00e676"
YELLOW = "#ffc107"
ORANGE = "#ff9100"
RED = "#ff1744"
TEXT_PRIMARY = "#e0e8f4"
TEXT_DIM = "#6b7fa3"


# ─────────────────── Global QSS ───────────────────
GLOBAL_QSS = f"""
QMainWindow {{
    background-color: {BG_COLOR};
}}
QWidget {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC", sans-serif;
}}
QFrame#card {{
    background-color: {CARD_BG};
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 8px;
}}
QFrame#topBar {{
    background-color: rgba(8, 22, 48, 0.92);
    border-bottom: 1px solid rgba(0, 212, 255, 0.2);
}}
QFrame#bottomBar {{
    background-color: rgba(8, 22, 48, 0.92);
    border-top: 1px solid rgba(0, 212, 255, 0.2);
}}
QLabel#title {{
    color: {ACCENT};
    font-size: 22px;
    font-weight: bold;
    letter-spacing: 2px;
}}
QLabel#subtitle {{
    color: {TEXT_DIM};
    font-size: 12px;
}}
QLabel#sectionTitle {{
    color: {ACCENT};
    font-size: 13px;
    font-weight: bold;
    padding: 6px 0px 2px 0px;
}}
QLabel#metricValue {{
    color: {TEXT_PRIMARY};
    font-size: 20px;
    font-weight: bold;
}}
QLabel#metricLabel {{
    color: {TEXT_DIM};
    font-size: 10px;
}}
QLabel#dimText {{
    color: {TEXT_DIM};
    font-size: 11px;
}}
QLabel#valueText {{
    color: {TEXT_PRIMARY};
    font-size: 13px;
    font-weight: bold;
}}
QLabel#recDot {{
    color: {RED};
    font-size: 13px;
    font-weight: bold;
}}
QListWidget {{
    background-color: rgba(5, 15, 35, 0.9);
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 4px;
    color: {TEXT_PRIMARY};
    font-size: 11px;
    outline: none;
}}
QListWidget::item {{
    padding: 3px 6px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.05);
}}
QListWidget::item:selected {{
    background-color: rgba(0, 212, 255, 0.1);
}}
QPushButton {{
    background-color: rgba(0, 212, 255, 0.15);
    border: 1px solid {ACCENT};
    border-radius: 4px;
    color: {ACCENT};
    padding: 6px 16px;
    font-size: 12px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: rgba(0, 212, 255, 0.3);
}}
QPushButton:pressed {{
    background-color: rgba(0, 212, 255, 0.45);
}}
QProgressBar {{
    background-color: rgba(10, 25, 55, 0.8);
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 4px;
    text-align: center;
    color: {TEXT_PRIMARY};
    font-size: 10px;
    height: 16px;
}}
QProgressBar::chunk {{
    border-radius: 3px;
}}
"""


def make_card(parent=None):
    """Create a styled card QFrame."""
    card = QFrame(parent)
    card.setObjectName("card")
    return card


def make_section_title(text, parent=None):
    """Create a section title label."""
    lbl = QLabel(text, parent)
    lbl.setObjectName("sectionTitle")
    return lbl


# ═══════════════════════════════════════════════════════
#  TOP BAR
# ═══════════════════════════════════════════════════════
class TopBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topBar")
        self.setFixedHeight(80)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 8, 20, 8)

        # Left: title + subtitle
        left = QVBoxLayout()
        left.setSpacing(2)
        title = QLabel("基于无人车平台的道路病害目标检测系统")
        title.setObjectName("title")
        subtitle = QLabel("Road Disease Detection System — YOLOv11")
        subtitle.setObjectName("subtitle")
        left.addWidget(title)
        left.addWidget(subtitle)
        left.addStretch()
        layout.addLayout(left, stretch=1)

        # Right: 4 metric cards
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(12)

        self.metrics = {}
        data = [
            ("系统状态", "在线", GREEN),
            ("检测总次数", "1,247", ACCENT),
            ("今日检出病害", "38", YELLOW),
            ("模型mAP", "96.3%", ACCENT),
        ]
        for label, value, color in data:
            card = make_card()
            card.setFixedSize(150, 56)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(10, 4, 10, 4)
            cl.setSpacing(0)
            vl = QLabel(value)
            vl.setObjectName("metricValue")
            vl.setStyleSheet(f"color: {color}; font-size: 19px; font-weight: bold;")
            vl.setAlignment(Qt.AlignCenter)
            ll = QLabel(label)
            ll.setObjectName("metricLabel")
            ll.setAlignment(Qt.AlignCenter)
            cl.addWidget(vl)
            cl.addWidget(ll)
            metrics_layout.addWidget(card)
            self.metrics[label] = vl

        layout.addLayout(metrics_layout)

    def update_detection_count(self, count):
        self.metrics["今日检出病害"].setText(str(count))

    def update_total_count(self, count):
        self.metrics["检测总次数"].setText(f"{count:,}")


# ═══════════════════════════════════════════════════════
#  VEHICLE MAP WIDGET (custom painted)
# ═══════════════════════════════════════════════════════
class VehicleMapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.vehicle_angle = 0.0
        self.pulse_phase = 0.0

    def advance_vehicle(self):
        self.vehicle_angle += 0.04
        self.pulse_phase += 0.15
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor(5, 15, 35))

        # Grid
        pen = QPen(QColor(0, 212, 255, 18), 1)
        p.setPen(pen)
        step = 30
        for x in range(0, w, step):
            p.drawLine(x, 0, x, h)
        for y in range(0, h, step):
            p.drawLine(0, y, w, y)

        # Roads
        road_pen = QPen(QColor(40, 60, 90), 8)
        p.setPen(road_pen)
        # Horizontal road
        p.drawLine(0, h // 2, w, h // 2)
        # Vertical road
        p.drawLine(w // 3, 0, w // 3, h)
        p.drawLine(2 * w // 3, h // 4, 2 * w // 3, h)

        # Road center dashes
        dash_pen = QPen(QColor(255, 255, 255, 60), 1, Qt.DashLine)
        p.setPen(dash_pen)
        p.drawLine(0, h // 2, w, h // 2)
        p.drawLine(w // 3, 0, w // 3, h)

        # Patrol path (dashed cyan)
        path_pen = QPen(QColor(0, 212, 255, 80), 2, Qt.DashDotLine)
        p.setPen(path_pen)
        path = QPainterPath()
        path.moveTo(20, h - 30)
        path.lineTo(w // 3, h // 2)
        path.lineTo(w // 3, 30)
        path.lineTo(2 * w // 3, 30)
        path.lineTo(2 * w // 3, h // 2)
        path.lineTo(w - 20, h // 2)
        p.drawPath(path)

        # Vehicle dot
        t = self.vehicle_angle
        # Move along path segments
        seg = int(t / (2 * math.pi) * 5) % 5
        frac = (t / (2 * math.pi) * 5) % 1.0
        pts = [
            ((20, h - 30), (w // 3, h // 2)),
            ((w // 3, h // 2), (w // 3, 30)),
            ((w // 3, 30), (2 * w // 3, 30)),
            ((2 * w // 3, 30), (2 * w // 3, h // 2)),
            ((2 * w // 3, h // 2), (w - 20, h // 2)),
        ]
        (x1, y1), (x2, y2) = pts[seg]
        vx = x1 + (x2 - x1) * frac
        vy = y1 + (y2 - y1) * frac

        # Pulse glow
        pulse = 0.5 + 0.5 * math.sin(self.pulse_phase)
        glow_r = int(14 + 8 * pulse)
        glow_alpha = int(50 + 40 * pulse)
        grad = QRadialGradient(vx, vy, glow_r)
        grad.setColorAt(0, QColor(0, 212, 255, glow_alpha + 60))
        grad.setColorAt(0.5, QColor(0, 212, 255, glow_alpha))
        grad.setColorAt(1, QColor(0, 212, 255, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(QPointF(vx, vy), glow_r, glow_r)

        # Core dot
        p.setBrush(QBrush(QColor(0, 212, 255)))
        p.drawEllipse(QPointF(vx, vy), 5, 5)

        # GPS overlay
        p.setPen(QColor(0, 212, 255, 200))
        f = QFont("monospace", 9)
        p.setFont(f)
        lng = 116.3912 + 0.001 * math.sin(t * 0.3)
        lat = 39.9062 + 0.001 * math.cos(t * 0.3)
        p.drawText(8, h - 8, f"GPS {lat:.4f}N  {lng:.4f}E")

        p.end()


# ═══════════════════════════════════════════════════════
#  SENSOR STATUS
# ═══════════════════════════════════════════════════════
class SensorStatusWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        sensors = [
            ("高清摄像头", "4K@60fps", GREEN),
            ("激光雷达", "LiDAR正常", GREEN),
            ("毫米波雷达", "77GHz正常", GREEN),
            ("惯性导航", "INS正常", GREEN),
            ("GPS定位", "RTK固定解", GREEN),
        ]
        for name, status, color in sensors:
            row = QHBoxLayout()
            row.setSpacing(6)
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 12px;")
            dot.setFixedWidth(16)
            dot.setAlignment(Qt.AlignCenter)
            nl = QLabel(name)
            nl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px;")
            sl = QLabel(status)
            sl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
            sl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(dot)
            row.addWidget(nl, stretch=1)
            row.addWidget(sl)
            w = QWidget()
            w.setLayout(row)
            layout.addWidget(w)


# ═══════════════════════════════════════════════════════
#  DRIVING INFO
# ═══════════════════════════════════════════════════════
class DrivingInfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self.speed = 18.5
        self.mileage = 12.8
        self.runtime_seconds = 4523  # start at ~1:15:23

        self.speed_label = QLabel()
        self.mileage_label = QLabel()
        self.runtime_label = QLabel()

        for lbl, label_text, val_label in [
            ("当前速度", "当前速度", self.speed_label),
            ("巡检里程", "巡检里程", self.mileage_label),
            ("运行时长", "运行时长", self.runtime_label),
        ]:
            row = QHBoxLayout()
            ll = QLabel(label_text)
            ll.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
            row.addWidget(ll)
            val_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: bold;")
            val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(val_label)
            w = QWidget()
            w.setLayout(row)
            layout.addWidget(w)

        self._update_display()

    def _update_display(self):
        self.speed_label.setText(f"{self.speed:.1f} km/h")
        self.mileage_label.setText(f"{self.mileage:.1f} km")
        h = self.runtime_seconds // 3600
        m = (self.runtime_seconds % 3600) // 60
        s = self.runtime_seconds % 60
        self.runtime_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def tick_speed(self):
        self.speed = max(5, min(35, self.speed + random.uniform(-1.5, 1.5)))
        self._update_display()

    def tick_runtime(self):
        self.runtime_seconds += 1
        self.mileage += 0.0005
        self._update_display()


# ═══════════════════════════════════════════════════════
#  CAMERA FEED WIDGET (custom painted)
# ═══════════════════════════════════════════════════════
class CameraFeedWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.scan_y = 0.0
        self.rec_visible = True
        self.clock = ""
        self._update_clock()

    def _update_clock(self):
        self.clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def advance_scan(self):
        self.scan_y = (self.scan_y + 2) % self.height()
        self.update()

    def blink_rec(self):
        self.rec_visible = not self.rec_visible
        self.update()

    def tick_clock(self):
        self._update_clock()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Dark road surface background
        bg = QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0, QColor(18, 28, 42))
        bg.setColorAt(0.5, QColor(25, 35, 50))
        bg.setColorAt(1, QColor(15, 22, 35))
        p.fillRect(0, 0, w, h, QBrush(bg))

        # Simulated road surface texture lines
        p.setPen(QPen(QColor(50, 60, 75, 60), 1))
        for y in range(0, h, 12):
            offset = random.randint(-1, 1)
            p.drawLine(0, y + offset, w, y + offset)

        # Road lane markings
        lane_pen = QPen(QColor(200, 200, 200, 40), 3, Qt.DashLine)
        p.setPen(lane_pen)
        p.drawLine(w // 3, 0, w // 3, h)
        p.drawLine(2 * w // 3, 0, 2 * w // 3, h)

        # Edge lines
        edge_pen = QPen(QColor(255, 255, 255, 50), 2)
        p.setPen(edge_pen)
        p.drawLine(30, 0, 30, h)
        p.drawLine(w - 30, 0, w - 30, h)

        # Detection boxes
        detections = [
            ("裂缝 Crack", "0.94", YELLOW, (0.15, 0.25, 0.28, 0.22)),
            ("坑洼 Pothole", "0.89", RED, (0.50, 0.45, 0.22, 0.20)),
            ("鼓包 Bump", "0.87", ORANGE, (0.30, 0.65, 0.25, 0.18)),
        ]
        for name, conf, color, (rx, ry, rw, rh) in detections:
            bx = int(rx * w)
            by = int(ry * h)
            bw = int(rw * w)
            bh = int(rh * h)
            c = QColor(color)

            # Semi-transparent fill
            fill = QColor(c.red(), c.green(), c.blue(), 25)
            p.fillRect(bx, by, bw, bh, fill)

            # Border
            pen = QPen(c, 2)
            p.setPen(pen)
            p.drawRect(bx, by, bw, bh)

            # Corner accents
            corner_len = 10
            p.setPen(QPen(c, 3))
            # Top-left
            p.drawLine(bx, by, bx + corner_len, by)
            p.drawLine(bx, by, bx, by + corner_len)
            # Top-right
            p.drawLine(bx + bw - corner_len, by, bx + bw, by)
            p.drawLine(bx + bw, by, bx + bw, by + corner_len)
            # Bottom-left
            p.drawLine(bx, by + bh - corner_len, bx, by + bh)
            p.drawLine(bx, by + bh, bx + corner_len, by + bh)
            # Bottom-right
            p.drawLine(bx + bw - corner_len, by + bh, bx + bw, by + bh)
            p.drawLine(bx + bw, by + bh - corner_len, bx + bw, by + bh)

            # Label background
            label_text = f"{name} {conf}"
            f = QFont("sans-serif", 10, QFont.Bold)
            p.setFont(f)
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label_text) + 10
            th = fm.height() + 6
            p.fillRect(bx, by - th, tw, th, QColor(c.red(), c.green(), c.blue(), 180))
            p.setPen(QColor(255, 255, 255))
            p.drawText(bx + 5, by - 5, label_text)

        # Scan line
        scan_c = QColor(0, 212, 255, 80)
        p.setPen(QPen(scan_c, 2))
        sy = int(self.scan_y)
        p.drawLine(0, sy, w, sy)
        # Scan glow
        scan_grad = QLinearGradient(0, sy - 15, 0, sy + 15)
        scan_grad.setColorAt(0, QColor(0, 212, 255, 0))
        scan_grad.setColorAt(0.5, QColor(0, 212, 255, 30))
        scan_grad.setColorAt(1, QColor(0, 212, 255, 0))
        p.fillRect(0, sy - 15, w, 30, QBrush(scan_grad))

        # === Overlays ===
        f_small = QFont("monospace", 10)
        f_badge = QFont("sans-serif", 10, QFont.Bold)

        # Top-left: REC + YOLOv11
        if self.rec_visible:
            p.setFont(f_badge)
            p.setPen(QColor(RED))
            p.drawText(12, 24, "● REC")
        # YOLOv11 badge
        p.setFont(f_badge)
        badge_x = 80
        p.fillRect(badge_x, 10, 70, 20, QColor(0, 212, 255, 60))
        p.setPen(QColor(ACCENT))
        p.drawRect(badge_x, 10, 70, 20)
        p.drawText(badge_x + 8, 25, "YOLOv11")

        # Top-right: cam info + clock + resolution
        p.setFont(f_small)
        p.setPen(QColor(200, 210, 225, 180))
        info1 = "CAM-01 | 前视主摄"
        p.drawText(w - 180, 22, info1)
        p.drawText(w - 180, 38, self.clock)
        p.drawText(w - 180, 54, "1920×1080")

        # Bottom-left: detection summary
        p.fillRect(0, h - 30, 320, 30, QColor(0, 0, 0, 140))
        p.setFont(f_badge)
        p.setPen(QColor(0, 212, 255))
        p.drawText(10, h - 10, "目标检测  3 个目标  |  置信度 0.92")

        p.end()


# ═══════════════════════════════════════════════════════
#  DETECTION LOG
# ═══════════════════════════════════════════════════════
class DetectionLogWidget(QWidget):
    new_entry = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.list_widget = QListWidget()
        self.list_widget.setFixedHeight(120)
        layout.addWidget(self.list_widget)

        self.streets = [
            "中关村大街", "长安街", "三环路", "学院路", "西直门北大街",
            "北四环中路", "建国路", "朝阳路", "望京街", "阜成路"
        ]
        self.types = [
            ("裂缝", YELLOW), ("坑洼", RED), ("鼓包", ORANGE), ("松散", ACCENT)
        ]
        # Pre-populate 3 entries
        for _ in range(3):
            self._add_entry()

    def _add_entry(self):
        now = datetime.now().strftime("%H:%M:%S")
        dtype, color = random.choice(self.types)
        conf = random.uniform(0.82, 0.98)
        street = random.choice(self.streets)
        lng = 116.39 + random.uniform(-0.01, 0.01)
        lat = 39.90 + random.uniform(-0.01, 0.01)
        text = f"[{now}]  {dtype}  {conf:.2f}  {lat:.4f}N,{lng:.4f}E · {street}"

        item = QListWidgetItem(text)
        item.setForeground(QColor(color))
        self.list_widget.insertItem(0, item)

        # Keep max 5
        while self.list_widget.count() > 5:
            self.list_widget.takeItem(self.list_widget.count() - 1)

    def add_random_entry(self):
        self._add_entry()
        self.new_entry.emit()


# ═══════════════════════════════════════════════════════
#  DISEASE TYPE DISTRIBUTION (Pie / Donut)
# ═══════════════════════════════════════════════════════
class DiseaseDistributionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = [
            ("裂缝", 42, YELLOW),
            ("坑洼", 23, RED),
            ("鼓包", 18, ORANGE),
            ("松散", 17, ACCENT),
        ]
        self.setMinimumHeight(180)

        if HAS_CHART:
            self._build_chart()
        # Otherwise custom paint

    def _build_chart(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        series = QPieSeries()
        for name, pct, color in self.data:
            slice_ = series.append(f"{name} {pct}%", pct)
            c = QColor(color)
            slice_.setColor(c)
            slice_.setLabelColor(QColor(TEXT_PRIMARY))
            slice_.setLabelVisible(True)
            slice_.setBorderWidth(1)
            slice_.setBorderColor(QColor(BG_COLOR))

        # Make it a donut
        series.setHoleSize(0.55)
        series.setPieSize(0.85)

        chart = QChart()
        chart.addSeries(series)
        chart.setBackgroundBrush(QBrush(QColor(0, 0, 0, 0)))
        chart.setBackgroundRoundness(0)
        chart.layout().setContentsMargins(0, 0, 0, 0)
        chart.legend().setVisible(False)

        view = QChartView(chart)
        view.setRenderHint(QPainter.Antialiasing)
        view.setStyleSheet("background: transparent;")
        layout.addWidget(view)

    def paintEvent(self, event):
        if HAS_CHART:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2 - 20
        inner = int(radius * 0.55)

        start = 0
        total = sum(d[1] for d in self.data)
        for name, pct, color in self.data:
            span = int(pct / total * 5760)  # 16ths of degree
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(color)))
            p.drawPie(cx - radius, cy - radius, 2 * radius, 2 * radius, start, span)
            start += span

        # Cut out center for donut
        p.setCompositionMode(QPainter.CompositionMode_Clear)
        p.setBrush(QBrush(QColor(0, 0, 0, 0)))
        p.drawEllipse(cx - inner, cy - inner, 2 * inner, 2 * inner)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        # Legend
        p.setFont(QFont("sans-serif", 10))
        lx = 8
        ly = h - 12 - len(self.data) * 18
        for name, pct, color in self.data:
            p.fillRect(lx, ly, 10, 10, QColor(color))
            p.setPen(QColor(TEXT_PRIMARY))
            p.drawText(lx + 16, ly + 10, f"{name} {pct}%")
            ly += 18

        p.end()


# ═══════════════════════════════════════════════════════
#  WEEKLY DETECTION TREND (Bar chart)
# ═══════════════════════════════════════════════════════
class WeeklyTrendWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dates = ["7/17", "7/18", "7/19", "7/20", "7/21", "7/22", "7/23"]
        self.values = [24, 31, 28, 42, 35, 39, 38]
        self.setMinimumHeight(150)

        if HAS_CHART:
            self._build_chart()

    def _build_chart(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        bar_set = QBarSet("检出")
        bar_set.setColor(QColor(ACCENT))
        bar_set.setLabelColor(QColor(TEXT_PRIMARY))
        for v in self.values:
            bar_set.append(v)

        series = QBarSeries()
        series.append(bar_set)

        chart = QChart()
        chart.addSeries(series)
        chart.setBackgroundBrush(QBrush(QColor(0, 0, 0, 0)))
        chart.layout().setContentsMargins(0, 0, 0, 0)
        chart.legend().setVisible(False)

        axis_x = QBarCategoryAxis()
        axis_x.append(self.dates)
        axis_x.setLabelsColor(QColor(TEXT_DIM))
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setRange(0, 50)
        axis_y.setLabelsColor(QColor(TEXT_DIM))
        axis_y.setGridLineColor(QColor(0, 212, 255, 25))
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        view = QChartView(chart)
        view.setRenderHint(QPainter.Antialiasing)
        view.setStyleSheet("background: transparent;")
        layout.addWidget(view)

    def paintEvent(self, event):
        if HAS_CHART:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        margin_l, margin_b, margin_t = 30, 24, 10
        chart_w = w - margin_l - 10
        chart_h = h - margin_b - margin_t
        max_val = max(self.values) * 1.15
        n = len(self.values)
        bar_w = chart_w // (n * 2)

        # Y axis
        p.setPen(QPen(QColor(TEXT_DIM), 1))
        for i in range(5):
            y = margin_t + chart_h - (i / 4) * chart_h
            val = int(max_val * i / 4)
            p.drawText(2, int(y) + 4, str(val))
            p.setPen(QPen(QColor(0, 212, 255, 20), 1))
            p.drawLine(margin_l, int(y), w - 10, int(y))
            p.setPen(QPen(QColor(TEXT_DIM), 1))

        # Bars
        p.setFont(QFont("sans-serif", 9))
        for i, (date, val) in enumerate(zip(self.dates, self.values)):
            x = margin_l + (i * 2 + 0.5) * (chart_w / (n * 2)) * 2
            x = margin_l + i * (chart_w / n) + (chart_w / n - bar_w) / 2
            bh = (val / max_val) * chart_h
            by = margin_t + chart_h - bh

            grad = QLinearGradient(x, by, x, by + bh)
            grad.setColorAt(0, QColor(0, 212, 255, 200))
            grad.setColorAt(1, QColor(0, 212, 255, 60))
            p.fillRect(QRectF(x, by, bar_w, bh), QBrush(grad))

            # Value on top
            p.setPen(QColor(TEXT_PRIMARY))
            p.drawText(int(x), int(by) - 4, str(val))

            # Date below
            p.setPen(QColor(TEXT_DIM))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(date)
            p.drawText(int(x + bar_w / 2 - tw / 2), h - 6, date)

        p.end()


# ═══════════════════════════════════════════════════════
#  SEVERITY LEVEL
# ═══════════════════════════════════════════════════════
class SeverityWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        levels = [
            ("轻微", GREEN, 186, 317),
            ("中等", YELLOW, 97, 317),
            ("严重", RED, 34, 317),
        ]
        for name, color, count, total in levels:
            row = QHBoxLayout()
            row.setSpacing(6)

            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 12px;")
            dot.setFixedWidth(16)
            dot.setAlignment(Qt.AlignCenter)

            nl = QLabel(name)
            nl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px;")
            nl.setFixedWidth(28)

            bar = QProgressBar()
            bar.setRange(0, total)
            bar.setValue(count)
            bar.setTextVisible(False)
            bar.setFixedHeight(14)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: rgba(10, 25, 55, 0.8);
                    border: 1px solid rgba(0, 212, 255, 0.1);
                    border-radius: 4px;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 3px;
                }}
            """)

            cl = QLabel(str(count))
            cl.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold;")
            cl.setFixedWidth(32)
            cl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            row.addWidget(dot)
            row.addWidget(nl)
            row.addWidget(bar, stretch=1)
            row.addWidget(cl)

            w = QWidget()
            w.setLayout(row)
            layout.addWidget(w)


# ═══════════════════════════════════════════════════════
#  LATEST REPORTS
# ═══════════════════════════════════════════════════════
class LatestReportsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(6)

        reports = [
            ("RPT-20250723-001", "14:32", "中关村大街巡检 · 检出12处病害"),
            ("RPT-20250723-002", "11:15", "三环路东段巡检 · 检出8处病害"),
            ("RPT-20250722-005", "16:48", "长安街全线巡检 · 检出19处病害"),
        ]
        for rid, time, desc in reports:
            card = make_card()
            card.setCursor(Qt.PointingHandCursor)
            card.setStyleSheet(f"""
                QFrame#card {{
                    background-color: rgba(8, 22, 48, 0.7);
                    border: 1px solid rgba(0, 212, 255, 0.1);
                    border-radius: 6px;
                }}
                QFrame#card:hover {{
                    border: 1px solid rgba(0, 212, 255, 0.4);
                    background-color: rgba(8, 22, 48, 0.95);
                }}
            """)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(8, 4, 8, 4)
            cl.setSpacing(2)

            top = QHBoxLayout()
            rl = QLabel(rid)
            rl.setStyleSheet(f"color: {ACCENT}; font-size: 11px; font-weight: bold;")
            tl = QLabel(time)
            tl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
            tl.setAlignment(Qt.AlignRight)
            top.addWidget(rl)
            top.addWidget(tl)
            cl.addLayout(top)

            dl = QLabel(desc)
            dl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
            cl.addWidget(dl)

            layout.addWidget(card)

        # Export button
        btn = QPushButton("导出报告")
        btn.setCursor(Qt.PointingHandCursor)
        layout.addWidget(btn)
        layout.addStretch()


# ═══════════════════════════════════════════════════════
#  BOTTOM BAR
# ═══════════════════════════════════════════════════════
class BottomBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("bottomBar")
        self.setFixedHeight(36)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)

        self.fps_label = QLabel("45")
        self.latency_label = QLabel("22")
        self.gpu_label = QLabel("67")

        items = [
            ("核心算法", "YOLOv11"),
            ("推理帧率", None, self.fps_label, " FPS"),
            ("推理延迟", None, self.latency_label, " ms"),
            ("传感器", "5/5在线"),
            ("GPU利用率", None, self.gpu_label, "%"),
            ("模型精度", "96.3%"),
            ("数据集", "12,480张"),
        ]

        for i, item in enumerate(items):
            if i > 0:
                sep = QLabel("|")
                sep.setStyleSheet(f"color: rgba(0, 212, 255, 0.25); font-size: 11px; margin: 0 4px;")
                layout.addWidget(sep)

            if len(item) == 2:
                label, value = item
                lbl = QLabel(f"{label}: {value}")
                lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
                layout.addWidget(lbl)
            else:
                label, _, val_widget, suffix = item
                row = QHBoxLayout()
                row.setSpacing(0)
                ll = QLabel(f"{label}: ")
                ll.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
                val_widget.setStyleSheet(f"color: {ACCENT}; font-size: 11px; font-weight: bold;")
                sl = QLabel(suffix)
                sl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
                row.addWidget(ll)
                row.addWidget(val_widget)
                row.addWidget(sl)
                w = QWidget()
                w.setLayout(row)
                layout.addWidget(w)

        layout.addStretch()

    def jitter_values(self):
        fps = random.randint(40, 50)
        lat = random.randint(18, 28)
        gpu = random.randint(60, 75)
        self.fps_label.setText(str(fps))
        self.latency_label.setText(str(lat))
        self.gpu_label.setText(str(gpu))


# ═══════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于无人车平台的道路病害目标检测系统")
        self.resize(1920, 1080)
        self.setStyleSheet(GLOBAL_QSS)

        self.detection_count = 38
        self.total_count = 1247

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Top Bar ──
        self.top_bar = TopBar()
        main_layout.addWidget(self.top_bar)

        # ── Content Area (3 panels) ──
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)

        # LEFT PANEL
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # Vehicle Map
        map_card = make_card()
        map_layout = QVBoxLayout(map_card)
        map_layout.setContentsMargins(4, 4, 4, 4)
        map_layout.setSpacing(2)
        map_layout.addWidget(make_section_title("车辆巡检态势"))
        self.vehicle_map = VehicleMapWidget()
        map_layout.addWidget(self.vehicle_map, stretch=1)
        left_panel.addWidget(map_card, stretch=3)

        # Sensor Status
        sensor_card = make_card()
        sensor_layout = QVBoxLayout(sensor_card)
        sensor_layout.setContentsMargins(4, 4, 4, 4)
        sensor_layout.setSpacing(2)
        sensor_layout.addWidget(make_section_title("传感器状态"))
        sensor_widget = SensorStatusWidget()
        sensor_layout.addWidget(sensor_widget)
        left_panel.addWidget(sensor_card, stretch=2)

        # Driving Info
        drive_card = make_card()
        drive_layout = QVBoxLayout(drive_card)
        drive_layout.setContentsMargins(4, 4, 4, 4)
        drive_layout.setSpacing(2)
        drive_layout.addWidget(make_section_title("行驶信息"))
        self.driving_info = DrivingInfoWidget()
        drive_layout.addWidget(self.driving_info)
        left_panel.addWidget(drive_card, stretch=1)

        content_layout.addLayout(left_panel, stretch=25)

        # CENTER PANEL
        center_panel = QVBoxLayout()
        center_panel.setSpacing(10)

        # Camera Feed
        cam_card = make_card()
        cam_layout = QVBoxLayout(cam_card)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_feed = CameraFeedWidget()
        cam_layout.addWidget(self.camera_feed, stretch=1)
        center_panel.addWidget(cam_card, stretch=1)

        # Detection Log
        log_card = make_card()
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(4, 4, 4, 4)
        log_layout.setSpacing(2)
        log_layout.addWidget(make_section_title("检测日志"))
        self.detection_log = DetectionLogWidget()
        self.detection_log.new_entry.connect(self._on_new_log_entry)
        log_layout.addWidget(self.detection_log)
        center_panel.addWidget(log_card)

        content_layout.addLayout(center_panel, stretch=50)

        # RIGHT PANEL
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # Disease Distribution
        dist_card = make_card()
        dist_layout = QVBoxLayout(dist_card)
        dist_layout.setContentsMargins(4, 4, 4, 4)
        dist_layout.setSpacing(2)
        dist_layout.addWidget(make_section_title("病害类型分布"))
        dist_widget = DiseaseDistributionWidget()
        dist_layout.addWidget(dist_widget, stretch=1)
        right_panel.addWidget(dist_card, stretch=2)

        # Weekly Trend
        trend_card = make_card()
        trend_layout = QVBoxLayout(trend_card)
        trend_layout.setContentsMargins(4, 4, 4, 4)
        trend_layout.setSpacing(2)
        trend_layout.addWidget(make_section_title("近7日检出趋势"))
        trend_widget = WeeklyTrendWidget()
        trend_layout.addWidget(trend_widget, stretch=1)
        right_panel.addWidget(trend_card, stretch=2)

        # Severity
        sev_card = make_card()
        sev_layout = QVBoxLayout(sev_card)
        sev_layout.setContentsMargins(4, 4, 4, 4)
        sev_layout.setSpacing(2)
        sev_layout.addWidget(make_section_title("病害严重程度"))
        sev_widget = SeverityWidget()
        sev_layout.addWidget(sev_widget)
        right_panel.addWidget(sev_card, stretch=1)

        # Latest Reports
        report_card = make_card()
        report_layout = QVBoxLayout(report_card)
        report_layout.setContentsMargins(4, 4, 4, 4)
        report_layout.setSpacing(2)
        report_layout.addWidget(make_section_title("最新检测报告"))
        report_widget = LatestReportsWidget()
        report_layout.addWidget(report_widget, stretch=1)
        right_panel.addWidget(report_card, stretch=2)

        content_layout.addLayout(right_panel, stretch=25)

        main_layout.addWidget(content, stretch=1)

        # ── Bottom Bar ──
        self.bottom_bar = BottomBar()
        main_layout.addWidget(self.bottom_bar)

        # ── Timers ──
        # 1s: clock + runtime + REC blink
        self.timer_1s = QTimer(self)
        self.timer_1s.timeout.connect(self._tick_1s)
        self.timer_1s.start(1000)

        # 50ms: scan line + vehicle movement (smooth animation)
        self.timer_anim = QTimer(self)
        self.timer_anim.timeout.connect(self._tick_anim)
        self.timer_anim.start(50)

        # 2s: speed jitter
        self.timer_2s = QTimer(self)
        self.timer_2s.timeout.connect(self.driving_info.tick_speed)
        self.timer_2s.start(2000)

        # 3s: FPS/latency/GPU jitter
        self.timer_3s = QTimer(self)
        self.timer_3s.timeout.connect(self.bottom_bar.jitter_values)
        self.timer_3s.start(3000)

        # 6s: new log entry
        self.timer_6s = QTimer(self)
        self.timer_6s.timeout.connect(self.detection_log.add_random_entry)
        self.timer_6s.start(6000)

    def _tick_1s(self):
        self.driving_info.tick_runtime()
        self.camera_feed.tick_clock()
        self.camera_feed.blink_rec()

    def _tick_anim(self):
        self.camera_feed.advance_scan()
        self.vehicle_map.advance_vehicle()

    def _on_new_log_entry(self):
        self.detection_count += 1
        self.total_count += 1
        self.top_bar.update_detection_count(self.detection_count)
        self.top_bar.update_total_count(self.total_count)


# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    # High DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Try to set a good font for Chinese
    font_families = [
        "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
        "Source Han Sans SC", "WenQuanYi Micro Hei", "SimHei"
    ]
    for ff in font_families:
        if ff in QFontDatabase().families():
            app.setFont(QFont(ff, 10))
            break

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
