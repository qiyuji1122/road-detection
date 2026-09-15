#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于YOLOv11与CLIP的道路病害智能检测系统
主窗口 + 侧边栏导航 + 启动入口
"""

import sys
import os

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QLabel, QFrame, QSizePolicy, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QFont

import theme as T
from storage_manager import StorageManager
from pages import (
    RealtimeDetectionPage, FileDetectionPage, OpenVocabDetectionPage,
    DashboardPage, HistoryPage, ReportPage, SettingsPage
)


class SidebarButton(QPushButton):
    """侧边栏导航按钮"""

    def __init__(self, text: str, icon_text: str = "", parent=None):
        super().__init__(parent)
        self.setProperty("class", "sidebar-btn")
        self.setText(f"  {icon_text}  {text}" if icon_text else f"  {text}")
        self.setFixedHeight(48)
        self.setCursor(Qt.PointingHandCursor)
        self._active = False

    def set_active(self, active: bool):
        self._active = active
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class Sidebar(QFrame):
    """侧边栏导航"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Logo区域
        logo_frame = QFrame()
        logo_frame.setFixedHeight(80)
        logo_layout = QVBoxLayout(logo_frame)
        logo_layout.setContentsMargins(16, 16, 16, 8)
        logo_layout.setSpacing(2)

        title = QLabel("道路病害检测")
        title.setStyleSheet(f"color: {T.ACCENT}; font-size: 16px; font-weight: 700;")
        logo_layout.addWidget(title)

        subtitle = QLabel("YOLOv11 + CLIP 智能检测")
        subtitle.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 11px; letter-spacing: 1px;")
        logo_layout.addWidget(subtitle)

        layout.addWidget(logo_frame)

        # 分隔线
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {T.BORDER};")
        layout.addWidget(sep)

        layout.addSpacing(8)

        # 导航按钮
        self._buttons = []
        nav_items = [
            ("数据总览", "◉"),
            ("实时检测", "◎"),
            ("文件检测", "◫"),
            ("开放词汇检测", "◈"),
            ("历史记录", "◷"),
            ("报告生成", "◧"),
            ("系统设置", "⚙"),
        ]

        for text, icon in nav_items:
            btn = SidebarButton(text, icon)
            btn.clicked.connect(lambda checked, b=btn: self._on_click(b))
            layout.addWidget(btn)
            self._buttons.append(btn)

        layout.addStretch()

        # 底部版本信息
        ver_label = QLabel("v2.0.0  |  YOLOv11 + CLIP")
        ver_label.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 10px; padding: 12px 16px;")
        ver_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(ver_label)

        # 默认选中第一个
        if self._buttons:
            self._buttons[0].set_active(True)

    def _on_click(self, clicked_btn):
        for btn in self._buttons:
            btn.set_active(btn is clicked_btn)
        idx = self._buttons.index(clicked_btn)
        if self._callback:
            self._callback(idx)

    def set_callback(self, callback):
        self._callback = callback

    _callback = None


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11与CLIP的道路病害智能检测系统")
        self.setMinimumSize(1280, 800)
        self.resize(1600, 900)

        # 初始化存储和配置
        project_dir = os.path.dirname(os.path.abspath(__file__))
        self.storage = StorageManager(project_dir)
        self.config = self.storage.load_config()

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 侧边栏
        self._sidebar = Sidebar()
        self._sidebar.set_callback(self._switch_page)
        main_layout.addWidget(self._sidebar)

        # 页面堆栈
        self._stack = QStackedWidget()

        # 创建页面
        self._pages = [
            DashboardPage(self.storage),
            RealtimeDetectionPage(self.storage, self.config),
            FileDetectionPage(self.storage, self.config),
            OpenVocabDetectionPage(self.storage),
            HistoryPage(self.storage),
            ReportPage(self.storage),
            SettingsPage(self.storage, self.config),
        ]

        for page in self._pages:
            self._stack.addWidget(page)

        # 连接设置页的配置保存信号
        settings_page = self._pages[6]
        settings_page.config_saved.connect(self._on_config_saved)

        main_layout.addWidget(self._stack, 1)

        # 默认显示数据总览页
        self._stack.setCurrentIndex(0)

    def _switch_page(self, index: int):
        self._stack.setCurrentIndex(index)
        # 切换到数据总览页时自动刷新
        if index == 0:
            self._pages[0].refresh_data()
        # 切换到历史记录页时自动搜索
        elif index == 4:
            self._pages[4]._do_search()

    def _on_config_saved(self, config: dict):
        """配置更新后通知其他页面"""
        self.config = config
        # 更新检测和文件页面的配置
        self._pages[1].config = config
        self._pages[2].config = config


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(T.get_global_qss())

    # HiDPI支持
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
