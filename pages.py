# -*- coding: utf-8 -*-
"""7个功能页面 — 实时检测、文件检测、开放词汇检测、数据总览、历史记录、报告生成、系统设置"""

import os
import cv2
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QSlider, QComboBox, QFileDialog, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QLineEdit, QDateEdit, QTextEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QScrollArea,
    QMessageBox, QSizePolicy, QSplitter, QAbstractItemView, QApplication
)
from PyQt5.QtCore import Qt, QDate, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

import theme as T
from widgets import SectionHeader, StatCard, PaintedPieChart, PaintedBarChart, SeverityBar, DetectionGallery
from detection_engine import DetectionWorker, ImageDetector, load_model, load_clip_classifier, get_available_cameras
from storage_manager import StorageManager


def _qimage_from_cv2(cv_img):
    """OpenCV BGR图像转QImage"""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


# ═══════════════════════════════════════════════════
#  1. 实时检测页
# ═══════════════════════════════════════════════════
class RealtimeDetectionPage(QWidget):

    def __init__(self, storage: StorageManager, config: dict, parent=None):
        super().__init__(parent)
        self.storage = storage
        self.config = config
        self.worker = None
        self.model = None
        self.clip_classifier = None
        self._current_result = None

        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # ── 左侧：视频画面 ──
        left = QVBoxLayout()
        left.setSpacing(8)

        self._video_label = QLabel("点击右侧【开始检测】启动摄像头")
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setStyleSheet(
            f"background-color: #0a1020; border: 1px solid {T.BORDER}; border-radius: 8px; "
            f"color: {T.TEXT_DIM}; font-size: 16px;"
        )
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self._video_label, 1)

        # 底部状态栏
        status_bar = QHBoxLayout()
        self._fps_label = QLabel("FPS: --")
        self._latency_label = QLabel("延迟: -- ms")
        self._count_label = QLabel("检测目标: 0")
        for lbl in [self._fps_label, self._latency_label, self._count_label]:
            lbl.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 12px; font-family: monospace;")
            status_bar.addWidget(lbl)
        status_bar.addStretch()
        left.addLayout(status_bar)

        main_layout.addLayout(left, 7)

        # ── 右侧：控制面板 ──
        right = QVBoxLayout()
        right.setSpacing(12)

        right.addWidget(SectionHeader("控制面板"))

        # 模型选择
        model_group = QVBoxLayout()
        model_group.addWidget(QLabel("模型文件:"))
        model_row = QHBoxLayout()
        self._model_path = QLineEdit(self.config.get("model_path", ""))
        self._model_path.setPlaceholderText("选择 .pt 模型文件")
        model_row.addWidget(self._model_path, 1)
        btn_browse_model = QPushButton("浏览")
        btn_browse_model.setFixedWidth(60)
        btn_browse_model.clicked.connect(self._browse_model)
        model_row.addWidget(btn_browse_model)
        model_group.addLayout(model_row)

        btn_load_model = QPushButton("加载模型", self)
        btn_load_model.setProperty("primary", True)
        btn_load_model.clicked.connect(self._load_model)
        model_group.addWidget(btn_load_model)

        self._clip_check = QCheckBox("启用CLIP零样本细分（裂缝/坑洼类型+严重程度）")
        self._clip_check.setChecked(False)
        self._clip_check.setToolTip("加载CLIP模型对YOLO检测区域进行细粒度分类，首次加载需下载模型约340MB")
        model_group.addWidget(self._clip_check)
        right.addLayout(model_group)

        # 摄像头
        cam_group = QVBoxLayout()
        cam_group.addWidget(QLabel("摄像头:"))
        self._cam_combo = QComboBox()
        self._cam_combo.addItem("摄像头 0", 0)
        self._cam_combo.addItem("摄像头 1", 1)
        cam_group.addWidget(self._cam_combo)
        right.addLayout(cam_group)

        # 置信度
        conf_group = QVBoxLayout()
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("置信度阈值:"))
        self._conf_val = QLabel("0.50")
        self._conf_val.setStyleSheet(f"color: {T.ACCENT}; font-weight: 700;")
        conf_row.addWidget(self._conf_val)
        conf_group.addLayout(conf_row)
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(10, 95)
        self._conf_slider.setValue(50)
        self._conf_slider.valueChanged.connect(lambda v: self._conf_val.setText(f"{v/100:.2f}"))
        conf_group.addWidget(self._conf_slider)
        right.addLayout(conf_group)

        # IoU
        iou_group = QVBoxLayout()
        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("IoU 阈值:"))
        self._iou_val = QLabel("0.45")
        self._iou_val.setStyleSheet(f"color: {T.ACCENT}; font-weight: 700;")
        iou_row.addWidget(self._iou_val)
        iou_group.addLayout(iou_row)
        self._iou_slider = QSlider(Qt.Horizontal)
        self._iou_slider.setRange(10, 95)
        self._iou_slider.setValue(45)
        self._iou_slider.valueChanged.connect(lambda v: self._iou_val.setText(f"{v/100:.2f}"))
        iou_group.addWidget(self._iou_slider)
        right.addLayout(iou_group)

        # 按钮
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("开始检测")
        self._btn_start.setProperty("primary", True)
        self._btn_start.clicked.connect(self._start_detection)
        self._btn_stop = QPushButton("停止检测")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_detection)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        right.addLayout(btn_row)

        self._btn_save = QPushButton("保存当前帧")
        self._btn_save.clicked.connect(self._save_frame)
        self._btn_save.setEnabled(False)
        right.addWidget(self._btn_save)

        # 实时结果列表
        right.addWidget(SectionHeader("实时检测结果"))
        self._result_list = QListWidget()
        self._result_list.setMaximumHeight(250)
        right.addWidget(self._result_list, 1)

        right.addStretch()

        right_panel = QWidget()
        right_panel.setLayout(right)
        right_panel.setMaximumWidth(320)
        main_layout.addWidget(right_panel, 3)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "YOLO模型 (*.pt);;所有文件 (*)")
        if path:
            self._model_path.setText(path)

    def _load_model(self):
        path = self._model_path.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "错误", "请选择有效的模型文件")
            return
        try:
            self.model = load_model(path)
            msg = f"YOLO模型加载成功\n类别: {list(self.model.names.values())}"

            # 加载CLIP
            if self._clip_check.isChecked():
                try:
                    self.clip_classifier = load_clip_classifier()
                    msg += "\nCLIP零样本分类已启用"
                except Exception as clip_e:
                    msg += f"\nCLIP加载失败: {clip_e}"
                    self.clip_classifier = None

            QMessageBox.information(self, "成功", msg)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{e}")

    def _start_detection(self):
        if self.model is None:
            QMessageBox.warning(self, "提示", "请先加载模型")
            return
        cam_idx = self._cam_combo.currentData()
        self.worker = DetectionWorker(
            model=self.model,
            source=cam_idx,
            conf=self._conf_slider.value() / 100,
            iou=self._iou_slider.value() / 100,
            clip_classifier=self.clip_classifier,
        )
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_save.setEnabled(True)

    def _stop_detection(self):
        if self.worker:
            self.worker.stop()

    def _on_frame(self, qimg: QImage):
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self._video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._video_label.setPixmap(scaled)
        self._video_label.setText("")

    def _on_result(self, result: dict):
        self._current_result = result
        self._fps_label.setText(f"FPS: {result['fps']:.1f}")
        self._latency_label.setText(f"延迟: {result['inference_ms']:.0f} ms")
        self._count_label.setText(f"检测目标: {result['count']}")

        # 更新结果列表（只保留最新5条去重）
        self._result_list.clear()
        seen = set()
        for det in result["detections"]:
            key = det["class"]
            if key not in seen:
                seen.add(key)
                color = T.CLASS_COLORS.get(det["class"], T.ACCENT)

                # 显示CLIP细分类型
                fine = det.get("fine_class", "")
                if fine:
                    text = f"  {det['class']} → {fine}    {det['confidence']:.1%}    [{det['severity']}]"
                else:
                    text = f"  {det['class']}    {det['confidence']:.1%}    [{det['severity']}]"

                item = QListWidgetItem(text)
                item.setForeground(QColor(color))
                self._result_list.addItem(item)

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "检测错误", msg)
        self._stop_detection()

    def _on_finished(self):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._video_label.clear()
        self._video_label.setText("检测已停止")

    def _save_frame(self):
        if self._current_result is None:
            return
        pixmap = self._video_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        # 保存截图
        record = {
            "source": "camera",
            "model": os.path.basename(self._model_path.text()),
            "detections": self._current_result["detections"],
            "summary": self._build_summary(self._current_result["detections"]),
        }
        record_id = self.storage.save_record(record)
        # 保存截图
        save_path = str(self.storage.images_dir / f"{record_id}.jpg")
        pixmap.save(save_path, "JPG")
        record["image_path"] = f"data/images/{record_id}.jpg"
        self.storage.update_record(record)  # 更新已有记录，不创建新记录
        # 保存CLIP特征用于语义检索
        self._save_clip_features(record_id, self._current_result["detections"])
        QMessageBox.information(self, "已保存", f"截图已保存\n记录ID: {record_id}")

    def _build_summary(self, detections):
        summary = {"total": len(detections)}
        for d in detections:
            cls = d.get("class", "unknown")
            summary[cls] = summary.get(cls, 0) + 1
        return summary

    def _save_clip_features(self, record_id, detections):
        """保存CLIP特征向量到索引，用于后续语义检索"""
        features_list = []
        for det in detections:
            feat = det.get("image_feature")
            if feat is not None:
                features_list.append({
                    "bbox": det.get("bbox", []),
                    "class": det.get("class", ""),
                    "fine_class": det.get("fine_class", ""),
                    "feature": feat,
                })
        if features_list:
            self.storage.save_features(record_id, features_list)


# ═══════════════════════════════════════════════════
#  2. 文件检测页
# ═══════════════════════════════════════════════════
class FileDetectionPage(QWidget):

    def __init__(self, storage: StorageManager, config: dict, parent=None):
        super().__init__(parent)
        self.storage = storage
        self.config = config
        self.model = None
        self.detector = None
        self.clip_classifier = None
        self._current_image = None
        self._current_annotated = None
        self._current_result = None

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 顶部：文件选择 + 模型加载
        top = QHBoxLayout()

        # 文件选择
        file_group = QVBoxLayout()
        file_row = QHBoxLayout()
        self._file_path = QLineEdit()
        self._file_path.setPlaceholderText("选择图片或视频文件...")
        file_row.addWidget(self._file_path, 1)
        btn_browse = QPushButton("选择文件")
        btn_browse.clicked.connect(self._browse_file)
        file_row.addWidget(btn_browse)
        btn_detect = QPushButton("开始检测")
        btn_detect.setProperty("primary", True)
        btn_detect.clicked.connect(self._run_detection)
        file_row.addWidget(btn_detect)
        file_group.addLayout(file_row)
        top.addLayout(file_group, 1)

        # 模型
        model_group = QVBoxLayout()
        model_row = QHBoxLayout()
        self._model_path = QLineEdit(self.config.get("model_path", ""))
        model_row.addWidget(self._model_path, 1)
        btn_load = QPushButton("加载模型")
        btn_load.clicked.connect(self._load_model)
        model_row.addWidget(btn_load)
        model_group.addLayout(model_row)
        self._clip_check2 = QCheckBox("CLIP细分")
        self._clip_check2.setToolTip("启用CLIP零样本细分类")
        model_group.addWidget(self._clip_check2)
        top.addLayout(model_group, 1)

        main_layout.addLayout(top)

        # 中部：原图 + 检测结果

        _img_style = (
            f"background-color: #0a1020; border: 1px solid {T.BORDER}; border-radius: 8px; "
            f"color: {T.TEXT_DIM}; font-size: 12px;"
        )
        _min_sz = (400, 300)

        # 第一行：原图 + 检测结果
        row1 = QHBoxLayout()
        g1 = QVBoxLayout()
        g1.addWidget(QLabel("原始图片"))
        self._orig_label = QLabel("请选择图片文件")
        self._orig_label.setAlignment(Qt.AlignCenter)
        self._orig_label.setMinimumSize(*_min_sz)
        self._orig_label.setStyleSheet(_img_style)
        g1.addWidget(self._orig_label, 1)
        row1.addLayout(g1, 1)

        g2 = QVBoxLayout()
        g2.addWidget(QLabel("检测结果"))
        self._det_label = QLabel("检测后图片")
        self._det_label.setAlignment(Qt.AlignCenter)
        self._det_label.setMinimumSize(*_min_sz)
        self._det_label.setStyleSheet(_img_style)
        g2.addWidget(self._det_label, 1)
        row1.addLayout(g2, 1)

        main_layout.addLayout(row1, 1)

        # 底部：结果表格 + 操作按钮
        bottom = QHBoxLayout()

        self._result_table = QTableWidget()
        self._result_table.setColumnCount(7)
        self._result_table.setHorizontalHeaderLabels(["序号", "类别", "CLIP细分", "置信度", "位置 (x1,y1,x2,y2)", "面积占比", "严重等级"])
        self._result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._result_table.verticalHeader().setVisible(False)
        bottom.addWidget(self._result_table, 1)

        btn_col = QVBoxLayout()
        self._btn_save_record = QPushButton("保存到历史记录")
        self._btn_save_record.setProperty("primary", True)
        self._btn_save_record.clicked.connect(self._save_to_history)
        self._btn_save_record.setEnabled(False)
        btn_col.addWidget(self._btn_save_record)

        self._btn_export_img = QPushButton("导出检测图片")
        self._btn_export_img.clicked.connect(self._export_image)
        self._btn_export_img.setEnabled(False)
        btn_col.addWidget(self._btn_export_img)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {T.GREEN}; font-size: 12px;")
        self._status_label.setWordWrap(True)
        btn_col.addWidget(self._status_label)
        btn_col.addStretch()

        bottom.addLayout(btn_col)
        main_layout.addLayout(bottom)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)"
        )
        if path:
            self._file_path.setText(path)
            # 预览原图
            img = cv2.imread(path)
            if img is not None:
                self._current_image = path
                qimg = _qimage_from_cv2(img)
                self._orig_label.setPixmap(
                    QPixmap.fromImage(qimg).scaled(self._orig_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self._orig_label.setText("")

    def _load_model(self):
        path = self._model_path.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "错误", "请选择有效的模型文件")
            return
        try:
            self.model = load_model(path)

            # 加载CLIP
            clip_cls = None
            if self._clip_check2.isChecked():
                try:
                    clip_cls = load_clip_classifier()
                    self.clip_classifier = clip_cls
                except Exception:
                    clip_cls = None

            self.detector = ImageDetector(self.model, clip_classifier=clip_cls)
            msg = f"模型加载成功\n类别: {list(self.model.names.values())}"
            if clip_cls:
                msg += "\nCLIP零样本分类已启用"
            QMessageBox.information(self, "成功", msg)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{e}")

    def _run_detection(self):
        if self.detector is None:
            QMessageBox.warning(self, "提示", "请先加载模型")
            return
        if not self._current_image:
            QMessageBox.warning(self, "提示", "请先选择图片文件")
            return

        result = self.detector.detect(
            self._current_image,
            conf=self.config.get("confidence_threshold", 0.5),
            iou=self.config.get("iou_threshold", 0.45),
        )

        if result.get("error"):
            QMessageBox.warning(self, "错误", result["error"])
            return

        self._current_result = result
        self._current_annotated = result["annotated"]

        # 显示检测后图片
        qimg = _qimage_from_cv2(result["annotated"])
        self._det_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(self._det_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self._det_label.setText("")

        # 填充结果表格
        detections = result["detections"]
        self._result_table.setRowCount(len(detections))
        for i, det in enumerate(detections):
            self._result_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._result_table.setItem(i, 1, QTableWidgetItem(det["class"]))
            # CLIP细分列
            fine = det.get("fine_class", "")
            self._result_table.setItem(i, 2, QTableWidgetItem(fine if fine else "-"))
            self._result_table.setItem(i, 3, QTableWidgetItem(f"{det['confidence']:.1%}"))
            bbox = det["bbox"]
            self._result_table.setItem(i, 4, QTableWidgetItem(f"{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}"))
            self._result_table.setItem(i, 5, QTableWidgetItem(f"{det['area_ratio']:.2%}"))
            sev_names = {"minor": "轻微", "moderate": "中等", "severe": "严重"}
            self._result_table.setItem(i, 6, QTableWidgetItem(sev_names.get(det["severity"], det["severity"])))

        status_parts = [f"检测完成！共 {len(detections)} 个目标，耗时 {result['inference_ms']:.0f}ms"]
        if result.get("grad_cam_success"):
            status_parts.append("，Grad-CAM热力图已生成")
        self._status_label.setText("".join(status_parts))
        self._btn_save_record.setEnabled(True)
        self._btn_export_img.setEnabled(True)

    def _save_to_history(self):
        if not self._current_result:
            return
        record = {
            "source": "file",
            "model": os.path.basename(self._model_path.text()),
            "detections": self._current_result["detections"],
            "summary": self._build_summary(self._current_result["detections"]),
        }
        record_id = self.storage.save_record(record)
        # 保存检测图片
        if self._current_annotated is not None:
            save_path = str(self.storage.images_dir / f"{record_id}.jpg")
            cv2.imwrite(save_path, self._current_annotated)
            record["image_path"] = f"data/images/{record_id}.jpg"
            self.storage.update_record(record)
        # 保存CLIP特征用于语义检索
        self._save_clip_features(record_id, self._current_result["detections"])

        self._status_label.setText(f"已保存到历史记录\n记录ID: {record_id}")
        QMessageBox.information(self, "已保存", f"检测记录已保存\nID: {record_id}")

    def _export_image(self):
        if self._current_annotated is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存检测图片", "detection_result.jpg", "JPEG图片 (*.jpg)")
        if path:
            cv2.imwrite(path, self._current_annotated)
            QMessageBox.information(self, "已导出", f"图片已保存到:\n{path}")

    def _build_summary(self, detections):
        summary = {"total": len(detections)}
        for d in detections:
            cls = d.get("class", "unknown")
            summary[cls] = summary.get(cls, 0) + 1
        return summary

    def _save_clip_features(self, record_id, detections):
        """保存CLIP特征向量到索引，用于后续语义检索"""
        features_list = []
        for det in detections:
            feat = det.get("image_feature")
            if feat is not None:
                features_list.append({
                    "bbox": det.get("bbox", []),
                    "class": det.get("class", ""),
                    "fine_class": det.get("fine_class", ""),
                    "feature": feat,
                })
        if features_list:
            self.storage.save_features(record_id, features_list)


# ═══════════════════════════════════════════════════
#  3. 开放词汇检测页
# ═══════════════════════════════════════════════════
class _OpenVocabWorker(QThread):
    """后台线程执行CLIP开放词汇检测，避免阻塞UI"""
    finished = pyqtSignal(list, float)  # detections, elapsed_ms
    error = pyqtSignal(str)

    def __init__(self, clip_classifier, image, prompts, threshold):
        super().__init__()
        self.clip = clip_classifier
        self.image = image
        self.prompts = prompts
        self.threshold = threshold

    def run(self):
        import time
        try:
            t0 = time.time()
            detections = self.clip.open_vocab_detect(
                self.image, self.prompts,
                patch_sizes=(128, 192, 256, 320),
                stride_ratio=0.45,
                threshold=self.threshold, top_k=15,
            )
            elapsed = (time.time() - t0) * 1000
            self.finished.emit(detections, elapsed)
        except Exception as e:
            self.error.emit(str(e))


class OpenVocabDetectionPage(QWidget):
    """基于CLIP的开放词汇检测 — 用户输入任意文字描述，CLIP在图片中定位匹配区域"""

    def __init__(self, storage: StorageManager, parent=None):
        super().__init__(parent)
        self.storage = storage
        self.clip_classifier = None
        self._current_image = None      # 原始BGR numpy图像
        self._current_detections = None  # 开放词汇检测结果
        self._worker = None
        self._init_ui()

    def _init_ui(self):
        # 外层滚动区域
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        outer_layout.addWidget(scroll)

        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        main_layout.addWidget(SectionHeader("开放词汇检测（CLIP Open-Vocabulary）"))

        # 说明
        hint = QLabel("输入任意文字描述（逗号分隔），CLIP将在图片中定位匹配区域。算法使用多尺度扫描+背景对比评分，能有效区分目标与正常路面。匹配阈值越高结果越精确，越低召回越多。检测可能需要10~60秒，请耐心等待。")
        hint.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 12px; padding: 8px; "
                           f"background: {T.BG_CARD}; border-radius: 6px;")
        hint.setWordWrap(True)
        main_layout.addWidget(hint)

        # 顶部控制栏
        top = QHBoxLayout()

        # 文件选择
        file_row = QHBoxLayout()
        self._file_path = QLineEdit()
        self._file_path.setPlaceholderText("选择一张图片...")
        file_row.addWidget(self._file_path, 1)
        btn_browse = QPushButton("选择文件")
        btn_browse.clicked.connect(self._browse_file)
        file_row.addWidget(btn_browse)
        top.addLayout(file_row, 1)

        # CLIP模型加载
        clip_row = QVBoxLayout()
        self._btn_load_clip = QPushButton("加载CLIP模型")
        self._btn_load_clip.setProperty("primary", True)
        self._btn_load_clip.clicked.connect(self._load_clip)
        clip_row.addWidget(self._btn_load_clip)
        self._clip_status = QLabel("未加载")
        self._clip_status.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 11px;")
        clip_row.addWidget(self._clip_status)
        top.addLayout(clip_row)
        main_layout.addLayout(top)

        # 文本输入
        prompt_frame = QFrame()
        prompt_frame.setStyleSheet(f"background: {T.BG_CARD}; border: 1px solid {T.BORDER}; border-radius: 8px; padding: 12px;")
        prompt_layout = QVBoxLayout(prompt_frame)

        prompt_layout.addWidget(QLabel("检测目标描述（逗号或顿号分隔多个目标）:"))
        self._prompt_input = QLineEdit()
        self._prompt_input.setPlaceholderText(
            "例如: 白色标线磨损, 路面积水, 井盖周围破损, 路面沉陷, 修补痕迹"
        )
        self._prompt_input.setStyleSheet(f"font-size: 14px; padding: 10px; border-color: {T.ACCENT};")
        prompt_layout.addWidget(self._prompt_input)

        # 参数行
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("匹配阈值:"))
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.10, 0.90)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setValue(0.40)
        param_row.addWidget(self._threshold_spin)

        param_row.addSpacing(20)
        param_row.addWidget(QLabel("预设模板:"))
        self._template_combo = QComboBox()
        self._template_combo.addItems([
            "自定义",
            "道路病害扩展（标线磨损/积水/沉陷/修补/井盖破损）",
            "交通设施（交通标志/护栏/路灯/减速带/排水沟）",
            "路面状况（路面油污/碎石散落/路面结冰/车辙印/轮胎痕迹）",
        ])
        self._template_combo.currentIndexChanged.connect(self._on_template_change)
        param_row.addWidget(self._template_combo, 1)
        prompt_layout.addLayout(param_row)

        # 检测按钮（独立一行，大按钮）
        self._btn_detect = QPushButton("开始检测")
        self._btn_detect.setProperty("primary", True)
        self._btn_detect.setFixedHeight(44)
        self._btn_detect.setStyleSheet(
            f"background-color: {T.ACCENT}; color: #001820; font-size: 16px; "
            f"font-weight: 700; border: none; border-radius: 6px;"
        )
        self._btn_detect.clicked.connect(self._run_detection)
        prompt_layout.addWidget(self._btn_detect)

        main_layout.addWidget(prompt_frame)

        # 中部：图片展示
        img_layout = QHBoxLayout()

        # 原图
        orig_group = QVBoxLayout()
        orig_group.addWidget(QLabel("原始图片"))
        self._orig_label = QLabel("请选择图片")
        self._orig_label.setAlignment(Qt.AlignCenter)
        self._orig_label.setMinimumSize(400, 320)
        self._orig_label.setStyleSheet(
            f"background-color: #0a1020; border: 1px solid {T.BORDER}; border-radius: 8px; "
            f"color: {T.TEXT_DIM}; font-size: 14px;"
        )
        orig_group.addWidget(self._orig_label, 1)
        img_layout.addLayout(orig_group, 1)

        # 检测结果
        det_group = QVBoxLayout()
        det_group.addWidget(QLabel("检测结果"))
        self._det_label = QLabel("检测后图片")
        self._det_label.setAlignment(Qt.AlignCenter)
        self._det_label.setMinimumSize(400, 320)
        self._det_label.setStyleSheet(
            f"background-color: #0a1020; border: 1px solid {T.BORDER}; border-radius: 8px; "
            f"color: {T.TEXT_DIM}; font-size: 14px;"
        )
        det_group.addWidget(self._det_label, 1)
        img_layout.addLayout(det_group, 1)

        main_layout.addLayout(img_layout, 1)

        # 底部结果表格
        bottom = QHBoxLayout()
        self._result_table = QTableWidget()
        self._result_table.setColumnCount(5)
        self._result_table.setHorizontalHeaderLabels(["序号", "匹配描述", "置信度", "位置 (x1,y1,x2,y2)", "窗口大小"])
        self._result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._result_table.verticalHeader().setVisible(False)
        self._result_table.setMaximumHeight(250)
        bottom.addWidget(self._result_table, 1)

        btn_col = QVBoxLayout()
        self._btn_save = QPushButton("保存结果")
        self._btn_save.setProperty("primary", True)
        self._btn_save.clicked.connect(self._save_result)
        self._btn_save.setEnabled(False)
        btn_col.addWidget(self._btn_save)

        self._btn_export_img = QPushButton("导出检测图片")
        self._btn_export_img.clicked.connect(self._export_image)
        self._btn_export_img.setEnabled(False)
        btn_col.addWidget(self._btn_export_img)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {T.GREEN}; font-size: 12px;")
        self._status_label.setWordWrap(True)
        btn_col.addWidget(self._status_label)
        btn_col.addStretch()
        bottom.addLayout(btn_col)

        main_layout.addLayout(bottom)
        main_layout.addStretch()

        scroll.setWidget(content_widget)

    # 预设模板内容
    _TEMPLATES = {
        1: "白色标线磨损, 路面积水, 路面沉陷, 修补痕迹, 井盖破损",
        2: "交通标志, 道路护栏, 路灯, 减速带, 排水沟",
        3: "路面油污, 碎石散落, 路面结冰, 车辙印, 轮胎痕迹",
    }

    def _on_template_change(self, idx):
        if idx in self._TEMPLATES:
            self._prompt_input.setText(self._TEMPLATES[idx])

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if path:
            self._file_path.setText(path)
            img = cv2.imread(path)
            if img is not None:
                self._current_image = img
                qimg = _qimage_from_cv2(img)
                self._orig_label.setPixmap(
                    QPixmap.fromImage(qimg).scaled(self._orig_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self._orig_label.setText("")

    def _load_clip(self):
        if self.clip_classifier is not None:
            self._clip_status.setText("已加载")
            self._clip_status.setStyleSheet(f"color: {T.GREEN}; font-size: 11px;")
            return
        try:
            self._clip_status.setText("加载中...")
            self._btn_load_clip.setEnabled(False)
            QApplication.processEvents()
            from clip_classifier import CLIPClassifier
            self.clip_classifier = CLIPClassifier()
            self._clip_status.setText("已加载")
            self._clip_status.setStyleSheet(f"color: {T.GREEN}; font-size: 11px;")
            self._btn_load_clip.setEnabled(True)
        except Exception as e:
            self._clip_status.setText(f"加载失败: {e}")
            self._clip_status.setStyleSheet(f"color: {T.RED}; font-size: 11px;")
            self._btn_load_clip.setEnabled(True)

    def _run_detection(self):
        if self.clip_classifier is None:
            QMessageBox.warning(self, "提示", "请先加载CLIP模型")
            return
        if self._current_image is None:
            QMessageBox.warning(self, "提示", "请先选择图片")
            return

        raw_text = self._prompt_input.text().strip()
        if not raw_text:
            QMessageBox.warning(self, "提示", "请输入检测目标描述")
            return

        # 解析文本（支持逗号、顿号、中文逗号分隔）
        prompts = [p.strip() for p in raw_text.replace("，", ",").replace("、", ",").split(",") if p.strip()]
        if not prompts:
            return

        # 缩放大图以加速检测（最长边不超过800px）
        img = self._current_image
        h, w = img.shape[:2]
        max_side = 800
        self._detect_scale = 1.0
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            self._detect_scale = 1.0 / scale  # 缩放回原图的倍率

        threshold = self._threshold_spin.value()
        self._status_label.setText("检测中，CLIP正在扫描图片... (可能需要10~30秒)")
        self._status_label.setStyleSheet(f"color: {T.YELLOW}; font-size: 12px;")
        self._btn_detect.setEnabled(False)
        self._btn_detect.setText("检测中...")
        QApplication.processEvents()

        # 后台线程执行检测
        self._worker = _OpenVocabWorker(self.clip_classifier, img, prompts, threshold)
        self._worker.finished.connect(self._on_detection_done)
        self._worker.error.connect(self._on_detection_error)
        self._worker.start()

    def _on_detection_done(self, detections, elapsed):
        # 缩放回原图坐标系
        s = getattr(self, '_detect_scale', 1.0)
        if s != 1.0:
            for det in detections:
                det["bbox"] = [v * s for v in det["bbox"]]

        self._current_detections = detections

        # 绘制结果（在原图上画，不是缩放后的图）
        annotated = self.clip_classifier.draw_detections(self._current_image, detections)
        qimg = _qimage_from_cv2(annotated)
        self._det_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(self._det_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self._det_label.setText("")

        # 填充表格
        self._result_table.setRowCount(len(detections))
        for i, det in enumerate(detections):
            self._result_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._result_table.setItem(i, 1, QTableWidgetItem(det["label"]))
            cos = det.get("cosine_sim", 0)
            self._result_table.setItem(i, 2, QTableWidgetItem(f"{det['confidence']:.1%} (cos={cos:.3f})"))
            bbox = det["bbox"]
            self._result_table.setItem(i, 3, QTableWidgetItem(
                f"{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}"
            ))
            self._result_table.setItem(i, 4, QTableWidgetItem(f"{det['patch_size']}px"))

        if len(detections) == 0:
            self._status_label.setText("未找到匹配区域。建议：降低匹配阈值（如0.25），或更换更具体的描述词。")
            self._status_label.setStyleSheet(f"color: {T.YELLOW}; font-size: 12px;")
        else:
            self._status_label.setText(f"检测完成！共匹配 {len(detections)} 个区域，耗时 {elapsed:.0f}ms")
            self._status_label.setStyleSheet(f"color: {T.GREEN}; font-size: 12px;")
        self._btn_save.setEnabled(True)
        self._btn_export_img.setEnabled(True)
        self._btn_detect.setEnabled(True)
        self._btn_detect.setText("开始检测")

    def _on_detection_error(self, msg):
        self._status_label.setText(f"检测失败: {msg}")
        self._status_label.setStyleSheet(f"color: {T.RED}; font-size: 12px;")
        self._btn_detect.setEnabled(True)
        self._btn_detect.setText("开始检测")
        QMessageBox.critical(self, "检测错误", msg)

    def _save_result(self):
        if self._current_detections is None:
            return
        detections = [{
            "class": "open_vocab",
            "fine_class": det["label"],
            "confidence": round(det["confidence"], 4),
            "bbox": det["bbox"],
            "area_ratio": 0,
            "severity": "moderate",
        } for det in self._current_detections]

        record = {
            "source": "open_vocab",
            "model": "CLIP-open-vocab",
            "detections": detections,
            "summary": {"total": len(detections), "open_vocab": len(detections)},
        }
        record_id = self.storage.save_record(record)
        # 保存检测图片
        if self._current_detections is not None:
            annotated = self.clip_classifier.draw_detections(self._current_image, self._current_detections)
            save_path = str(self.storage.images_dir / f"{record_id}.jpg")
            cv2.imwrite(save_path, annotated)
            record["image_path"] = f"data/images/{record_id}.jpg"
            self.storage.update_record(record)

        self._status_label.setText(f"已保存，记录ID: {record_id}")
        QMessageBox.information(self, "已保存", f"检测记录已保存\nID: {record_id}")

    def _export_image(self):
        if self._current_detections is None or self._current_image is None:
            return
        annotated = self.clip_classifier.draw_detections(self._current_image, self._current_detections)
        path, _ = QFileDialog.getSaveFileName(self, "保存图片", "open_vocab_result.jpg", "JPEG图片 (*.jpg)")
        if path:
            cv2.imwrite(path, annotated)
            QMessageBox.information(self, "已导出", f"图片已保存到:\n{path}")


# ═══════════════════════════════════════════════════
#  4. 数据总览页
# ═══════════════════════════════════════════════════
class DashboardPage(QWidget):

    def __init__(self, storage: StorageManager, parent=None):
        super().__init__(parent)
        self.storage = storage
        self._init_ui()

    def _init_ui(self):
        # 外层滚动区域
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        layout.addWidget(SectionHeader("数据总览"))

        # 统计卡片行
        cards_layout = QHBoxLayout()
        self._card_total_rec = StatCard("总检测次数", "0", color=T.ACCENT)
        self._card_total_det = StatCard("病害检出总数", "0", color=T.YELLOW)
        self._card_today = StatCard("今日检出", "0", color=T.GREEN)
        self._card_conf = StatCard("平均置信度", "0%", color=T.ORANGE)

        for card in [self._card_total_rec, self._card_total_det, self._card_today, self._card_conf]:
            cards_layout.addWidget(card)
        layout.addLayout(cards_layout)

        # 图表行
        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(12)

        # 饼图
        pie_frame = QFrame()
        pie_frame.setStyleSheet(f"background-color: {T.BG_CARD}; border: 1px solid {T.BORDER}; border-radius: 8px;")
        pie_layout = QVBoxLayout(pie_frame)
        pie_layout.addWidget(SectionHeader("病害类型分布"))
        self._pie_chart = PaintedPieChart()
        pie_layout.addWidget(self._pie_chart, 1)
        charts_layout.addWidget(pie_frame, 1)

        # 柱状图
        bar_frame = QFrame()
        bar_frame.setStyleSheet(f"background-color: {T.BG_CARD}; border: 1px solid {T.BORDER}; border-radius: 8px;")
        bar_layout = QVBoxLayout(bar_frame)
        bar_layout.addWidget(SectionHeader("近7日检出趋势"))
        self._bar_chart = PaintedBarChart()
        bar_layout.addWidget(self._bar_chart, 1)
        charts_layout.addWidget(bar_frame, 1)

        # 严重程度
        sev_frame = QFrame()
        sev_frame.setStyleSheet(f"background-color: {T.BG_CARD}; border: 1px solid {T.BORDER}; border-radius: 8px;")
        sev_layout = QVBoxLayout(sev_frame)
        sev_layout.addWidget(SectionHeader("严重程度分布"))
        self._sev_minor = SeverityBar("轻微", 0, "minor")
        self._sev_moderate = SeverityBar("中等", 0, "moderate")
        self._sev_severe = SeverityBar("严重", 0, "severe")
        sev_layout.addWidget(self._sev_minor)
        sev_layout.addWidget(self._sev_moderate)
        sev_layout.addWidget(self._sev_severe)
        sev_layout.addStretch()
        charts_layout.addWidget(sev_frame, 1)

        layout.addLayout(charts_layout)

        # ── 历史检测结果图片 ──
        gallery_frame = QFrame()
        gallery_frame.setStyleSheet(f"background-color: {T.BG_CARD}; border: 1px solid {T.BORDER}; border-radius: 8px;")
        gallery_layout = QVBoxLayout(gallery_frame)
        gallery_layout.setContentsMargins(12, 12, 12, 12)
        gallery_layout.addWidget(SectionHeader("历史检测结果"))
        self._gallery = DetectionGallery(self.storage)
        gallery_layout.addWidget(self._gallery)
        layout.addWidget(gallery_frame)

        layout.addStretch()
        scroll.setWidget(content_widget)
        outer_layout.addWidget(scroll)

    def refresh_data(self):
        """从存储中刷新统计数据"""
        stats = self.storage.get_statistics()

        self._card_total_rec.set_value(str(stats["total_records"]))
        self._card_total_det.set_value(str(stats["total_detections"]))
        self._card_today.set_value(str(stats["today_detections"]))
        self._card_conf.set_value(f"{stats['avg_confidence']:.0%}")

        # 饼图
        class_colors = {"crack": T.YELLOW, "pothole": T.RED, "bump": T.ORANGE, "loose": T.ACCENT}
        pie_data = []
        for cls, count in stats["class_counts"].items():
            color = class_colors.get(cls, T.TEXT_DIM)
            pie_data.append((cls, count, color))
        self._pie_chart.set_data(pie_data)

        # 柱状图
        bar_data = [(item["date"], item["count"]) for item in stats["daily_trend"]]
        self._bar_chart.set_data(bar_data, T.ACCENT)

        # 严重程度
        sev = stats["severity_counts"]
        max_sev = max(sev.values()) if sev else 1
        self._sev_minor.set_max(max_sev)
        self._sev_moderate.set_max(max_sev)
        self._sev_severe.set_max(max_sev)
        self._sev_minor.update_value(sev.get("minor", 0))
        self._sev_moderate.update_value(sev.get("moderate", 0))
        self._sev_severe.update_value(sev.get("severe", 0))

        # 刷新历史检测图片
        records = self.storage.get_records()
        self._gallery.load_records(records)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_data()


# ═══════════════════════════════════════════════════
#  5. 历史记录页（含CLIP语义检索）
# ═══════════════════════════════════════════════════
class HistoryPage(QWidget):

    def __init__(self, storage: StorageManager, parent=None):
        super().__init__(parent)
        self.storage = storage
        self._clip_classifier = None  # 用于语义检索的CLIP模型（按需加载）
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(SectionHeader("历史记录"))

        # 传统筛选栏
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("从:"))
        self._date_from = QDateEdit()
        self._date_from.setCalendarPopup(True)
        self._date_from.setDate(QDate.currentDate().addDays(-30))
        filter_layout.addWidget(self._date_from)

        filter_layout.addWidget(QLabel("至:"))
        self._date_to = QDateEdit()
        self._date_to.setCalendarPopup(True)
        self._date_to.setDate(QDate.currentDate())
        filter_layout.addWidget(self._date_to)

        filter_layout.addWidget(QLabel("类别:"))
        self._class_filter = QComboBox()
        self._class_filter.addItems(["全部", "crack", "pothole", "bump", "loose"])
        filter_layout.addWidget(self._class_filter)

        self._keyword = QLineEdit()
        self._keyword.setPlaceholderText("关键词搜索...")
        self._keyword.setMaximumWidth(200)
        filter_layout.addWidget(self._keyword)

        btn_search = QPushButton("查询")
        btn_search.setProperty("primary", True)
        btn_search.clicked.connect(self._do_search)
        filter_layout.addWidget(btn_search)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # ── CLIP语义搜索栏 ──
        semantic_frame = QFrame()
        semantic_frame.setStyleSheet(
            f"background: {T.BG_CARD}; border: 1px solid {T.ACCENT_DIM}; border-radius: 8px; padding: 8px;"
        )
        semantic_layout = QHBoxLayout(semantic_frame)
        semantic_layout.addWidget(QLabel("语义搜索:"))
        self._semantic_input = QLineEdit()
        self._semantic_input.setPlaceholderText(
            "用自然语言描述你要找的检测记录，例如: 严重的网状裂缝, 大型坑洼, 路面标线磨损..."
        )
        self._semantic_input.setStyleSheet(f"border-color: {T.ACCENT};")
        semantic_layout.addWidget(self._semantic_input, 1)
        btn_semantic = QPushButton("搜索")
        btn_semantic.setProperty("primary", True)
        btn_semantic.clicked.connect(self._do_semantic_search)
        semantic_layout.addWidget(btn_semantic)
        self._semantic_status = QLabel("")
        self._semantic_status.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 11px;")
        self._semantic_status.setMaximumWidth(200)
        semantic_layout.addWidget(self._semantic_status)
        layout.addWidget(semantic_frame)

        # 内容区：左列表 + 右详情
        content = QHBoxLayout()

        # 左列表
        self._record_list = QListWidget()
        self._record_list.setMaximumWidth(350)
        self._record_list.currentRowChanged.connect(self._on_select)
        content.addWidget(self._record_list, 1)

        # 右详情
        detail_layout = QVBoxLayout()

        self._detail_image = QLabel("选择一条记录查看详情")
        self._detail_image.setAlignment(Qt.AlignCenter)
        self._detail_image.setMinimumSize(400, 250)
        self._detail_image.setStyleSheet(
            f"background-color: #0a1020; border: 1px solid {T.BORDER}; border-radius: 8px; "
            f"color: {T.TEXT_DIM}; font-size: 14px;"
        )
        detail_layout.addWidget(self._detail_image, 1)

        self._detail_table = QTableWidget()
        self._detail_table.setColumnCount(5)
        self._detail_table.setHorizontalHeaderLabels(["类别", "置信度", "位置", "面积占比", "严重等级"])
        self._detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._detail_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._detail_table.verticalHeader().setVisible(False)
        self._detail_table.setMaximumHeight(200)
        detail_layout.addWidget(self._detail_table)

        # 操作按钮
        btn_row = QHBoxLayout()
        self._detail_info = QLabel("")
        self._detail_info.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 12px;")
        btn_row.addWidget(self._detail_info, 1)

        btn_delete = QPushButton("删除记录")
        btn_delete.setProperty("danger", True)
        btn_delete.clicked.connect(self._delete_record)
        btn_row.addWidget(btn_delete)

        btn_export = QPushButton("导出记录")
        btn_export.clicked.connect(self._export_record)
        btn_row.addWidget(btn_export)

        detail_layout.addLayout(btn_row)
        content.addLayout(detail_layout, 2)

        layout.addLayout(content, 1)

    def _do_search(self):
        filters = {
            "date_from": self._date_from.date().toString("yyyy-MM-dd"),
            "date_to": self._date_to.date().toString("yyyy-MM-dd"),
            "class_name": self._class_filter.currentText(),
            "keyword": self._keyword.text().strip(),
        }
        records = self.storage.get_records(filters)
        self._record_list.clear()
        self._current_records = records
        for r in records:
            ts = r.get("timestamp", "")[:19]
            n = len(r.get("detections", []))
            classes = set(d.get("class", "") for d in r.get("detections", []))
            text = f"{ts}  |  {n}个目标  |  {', '.join(classes)}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, r.get("id"))
            self._record_list.addItem(item)

    def _do_semantic_search(self):
        """CLIP语义检索：用自然语言描述搜索历史记录"""
        query = self._semantic_input.text().strip()
        if not query:
            QMessageBox.warning(self, "提示", "请输入搜索描述")
            return

        # 按需加载CLIP
        if self._clip_classifier is None:
            self._semantic_status.setText("加载CLIP中...")
            self._semantic_status.setStyleSheet(f"color: {T.YELLOW}; font-size: 11px;")
            QApplication.processEvents()
            try:
                from clip_classifier import CLIPClassifier
                self._clip_classifier = CLIPClassifier()
            except Exception as e:
                self._semantic_status.setText(f"加载失败")
                self._semantic_status.setStyleSheet(f"color: {T.RED}; font-size: 11px;")
                QMessageBox.critical(self, "错误", f"CLIP加载失败:\n{e}")
                return

        self._semantic_status.setText("搜索中...")
        QApplication.processEvents()

        # 编码查询文本
        prompts = [p.strip() for p in query.replace("，", ",").replace("、", ",").split(",") if p.strip()]
        text_features = self._clip_classifier.encode_text_prompts(prompts)

        # 语义检索
        results = self.storage.semantic_search(text_features, top_k=20)

        if not results:
            self._semantic_status.setText("无匹配记录")
            self._semantic_status.setStyleSheet(f"color: {T.YELLOW}; font-size: 11px;")
            self._record_list.clear()
            self._current_records = []
            return

        # 显示结果
        self._record_list.clear()
        self._current_records = []

        for sr in results:
            record = self.storage.get_record(sr["record_id"])
            if record is None:
                continue
            self._current_records.append(record)
            ts = record.get("timestamp", "")[:19]
            score_text = f"{sr['score']:.0%}"
            matched = sr.get("matched_fine") or sr.get("matched_class") or ""
            n = len(record.get("detections", []))
            text = f"[{score_text}] {ts}  |  {n}个目标  |  匹配: {matched}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, sr["record_id"])
            self._record_list.addItem(item)

        self._semantic_status.setText(f"找到 {len(self._current_records)} 条记录")
        self._semantic_status.setStyleSheet(f"color: {T.GREEN}; font-size: 11px;")

    def _on_select(self, row):
        if row < 0 or not hasattr(self, '_current_records') or row >= len(self._current_records):
            return
        record = self._current_records[row]
        detections = record.get("detections", [])

        # 显示图片
        img_path = record.get("image_path", "")
        if img_path:
            full_path = os.path.join(str(self.storage.base_dir), img_path)
            if os.path.exists(full_path):
                pixmap = QPixmap(full_path)
                self._detail_image.setPixmap(
                    pixmap.scaled(self._detail_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self._detail_image.setText("")
            else:
                self._detail_image.setText(f"图片不存在: {img_path}")
        else:
            self._detail_image.setText("无检测图片")

        # 填充表格
        self._detail_table.setRowCount(len(detections))
        sev_names = {"minor": "轻微", "moderate": "中等", "severe": "严重"}
        for i, det in enumerate(detections):
            self._detail_table.setItem(i, 0, QTableWidgetItem(det.get("class", "")))
            self._detail_table.setItem(i, 1, QTableWidgetItem(f"{det.get('confidence', 0):.1%}"))
            bbox = det.get("bbox", [])
            self._detail_table.setItem(i, 2, QTableWidgetItem(
                f"{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}" if len(bbox) == 4 else ""
            ))
            self._detail_table.setItem(i, 3, QTableWidgetItem(f"{det.get('area_ratio', 0):.2%}"))
            self._detail_table.setItem(i, 4, QTableWidgetItem(sev_names.get(det.get("severity", ""), "")))

        self._detail_info.setText(
            f"ID: {record.get('id', '')}  |  {record.get('timestamp', '')[:19]}  |  "
            f"来源: {record.get('source', '')}  |  模型: {record.get('model', '')}"
        )

    def _delete_record(self):
        row = self._record_list.currentRow()
        if row < 0:
            return
        record_id = self._record_list.item(row).data(Qt.UserRole)
        reply = QMessageBox.question(self, "确认删除", f"确定要删除记录 {record_id} 吗？")
        if reply == QMessageBox.Yes:
            self.storage.delete_record(record_id)
            self._do_search()

    def _export_record(self):
        row = self._record_list.currentRow()
        if row < 0 or not hasattr(self, '_current_records'):
            return
        record = self._current_records[row]
        path, _ = QFileDialog.getSaveFileName(self, "导出记录", f"{record.get('id', 'record')}.csv", "CSV文件 (*.csv)")
        if path:
            self.storage.export_csv([record], path)
            QMessageBox.information(self, "已导出", f"记录已导出到:\n{path}")

    def showEvent(self, event):
        super().showEvent(event)
        self._do_search()


# ═══════════════════════════════════════════════════
#  6. 报告生成页
# ═══════════════════════════════════════════════════
class ReportPage(QWidget):

    def __init__(self, storage: StorageManager, parent=None):
        super().__init__(parent)
        self.storage = storage
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(SectionHeader("报告生成"))

        # 配置区
        config_frame = QFrame()
        config_frame.setStyleSheet(f"background-color: {T.BG_CARD}; border: 1px solid {T.BORDER}; border-radius: 8px;")
        config_layout = QVBoxLayout(config_frame)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("报告标题:"))
        self._title_input = QLineEdit("道路病害检测报告")
        row1.addWidget(self._title_input, 1)
        config_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("时间范围:"))
        self._date_from = QDateEdit()
        self._date_from.setCalendarPopup(True)
        self._date_from.setDate(QDate.currentDate().addDays(-30))
        row2.addWidget(self._date_from)
        row2.addWidget(QLabel("至"))
        self._date_to = QDateEdit()
        self._date_to.setCalendarPopup(True)
        self._date_to.setDate(QDate.currentDate())
        row2.addWidget(self._date_to)
        row2.addStretch()
        config_layout.addLayout(row2)

        row3 = QHBoxLayout()
        btn_generate = QPushButton("生成报告预览")
        btn_generate.setProperty("primary", True)
        btn_generate.clicked.connect(self._generate)
        row3.addWidget(btn_generate)

        btn_csv = QPushButton("导出 CSV")
        btn_csv.clicked.connect(self._export_csv)
        row3.addWidget(btn_csv)

        btn_txt = QPushButton("导出 TXT 报告")
        btn_txt.clicked.connect(self._export_txt)
        row3.addWidget(btn_txt)
        row3.addStretch()
        config_layout.addLayout(row3)

        layout.addWidget(config_frame)

        # 预览区
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setStyleSheet(
            f"background-color: #0a1020; border: 1px solid {T.BORDER}; border-radius: 8px; "
            f"color: {T.TEXT}; font-size: 13px; padding: 16px;"
        )
        self._preview.setPlaceholderText("点击【生成报告预览】查看报告内容...")
        layout.addWidget(self._preview, 1)

    def _get_filtered_records(self):
        filters = {
            "date_from": self._date_from.date().toString("yyyy-MM-dd"),
            "date_to": self._date_to.date().toString("yyyy-MM-dd"),
        }
        return self.storage.get_records(filters)

    def _generate(self):
        records = self._get_filtered_records()
        if not records:
            self._preview.setText("所选时间范围内无检测记录。")
            return

        stats = self.storage.get_statistics()
        title = self._title_input.text().strip() or "检测报告"
        date_range = f"{self._date_from.date().toString('yyyy-MM-dd')} ~ {self._date_to.date().toString('yyyy-MM-dd')}"

        text = []
        text.append(f"{'=' * 50}")
        text.append(f"  {title}")
        text.append(f"{'=' * 50}")
        text.append(f"")
        text.append(f"报告时间范围：{date_range}")
        text.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        text.append(f"记录数：{len(records)} 条")
        text.append(f"检出病害总数：{stats['total_detections']} 个")
        text.append(f"平均置信度：{stats['avg_confidence']:.1%}")
        text.append(f"")
        text.append(f"─── 病害类型分布 ───")
        for cls, count in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
            text.append(f"  {cls}: {count} 个")
        text.append(f"")
        text.append(f"─── 严重程度分布 ───")
        sev_names = {"minor": "轻微", "moderate": "中等", "severe": "严重"}
        for sev, count in stats["severity_counts"].items():
            text.append(f"  {sev_names.get(sev, sev)}: {count} 个")
        text.append(f"")
        text.append(f"─── 检测明细（最近20条）───")
        for i, r in enumerate(records[:20], 1):
            text.append(f"")
            text.append(f"  [{i}] {r.get('id', '')} | {r.get('timestamp', '')[:19]}")
            text.append(f"      来源: {r.get('source', '')} | 模型: {r.get('model', '')}")
            for det in r.get("detections", []):
                text.append(
                    f"      - {det.get('class', '')} ({det.get('confidence', 0):.1%}) "
                    f"[{sev_names.get(det.get('severity', ''), '')}]"
                )
        text.append(f"")
        text.append(f"{'=' * 50}")
        text.append(f"  报告结束")
        text.append(f"{'=' * 50}")

        self._preview.setText("\n".join(text))

    def _export_csv(self):
        records = self._get_filtered_records()
        if not records:
            QMessageBox.information(self, "提示", "所选时间范围内无记录")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出CSV", "detection_report.csv", "CSV文件 (*.csv)")
        if path:
            self.storage.export_csv(records, path)
            QMessageBox.information(self, "已导出", f"CSV已保存到:\n{path}")

    def _export_txt(self):
        records = self._get_filtered_records()
        if not records:
            QMessageBox.information(self, "提示", "所选时间范围内无记录")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出TXT报告", "detection_report.txt", "文本文件 (*.txt)")
        if path:
            title = self._title_input.text().strip() or "道路病害检测报告"
            self.storage.export_txt_report(records, path, title)
            QMessageBox.information(self, "已导出", f"报告已保存到:\n{path}")


# ═══════════════════════════════════════════════════
#  7. 系统设置页
# ═══════════════════════════════════════════════════
class SettingsPage(QWidget):

    config_saved = pyqtSignal(dict)  # 通知其他页面配置已更新

    def __init__(self, storage: StorageManager, config: dict, parent=None):
        super().__init__(parent)
        self.storage = storage
        self.config = config
        self._init_ui()
        self._load_values()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        layout.addWidget(SectionHeader("系统设置"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(16)

        # 模型设置
        model_group = QGroupBox("模型设置")
        mg = QVBoxLayout(model_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("默认模型路径:"))
        self._model_path = QLineEdit()
        row1.addWidget(self._model_path, 1)
        btn = QPushButton("浏览")
        btn.setFixedWidth(60)
        btn.clicked.connect(self._browse_model)
        row1.addWidget(btn)
        mg.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("默认置信度阈值:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        row2.addWidget(self._conf_spin)
        row2.addStretch()
        mg.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("默认IoU阈值:"))
        self._iou_spin = QDoubleSpinBox()
        self._iou_spin.setRange(0.05, 0.95)
        self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setDecimals(2)
        row3.addWidget(self._iou_spin)
        row3.addStretch()
        mg.addLayout(row3)

        scroll_layout.addWidget(model_group)

        # 摄像头设置
        cam_group = QGroupBox("摄像头设置")
        cg = QVBoxLayout(cam_group)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("默认摄像头:"))
        self._cam_spin = QSpinBox()
        self._cam_spin.setRange(0, 10)
        row4.addWidget(self._cam_spin)
        row4.addStretch()
        cg.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("分辨率:"))
        self._resolution = QComboBox()
        self._resolution.addItems(["640x480", "1280x720", "1920x1080"])
        row5.addWidget(self._resolution)
        row5.addStretch()
        cg.addLayout(row5)

        scroll_layout.addWidget(cam_group)

        # 存储设置
        store_group = QGroupBox("存储设置")
        sg = QVBoxLayout(store_group)
        row6 = QHBoxLayout()
        row6.addWidget(QLabel("记录保存路径:"))
        self._data_path = QLineEdit()
        self._data_path.setReadOnly(True)
        row6.addWidget(self._data_path, 1)
        sg.addLayout(row6)
        scroll_layout.addWidget(store_group)

        # 关于
        about_group = QGroupBox("关于系统")
        ag = QVBoxLayout(about_group)
        info_lines = [
            ("系统名称", "基于YOLOv11与CLIP的道路病害智能检测系统"),
            ("系统版本", "v2.0.0"),
            ("核心算法", "YOLOv11 (ultralytics) + CLIP (open_clip)"),
            ("特色功能", "开放词汇检测 / CLIP语义检索"),
            ("框架", "PyQt5"),
        ]
        for label, value in info_lines:
            row = QHBoxLayout()
            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet(f"color: {T.TEXT_DIM}; min-width: 100px;")
            row.addWidget(lbl)
            val = QLabel(value)
            val.setStyleSheet(f"color: {T.ACCENT}; font-weight: 600;")
            row.addWidget(val)
            row.addStretch()
            ag.addLayout(row)
        scroll_layout.addWidget(about_group)

        # 保存按钮
        btn_save = QPushButton("保存设置")
        btn_save.setProperty("primary", True)
        btn_save.clicked.connect(self._save_config)
        scroll_layout.addWidget(btn_save)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "YOLO模型 (*.pt)")
        if path:
            self._model_path.setText(path)

    def _load_values(self):
        self._model_path.setText(self.config.get("model_path", ""))
        self._conf_spin.setValue(self.config.get("confidence_threshold", 0.5))
        self._iou_spin.setValue(self.config.get("iou_threshold", 0.45))
        self._cam_spin.setValue(self.config.get("camera_index", 0))
        res = self.config.get("resolution", "1280x720")
        idx = self._resolution.findText(res)
        if idx >= 0:
            self._resolution.setCurrentIndex(idx)
        self._data_path.setText(str(self.storage.data_dir))

    def _save_config(self):
        self.config["model_path"] = self._model_path.text().strip()
        self.config["confidence_threshold"] = self._conf_spin.value()
        self.config["iou_threshold"] = self._iou_spin.value()
        self.config["camera_index"] = self._cam_spin.value()
        self.config["resolution"] = self._resolution.currentText()
        self.storage.save_config(self.config)
        self.config_saved.emit(self.config)
        QMessageBox.information(self, "已保存", "设置已保存")
