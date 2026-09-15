# -*- coding: utf-8 -*-
"""YOLO推理引擎 — 摄像头/视频/图片检测 + QThread异步推理"""

import time
import numpy as np
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage

from storage_manager import classify_severity


class DetectionWorker(QThread):
    """摄像头/视频实时检测线程"""

    frame_ready = pyqtSignal(QImage)           # 标注后的帧
    result_ready = pyqtSignal(dict)            # 检测结果
    error_occurred = pyqtSignal(str)           # 错误
    finished_signal = pyqtSignal()             # 完成信号

    def __init__(self, model, source, conf=0.5, iou=0.45, clip_classifier=None, parent=None):
        super().__init__(parent)
        self.model = model          # ultralytics YOLO model instance
        self.source = source        # 摄像头索引(int) 或 视频路径(str)
        self.conf = conf
        self.iou = iou
        self.clip = clip_classifier # CLIP zero-shot classifier
        self._running = True
        self._class_names = []
        self._class_colors = {}

    def stop(self):
        self._running = False

    def run(self):
        import cv2
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.error_occurred.emit(f"无法打开视频源: {self.source}")
                return

            # 获取类别信息
            self._class_names = list(self.model.names.values())
            from theme import CLASS_COLORS_LIST
            for i, name in enumerate(self._class_names):
                self._class_colors[name] = CLASS_COLORS_LIST[i % len(CLASS_COLORS_LIST)]

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.time()
                results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
                inference_time = (time.time() - t0) * 1000  # ms

                annotated = results[0].plot()
                detections = self._parse_results(results[0], frame.shape, frame=frame)
                fps = 1.0 / max((time.time() - t0), 0.001)

                # 转QImage
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

                self.frame_ready.emit(qimg)
                self.result_ready.emit({
                    "detections": detections,
                    "fps": fps,
                    "inference_ms": inference_time,
                    "count": len(detections),
                    "class_names": self._class_names,
                })

            cap.release()
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.finished_signal.emit()

    def _parse_results(self, result, frame_shape, frame=None):
        """解析YOLO结果为结构化检测结果"""
        import cv2
        detections = []
        if result.boxes is None:
            return detections

        h, w = frame_shape[:2]
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = result.names.get(cls_id, f"class_{cls_id}")
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            # 面积占比
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = h * w
            area_ratio = bbox_area / frame_area if frame_area > 0 else 0

            severity = classify_severity(cls_name, conf, area_ratio)

            # CLIP零样本分类
            clip_result = {}
            if self.clip is not None and frame is not None:
                try:
                    x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                    x1_i = max(0, x1_i)
                    y1_i = max(0, y1_i)
                    x2_i = min(w, x2_i)
                    y2_i = min(h, y2_i)
                    crop = frame[y1_i:y2_i, x1_i:x2_i]
                    clip_result = self.clip.classify(crop, cls_name)
                except Exception:
                    clip_result = {}

            det = {
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "area_ratio": round(area_ratio, 4),
                "severity": severity,
            }
            if clip_result:
                det["fine_class"] = clip_result.get("fine_class", "")
                det["fine_confidence"] = clip_result.get("fine_confidence", 0)
                det["clip_severity"] = clip_result.get("clip_severity", "")
                det["severity_confidence"] = clip_result.get("severity_confidence", 0)
                if "image_feature" in clip_result:
                    det["image_feature"] = clip_result["image_feature"]

            detections.append(det)
        return detections


class ImageDetector(QObject):
    """单张图片检测（非线程，同步执行）"""

    def __init__(self, model, clip_classifier=None):
        super().__init__()
        self.model = model
        self.clip = clip_classifier
        self._class_names = list(model.names.values())

    def detect(self, image_path: str, conf=0.5, iou=0.45) -> dict:
        """检测单张图片，返回 {annotated_image(np.ndarray), detections(list)}"""
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return {"annotated": None, "detections": [], "error": "无法读取图片"}

        t0 = time.time()
        results = self.model(img, conf=conf, iou=iou, verbose=False)
        inference_time = (time.time() - t0) * 1000

        annotated = results[0].plot()
        detections = self._parse_results(results[0], img.shape, frame=img)

        return {
            "annotated": annotated,
            "detections": detections,
            "inference_ms": inference_time,
            "count": len(detections),
            "class_names": self._class_names,
        }

    def _parse_results(self, result, frame_shape, frame=None):
        import cv2
        detections = []
        if result.boxes is None:
            return detections
        h, w = frame_shape[:2]
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = result.names.get(cls_id, f"class_{cls_id}")
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = h * w
            area_ratio = bbox_area / frame_area if frame_area > 0 else 0
            severity = classify_severity(cls_name, conf, area_ratio)

            # CLIP零样本分类
            clip_result = {}
            if self.clip is not None and frame is not None:
                try:
                    x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                    x1_i = max(0, x1_i)
                    y1_i = max(0, y1_i)
                    x2_i = min(w, x2_i)
                    y2_i = min(h, y2_i)
                    crop = frame[y1_i:y2_i, x1_i:x2_i]
                    clip_result = self.clip.classify(crop, cls_name)
                except Exception:
                    clip_result = {}

            det = {
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "area_ratio": round(area_ratio, 4),
                "severity": severity,
            }
            if clip_result:
                det["fine_class"] = clip_result.get("fine_class", "")
                det["fine_confidence"] = clip_result.get("fine_confidence", 0)
                det["clip_severity"] = clip_result.get("clip_severity", "")
                det["severity_confidence"] = clip_result.get("severity_confidence", 0)
                if "image_feature" in clip_result:
                    det["image_feature"] = clip_result["image_feature"]

            detections.append(det)
        return detections


def load_model(model_path: str):
    """加载YOLO模型"""
    from ultralytics import YOLO
    model = YOLO(model_path)
    return model


def load_clip_classifier(device=None):
    """加载CLIP零样本分类器"""
    from clip_classifier import CLIPClassifier
    return CLIPClassifier(device=device)


def get_available_cameras() -> list:
    """检测可用的摄像头"""
    import cv2
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available
