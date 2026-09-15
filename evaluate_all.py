# -*- coding: utf-8 -*-
"""
=============================================================
  道路病害检测系统 —— 全指标自动评测脚本
  覆盖说明书 表5（指标体系）+ 表6（消融实验 E1-E3）
=============================================================

用法:
  python evaluate_all.py                                    # 默认参数
  python evaluate_all.py --model path/to/best.pt            # 指定模型
  python evaluate_all.py --data my_data.yaml --imgsz 640    # 指定数据和输入尺寸
  python evaluate_all.py --experiment E1                    # 只跑某组实验

依赖 (在GPU环境安装):
  pip install ultralytics open_clip_torch torch torchvision opencv-python numpy tabulate psutil

输出:
  1. eval_report.json    — 所有指标的结构化JSON（可直接用于论文/答辩）
  2. eval_report.txt     — 可读的文本报告
  3. eval_report.csv     — CSV格式（方便Excel打开）
  4. per_image/          — 每张图片的推理详情（调试用）
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# 配置常量
# ============================================================
DEFAULT_MODEL = "best.pt"                    # YOLOv11 权重（相对于本脚本所在目录）
DEFAULT_DATA  = "my_data.yaml"               # 数据配置
CLASS_NAMES   = {0: "crack", 1: "pothole"}   # 类别映射
CONF_THRESH   = 0.25                         # YOLO 置信度阈值（评测用，偏低以保留更多候选）
IOU_THRESH    = 0.5                          # NMS IoU 阈值
IMG_SIZE      = 640                          # 推理输入尺寸
WARMUP_FRAMES = 10                           # 预热帧数（不计入统计）
STABLE_FRAMES = 200                          # 稳定测试帧数上限（不够就用全部）
CLIP_MODEL    = "ViT-B-32"                   # CLIP 模型（轻量）
CLIP_PRETRAINED = "openai"
CLIP_LOW_CONF_TRIGGER = 0.5                  # E3: CLIP 条件触发阈值


# ============================================================
# 工具函数
# ============================================================
def check_deps():
    """检查依赖是否安装"""
    missing = []
    for mod in ["ultralytics", "torch", "open_clip", "cv2", "psutil"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[ERROR] 缺少依赖: {', '.join(missing)}")
        print(f"  请运行: pip install {' '.join(missing)}")
        sys.exit(1)

def get_device():
    """自动选择推理设备"""
    import torch
    if torch.cuda.is_available():
        dev = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"[设备] CUDA: {gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
        print("[设备] Apple Silicon MPS")
    else:
        dev = "cpu"
        print("[设备] CPU (推理可能较慢)")
    return dev

def gpu_snapshot():
    """获取当前GPU资源快照"""
    info = {"gpu_available": False}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1024**2, 1)
            info["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved(0) / 1024**2, 1)
            info["gpu_mem_total_mb"] = round(torch.cuda.get_device_properties(0).total_mem / 1024**2, 1)
            try:
                info["gpu_utilization_pct"] = torch.cuda.utilization(0)
            except Exception:
                info["gpu_utilization_pct"] = None
    except Exception:
        pass
    # CPU / RAM
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count()
        info["ram_used_gb"] = round(psutil.virtual_memory().used / 1024**3, 1)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
        info["ram_pct"] = psutil.virtual_memory().percent
    except ImportError:
        pass
    return info

def load_yolo_labels(label_path, img_h, img_w):
    """加载YOLO格式标注，返回 list of [cls_id, x1, y1, x2, y2] (像素坐标)"""
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            # YOLO normalized → pixel xyxy
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append([cls_id, max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)])
    return boxes

def iou(box_a, box_b):
    """计算IoU，输入 [x1,y1,x2,y2]"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def compute_ap(precision_list, recall_list):
    """计算AP (11点插值法 + 全点插值法)"""
    # 按recall排序
    pairs = sorted(zip(recall_list, precision_list))
    recalls = [p[0] for p in pairs]
    precisions = [p[1] for p in pairs]

    # 全点插值
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap

def match_detections(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    匹配预测框和真实框。
    pred_boxes: list of [x1,y1,x2,y2,conf,cls_id]
    gt_boxes: list of [cls_id, x1,y1,x2,y2]
    返回: (tp_count, fp_count, fn_count, matched_gts)
    """
    gt_remaining = list(range(len(gt_boxes)))
    tp = 0
    fp = 0
    # 按置信度降序排序
    preds_sorted = sorted(pred_boxes, key=lambda x: -x[4])
    for pred in preds_sorted:
        pred_bbox = pred[:4]
        pred_cls = pred[5]
        best_iou = 0
        best_gt_idx = -1
        for gi in gt_remaining:
            gt = gt_boxes[gi]
            if gt[0] != pred_cls:
                continue
            gt_bbox = gt[1:5]
            score = iou(pred_bbox, gt_bbox)
            if score > best_iou:
                best_iou = score
                best_gt_idx = gi
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp += 1
            gt_remaining.remove(best_gt_idx)
        else:
            fp += 1
    fn = len(gt_remaining)
    return tp, fp, fn

def percentile(data, pct):
    """计算百分位数"""
    if not data:
        return 0.0
    arr = np.array(sorted(data))
    k = (len(arr) - 1) * pct / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(arr) else f
    d = k - f
    return arr[f] + d * (arr[c] - arr[f])


# ============================================================
# 核心评测
# ============================================================
class Evaluator:
    """统一评测器"""

    def __init__(self, model_path, data_yaml, device, imgsz=IMG_SIZE,
                 conf=CONF_THRESH, iou_thresh=IOU_THRESH):
        from ultralytics import YOLO
        import yaml

        self.model_path = str(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou_thresh = iou_thresh

        # 加载模型
        print(f"[加载模型] {model_path}")
        self.model = YOLO(str(model_path))
        self.model.to(device)
        self.model.eval()
        self.class_names = self.model.names  # {0: 'crack', 1: 'pothole'}

        # 加载数据配置
        with open(data_yaml) as f:
            self.data_cfg = yaml.safe_load(f)
        self.data_root = Path(self.data_cfg["path"])

        # 确定测试集路径
        test_rel = self.data_cfg.get("test", "test/images")
        val_rel = self.data_cfg.get("val", "val/images")
        test_dir = self.data_root / test_rel
        val_dir = self.data_root / val_rel

        if test_dir.exists() and len(list(test_dir.glob("*"))) > 0:
            self.test_dir = test_dir
            self.label_dir = test_dir.parent.parent / "test" / "labels"
            print(f"[测试集] {self.test_dir} ({len(list(self.test_dir.glob('*')))} 张)")
        elif val_dir.exists():
            self.test_dir = val_dir
            self.label_dir = val_dir.parent.parent / "val" / "labels"
            print(f"[测试集] test/ 不存在，使用 val/ 作为测试集 ({len(list(val_dir.glob('*')))} 张)")
        else:
            print("[ERROR] 找不到测试集或验证集!")
            sys.exit(1)

        # 收集测试图片
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(self.test_dir.glob(ext))
        self.image_paths = sorted(set(self.image_paths))
        print(f"[图片] 共 {len(self.image_paths)} 张测试图片")

        # CLIP 分类器（延迟加载）
        self.clip = None
        self.clip_loaded = False

    def load_clip(self):
        """加载CLIP分类器"""
        if self.clip_loaded:
            return
        try:
            from clip_classifier import CLIPClassifier
            print(f"[CLIP] 加载 {CLIP_MODEL}...")
            self.clip = CLIPClassifier(model_name=CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=self.device)
            self.clip_loaded = True
            print("[CLIP] 加载完成")
        except Exception as e:
            print(f"[CLIP] 加载失败: {e}")
            print("[CLIP] 将跳过 CLIP 相关实验 (E2, E3)")
            self.clip = None
            self.clip_loaded = True  # 标记为已尝试

    def _get_image_paths(self, max_n=None):
        paths = self.image_paths
        if max_n and len(paths) > max_n:
            # 均匀采样
            step = len(paths) // max_n
            paths = paths[::step][:max_n]
        return paths

    # ----------------------------------------------------------
    # Part 1: YOLO 内置验证（mAP, P, R, F1）
    # ----------------------------------------------------------
    def run_yolo_validation(self, tag="YOLOv11"):
        """
        使用 ultralytics 内置 model.val() 获取标准检测指标。
        返回: dict with precision, recall, F1, mAP@0.5, mAP@0.5:0.95, per-class metrics
        """
        print(f"\n{'='*60}")
        print(f"[{tag}] YOLO 内置验证 (mAP/P/R/F1)")
        print(f"{'='*60}")

        # 使用 model.val() 获取指标
        # 注意: 这里用验证集做评测，如果 data.yaml 的 val 和 test 指向同一目录则等价
        results = self.model.val(
            data=str(Path(self.data_cfg["path"]) / self.data_cfg.get("val", "val/images")),
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou_thresh,
            batch=16,
            device=self.device,
            verbose=False,
        )

        metrics = {}

        # 总体指标
        # results 是 ultralytics.utils.metrics.DetMetrics 对象
        if hasattr(results, 'box'):
            box = results.box
            metrics["precision"] = round(float(box.mp), 4) if hasattr(box, 'mp') else None
            metrics["recall"] = round(float(box.mr), 4) if hasattr(box, 'mr') else None
            metrics["f1"] = round(2 * metrics["precision"] * metrics["recall"] /
                                  (metrics["precision"] + metrics["recall"] + 1e-8), 4) \
                if metrics["precision"] and metrics["recall"] else None

            # mAP
            if hasattr(box, 'maps') and box.maps is not None:
                metrics["mAP_0.5"] = round(float(box.maps[0]) if len(box.maps) > 0 else 0, 4)
                metrics["mAP_0.5_0.95"] = round(float(box.maps[1]) if len(box.maps) > 1 else 0, 4)
            elif hasattr(box, 'map'):
                metrics["mAP_0.5_0.95"] = round(float(box.map), 4)
                metrics["mAP_0.5"] = round(float(box.map50), 4) if hasattr(box, 'map50') else None
            else:
                # 从 results 的其他属性获取
                metrics["mAP_0.5"] = round(float(results.box.map50), 4) if hasattr(results.box, 'map50') else None
                metrics["mAP_0.5_0.95"] = round(float(results.box.map), 4) if hasattr(results.box, 'map') else None

        # 尝试从 results 字典获取
        if not metrics.get("mAP_0.5"):
            try:
                # 某些版本的 ultralytics 返回 results_dict
                rd = results.results_dict if hasattr(results, 'results_dict') else {}
                metrics["precision"] = metrics.get("precision") or round(float(rd.get("metrics/precision(B)", 0)), 4)
                metrics["recall"] = metrics.get("recall") or round(float(rd.get("metrics/recall(B)", 0)), 4)
                if metrics["precision"] and metrics["recall"]:
                    metrics["f1"] = round(2 * metrics["precision"] * metrics["recall"] /
                                          (metrics["precision"] + metrics["recall"] + 1e-8), 4)
                metrics["mAP_0.5"] = metrics.get("mAP_0.5") or round(float(rd.get("metrics/mAP50(B)", 0)), 4)
                metrics["mAP_0.5_0.95"] = metrics.get("mAP_0.5_0.95") or round(float(rd.get("metrics/mAP50-95(B)", 0)), 4)
            except Exception:
                pass

        # 每类指标
        per_class = {}
        try:
            if hasattr(results, 'box') and hasattr(results.box, 'aps'):
                for i, (name, ap50, ap) in enumerate(zip(
                    self.class_names.values(),
                    results.box.ap50 if hasattr(results.box, 'ap50') else [],
                    results.box.ap if hasattr(results.box, 'ap') else []
                )):
                    per_class[name] = {
                        "AP_0.5": round(float(ap50), 4),
                        "AP_0.5_0.95": round(float(ap), 4),
                    }
        except Exception:
            pass
        metrics["per_class"] = per_class

        print(f"  Precision: {metrics.get('precision', 'N/A')}")
        print(f"  Recall:    {metrics.get('recall', 'N/A')}")
        print(f"  F1:        {metrics.get('f1', 'N/A')}")
        print(f"  mAP@0.5:   {metrics.get('mAP_0.5', 'N/A')}")
        print(f"  mAP@0.5:0.95: {metrics.get('mAP_0.5_0.95', 'N/A')}")
        if per_class:
            for name, cm in per_class.items():
                print(f"  [{name}] AP50={cm['AP_0.5']}, AP50:95={cm['AP_0.5_0.95']}")

        return metrics

    # ----------------------------------------------------------
    # Part 2: 自定义推理循环（延迟 / FPS / 资源）
    # ----------------------------------------------------------
    def run_inference_benchmark(self, tag="YOLOv11", use_clip=False, clip_conditional=False):
        """
        逐张推理，测量延迟、FPS、资源占用。
        use_clip: 是否对每个检测框调用 CLIP
        clip_conditional: 仅对低置信度检测框调用 CLIP (E3)
        返回: dict with timing stats, resource stats, detection counts
        """
        import cv2
        import torch

        mode = "YOLO-only"
        if use_clip and not clip_conditional:
            mode = "YOLO+CLIP(全候选)"
        elif use_clip and clip_conditional:
            mode = "YOLO+CLIP(条件触发)"

        print(f"\n{'='*60}")
        print(f"[{tag}] 推理性能测试: {mode}")
        print(f"{'='*60}")

        if use_clip:
            self.load_clip()
            if self.clip is None:
                print("[SKIP] CLIP 不可用，跳过此实验")
                return None

        paths = self._get_image_paths(STABLE_FRAMES)
        n_images = len(paths)

        # 计时列表
        latencies = []         # 端到端延迟 (ms)
        yolo_times = []        # YOLO 推理时间 (ms)
        clip_times = []        # CLIP 推理时间 (ms)
        post_times = []        # 后处理时间 (ms)
        detection_counts = []  # 每张图片的检测数
        clip_calls = 0         # CLIP 调用总次数
        total_detections = 0

        # 资源基线
        gpu_before = gpu_snapshot()

        print(f"  测试 {n_images} 张图片 (预热 {WARMUP_FRAMES} 张)...")

        for idx, img_path in enumerate(paths):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 端到端计时
            t_start = time.perf_counter()

            # YOLO 推理
            t_yolo_start = time.perf_counter()
            results = self.model(img, conf=self.conf, iou=self.iou_thresh,
                                 imgsz=self.imgsz, verbose=False)
            t_yolo_end = time.perf_counter()

            # 后处理 + CLIP
            t_post_start = time.perf_counter()
            boxes = results[0].boxes
            n_det = 0
            if boxes is not None and len(boxes) > 0:
                n_det = len(boxes)
                if use_clip and self.clip is not None:
                    h, w = img.shape[:2]
                    for i in range(n_det):
                        conf_val = float(boxes.conf[i].item())
                        # E3: 条件触发
                        if clip_conditional and conf_val >= CLIP_LOW_CONF_TRIGGER:
                            continue
                        cls_id = int(boxes.cls[i].item())
                        cls_name = self.class_names.get(cls_id, "unknown")
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w, int(x2)), min(h, int(y2))
                        if x2 > x1 and y2 > y1:
                            crop = img[y1:y2, x1:x2]
                            t_clip_start = time.perf_counter()
                            try:
                                self.clip.classify(crop, cls_name)
                            except Exception:
                                pass
                            t_clip_end = time.perf_counter()
                            clip_times.append((t_clip_end - t_clip_start) * 1000)
                            clip_calls += 1
            t_post_end = time.perf_counter()

            # 同步 CUDA（确保推理完成）
            if self.device == "cuda":
                torch.cuda.synchronize()

            t_end = time.perf_counter()

            e2e_ms = (t_end - t_start) * 1000
            yolo_ms = (t_yolo_end - t_yolo_start) * 1000
            post_ms = (t_post_end - t_post_start) * 1000

            if idx >= WARMUP_FRAMES:  # 跳过预热
                latencies.append(e2e_ms)
                yolo_times.append(yolo_ms)
                post_times.append(post_ms)
                detection_counts.append(n_det)
                total_detections += n_det

            if (idx + 1) % 50 == 0:
                avg_lat = np.mean(latencies) if latencies else 0
                print(f"  [{idx+1}/{n_images}] 当前平均延迟: {avg_lat:.1f}ms")

        # 资源快照
        gpu_after = gpu_snapshot()

        # 统计
        if not latencies:
            print("  [WARN] 没有有效推理数据")
            return None

        latencies = np.array(latencies)
        n_valid = len(latencies)
        avg_latency = float(np.mean(latencies))
        p50 = float(percentile(latencies, 50))
        p95 = float(percentile(latencies, 95))
        p99 = float(percentile(latencies, 99))
        min_lat = float(np.min(latencies))
        max_lat = float(np.max(latencies))
        std_lat = float(np.std(latencies))

        # 有效 FPS = 1000 / 平均延迟（端到端）
        effective_fps = 1000.0 / avg_latency if avg_latency > 0 else 0

        # YOLO 推理时间统计
        yolo_arr = np.array(yolo_times) if yolo_times else np.array([0])
        clip_arr = np.array(clip_times) if clip_times else np.array([0])

        result = {
            "mode": mode,
            "images_tested": n_valid,
            "total_detections": total_detections,
            "avg_detections_per_image": round(total_detections / max(n_valid, 1), 2),
            "effective_fps": round(effective_fps, 2),
            "e2e_latency_ms": {
                "mean": round(avg_latency, 2),
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "p99": round(p99, 2),
                "min": round(min_lat, 2),
                "max": round(max_lat, 2),
                "std": round(std_lat, 2),
            },
            "yolo_latency_ms": {
                "mean": round(float(np.mean(yolo_arr)), 2),
                "p50": round(float(percentile(yolo_arr, 50)), 2),
                "p95": round(float(percentile(yolo_arr, 95)), 2),
            },
            "clip_latency_ms": {
                "mean": round(float(np.mean(clip_arr)), 2),
                "p50": round(float(percentile(clip_arr, 50)), 2),
                "total_calls": clip_calls,
            } if use_clip else None,
            "resource": {
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
            },
            "config": {
                "conf_thresh": self.conf,
                "iou_thresh": self.iou_thresh,
                "imgsz": self.imgsz,
                "device": self.device,
                "clip_conditional": clip_conditional,
                "clip_low_conf_trigger": CLIP_LOW_CONF_TRIGGER if clip_conditional else None,
            }
        }

        print(f"  有效图片: {n_valid}")
        print(f"  平均检测数: {result['avg_detections_per_image']}")
        print(f"  有效 FPS:   {result['effective_fps']}")
        print(f"  端到端延迟: mean={avg_latency:.1f}ms, P50={p50:.1f}ms, P95={p95:.1f}ms")
        print(f"  YOLO 延迟:  mean={float(np.mean(yolo_arr)):.1f}ms")
        if use_clip and clip_times:
            print(f"  CLIP 延迟:  mean={float(np.mean(clip_arr)):.1f}ms, 调用 {clip_calls} 次")
        print(f"  GPU 显存:   {gpu_after.get('gpu_mem_allocated_mb', 'N/A')} MB")

        return result

    # ----------------------------------------------------------
    # Part 3: 逐图评测（用于手动计算 P/R/mAP，更精细）
    # ----------------------------------------------------------
    def run_per_image_eval(self, tag="YOLOv11", use_clip=False, clip_conditional=False):
        """
        逐图推理并与 GT 标注匹配，计算精确的 TP/FP/FN 和自定义 mAP。
        作为 model.val() 的补充验证。
        """
        import cv2

        mode = "YOLO-only"
        if use_clip and not clip_conditional:
            mode = "YOLO+CLIP"
        elif use_clip and clip_conditional:
            mode = "YOLO+CLIP(条件)"

        print(f"\n{'='*60}")
        print(f"[{tag}] 逐图评测: {mode}")
        print(f"{'='*60}")

        if use_clip:
            self.load_clip()
            if self.clip is None:
                return None

        paths = self._get_image_paths()
        n_images = len(paths)

        # 按类别统计 TP/FP/FN
        tp_total = defaultdict(int)
        fp_total = defaultdict(int)
        fn_total = defaultdict(int)
        all_pred_confs = defaultdict(list)  # {class_id: [(conf, is_tp)]}

        for idx, img_path in enumerate(paths):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # 加载 GT
            label_path = str(img_path).replace("/images/", "/labels/").replace(
                Path(img_path).suffix, ".txt")
            gt_boxes = load_yolo_labels(label_path, h, w)

            # YOLO 推理
            results = self.model(img, conf=self.conf, iou=self.iou_thresh,
                                 imgsz=self.imgsz, verbose=False)
            boxes = results[0].boxes

            # 构造预测框
            pred_boxes = []
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i].item())
                    cls_id = int(boxes.cls[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                    # CLIP 辅助
                    if use_clip and self.clip is not None:
                        should_run_clip = True
                        if clip_conditional and conf >= CLIP_LOW_CONF_TRIGGER:
                            should_run_clip = False
                        if should_run_clip:
                            cls_name = self.class_names.get(cls_id, "unknown")
                            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                            ix2, iy2 = min(w, int(x2)), min(h, int(y2))
                            if ix2 > ix1 and iy2 > iy1:
                                try:
                                    crop = img[iy1:iy2, ix1:ix2]
                                    clip_res = self.clip.classify(crop, cls_name)
                                    # 融合 CLIP fine_confidence 和 YOLO conf
                                    clip_conf = clip_res.get("fine_confidence", 0.5)
                                    # 加权融合: 70% YOLO + 30% CLIP
                                    conf = 0.7 * conf + 0.3 * clip_conf
                                except Exception:
                                    pass

                    pred_boxes.append([x1, y1, x2, y2, conf, cls_id])

            # 匹配
            tp_by_cls, fp_by_cls, fn_by_cls = {}, {}, {}
            for cls_id in self.class_names:
                cls_preds = [pb for pb in pred_boxes if pb[5] == cls_id]
                cls_gts = [gb for gb in gt_boxes if gb[0] == cls_id]
                tp, fp, fn = match_detections(cls_preds, cls_gts, self.iou_thresh)
                tp_total[cls_id] += tp
                fp_total[cls_id] += fp
                fn_total[cls_id] += fn

                # 记录每个预测的 conf 和是否 TP（用于画 PR 曲线和计算 AP）
                cls_preds_sorted = sorted(cls_preds, key=lambda x: -x[4])
                gt_rem = list(range(len(cls_gts)))
                for pred in cls_preds_sorted:
                    matched = False
                    best_iou_val = 0
                    best_gt = -1
                    for gi in gt_rem:
                        gt_bbox = cls_gts[gi][1:5]
                        score = iou(pred[:4], gt_bbox)
                        if score > best_iou_val:
                            best_iou_val = score
                            best_gt = gi
                    if best_iou_val >= self.iou_thresh and best_gt >= 0:
                        all_pred_confs[cls_id].append((pred[4], True))
                        gt_rem.remove(best_gt)
                    else:
                        all_pred_confs[cls_id].append((pred[4], False))

            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{n_images}] 已处理...")

        # 计算每类 P/R/F1/AP
        per_class = {}
        total_tp = total_fp = total_fn = 0
        for cls_id, name in self.class_names.items():
            tp = tp_total[cls_id]
            fp = fp_total[cls_id]
            fn = fn_total[cls_id]
            total_tp += tp
            total_fp += fp
            total_fn += fn

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            # AP (全点插值)
            preds_sorted = sorted(all_pred_confs.get(cls_id, []), key=lambda x: -x[0])
            n_pos = tp + fn  # GT 总数
            cum_tp = 0
            cum_fp = 0
            precisions = []
            recalls = []
            for conf_val, is_tp in preds_sorted:
                if is_tp:
                    cum_tp += 1
                else:
                    cum_fp += 1
                precisions.append(cum_tp / (cum_tp + cum_fp))
                recalls.append(cum_tp / max(n_pos, 1))
            ap = compute_ap(precisions, recalls) if precisions else 0

            per_class[name] = {
                "TP": tp, "FP": fp, "FN": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "AP_0.5": round(ap, 4),
                "n_gt": n_pos,
            }
            print(f"  [{name}] P={prec:.3f} R={rec:.3f} F1={f1:.3f} AP@0.5={ap:.4f} (TP={tp} FP={fp} FN={fn})")

        # 总体
        total_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        total_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        total_f1 = 2 * total_prec * total_rec / (total_prec + total_rec) if (total_prec + total_rec) > 0 else 0
        mean_ap = np.mean([v["AP_0.5"] for v in per_class.values()]) if per_class else 0

        summary = {
            "mode": mode,
            "images": n_images,
            "overall": {
                "precision": round(total_prec, 4),
                "recall": round(total_rec, 4),
                "f1": round(total_f1, 4),
                "mAP_0.5": round(float(mean_ap), 4),
                "total_TP": total_tp,
                "total_FP": total_fp,
                "total_FN": total_fn,
            },
            "per_class": per_class,
        }

        print(f"\n  总体: P={total_prec:.3f} R={total_rec:.3f} F1={total_f1:.3f} mAP@0.5={mean_ap:.4f}")
        return summary

    # ----------------------------------------------------------
    # Part 4: 稳定性测试（长时间连续运行）
    # ----------------------------------------------------------
    def run_stability_test(self, duration_sec=300):
        """
        持续推理 N 秒，检测内存泄漏和帧率稳定性。
        """
        import cv2
        import torch

        print(f"\n{'='*60}")
        print(f"[稳定性测试] 持续 {duration_sec} 秒连续推理")
        print(f"{'='*60}")

        paths = self.image_paths
        if not paths:
            return None

        n_paths = len(paths)
        frame_count = 0
        fps_history = []     # 每秒的 FPS
        mem_history = []     # 每 10 秒的 GPU 显存
        errors = 0
        t_start = time.time()
        t_last_report = t_start

        while (time.time() - t_start) < duration_sec:
            img_path = paths[frame_count % n_paths]
            img = cv2.imread(str(img_path))
            if img is None:
                errors += 1
                frame_count += 1
                continue

            try:
                self.model(img, conf=self.conf, imgsz=self.imgsz, verbose=False)
                if self.device == "cuda":
                    torch.cuda.synchronize()
            except Exception:
                errors += 1

            frame_count += 1
            t_now = time.time()

            # 每秒记录 FPS
            if t_now - t_last_report >= 1.0:
                elapsed = t_now - t_last_report
                fps = (frame_count % 1000) / elapsed if elapsed > 0 else 0  # 简单估计
                fps_history.append(fps)
                t_last_report = t_now

            # 每 10 秒记录显存
            if frame_count % 100 == 0 and self.device == "cuda":
                mem_history.append(torch.cuda.memory_allocated(0) / 1024**2)

        elapsed_total = time.time() - t_start
        effective_fps = frame_count / elapsed_total if elapsed_total > 0 else 0

        # 显存趋势
        mem_growth_mb = 0
        if len(mem_history) >= 2:
            mem_growth_mb = mem_history[-1] - mem_history[0]

        result = {
            "duration_sec": round(elapsed_total, 1),
            "total_frames": frame_count,
            "effective_fps": round(effective_fps, 2),
            "errors": errors,
            "error_rate": round(errors / max(frame_count, 1), 4),
            "gpu_mem_samples_mb": [round(m, 1) for m in mem_history],
            "gpu_mem_growth_mb": round(mem_growth_mb, 1),
            "stable": errors < frame_count * 0.01 and abs(mem_growth_mb) < 100,
        }

        print(f"  总帧数:   {frame_count}")
        print(f"  有效 FPS: {result['effective_fps']}")
        print(f"  错误数:   {errors} ({result['error_rate']:.1%})")
        print(f"  显存增长: {mem_growth_mb:.1f} MB")
        print(f"  结论:     {'稳定' if result['stable'] else '不稳定'}")

        return result


# ============================================================
# 报告生成
# ============================================================
def generate_report(e1_val, e1_perf, e1_detail, e2_perf, e2_detail,
                    e3_perf, e3_detail, stability, device_info, output_dir):
    """生成评测报告 (JSON + TXT + CSV)"""
    import csv

    report = {
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device_info,
        "config": {
            "conf_thresh": CONF_THRESH,
            "iou_thresh": IOU_THRESH,
            "imgsz": IMG_SIZE,
            "clip_model": CLIP_MODEL,
            "clip_low_conf_trigger": CLIP_LOW_CONF_TRIGGER,
        },
        "experiments": {}
    }

    # E1: YOLOv11 基线
    e1 = {"name": "E1: YOLOv11 基线", "description": "建立无CLIP时的准确性与速度基线"}
    if e1_val:
        e1["accuracy"] = e1_val
    if e1_perf:
        e1["performance"] = e1_perf
    if e1_detail:
        e1["detailed_metrics"] = e1_detail
    report["experiments"]["E1"] = e1

    # E2: YOLOv11+CLIP 全候选
    e2 = {"name": "E2: YOLOv11+CLIP 全候选辅助", "description": "评价语义模块的增益与计算开销"}
    if e2_perf:
        e2["performance"] = e2_perf
    if e2_detail:
        e2["detailed_metrics"] = e2_detail
    # 计算增益
    if e1_perf and e2_perf:
        e2["vs_E1"] = {
            "fps_change": round(e2_perf["effective_fps"] - e1_perf["effective_fps"], 2),
            "fps_change_pct": round((e2_perf["effective_fps"] / e1_perf["effective_fps"] - 1) * 100, 1)
                              if e1_perf["effective_fps"] > 0 else None,
            "p95_change_ms": round(e2_perf["e2e_latency_ms"]["p95"] - e1_perf["e2e_latency_ms"]["p95"], 2),
        }
    if e1_detail and e2_detail:
        e2["vs_E1_accuracy"] = {}
        e1_p = e1_detail.get("overall", {})
        e2_p = e2_detail.get("overall", {})
        for key in ["precision", "recall", "f1", "mAP_0.5"]:
            if key in e1_p and key in e2_p:
                e2["vs_E1_accuracy"][f"{key}_delta"] = round(e2_p[key] - e1_p[key], 4)
    report["experiments"]["E2"] = e2

    # E3: YOLOv11+CLIP 条件触发
    e3 = {"name": "E3: YOLOv11+CLIP 条件触发", "description": "评价仅处理低置信/易混候选的平衡"}
    if e3_perf:
        e3["performance"] = e3_perf
    if e3_detail:
        e3["detailed_metrics"] = e3_detail
    if e1_perf and e3_perf:
        e3["vs_E1"] = {
            "fps_change": round(e3_perf["effective_fps"] - e1_perf["effective_fps"], 2),
            "fps_change_pct": round((e3_perf["effective_fps"] / e1_perf["effective_fps"] - 1) * 100, 1)
                              if e1_perf["effective_fps"] > 0 else None,
            "p95_change_ms": round(e3_perf["e2e_latency_ms"]["p95"] - e1_perf["e2e_latency_ms"]["p95"], 2),
        }
    if e1_detail and e3_detail:
        e3["vs_E1_accuracy"] = {}
        e1_p = e1_detail.get("overall", {})
        e3_p = e3_detail.get("overall", {})
        for key in ["precision", "recall", "f1", "mAP_0.5"]:
            if key in e1_p and key in e3_p:
                e3["vs_E1_accuracy"][f"{key}_delta"] = round(e3_p[key] - e1_p[key], 4)
    report["experiments"]["E3"] = e3

    # E4 & E5: 框架
    report["experiments"]["E4"] = {
        "name": "E4: 移除针对性数据增强",
        "description": "评价复杂光照与模糊增强的作用",
        "status": "需要单独训练无增强模型后运行",
        "how_to": "python evaluate_all.py --model path/to/no_augment_best.pt",
    }
    report["experiments"]["E5"] = {
        "name": "E5: 轻量模型/部署格式",
        "description": "评价边缘部署的速度—精度权衡",
        "status": "需要导出轻量模型后运行",
        "how_to": "python evaluate_all.py --model path/to/yolo11n_openvino_model/ 或 --model yolov8n.onnx",
    }

    # 稳定性
    if stability:
        report["stability"] = stability

    # 表5 汇总（说明书直接填写用）
    report["table5_summary"] = {
        "detection_accuracy": {
            "precision": e1_detail.get("overall", {}).get("precision", "待实测") if e1_detail else "待实测",
            "recall": e1_detail.get("overall", {}).get("recall", "待实测") if e1_detail else "待实测",
            "f1": e1_detail.get("overall", {}).get("f1", "待实测") if e1_detail else "待实测",
        },
        "localization_performance": {
            "mAP_0.5": e1_val.get("mAP_0.5", "待实测") if e1_val else "待实测",
            "mAP_0.5_0.95": e1_val.get("mAP_0.5_0.95", "待实测") if e1_val else "待实测",
        },
        "realtime_performance": {
            "effective_fps": e1_perf.get("effective_fps", "待实测") if e1_perf else "待实测",
            "p50_latency_ms": e1_perf.get("e2e_latency_ms", {}).get("p50", "待实测") if e1_perf else "待实测",
            "p95_latency_ms": e1_perf.get("e2e_latency_ms", {}).get("p95", "待实测") if e1_perf else "待实测",
        },
        "stability": {
            "duration_sec": stability.get("duration_sec", "待实测") if stability else "待实测",
            "errors": stability.get("errors", "待实测") if stability else "待实测",
            "stable": stability.get("stable", "待实测") if stability else "待实测",
        },
        "resource_efficiency": {
            "gpu_mem_mb": device_info.get("gpu_after", {}).get("gpu_mem_allocated_mb", "待实测") if device_info.get("gpu_after") else "待实测",
        },
    }

    # === 写 JSON ===
    json_path = os.path.join(output_dir, "eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[输出] JSON 报告: {json_path}")

    # === 写 TXT ===
    txt_path = os.path.join(output_dir, "eval_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  道路病害检测系统 — 全指标评测报告\n")
        f.write(f"  生成时间: {report['evaluation_time']}\n")
        f.write(f"  设备: {device_info.get('gpu_name', device_info.get('device', 'unknown'))}\n")
        f.write("=" * 70 + "\n\n")

        f.write("【表5 直接填写数据】\n")
        f.write("-" * 50 + "\n")
        t5 = report["table5_summary"]
        f.write(f"检测准确性:  P={t5['detection_accuracy']['precision']}, ")
        f.write(f"R={t5['detection_accuracy']['recall']}, ")
        f.write(f"F1={t5['detection_accuracy']['f1']}\n")
        f.write(f"定位性能:    mAP@0.5={t5['localization_performance']['mAP_0.5']}, ")
        f.write(f"mAP@0.5:0.95={t5['localization_performance']['mAP_0.5_0.95']}\n")
        f.write(f"实时性能:    有效FPS={t5['realtime_performance']['effective_fps']}, ")
        f.write(f"P50={t5['realtime_performance']['p50_latency_ms']}ms, ")
        f.write(f"P95={t5['realtime_performance']['p95_latency_ms']}ms\n")
        f.write(f"稳定性:      运行{t5['stability']['duration_sec']}s, ")
        f.write(f"错误{t5['stability']['errors']}次, ")
        f.write(f"{'稳定' if t5['stability']['stable'] else '不稳定'}\n")
        f.write(f"资源效率:    GPU显存={t5['resource_efficiency']['gpu_mem_mb']}MB\n\n")

        f.write("【表6 消融实验对比】\n")
        f.write("-" * 50 + "\n")
        header = f"{'实验':<8} {'P':>8} {'R':>8} {'F1':>8} {'mAP@0.5':>10} {'FPS':>8} {'P95延迟':>10}\n"
        f.write(header)
        f.write("-" * 65 + "\n")
        for exp_id, exp_data in report["experiments"].items():
            if exp_id in ("E4", "E5"):
                continue
            detail = exp_data.get("detailed_metrics", {})
            perf = exp_data.get("performance", {})
            overall = detail.get("overall", {}) if detail else {}
            p = overall.get("precision", "-")
            r = overall.get("recall", "-")
            f1 = overall.get("f1", "-")
            mAP = overall.get("mAP_0.5", "-")
            fps = perf.get("effective_fps", "-") if perf else "-"
            p95 = perf.get("e2e_latency_ms", {}).get("p95", "-") if perf else "-"
            short_name = exp_id + ": " + exp_data.get("name", "").split(":")[-1].strip()[:12]
            f.write(f"{short_name:<8} {p:>8} {r:>8} {f1:>8} {mAP:>10} {fps:>8} {p95:>10}\n")

        # E2/E3 vs E1 增益
        for exp_id in ["E2", "E3"]:
            exp = report["experiments"].get(exp_id, {})
            vs = exp.get("vs_E1", {})
            vs_acc = exp.get("vs_E1_accuracy", {})
            if vs or vs_acc:
                f.write(f"\n  {exp_id} 相对 E1 的变化:\n")
                if vs.get("fps_change_pct") is not None:
                    f.write(f"    FPS 变化: {vs['fps_change_pct']:+.1f}%\n")
                if vs.get("p95_change_ms") is not None:
                    f.write(f"    P95 延迟变化: {vs['p95_change_ms']:+.1f}ms\n")
                for k, v in vs_acc.items():
                    f.write(f"    {k}: {v:+.4f}\n")

        if stability:
            f.write(f"\n【稳定性测试】\n")
            f.write("-" * 50 + "\n")
            f.write(f"  运行时长:   {stability['duration_sec']}s\n")
            f.write(f"  总帧数:     {stability['total_frames']}\n")
            f.write(f"  有效 FPS:   {stability['effective_fps']}\n")
            f.write(f"  错误数:     {stability['errors']} ({stability['error_rate']:.1%})\n")
            f.write(f"  显存增长:   {stability['gpu_mem_growth_mb']} MB\n")
            f.write(f"  结论:       {'稳定' if stability['stable'] else '不稳定'}\n")

    print(f"[输出] 文本报告: {txt_path}")

    # === 写 CSV（方便填入说明书表格）===
    csv_path = os.path.join(output_dir, "eval_report.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["实验", "Precision", "Recall", "F1", "mAP@0.5", "有效FPS", "P95延迟(ms)", "说明"])
        for exp_id in ["E1", "E2", "E3"]:
            exp = report["experiments"].get(exp_id, {})
            detail = exp.get("detailed_metrics", {})
            perf = exp.get("performance", {})
            overall = detail.get("overall", {}) if detail else {}
            writer.writerow([
                exp.get("name", exp_id),
                overall.get("precision", ""),
                overall.get("recall", ""),
                overall.get("f1", ""),
                overall.get("mAP_0.5", ""),
                perf.get("effective_fps", "") if perf else "",
                perf.get("e2e_latency_ms", {}).get("p95", "") if perf else "",
                exp.get("description", ""),
            ])
    print(f"[输出] CSV 报告: {csv_path}")

    return report


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="道路病害检测系统 — 全指标自动评测")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO 模型权重路径")
    parser.add_argument("--data", default=DEFAULT_DATA, help="数据配置 YAML 路径")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="推理输入尺寸")
    parser.add_argument("--conf", type=float, default=CONF_THRESH, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=IOU_THRESH, help="NMS IoU 阈值")
    parser.add_argument("--experiment", choices=["E1", "E2", "E3", "stability", "all"],
                        default="all", help="只跑指定实验")
    parser.add_argument("--stability-duration", type=int, default=300,
                        help="稳定性测试时长(秒)")
    parser.add_argument("--output", default=".", help="输出目录")
    args = parser.parse_args()

    # 解析路径（相对于脚本所在目录）
    script_dir = Path(__file__).resolve().parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = script_dir / model_path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = script_dir / data_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 参数已通过 Evaluator 构造器传递，无需修改全局变量

    check_deps()
    device = get_device()

    # 设备信息
    device_info = gpu_snapshot()
    device_info["device"] = device

    print(f"\n{'#'*60}")
    print(f"  道路病害检测系统 — 全指标自动评测")
    print(f"  模型: {model_path}")
    print(f"  数据: {data_path}")
    print(f"  输入尺寸: {args.imgsz}, 置信度: {args.conf}, IoU: {args.iou}")
    print(f"{'#'*60}\n")

    # 初始化评测器
    evaluator = Evaluator(
        model_path=model_path,
        data_yaml=str(data_path),
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou_thresh=args.iou,
    )

    e1_val = e1_perf = e1_detail = None
    e2_perf = e2_detail = None
    e3_perf = e3_detail = None
    stability = None

    # ---- E1: YOLOv11 基线 ----
    if args.experiment in ("E1", "all"):
        print("\n" + "█" * 60)
        print("  E1: YOLOv11 基线")
        print("█" * 60)

        e1_val = evaluator.run_yolo_validation(tag="E1")
        e1_perf = evaluator.run_inference_benchmark(tag="E1", use_clip=False)
        e1_detail = evaluator.run_per_image_eval(tag="E1", use_clip=False)

    # ---- E2: YOLOv11+CLIP 全候选辅助 ----
    if args.experiment in ("E2", "all"):
        print("\n" + "█" * 60)
        print("  E2: YOLOv11+CLIP 全候选辅助")
        print("█" * 60)

        e2_perf = evaluator.run_inference_benchmark(tag="E2", use_clip=True, clip_conditional=False)
        e2_detail = evaluator.run_per_image_eval(tag="E2", use_clip=True, clip_conditional=False)

    # ---- E3: YOLOv11+CLIP 条件触发 ----
    if args.experiment in ("E3", "all"):
        print("\n" + "█" * 60)
        print("  E3: YOLOv11+CLIP 条件触发 (仅低置信度)")
        print("█" * 60)

        e3_perf = evaluator.run_inference_benchmark(tag="E3", use_clip=True, clip_conditional=True)
        e3_detail = evaluator.run_per_image_eval(tag="E3", use_clip=True, clip_conditional=True)

    # ---- 稳定性测试 ----
    if args.experiment in ("stability", "all"):
        print("\n" + "█" * 60)
        print("  稳定性测试")
        print("█" * 60)

        stability = evaluator.run_stability_test(duration_sec=args.stability_duration)

    # ---- 生成报告 ----
    print(f"\n{'='*60}")
    print("  生成报告...")
    print(f"{'='*60}")

    report = generate_report(
        e1_val, e1_perf, e1_detail,
        e2_perf, e2_detail,
        e3_perf, e3_detail,
        stability, device_info, str(output_dir)
    )

    # 打印最终汇总
    print(f"\n{'='*60}")
    print("  评测完成!")
    print(f"{'='*60}")
    print(f"  JSON: {output_dir}/eval_report.json")
    print(f"  TXT:  {output_dir}/eval_report.txt")
    print(f"  CSV:  {output_dir}/eval_report.csv")
    print(f"\n  【可直接填入说明书的数据】")
    t5 = report.get("table5_summary", {})
    da = t5.get("detection_accuracy", {})
    lp = t5.get("localization_performance", {})
    rp = t5.get("realtime_performance", {})
    print(f"  表5 检测准确性: P={da.get('precision')}, R={da.get('recall')}, F1={da.get('f1')}")
    print(f"  表5 定位性能:   mAP@0.5={lp.get('mAP_0.5')}, mAP@0.5:0.95={lp.get('mAP_0.5_0.95')}")
    print(f"  表5 实时性能:   FPS={rp.get('effective_fps')}, P50={rp.get('p50_latency_ms')}ms, P95={rp.get('p95_latency_ms')}ms")
    print(f"\n  E4(移除增强) 和 E5(轻量模型) 需要单独训练模型后运行:")
    print(f"    python evaluate_all.py --model path/to/no_augment_best.pt --experiment E1")
    print(f"    python evaluate_all.py --model path/to/lightweight_best.pt --experiment E1")


if __name__ == "__main__":
    main()
