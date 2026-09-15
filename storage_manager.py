# -*- coding: utf-8 -*-
"""存储管理 — JSON文件持久化 + CSV导出 + 统计查询"""

import json
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path


class StorageManager:
    """检测记录的存储管理器"""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.records_file = self.data_dir / "records.json"
        self.images_dir = self.data_dir / "images"
        self.features_dir = self.data_dir / "features"
        self.config_file = self.base_dir / "config.json"
        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        # 加载或初始化记录
        self._records = self._load_records()

    # ─── 记录 CRUD ───

    def _load_records(self) -> list:
        if self.records_file.exists():
            try:
                with open(self.records_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_records(self):
        with open(self.records_file, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)

    def _strip_features(self, record: dict):
        """从检测记录中移除 image_feature（CLIP特征向量）
        特征向量已单独存储在 data/features/*.npz 中，
        写入 JSON 会导致文件过大且容易截断损坏"""
        for det in record.get("detections", []):
            det.pop("image_feature", None)

    def save_record(self, record: dict) -> str:
        """保存一条检测记录，返回record_id"""
        self._strip_features(record)
        now = datetime.now()
        # 生成唯一ID
        today_count = sum(
            1 for r in self._records
            if r.get("timestamp", "").startswith(now.strftime("%Y-%m-%d"))
        )
        record_id = f"REC-{now.strftime('%Y%m%d')}-{today_count + 1:03d}"
        record["id"] = record_id
        record["timestamp"] = now.isoformat(timespec="seconds")
        self._records.insert(0, record)  # 最新的在前面
        self._save_records()
        return record_id

    def update_record(self, record: dict):
        """更新已有记录（通过 id 匹配），不创建新记录。
        用于在 save_record 之后补充 image_path 等字段。"""
        self._strip_features(record)
        record_id = record.get("id")
        if not record_id:
            return
        for i, existing in enumerate(self._records):
            if existing.get("id") == record_id:
                self._records[i] = record
                self._save_records()
                return
        # 如果找不到，当作新记录保存
        self._records.insert(0, record)
        self._save_records()

    def save_detection_image(self, image_path: str, record_id: str) -> str:
        """保存检测图片到data/images/目录，返回相对路径"""
        if not os.path.exists(image_path):
            return ""
        ext = os.path.splitext(image_path)[1] or ".jpg"
        dest = self.images_dir / f"{record_id}{ext}"
        import shutil
        shutil.copy2(image_path, dest)
        return str(dest.relative_to(self.base_dir))

    def get_records(self, filters: dict = None) -> list:
        """获取记录列表，支持过滤"""
        records = self._records[:]
        if not filters:
            return records

        if "date_from" in filters and filters["date_from"]:
            records = [r for r in records if r.get("timestamp", "") >= filters["date_from"]]
        if "date_to" in filters and filters["date_to"]:
            records = [r for r in records if r.get("timestamp", "") <= filters["date_to"] + "T23:59:59"]
        if "class_name" in filters and filters["class_name"] and filters["class_name"] != "全部":
            records = [
                r for r in records
                if any(d.get("class") == filters["class_name"] for d in r.get("detections", []))
            ]
        if "keyword" in filters and filters["keyword"]:
            kw = filters["keyword"].lower()
            records = [
                r for r in records
                if kw in r.get("id", "").lower()
                or kw in r.get("model", "").lower()
                or kw in r.get("source", "").lower()
                or any(kw in d.get("class", "").lower() for d in r.get("detections", []))
            ]
        return records

    def get_record(self, record_id: str) -> dict:
        """获取单条记录"""
        for r in self._records:
            if r.get("id") == record_id:
                return r
        return None

    def delete_record(self, record_id: str) -> bool:
        """删除一条记录"""
        original_len = len(self._records)
        self._records = [r for r in self._records if r.get("id") != record_id]
        if len(self._records) < original_len:
            self._save_records()
            return True
        return False

    # ─── 统计 ───

    def get_statistics(self) -> dict:
        """从所有记录中汇总统计数据"""
        total_records = len(self._records)
        total_detections = 0
        class_counts = {}
        severity_counts = {"minor": 0, "moderate": 0, "severe": 0}
        daily_counts = {}
        confidence_sum = 0.0
        confidence_count = 0

        for record in self._records:
            detections = record.get("detections", [])
            total_detections += len(detections)
            # 按日期统计
            date_str = record.get("timestamp", "")[:10]
            daily_counts[date_str] = daily_counts.get(date_str, 0) + len(detections)
            # 按类别统计
            for det in detections:
                cls = det.get("class", "unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1
                sev = det.get("severity", "minor")
                if sev in severity_counts:
                    severity_counts[sev] += 1
                conf = det.get("confidence", 0)
                if conf > 0:
                    confidence_sum += conf
                    confidence_count += 1

        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0

        # 今日统计
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = daily_counts.get(today, 0)

        # 近7日趋势
        trend = []
        for i in range(6, -1, -1):
            d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            trend.append({"date": d[5:], "count": daily_counts.get(d, 0)})

        return {
            "total_records": total_records,
            "total_detections": total_detections,
            "today_detections": today_count,
            "avg_confidence": avg_confidence,
            "class_counts": class_counts,
            "severity_counts": severity_counts,
            "daily_trend": trend,
        }

    # ─── 导出 ───

    def export_csv(self, records: list, filepath: str):
        """将记录导出为CSV文件"""
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "记录ID", "时间", "来源", "模型", "检测数量",
                "类别", "置信度", "位置(x1,y1,x2,y2)", "严重等级"
            ])
            for record in records:
                base_row = [
                    record.get("id", ""),
                    record.get("timestamp", ""),
                    record.get("source", ""),
                    record.get("model", ""),
                    len(record.get("detections", [])),
                ]
                detections = record.get("detections", [])
                if detections:
                    for det in detections:
                        bbox = det.get("bbox", [])
                        bbox_str = f"{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}" if len(bbox) == 4 else ""
                        writer.writerow(base_row + [
                            det.get("class", ""),
                            f"{det.get('confidence', 0):.3f}",
                            bbox_str,
                            det.get("severity", ""),
                        ])
                else:
                    writer.writerow(base_row + ["", "", "", ""])

    def export_txt_report(self, records: list, filepath: str, title: str = "道路病害检测报告"):
        """导出为TXT文本报告"""
        stats = self.get_statistics()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"  {title}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"报告记录数：{len(records)} 条\n")
            f.write(f"病害检出总数：{stats['total_detections']} 个\n")
            f.write(f"平均检测置信度：{stats['avg_confidence']:.1%}\n\n")

            f.write(f"{'─' * 40}\n")
            f.write("一、病害类型分布\n")
            f.write(f"{'─' * 40}\n")
            for cls, count in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
                f.write(f"  {cls}: {count} 个\n")

            f.write(f"\n{'─' * 40}\n")
            f.write("二、严重程度分布\n")
            f.write(f"{'─' * 40}\n")
            sev_names = {"minor": "轻微", "moderate": "中等", "severe": "严重"}
            for sev, count in stats["severity_counts"].items():
                f.write(f"  {sev_names.get(sev, sev)}: {count} 个\n")

            f.write(f"\n{'─' * 40}\n")
            f.write("三、检测明细\n")
            f.write(f"{'─' * 40}\n")
            for i, record in enumerate(records[:50], 1):
                f.write(f"\n  [{i}] {record.get('id', '')} | {record.get('timestamp', '')[:19]}\n")
                f.write(f"      来源: {record.get('source', '')} | 模型: {record.get('model', '')}\n")
                for det in record.get("detections", []):
                    f.write(f"      - {det.get('class', '')} ({det.get('confidence', 0):.1%}) "
                            f"[{det.get('severity', '')}]\n")

            f.write(f"\n{'=' * 60}\n")
            f.write("  报告结束\n")
            f.write(f"{'=' * 60}\n")

    # ─── CLIP特征索引与语义检索 ───

    def save_features(self, record_id: str, features_list: list):
        """
        为一条记录保存CLIP图像特征向量
        features_list: [{"bbox": [...], "class": str, "fine_class": str, "feature": np.ndarray}, ...]
        """
        import numpy as np
        feat_file = self.features_dir / f"{record_id}.npz"
        if not features_list:
            return
        arrays = {}
        metadata = []
        for i, item in enumerate(features_list):
            arrays[f"feat_{i}"] = item["feature"]
            metadata.append({
                "bbox": item.get("bbox", []),
                "class": item.get("class", ""),
                "fine_class": item.get("fine_class", ""),
                "index": i,
            })
        # 存metadata为json字符串
        arrays["meta"] = np.array([json.dumps(metadata, ensure_ascii=False)], dtype="U")
        np.savez(feat_file, **arrays)

    def load_all_features(self) -> list:
        """
        加载所有记录的特征，返回:
        [{"record_id": str, "features": [(feature_vec, metadata), ...]}, ...]
        """
        import numpy as np
        all_data = []
        for feat_file in self.features_dir.glob("*.npz"):
            record_id = feat_file.stem
            try:
                data = np.load(feat_file, allow_pickle=False)
                meta_str = str(data["meta"][0])
                metadata = json.loads(meta_str)
                items = []
                for m in metadata:
                    key = f"feat_{m['index']}"
                    if key in data:
                        items.append((data[key], m))
                all_data.append({"record_id": record_id, "features": items})
            except Exception:
                continue
        return all_data

    def semantic_search(self, text_features, top_k=10) -> list:
        """
        用CLIP文本特征向量检索历史记录
        text_features: numpy数组 (N_text, 512) 或 (512,)
        返回按最高相似度排序的record_id列表:
        [{"record_id": str, "score": float, "matched_class": str, "matched_fine": str}, ...]
        """
        import numpy as np
        if text_features.ndim == 1:
            text_features = text_features.reshape(1, -1)

        all_features = self.load_all_features()
        results = []

        for entry in all_features:
            best_score = 0.0
            best_meta = {}
            for feat_vec, meta in entry["features"]:
                # 计算与所有文本特征的最大相似度
                sims = feat_vec @ text_features.T  # (N_text,)
                max_sim = float(sims.max())
                if max_sim > best_score:
                    best_score = max_sim
                    best_meta = meta
            results.append({
                "record_id": entry["record_id"],
                "score": best_score,
                "matched_class": best_meta.get("class", ""),
                "matched_fine": best_meta.get("fine_class", ""),
            })

        results.sort(key=lambda x: -x["score"])
        return results[:top_k]

    # ─── 配置管理 ───

    def load_config(self) -> dict:
        """加载配置文件"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return self._default_config()

    def save_config(self, config: dict):
        """保存配置文件"""
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _default_config(self) -> dict:
        return {
            "model_path": str(self.base_dir / "best.pt"),
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
            "camera_index": 0,
            "resolution": "1280x720",
            "image_save_path": str(self.images_dir),
            "auto_save_detections": True,
        }


def classify_severity(class_name: str, confidence: float, bbox_area_ratio: float) -> str:
    """根据类别、置信度和面积占比自动判定严重等级"""
    if confidence > 0.9 and bbox_area_ratio > 0.1:
        return "severe"
    elif confidence > 0.7 and bbox_area_ratio > 0.03:
        return "moderate"
    else:
        return "minor"
