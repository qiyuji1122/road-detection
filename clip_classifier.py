# -*- coding: utf-8 -*-
"""CLIP零样本分类引擎 — 对YOLO检测区域进行细分类和严重程度评估"""

import os
# 国内使用HuggingFace镜像
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
from PIL import Image
import torch
import open_clip


# CLIP细分标签定义
# YOLO检测到"crack"后，CLIP进一步判断是哪种裂缝
CRACK_LABELS = [
    "a photo of longitudinal crack on road surface",
    "a photo of transverse crack on asphalt",
    "a photo of alligator cracking pattern on road",
    "a photo of minor thin crack on pavement",
    "a photo of severe wide crack with spalling on road",
]

CRACK_LABEL_NAMES = [
    "纵向裂缝",
    "横向裂缝",
    "网状裂缝",
    "轻微裂缝",
    "严重裂缝",
]

POTHOLE_LABELS = [
    "a photo of small shallow pothole on road",
    "a photo of large deep pothole on asphalt",
    "a photo of pothole with exposed aggregate",
    "a photo of minor surface depression on road",
    "a photo of severe pothole with broken edges",
]

POTHOLE_LABEL_NAMES = [
    "小型坑洼",
    "大型坑洼",
    "骨料外露坑洼",
    "浅层凹陷",
    "严重破损坑洼",
]

# 通用严重度评估标签
SEVERITY_LABELS = [
    "a photo of minor road damage, barely visible",
    "a photo of moderate road damage, clearly visible",
    "a photo of severe road damage, dangerous condition",
]

SEVERITY_LABEL_NAMES = ["轻微", "中等", "严重"]


class CLIPClassifier:
    """CLIP零样本分类器"""

    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=None):
        """
        初始化CLIP模型
        Args:
            model_name: CLIP模型名称 (ViT-B-32最轻量，ViT-B-16更精确)
            pretrained: 预训练权重
            device: 推理设备 (mps/cuda/cpu)
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # 预编码文本特征（避免每次推理重复计算）
        self._text_cache = {}
        self._precompute_text_features()

    def _precompute_text_features(self):
        """预计算所有标签的文本特征向量"""
        with torch.no_grad():
            for name, labels in [
                ("crack", CRACK_LABELS),
                ("pothole", POTHOLE_LABELS),
                ("severity", SEVERITY_LABELS),
            ]:
                tokens = self.tokenizer(labels).to(self.device)
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                self._text_cache[name] = features

    def classify(self, image_crop: np.ndarray, yolo_class: str) -> dict:
        """
        对YOLO检测区域进行CLIP零样本分类

        Args:
            image_crop: numpy数组，YOLO裁剪出的BGR图像区域
            yolo_class: YOLO检测的类别名 ("crack" 或 "pothole")

        Returns:
            {
                "fine_class": "纵向裂缝",       # 细分类型
                "fine_confidence": 0.65,        # 细分置信度
                "clip_severity": "中等",         # CLIP评估的严重程度
                "severity_confidence": 0.52,    # 严重度置信度
                "all_scores": {...},            # 所有类别的得分
            }
        """
        import cv2

        if image_crop is None or image_crop.size == 0:
            return self._empty_result()

        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # CLIP预处理
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        result = {}

        with torch.no_grad():
            img_features = self.model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # 保存原始特征向量（用于语义检索索引）
        result["image_feature"] = img_features.squeeze(0).cpu().numpy().astype(np.float32)

        # 1. 细分类型分类
        class_key = yolo_class if yolo_class in self._text_cache else "crack"
        text_features = self._text_cache[class_key]
        similarity = (img_features @ text_features.T).squeeze(0)
        probs = similarity.softmax(dim=0).cpu().numpy()

        label_names = (
            CRACK_LABEL_NAMES if class_key == "crack" else POTHOLE_LABEL_NAMES
        )
        best_idx = int(np.argmax(probs))

        result["fine_class"] = label_names[best_idx]
        result["fine_confidence"] = round(float(probs[best_idx]), 4)
        result["all_fine_scores"] = {
            name: round(float(score), 4)
            for name, score in zip(label_names, probs)
        }

        # 2. 严重程度评估
        sev_features = self._text_cache["severity"]
        sev_sim = (img_features @ sev_features.T).squeeze(0)
        sev_probs = sev_sim.softmax(dim=0).cpu().numpy()

        sev_idx = int(np.argmax(sev_probs))
        result["clip_severity"] = SEVERITY_LABEL_NAMES[sev_idx]
        result["severity_confidence"] = round(float(sev_probs[sev_idx]), 4)

        return result

    def classify_batch(self, crops: list, yolo_classes: list) -> list:
        """批量分类多个检测区域"""
        results = []
        for crop, cls in zip(crops, yolo_classes):
            results.append(self.classify(crop, cls))
        return results

    def encode_image_np(self, image_crop: np.ndarray) -> np.ndarray:
        """
        编码一张图片为CLIP特征向量（numpy数组）
        用于存储到索引中做语义检索
        """
        import cv2
        if image_crop is None or image_crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    # ── 中文→英文翻译表（CLIP只理解英文）──
    _ZH_EN_DICT = {
        # 道路病害
        "白色标线磨损": "faded white road marking paint wear",
        "标线磨损": "faded road lane marking paint",
        "路面积水": "water puddle accumulation on road surface",
        "路面沉陷": "road surface subsidence depression",
        "修补痕迹": "road surface patch repair mark",
        "井盖破损": "damaged broken manhole cover",
        "井盖周围破损": "broken pavement around manhole cover",
        # 交通设施
        "交通标志": "traffic road sign",
        "道路护栏": "road guardrail barrier",
        "路灯": "street light lamp post",
        "减速带": "speed bump road hump",
        "排水沟": "road drainage ditch gutter",
        # 路面状况
        "路面油污": "oil stain spill on road surface",
        "碎石散落": "scattered gravel loose stones on road",
        "路面结冰": "icy frozen road surface",
        "车辙印": "vehicle tire rut track marks on road",
        "轮胎痕迹": "tire track marks on asphalt road",
        # 裂缝类型
        "纵向裂缝": "longitudinal crack along road direction",
        "横向裂缝": "transverse crack across road",
        "网状裂缝": "alligator cracking pattern on road",
        "龟裂": "alligator fatigue cracking pattern",
        "轻微裂缝": "minor thin hairline crack on pavement",
        "严重裂缝": "severe wide deep crack with spalling",
        # 坑洼类型
        "小型坑洼": "small shallow pothole on road",
        "大型坑洼": "large deep pothole on asphalt",
        "骨料外露": "exposed aggregate on road surface",
        "浅层凹陷": "minor surface depression on road",
        "严重破损": "severe pothole with broken edges",
        # 通用描述
        "裂缝": "crack on road surface",
        "坑洼": "pothole on road",
        "破损": "road damage distress",
        "路面": "asphalt road surface pavement",
        "严重": "severe serious damage",
        "轻微": "minor light damage",
    }

    def _translate_prompt(self, text: str) -> str:
        """将中文提示翻译为英文，优先查内置词典，否则简单替换"""
        text = text.strip()
        # 先精确匹配
        if text in self._ZH_EN_DICT:
            return self._ZH_EN_DICT[text]
        # 再尝试部分匹配（词典中的中文key是否包含在text中）
        for zh, en in self._ZH_EN_DICT.items():
            if zh in text:
                return en
        # 如果已经是英文（不含中文字符），直接返回
        if not any('\u4e00' <= c <= '\u9fff' for c in text):
            return text
        # 最后的兜底：返回原文，CLIP可能无法理解
        return text

    def encode_text_prompts(self, prompts: list) -> np.ndarray:
        """
        编码任意文本列表为CLIP文本特征向量
        自动将中文翻译为英文（CLIP只理解英文）
        返回 (N, 512) 的numpy数组
        """
        translated = [self._translate_prompt(p) for p in prompts]
        wrapped = [f"a photo of {p}" if not p.startswith("a photo") else p for p in translated]
        tokens = self.tokenizer(wrapped).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    # ── 背景负样本提示（用于对比评分，抑制误检）──
    _BACKGROUND_PROMPTS_EN = [
        "a photo of normal clean undamaged asphalt road surface",
        "a photo of empty road with no damage or defects",
        "a photo of smooth intact pavement in good condition",
        "a photo of clear sky or roadside scenery",
    ]

    # ── 提示增强模板（自动扩充用户输入的描述变体）──
    _ENRICH_TEMPLATES = [
        "a photo of {p} on road surface",
        "a photo of {p} on asphalt pavement",
        "a close-up photo showing {p}",
    ]

    def open_vocab_detect(self, image: np.ndarray, text_prompts: list,
                          patch_sizes=(128, 192, 256, 320), stride_ratio=0.45,
                          threshold=0.35, top_k=20):
        """
        开放词汇检测：纯CLIP实现，无需YOLO
        改进版：多尺度扫描 + 背景对比评分 + 提示增强

        Args:
            image: BGR numpy图像
            text_prompts: 用户输入的文本描述列表（中文或英文）
            patch_sizes: 多尺度窗口大小列表
            stride_ratio: 滑动步长占窗口大小的比例（越小越精细）
            threshold: 对比评分阈值（0~1，默认0.35）
            top_k: 最多返回多少个检测结果

        Returns:
            list of dict, 每个dict包含:
              bbox: [x1, y1, x2, y2]
              label: 匹配的文本
              confidence: 对比评分（0~1）
              patch_size: 使用的窗口大小
        """
        import cv2
        if image is None or image.size == 0 or not text_prompts:
            return []

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ── 1. 编码增强后的正样本提示 ──
        enriched_prompts = []
        prompt_to_orig = []  # 记录每个增强提示对应的原始prompt索引
        for idx, p in enumerate(text_prompts):
            en = self._translate_prompt(p)
            # 用多个描述模板扩充
            for tmpl in self._ENRICH_TEMPLATES:
                enriched_prompts.append(tmpl.format(p=en))
                prompt_to_orig.append(idx)

        # 编码正样本
        pos_tokens = self.tokenizer(enriched_prompts).to(self.device)
        with torch.no_grad():
            pos_features = self.model.encode_text(pos_tokens)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
        pos_features_np = pos_features.cpu().numpy().astype(np.float32)

        # ── 2. 编码背景负样本 ──
        bg_tokens = self.tokenizer(self._BACKGROUND_PROMPTS_EN).to(self.device)
        with torch.no_grad():
            bg_features = self.model.encode_text(bg_tokens)
            bg_features = bg_features / bg_features.norm(dim=-1, keepdim=True)
        bg_features_np = bg_features.cpu().numpy().astype(np.float32)

        # 合并正+负特征用于softmax
        all_text_features = np.concatenate([pos_features_np, bg_features_np], axis=0)
        n_pos = pos_features_np.shape[0]
        n_bg = bg_features_np.shape[0]

        # 获取CLIP的logit_scale用于温度缩放
        logit_scale = self.model.logit_scale.exp().item() if hasattr(self.model, 'logit_scale') else 100.0

        all_detections = []

        # ── 3. 多尺度滑动窗口 ──
        for psize in patch_sizes:
            stride = max(int(psize * stride_ratio), 8)
            patches = []
            coords = []

            for y in range(0, h - psize + 1, stride):
                for x in range(0, w - psize + 1, stride):
                    patch = rgb[y:y+psize, x:x+psize]
                    patches.append(Image.fromarray(patch))
                    coords.append((x, y, x + psize, y + psize))

            if not patches:
                continue

            # 批量编码patches
            batch_size = 32
            all_img_features = []
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                tensors = torch.stack([self.preprocess(p) for p in batch]).to(self.device)
                with torch.no_grad():
                    feats = self.model.encode_image(tensors)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                all_img_features.append(feats.cpu().numpy())

            img_features = np.concatenate(all_img_features, axis=0)  # (N_patches, 512)

            # ── 4. 对比评分：正样本 vs 背景 ──
            # 与所有文本（正+负）做余弦相似度
            cosine_sim = img_features @ all_text_features.T  # (N_patches, n_pos + n_bg)

            # 温度缩放 + softmax
            logits = cosine_sim * logit_scale
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # 正样本概率和 vs 背景概率和
            pos_prob = probs[:, :n_pos].sum(axis=1)       # 属于某个正样本的总概率
            bg_prob = probs[:, n_pos:].sum(axis=1)        # 属于背景的总概率

            # 对比评分 = 正样本概率 / (正样本概率 + 背景概率)
            contrast_score = pos_prob / (pos_prob + bg_prob + 1e-8)

            # 对每个patch，找最匹配的正样本prompt
            pos_probs_only = probs[:, :n_pos]  # (N_patches, n_pos)
            best_enriched_idx = pos_probs_only.argmax(axis=1)  # 最佳增强提示索引
            best_orig_idx = np.array([prompt_to_orig[i] for i in best_enriched_idx])

            for j in range(len(coords)):
                if contrast_score[j] >= threshold:
                    orig_idx = int(best_orig_idx[j])
                    all_detections.append({
                        "bbox": list(coords[j]),
                        "label": text_prompts[orig_idx],
                        "label_index": orig_idx,
                        "confidence": float(contrast_score[j]),
                        "cosine_sim": float(cosine_sim[j, best_enriched_idx[j]]),
                        "patch_size": psize,
                    })

        # 按置信度排序
        all_detections.sort(key=lambda d: -d["confidence"])

        # ── 5. NMS：去掉高度重叠的检测框 ──
        final = []
        for det in all_detections:
            if len(final) >= top_k:
                break
            overlap = False
            for kept in final:
                if self._iou(det["bbox"], kept["bbox"]) > 0.35:
                    overlap = True
                    break
            if not overlap:
                final.append(det)

        return final

    @staticmethod
    def _iou(box1, box2):
        """计算两个bbox的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """在图片上绘制开放词汇检测结果（支持中文标签）"""
        import cv2
        from PIL import ImageDraw, ImageFont

        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        # 加载中文字体
        font = None
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, 14)
                    break
                except Exception:
                    continue
        if font is None:
            try:
                font = ImageFont.truetype("PingFang SC", 14)
            except Exception:
                font = ImageFont.load_default()

        # RGB颜色列表（PIL用RGB，不是BGR）
        colors = [
            (0, 212, 255), (0, 230, 118), (255, 193, 7), (255, 145, 0),
            (255, 23, 68), (224, 64, 251), (124, 77, 255), (255, 110, 64),
        ]

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            color = colors[i % len(colors)]

            # 画矩形框
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

            # 标签文字
            label_text = f"{det['label']} {det['confidence']:.0%}"

            # 计算文字尺寸
            bbox = draw.textbbox((0, 0), label_text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # 标签背景矩形
            label_y = max(0, y1 - th - 10)
            draw.rectangle([(x1, label_y), (x1 + tw + 8, label_y + th + 8)], fill=color)

            # 画文字（黑色）
            draw.text((x1 + 4, label_y + 3), label_text, fill=(0, 0, 0), font=font)

        # PIL RGB -> numpy BGR
        canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return canvas

    @staticmethod
    def _empty_result():
        return {
            "fine_class": "未知",
            "fine_confidence": 0.0,
            "clip_severity": "未知",
            "severity_confidence": 0.0,
            "all_fine_scores": {},
        }


def load_clip_classifier(device=None):
    """便捷函数：加载CLIP分类器"""
    return CLIPClassifier(device=device)
