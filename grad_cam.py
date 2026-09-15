# -*- coding: utf-8 -*-
"""Grad-CAM 热力图可视化工具 — 用于YOLO模型的可解释性分析

方案：
  1. deepcopy nn.Module 隔离 inference_mode
  2. monkey-patch 目标层 forward → 输出 .clone() 替换原始输出，
     使计算图直接经过 clone tensor，解决 inplace 冲突
  3. torch.autograd.grad() 计算梯度
"""

import copy
import types
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMExtractor:

    def __init__(self, model):
        self.model = model
        self._target_layer_ref, self._target_idx = self._find_target_layer()

    def _find_target_layer(self):
        net = self.model.model
        convs = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            raise RuntimeError("未找到卷积层")

        layer_info = {}

        def _mk(tid):
            def hook(mod, inp, out):
                if isinstance(out, torch.Tensor) and out.dim() >= 4:
                    layer_info[tid] = {
                        "c": out.shape[1],
                        "h": out.shape[2],
                        "w": out.shape[3],
                    }
            return hook

        handles = [c.register_forward_hook(_mk(i)) for i, c in enumerate(convs)]
        try:
            dev = next(net.parameters()).device
            dummy = torch.zeros(1, 3, 640, 640, device=dev)
            net.eval()
            with torch.no_grad():
                net(dummy)
        except Exception:
            pass
        finally:
            for h in handles:
                h.remove()

        for i in range(len(convs) - 1, -1, -1):
            if i in layer_info:
                info = layer_info[i]
                if info["h"] >= 8 and info["w"] >= 8 and info["c"] >= 64:
                    print(f"[GradCAM] 选择第 {i}/{len(convs)} 个Conv2d, "
                          f"通道={info['c']}, 空间={info['h']}x{info['w']}")
                    return convs[i], i

        for i in range(len(convs) - 1, -1, -1):
            if i in layer_info:
                info = layer_info[i]
                if info["h"] >= 8 and info["w"] >= 8 and info["c"] >= 16:
                    print(f"[GradCAM] 兜底层 {i}/{len(convs)}, "
                          f"通道={info['c']}, 空间={info['h']}x{info['w']}")
                    return convs[i], i

        idx = max(0, len(convs) - 3)
        print(f"[GradCAM] 最终兜底层 {idx}")
        return convs[idx], idx

    def generate(self, image, target_class=None):
        img_h, img_w = image.shape[:2]
        print(f"[GradCAM] ===== 开始生成, 图片 {img_w}x{img_h} =====")

        try:
            net = copy.deepcopy(self.model.model)
            net.eval()

            convs = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
            if self._target_idx >= len(convs):
                return self._fail("Conv2d数量不匹配")
            target_layer = convs[self._target_idx]

            # ── monkey-patch forward：输出 clone 替换原始输出 ──
            saved_activation = [None]
            original_forward = target_layer.forward

            def patched_forward(self_layer, x):
                result = original_forward(x)
                # clone 后计算图经过此 tensor，autograd.grad 可达
                cloned = result.clone()
                saved_activation[0] = cloned
                return cloned

            target_layer.forward = types.MethodType(patched_forward, target_layer)

            try:
                img_tensor = self._preprocess(image)
                device = next(net.parameters()).device
                img_tensor = img_tensor.to(device).requires_grad_(True)

                with torch.enable_grad():
                    preds = net.forward(img_tensor)
                    if isinstance(preds, (list, tuple)):
                        preds = preds[0]

                    print(f"[GradCAM] preds shape: {preds.shape}")

                    if preds.dim() == 3 and preds.shape[1] < preds.shape[2]:
                        pass
                    elif preds.dim() == 3:
                        preds = preds.permute(0, 2, 1)

                    num_classes = preds.shape[1] - 4

                    if target_class is None:
                        all_cls = preds[:, 4:, :]
                        score_val, flat_idx = all_cls.max()
                        tc = flat_idx.item() % num_classes
                        print(f"[GradCAM] 自动类别: {tc}, score={score_val.item():.4f}")
                    else:
                        tc = min(target_class, num_classes - 1)

                    cls_scores = preds[:, 4 + tc, :]
                    score = cls_scores.max()
                    print(f"[GradCAM] 目标得分: {score.item():.4f}")

                if saved_activation[0] is None:
                    return self._fail("patched forward 未捕获激活值")

                act = saved_activation[0]
                print(f"[GradCAM] activation shape: {act.shape}, requires_grad: {act.requires_grad}")

                grads = torch.autograd.grad(
                    score, act,
                    retain_graph=False,
                    allow_unused=False,
                )

                grad = grads[0]
                print(f"[GradCAM] gradient shape: {grad.shape}")

                # Grad-CAM
                act_b = act[0]
                grad_b = grad[0]

                weights = grad_b.mean(dim=(1, 2))
                cam = (weights.unsqueeze(-1).unsqueeze(-1) * act_b).sum(dim=0)
                cam = F.relu(cam)

                if cam.dim() != 2:
                    cam = cam.squeeze()
                if cam.dim() != 2:
                    return self._fail(f"CAM 维度异常: {cam.shape}")

                cam_max = cam.max().item()
                print(f"[GradCAM] cam max={cam_max:.4f}, min={cam.min().item():.4f}")

                if cam_max <= 1e-8:
                    return self._fail("CAM 全零")

                cam = cam / cam_max
                cam_np = cam.detach().cpu().numpy().astype(np.float32)

                cam_resized = cv2.resize(cam_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

                print(f"[GradCAM] ===== 生成成功 =====")
                return {"heatmap": heatmap, "overlay": overlay, "success": True}

            finally:
                target_layer.forward = original_forward
                del net

        except Exception as e:
            import traceback
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[GradCAM] 错误: {err_msg}")
            traceback.print_exc()
            return {"heatmap": None, "overlay": None, "success": False, "error": err_msg}

    def _fail(self, msg):
        print(f"[GradCAM] 失败: {msg}")
        return {"heatmap": None, "overlay": None, "success": False, "error": msg}

    def _preprocess(self, image):
        imgsz = 640
        try:
            val = self.model.args.get("imgsz", 640)
            if isinstance(val, (list, tuple)):
                imgsz = val[0]
            else:
                imgsz = int(val)
        except Exception:
            pass
        resized = cv2.resize(image, (imgsz, imgsz))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)
