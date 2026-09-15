# -*- coding: utf-8 -*-
"""视频检测页面 — 上传视频文件逐帧检测"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import cv2
import numpy as np
import tempfile
import zipfile
import io
from streamlit_utils import (
    apply_theme, init_session_state, auto_load_model,
    load_clip_classifier, run_detection, save_record,
    image_to_b64, SEVERITY_EMOJI,
)

# ── 初始化 ──
apply_theme()
init_session_state()

st.markdown("# 🎬 视频检测")
st.markdown("上传视频文件，逐帧进行道路病害检测并汇总结果。")
st.markdown("---")

# ═══════════════════════════════════════════════════
#  侧边栏
# ═══════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎬 视频检测")
    st.caption("逐帧检测 · 结果汇总")

    st.markdown("---")
    st.markdown("### ⚙️ 检测参数")
    conf_threshold = st.slider("置信度", 0.1, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("IoU", 0.1, 0.95, 0.45, 0.05)
    sample_rate = st.slider(
        "采样间隔（每N帧检测一次）", 1, 30, 5,
        help="值越小检测越精细但越慢。1=每帧都检测，5=每5帧检测一次"
    )

    st.markdown("---")
    enable_clip = st.checkbox("🧠 启用CLIP细粒度分类", value=False,
                               help="对每个检测区域进行子类细分（首次加载约30秒，会显著增加处理时间）")

    st.markdown("---")
    st.caption("© 2025 智巡路网 — 道路病害智能检测系统")

# ═══════════════════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════════════════

model, _ = auto_load_model()
if model:
    st.success(f"✅ 模型加载成功 — 类别: {list(model.names.values())}")
else:
    st.warning("⚠️ 未找到模型权重文件")
    st.stop()

# ═══════════════════════════════════════════════════
#  视频上传
# ═══════════════════════════════════════════════════

st.markdown("### 📤 上传视频")
uploaded_video = st.file_uploader(
    "选择视频文件",
    type=["mp4", "avi", "mov", "mkv"],
    help="支持 MP4/AVI/MOV/MKV 格式，建议文件 < 100MB"
)

if uploaded_video is not None:
    # 保存到临时文件
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ 无法打开视频文件")
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    # 显示视频信息
    st.markdown(f"**{uploaded_video.name}**")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.metric("总帧数", f"{total_frames:,}")
    with info_col2:
        st.metric("帧率", f"{fps:.1f} FPS")
    with info_col3:
        st.metric("分辨率", f"{width}×{height}")
    with info_col4:
        st.metric("时长", f"{duration:.1f} 秒")

    frames_to_process = total_frames // sample_rate
    st.caption(f"将检测约 **{frames_to_process}** 帧（采样间隔 {sample_rate}）")

    # ═══════════════════════════════════════════════════
    #  开始检测
    # ═══════════════════════════════════════════════════

    st.markdown("---")
    if st.button("🚀 开始视频检测", type="primary", use_container_width=True):
        # 加载 CLIP（如需要）
        clip = None
        if enable_clip:
            with st.spinner("正在加载CLIP模型..."):
                try:
                    clip = load_clip_classifier()
                except Exception as e:
                    st.warning(f"CLIP加载失败，将跳过细分: {e}")

        # 处理视频
        cap = cv2.VideoCapture(video_path)
        all_detections = []       # 所有帧的检测结果
        annotated_frames = []   # 有检测结果的标注帧 (frame_idx, annotated_bgr, detections)
        total_objects = 0
        class_counts = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                annotated, detections, inference_ms = run_detection(
                    model, frame, conf=conf_threshold, iou=iou_threshold, clip=clip
                )

                if detections:
                    annotated_frames.append({
                        "frame": frame_idx,
                        "time": frame_idx / fps if fps > 0 else 0,
                        "image": annotated,
                        "detections": detections,
                    })
                    total_objects += len(detections)
                    for d in detections:
                        cls = d.get("类别", "unknown")
                        class_counts[cls] = class_counts.get(cls, 0) + 1

                all_detections.append({
                    "frame": frame_idx,
                    "count": len(detections),
                })
                processed += 1

                # 更新进度
                progress = min(processed / max(frames_to_process, 1), 1.0)
                progress_bar.progress(progress)
                status_text.text(
                    f"已处理 {processed}/{frames_to_process} 帧 | "
                    f"当前帧: #{frame_idx} | "
                    f"累计检出: {total_objects}"
                )

            frame_idx += 1

        cap.release()
        progress_bar.progress(1.0)
        status_text.text(f"✅ 处理完成！共处理 {processed} 帧")

        # ═══════════════════════════════════════════════════
        #  结果汇总
        # ═══════════════════════════════════════════════════

        st.markdown("---")
        st.markdown("### 📊 检测汇总")

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("处理帧数", processed)
        with s2:
            st.metric("检出帧数", len(annotated_frames))
        with s3:
            st.metric("检出目标总数", total_objects)
        with s4:
            hit_rate = len(annotated_frames) / max(processed, 1)
            st.metric("检出率", f"{hit_rate:.1%}")

        # 类别分布
        if class_counts:
            st.markdown("#### 类别分布")
            cls_cols = st.columns(len(class_counts))
            for i, (cls, count) in enumerate(sorted(class_counts.items(), key=lambda x: -x[1])):
                with cls_cols[i]:
                    st.metric(cls, count)

        # ═══════════════════════════════════════════════════
        #  标注帧展示（最多显示 20 帧）
        # ═══════════════════════════════════════════════════

        if annotated_frames:
            st.markdown("---")
            show_count = min(len(annotated_frames), 20)
            st.markdown(f"#### 🖼️ 检测帧展示（共 {len(annotated_frames)} 帧有检出，展示前 {show_count} 帧）")

            for i in range(0, show_count, 2):
                cols = st.columns(2)
                for j in range(2):
                    idx = i + j
                    if idx >= show_count:
                        break
                    af = annotated_frames[idx]
                    with cols[j]:
                        img_rgb = cv2.cvtColor(af["image"], cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, use_container_width=True)
                        det_count = len(af["detections"])
                        st.caption(
                            f"**帧 #{af['frame']}** | "
                            f"时间: {af['time']:.1f}s | "
                            f"检出: {det_count} 个目标"
                        )

            # 保存最佳帧到会话历史
            best_frame = max(annotated_frames, key=lambda x: len(x["detections"]))
            img_b64 = image_to_b64(best_frame["image"])
            rec_id = save_record("视频检测", best_frame["detections"], img_b64)
            st.caption(f"最佳检出帧已保存至会话历史: {rec_id}")

            # ═══════════════════════════════════════════════════
            #  导出：所有标注帧打包为 ZIP
            # ═══════════════════════════════════════════════════

            st.markdown("---")
            st.markdown("### 📥 导出")

            dl1, dl2 = st.columns(2)
            with dl1:
                # 打包所有标注帧为 ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for af in annotated_frames:
                        _, buf = cv2.imencode(".jpg", af["image"])
                        zf.writestr(
                            f"frame_{af['frame']:06d}_{len(af['detections'])}det.jpg",
                            buf.tobytes(),
                        )
                zip_buffer.seek(0)
                st.download_button(
                    f"📦 下载全部标注帧（{len(annotated_frames)}张 ZIP）",
                    data=zip_buffer.getvalue(),
                    file_name=f"video_detections_{uploaded_video.name}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

            with dl2:
                # 导出检测统计 CSV
                csv_lines = ["帧号,时间(秒),检出数,类别,置信度,严重等级"]
                for af in annotated_frames:
                    for d in af["detections"]:
                        conf_str = f"{d['置信度']:.4f}" if isinstance(d.get("置信度"), (int, float)) else str(d.get("置信度", ""))
                        csv_lines.append(
                            f"{af['frame']},{af['time']:.2f},{len(af['detections'])},"
                            f"{d.get('类别', '')},{conf_str},{d.get('严重等级', '')}"
                        )
                csv_bytes = "\n".join(csv_lines).encode("utf-8-sig")
                st.download_button(
                    "📥 下载检测统计 CSV",
                    data=csv_bytes,
                    file_name=f"video_report_{uploaded_video.name}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        else:
            st.info("未在视频帧中检测到道路病害目标")

    # 清理临时文件
    try:
        os.unlink(video_path)
    except Exception:
        pass

# ═══════════════════════════════════════════════════
#  底部说明
# ═══════════════════════════════════════════════════

st.markdown("---")
with st.expander("📖 视频检测说明"):
    st.markdown("""
    **视频检测** 对上传的视频文件进行逐帧采样检测：

    1. **采样间隔：** 可设置每 N 帧检测一次（默认每5帧），降低计算量同时保持检测覆盖率
    2. **逐帧推理：** 对每个采样帧执行 YOLO 目标检测，可选启用 CLIP 细粒度分类
    3. **结果汇总：** 统计总检出数、检出率、类别分布等指标
    4. **标注帧导出：** 将所有检出目标的帧标注后打包为 ZIP 下载
    5. **统计导出：** 每帧的检测详情导出为 CSV 文件

    **建议：** 视频文件建议 < 100MB，采样间隔设为 5-10 可在速度和质量间取得平衡。
    """)
