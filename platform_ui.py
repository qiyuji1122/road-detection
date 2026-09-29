# -*- coding: utf-8 -*-
"""Streamlit platform pages for the autonomous road inspection console."""

from __future__ import annotations

import io
import os
import platform
import re
import time
import zipfile
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from streamlit_utils import (
    auto_load_model,
    b64_to_image,
    export_state_backup,
    export_csv,
    generate_report,
    get_records,
    get_storage_info,
    get_statistics,
    image_to_b64,
    load_clip_classifier,
    persist_session_state,
    restore_state_backup,
    run_detection,
    save_record,
)


APP_DIR = os.path.dirname(os.path.abspath(__file__))

NAV_ITEMS = [
    ("数据总览", "⌂"),
    ("检测中心", "⌁"),
    ("实时检测", "◉"),
    ("CLIP 创新中心", "✦"),
    ("检测任务", "✓"),
    ("往期数据", "◷"),
    ("病害档案", "▤"),
    ("数据看板", "▥"),
    ("模型与实验", "◇"),
    ("报告中心", "▣"),
    ("系统管理", "⚙"),
]


def page_header(eyebrow: str, title: str, description: str, status: str | None = None):
    status_html = ""
    if status:
        status_html = (
            '<span class="status-pill"><span class="status-dot"></span>'
            f"{status}</span>"
        )
    st.markdown(
        f"""
        <div style="margin-bottom:1.35rem">
          <div class="platform-eyebrow">{eyebrow}</div>
          <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
            <h1 style="margin:0">{title}</h1>{status_html}
          </div>
          <p class="platform-subtitle">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(icon: str, title: str, body: str):
    st.markdown(
        f'<div class="surface-card"><div style="font-size:1.45rem;margin-bottom:14px">{icon}</div>'
        f"<h4>{title}</h4><p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def empty_panel(title: str, body: str):
    st.markdown(
        f'<div class="empty-panel"><div style="font-size:1.8rem;margin-bottom:8px">◌</div>'
        f"<b style='color:#dceaf5'>{title}</b><br><span>{body}</span></div>",
        unsafe_allow_html=True,
    )


def _settings() -> dict:
    return st.session_state["system_settings"]


def _read_image(uploaded_file):
    if uploaded_file is None:
        return None
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _next_id(prefix: str, collection: list) -> str:
    return f"{prefix}-{datetime.now():%m%d}-{len(collection) + 1:03d}"


def _register_task(name: str, source: str, status: str = "已完成", result_id: str = ""):
    tasks = st.session_state["tasks"]
    tasks.insert(
        0,
        {
            "任务编号": _next_id("TASK", tasks),
            "任务名称": name,
            "来源": source,
            "状态": status,
            "进度": 100 if status == "已完成" else 0,
            "创建时间": datetime.now().strftime("%m-%d %H:%M"),
            "结果记录": result_id,
        },
    )
    persist_session_state()


def _register_archive(record_id: str, detections: list, source: str):
    archive = st.session_state["archives"]
    record = next((item for item in get_records() if item.get("id") == record_id), {})
    detected_at = record.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    existing = {(item.get("记录编号"), item.get("目标序号")) for item in archive}
    for index, det in enumerate(detections, 1):
        key = (record_id, index)
        if key in existing:
            continue
        archive.insert(
            0,
            {
                "档案编号": _next_id("CASE", archive),
                "记录编号": record_id,
                "目标序号": index,
                "来源": source,
                "检测时间": detected_at,
                "原始分类": det.get("类别", "unknown"),
                "当前分类": det.get("CLIP细分") or det.get("类别", "unknown"),
                "置信度": float(det.get("置信度", 0) or 0),
                "严重程度": det.get("严重等级", "待复核"),
                "复核状态": "待复核",
                "修正说明": "",
                "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
        )
    persist_session_state()


def _save_detection_result(source: str, file_name: str, annotated, detections: list) -> str:
    record_id = save_record(source, detections, image_to_b64(annotated))
    _register_task(file_name, source, "已完成", record_id)
    _register_archive(record_id, detections, source)
    return record_id


def _model_state():
    model, model_path = auto_load_model()
    return model, model_path


def _render_detection_metrics(detections: list, inference_ms: float):
    severe = sum(d.get("严重等级") == "严重" for d in detections)
    avg_conf = np.mean([d.get("置信度", 0) for d in detections]) if detections else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("检出目标", len(detections))
    c2.metric("平均置信度", f"{avg_conf:.1%}" if detections else "—")
    c3.metric("高风险目标", severe)
    c4.metric("推理耗时", f"{inference_ms:.0f} ms")


def _render_image_result(image, annotated, detections, inference_ms, key: str):
    _render_detection_metrics(detections, inference_ms)
    left, right = st.columns(2)
    left.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="原始影像", use_container_width=True)
    right.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="检测标注", use_container_width=True)
    if detections:
        rows = []
        for i, det in enumerate(detections, 1):
            rows.append(
                {
                    "序号": i,
                    "基础类别": det.get("类别", ""),
                    "CLIP 细分类": det.get("CLIP细分", "—"),
                    "置信度": f"{det.get('置信度', 0):.1%}",
                    "严重程度": det.get("严重等级", ""),
                    "位置": det.get("位置", ""),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("当前阈值下未检出道路病害，可适当降低置信度后重试。")
    _, encoded = cv2.imencode(".jpg", annotated)
    st.download_button(
        "下载标注结果",
        encoded.tobytes(),
        file_name=f"inspection_{key}.jpg",
        mime="image/jpeg",
        key=f"download_{key}",
    )


def render_overview():
    stats = get_statistics()
    model, _ = _model_state()
    left, right = st.columns([1.12, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="hero-copy">
              <span class="status-pill"><span class="status-dot"></span>AI 巡检节点在线</span>
              <div class="platform-eyebrow" style="margin-top:1.2rem">AUTONOMOUS ROAD INSPECTION OS</div>
              <h1>把每一公里道路，变成可检索、可复核、可追溯的数据。</h1>
              <p class="lead">智巡路网连接车载视觉、YOLOv11 与 CLIP 语义能力，从发现病害到生成报告形成完整闭环。</p>
              <div class="hero-points">● 多源接入 &nbsp;&nbsp; ● 细粒度语义识别 &nbsp;&nbsp; ● 全流程任务追踪</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        q1, q2 = st.columns(2)
        if q1.button("开始检测", type="primary", use_container_width=True):
            st.session_state["nav_page"] = "检测中心"
            st.rerun()
        if q2.button("进入 CLIP 创新中心", use_container_width=True):
            st.session_state["nav_page"] = "CLIP 创新中心"
            st.rerun()
    with right:
        # Keep this side intentionally open so the autonomous inspection vehicle
        # remains visible in the full-page background artwork.
        st.markdown('<div aria-hidden="true" style="min-height:385px"></div>', unsafe_allow_html=True)

    st.markdown("### 今日运行概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("检测任务", stats["total_records"], "+ 实时会话")
    c2.metric("病害目标", stats["total_detections"], f"今日 {stats['today_count']}")
    c3.metric("平均置信度", f"{stats['avg_confidence']:.1%}" if stats["avg_confidence"] else "—")
    c4.metric("模型状态", "在线" if model else "待配置", "YOLOv11")

    st.markdown("### 平台能力")
    cols = st.columns(4)
    cards = [
        ("⌁", "多模态检测", "支持图片、批量图片、视频和摄像头输入，复用现有 YOLO 推理链路。"),
        ("✦", "CLIP 语义中枢", "病害细分类、文字查找目标、相似案例检索集中在同一工作台。"),
        ("▤", "人工复核闭环", "检测结果进入档案后可修正类别、标记复核状态并保留来源。"),
        ("▣", "报告与治理", "任务、看板、模型实验、报告与系统状态采用统一信息架构。"),
    ]
    for col, card in zip(cols, cards):
        with col:
            info_card(*card)


def render_detection_center():
    page_header("DETECTION HUB", "检测中心", "面向单张、批量与视频的统一检测入口；沿用当前 YOLOv11 权重与结果格式。", "YOLOv11 已连接")
    model, _ = _model_state()
    if model is None:
        st.error("未找到 best.pt，检测功能暂不可用。CLIP 独立能力仍可在创新中心使用。")
        return
    cfg = _settings()
    single, batch, video = st.tabs(["单张图片", "批量图片", "上传视频"])

    with single:
        uploaded = st.file_uploader("上传一张道路影像", type=["jpg", "jpeg", "png", "bmp"], key="single_image")
        image = _read_image(uploaded)
        if image is not None:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=520, caption=f"待检测 · {uploaded.name}")
        if st.button("开始单图检测", type="primary", disabled=image is None, key="single_detect"):
            clip = None
            if cfg["enable_clip"]:
                try:
                    with st.spinner("加载 CLIP 细分类模型…"):
                        clip = load_clip_classifier()
                except Exception as exc:
                    st.warning(f"CLIP 暂不可用，继续使用 YOLO 检测：{exc}")
            with st.spinner("正在分析道路影像…"):
                annotated, detections, latency = run_detection(model, image, cfg["confidence"], cfg["iou"], clip)
            rec_id = _save_detection_result("单张图片", uploaded.name, annotated, detections)
            st.success(f"检测完成，结果已归档为 {rec_id}")
            _render_image_result(image, annotated, detections, latency, "single")

    with batch:
        files = st.file_uploader(
            "上传多张道路影像", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True, key="batch_images"
        )
        st.caption("批次内每张图片独立生成任务和档案，可在“检测任务”与“病害档案”继续处理。")
        if st.button("启动批量检测", type="primary", disabled=not files, key="batch_detect"):
            progress = st.progress(0)
            manifest, bundle = [], io.BytesIO()
            with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
                for index, uploaded in enumerate(files):
                    image = _read_image(uploaded)
                    if image is None:
                        _register_task(uploaded.name, "批量图片", "失败")
                        continue
                    annotated, detections, latency = run_detection(model, image, cfg["confidence"], cfg["iou"])
                    rec_id = _save_detection_result("批量图片", uploaded.name, annotated, detections)
                    _, encoded = cv2.imencode(".jpg", annotated)
                    archive.writestr(f"{index + 1:03d}_{uploaded.name}.jpg", encoded.tobytes())
                    manifest.append({"文件": uploaded.name, "记录": rec_id, "目标数": len(detections), "耗时ms": round(latency)})
                    progress.progress((index + 1) / len(files))
            st.success(f"批次完成，共处理 {len(manifest)} 张影像。")
            st.dataframe(manifest, use_container_width=True, hide_index=True)
            st.download_button("下载批量标注结果 ZIP", bundle.getvalue(), "batch_detection_results.zip", "application/zip")

    with video:
        uploaded_video = st.file_uploader("上传巡检视频", type=["mp4", "avi", "mov", "mkv"], key="video_upload")
        sample_rate = st.slider("抽帧间隔", 1, 30, 8, help="每隔多少帧执行一次检测；数值越小越精细。")
        if uploaded_video:
            st.video(uploaded_video)
        if st.button("开始视频检测", type="primary", disabled=uploaded_video is None, key="video_detect"):
            import tempfile

            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_video.getvalue())
                    tmp_path = tmp.name
                cap = cv2.VideoCapture(tmp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                sampled, hits, summary = 0, 0, []
                previews = []
                progress = st.progress(0)
                frame_no = 0
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_no % sample_rate == 0:
                        annotated, detections, latency = run_detection(model, frame, cfg["confidence"], cfg["iou"])
                        sampled += 1
                        hits += len(detections)
                        summary.append({"帧": frame_no, "时间(秒)": round(frame_no / fps, 2), "目标数": len(detections), "耗时ms": round(latency)})
                        if detections and len(previews) < 6:
                            previews.append((frame_no, annotated, detections))
                    frame_no += 1
                    if frame_no % 20 == 0:
                        progress.progress(min(frame_no / total_frames, 1.0))
                cap.release()
                progress.progress(1.0)
                combined = [det for _, _, detections in previews for det in detections]
                if previews:
                    record_id = _save_detection_result("视频检测", uploaded_video.name, previews[0][1], combined)
                else:
                    record_id = save_record("视频检测", [], None)
                    _register_task(uploaded_video.name, "视频检测", "已完成", record_id)
                c1, c2, c3 = st.columns(3)
                c1.metric("采样帧", sampled)
                c2.metric("病害目标", hits)
                c3.metric("命中帧", sum(row["目标数"] > 0 for row in summary))
                for idx, (frame_id, image, detections) in enumerate(previews[:4]):
                    if idx % 2 == 0:
                        cols = st.columns(2)
                    cols[idx % 2].image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"帧 {frame_id} · {len(detections)} 个目标", use_container_width=True)
                report_bytes = pd.DataFrame(summary).to_csv(index=False).encode("utf-8-sig")
                st.download_button("下载视频检测明细", report_bytes, "video_detection.csv", "text/csv")
            except Exception as exc:
                _register_task(uploaded_video.name, "视频检测", "失败")
                st.error(f"视频处理失败：{exc}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)


def render_realtime():
    page_header("LIVE VISION", "实时检测", "接入电脑、手机浏览器摄像头或网络摄像头，抓取画面后进入同一检测与归档流程。", "等待视频源")
    model, _ = _model_state()
    if model is None:
        st.error("实时检测需要 best.pt 模型权重。")
        return
    cfg = _settings()
    local_tab, network_tab = st.tabs(["电脑 / 手机摄像头", "网络摄像头"])
    with local_tab:
        st.markdown("#### 浏览器摄像头")
        st.caption("在电脑或手机浏览器打开本站，允许一次摄像头权限即可拍摄当前路况。")
        snapshot = st.camera_input("拍摄道路画面", key="camera_capture")
        image = _read_image(snapshot)
        if st.button("检测当前画面", type="primary", disabled=image is None, key="camera_detect"):
            annotated, detections, latency = run_detection(model, image, cfg["confidence"], cfg["iou"])
            rec_id = _save_detection_result("实时摄像头", "camera_snapshot.jpg", annotated, detections)
            st.success(f"实时快照已进入档案 {rec_id}")
            _render_image_result(image, annotated, detections, latency, "camera")
    with network_tab:
        st.markdown("#### RTSP / HTTP 视频源")
        source = st.text_input("网络摄像头地址", placeholder="rtsp://camera.example/live 或 https://…/frame.jpg", type="password")
        st.caption("地址仅用于当前会话抓取一帧，不写入检测记录。请在可信网络内使用。")
        if st.button("连接并抓取一帧", type="primary", disabled=not source, key="network_capture"):
            with st.spinner("正在连接视频源…"):
                cap = cv2.VideoCapture(source)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
                ok, image = cap.read()
                cap.release()
            if not ok or image is None:
                st.error("未能读取视频源，请检查地址、网络和摄像头权限。")
            else:
                annotated, detections, latency = run_detection(model, image, cfg["confidence"], cfg["iou"])
                rec_id = _save_detection_result("网络摄像头", "network_camera", annotated, detections)
                st.success(f"网络画面已完成检测，记录 {rec_id}")
                _render_image_result(image, annotated, detections, latency, "network")


def _bbox_from_position(position: str):
    nums = [int(float(x)) for x in re.findall(r"-?\d+(?:\.\d+)?", position or "")]
    return nums[:4] if len(nums) >= 4 else None


def render_clip_center():
    page_header("SEMANTIC INTELLIGENCE", "CLIP 创新中心", "围绕道路病害构建视觉—语言工作流：细粒度分类、文字查找目标与相似案例检索。", "CLIP 零样本能力")
    fine_tab, text_tab, similar_tab = st.tabs(["病害细分类", "文字查找目标", "相似案例检索"])

    with fine_tab:
        st.markdown("#### YOLO 定位 + CLIP 细分")
        st.caption("先用 YOLO 定位裂缝/坑洞，再由 CLIP 判断纵向裂缝、网状裂缝、小型坑洼等子类。")
        uploaded = st.file_uploader("上传待细分图片", type=["jpg", "jpeg", "png", "bmp"], key="clip_fine_upload")
        image = _read_image(uploaded)
        if st.button("运行细粒度识别", type="primary", disabled=image is None, key="clip_fine_run"):
            model, _ = _model_state()
            if model is None:
                st.error("细分类流程需要 YOLO 权重先定位目标。")
            else:
                try:
                    with st.spinner("正在加载并运行 CLIP…"):
                        clip = load_clip_classifier()
                        annotated, detections, latency = run_detection(model, image, _settings()["confidence"], _settings()["iou"], clip)
                    rec_id = _save_detection_result("CLIP 病害细分类", uploaded.name, annotated, detections)
                    cases = st.session_state["clip_cases"]
                    for idx, det in enumerate(detections, 1):
                        bbox = _bbox_from_position(det.get("位置"))
                        if not bbox:
                            continue
                        x1, y1, x2, y2 = bbox
                        crop = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
                        if crop.size == 0:
                            continue
                        feature = clip.encode_image_np(crop)
                        cases.append({
                            "case_id": f"{rec_id}-{idx:02d}",
                            "feature": feature,
                            "image_b64": image_to_b64(crop),
                            "class": det.get("CLIP细分") or det.get("类别"),
                            "severity": det.get("严重等级"),
                            "source": uploaded.name,
                        })
                    persist_session_state()
                    st.success(f"细分类完成，{len(detections)} 个目标已进入相似案例索引。")
                    _render_image_result(image, annotated, detections, latency, "clip_fine")
                except Exception as exc:
                    st.error(f"CLIP 细分类暂不可用：{exc}")

    with text_tab:
        st.markdown("#### 用自然语言在图片中找目标")
        uploaded = st.file_uploader("上传搜索图片", type=["jpg", "jpeg", "png", "bmp"], key="clip_text_upload")
        image = _read_image(uploaded)
        prompt = st.text_input("目标描述", placeholder="例如：井盖周围破损，路面积水，纵向裂缝", key="clip_text_prompt")
        threshold = st.slider("语义匹配阈值", 0.10, 0.90, 0.38, 0.05, key="clip_text_threshold")
        if st.button("开始文字查找", type="primary", disabled=image is None or not prompt.strip(), key="clip_text_run"):
            prompts = [p.strip() for p in re.split(r"[,，、]", prompt) if p.strip()]
            try:
                with st.spinner("正在进行多尺度语义搜索…"):
                    clip = load_clip_classifier()
                    results = clip.open_vocab_detect(image, prompts, threshold=threshold)
                annotated = clip.draw_detections(image, results)
                c1, c2, c3 = st.columns(3)
                c1.metric("匹配区域", len(results))
                c2.metric("覆盖描述", len({r["label"] for r in results}))
                c3.metric("最高匹配度", f"{max((r['confidence'] for r in results), default=0):.1%}")
                left, right = st.columns(2)
                left.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="原始影像", use_container_width=True)
                right.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="语义定位结果", use_container_width=True)
                if results:
                    st.dataframe([
                        {"描述": r["label"], "匹配度": f"{r['confidence']:.1%}", "窗口": r["patch_size"], "位置": r["bbox"]}
                        for r in results
                    ], use_container_width=True, hide_index=True)
                    record_detections = [
                        {"类别": r["label"], "置信度": r["confidence"], "位置": str(tuple(r["bbox"])), "面积占比": 0, "严重等级": "待复核"}
                        for r in results
                    ]
                    _save_detection_result("CLIP 文字查找", uploaded.name, annotated, record_detections)
                else:
                    st.info("未找到匹配区域，可降低阈值或使用更具体的描述。")
            except Exception as exc:
                st.error(f"文字查找暂不可用：{exc}")

    with similar_tab:
        cases = st.session_state["clip_cases"]
        st.markdown("#### 以图搜图 / 以文搜案例")
        st.caption(f"当前会话已建立 {len(cases)} 条 CLIP 病害特征索引。先在“病害细分类”完成一次识别即可加入案例。")
        mode = st.radio("查询方式", ["上传参考图片", "输入文字描述"], horizontal=True)
        query_image, query_text = None, ""
        if mode == "上传参考图片":
            query_file = st.file_uploader("上传参考病害图片", type=["jpg", "jpeg", "png"], key="similar_image")
            query_image = _read_image(query_file)
        else:
            query_text = st.text_input("描述希望寻找的病害", placeholder="严重网状裂缝", key="similar_text")
        can_search = bool(cases) and (query_image is not None or bool(query_text.strip()))
        if st.button("检索相似案例", type="primary", disabled=not can_search, key="similar_run"):
            try:
                clip = load_clip_classifier()
                query = clip.encode_image_np(query_image) if query_image is not None else clip.encode_text_prompts([query_text])[0]
                scored = []
                for case in cases:
                    score = float(np.dot(query, case["feature"]) / (np.linalg.norm(query) * np.linalg.norm(case["feature"]) + 1e-8))
                    scored.append((score, case))
                scored.sort(key=lambda item: item[0], reverse=True)
                cols = st.columns(min(3, len(scored)))
                for idx, (score, case) in enumerate(scored[:6]):
                    with cols[idx % len(cols)]:
                        image = b64_to_image(case["image_b64"])
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.markdown(f"**{case['class']}** · 相似度 {score:.1%}")
                        st.caption(f"{case['case_id']} · {case['severity']} · {case['source']}")
            except Exception as exc:
                st.error(f"相似检索暂不可用：{exc}")
        elif not cases:
            empty_panel("案例索引尚为空", "完成一次 CLIP 病害细分类后，系统会自动保存目标特征用于相似检索。")


def render_tasks():
    page_header("TASK ORCHESTRATION", "检测任务", "集中查看任务进度、失败原因、重试状态与结果下载。", "队列运行正常")
    tasks = st.session_state["tasks"]
    if not tasks:
        empty_panel("暂无检测任务", "从检测中心或实时检测发起任务后，进度和结果会显示在这里。")
        return
    status_filter = st.segmented_control("状态筛选", ["全部", "已完成", "失败", "处理中"], default="全部")
    shown = tasks if status_filter == "全部" else [t for t in tasks if t["状态"] == status_filter]
    c1, c2, c3 = st.columns(3)
    c1.metric("任务总数", len(tasks))
    c2.metric("完成率", f"{sum(t['状态'] == '已完成' for t in tasks) / len(tasks):.0%}")
    c3.metric("失败待重试", sum(t["状态"] == "失败" for t in tasks))
    st.dataframe(shown, use_container_width=True, hide_index=True)
    failed = [t for t in tasks if t["状态"] == "失败"]
    if failed:
        selected = st.selectbox("选择失败任务", [t["任务编号"] for t in failed])
        if st.button("重试所选任务", type="primary"):
            task = next(t for t in tasks if t["任务编号"] == selected)
            task.update({"状态": "处理中", "进度": 5})
            persist_session_state()
            st.info("任务已重新进入队列。请回到原始输入页面重新提供检测源。")
    records = get_records()
    if records:
        st.download_button("下载全部任务结果 CSV", export_csv(records), "inspection_task_results.csv", "text/csv")


def _backfill_archive():
    if st.session_state["archives"]:
        return
    for record in get_records():
        _register_archive(record.get("id", "REC"), record.get("detections", []), record.get("source", "历史记录"))


def render_history():
    records = sorted(get_records(), key=lambda item: item.get("timestamp", ""), reverse=True)
    page_header("HISTORICAL DATA", "往期数据", "按日期保留每天的检测结果，支持追溯、筛选、查看明细与完整备份。", "自动持久化")
    storage = get_storage_info()
    if storage["error"]:
        st.error(f"历史数据写入异常：{storage['error']}")
    else:
        updated = storage["updated_at"].replace("T", " ") if storage["updated_at"] else "尚未产生记录"
        st.caption(f"数据会在每次检测、任务变更和人工复核后自动保存 · 最近保存：{updated}")

    if not records:
        empty_panel("暂无往期数据", "完成一次图片、视频、实时或 CLIP 检测后，记录会按日期自动归档，并在刷新后恢复。")
        st.download_button("下载空白数据备份", export_state_backup(), "road_inspection_backup.json", "application/json")
        backup = st.file_uploader("恢复历史备份", type=["json"], key="history_restore_empty")
        if st.button("导入备份", disabled=backup is None, key="history_restore_empty_button"):
            try:
                restore_state_backup(backup.getvalue())
                st.success("历史备份已恢复。")
                st.rerun()
            except Exception as exc:
                st.error(f"备份恢复失败：{exc}")
        return

    record_days = [item.get("timestamp", "")[:10] for item in records if item.get("timestamp")]
    parsed_days = [datetime.strptime(day, "%Y-%m-%d").date() for day in record_days]
    min_day, max_day = min(parsed_days), max(parsed_days)
    filters = st.columns([1, 1, 1.15, 1.4])
    date_from = filters[0].date_input("开始日期", value=min_day, min_value=min_day, max_value=max_day, key="history_from")
    date_to = filters[1].date_input("结束日期", value=max_day, min_value=min_day, max_value=max_day, key="history_to")
    all_sources = sorted({item.get("source", "未知来源") for item in records})
    sources = filters[2].multiselect("检测来源", all_sources, default=all_sources, key="history_sources")
    keyword = filters[3].text_input("搜索", placeholder="记录编号 / 病害类别 / 严重程度", key="history_keyword").strip().lower()

    filtered = []
    for record in records:
        day = record.get("timestamp", "")[:10]
        if not day or not (date_from.isoformat() <= day <= date_to.isoformat()):
            continue
        if sources and record.get("source", "未知来源") not in sources:
            continue
        searchable = " ".join([
            record.get("id", ""), record.get("source", ""),
            *[str(value) for det in record.get("detections", []) for value in det.values()],
        ]).lower()
        if keyword and keyword not in searchable:
            continue
        filtered.append(record)

    selected_ids = {record.get("id") for record in filtered}
    related_archives = [item for item in st.session_state["archives"] if item.get("记录编号") in selected_ids]
    total_targets = sum(item.get("detection_count", len(item.get("detections", []))) for item in filtered)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("历史批次", len(filtered))
    c2.metric("病害目标", total_targets)
    c3.metric("覆盖天数", len({item.get("timestamp", "")[:10] for item in filtered}))
    c4.metric("已复核档案", sum(item.get("复核状态") == "已复核" for item in related_archives))

    daily = {}
    for record in filtered:
        day = record.get("timestamp", "")[:10]
        bucket = daily.setdefault(day, {"日期": day, "检测批次": 0, "病害目标": 0, "来源": set()})
        bucket["检测批次"] += 1
        bucket["病害目标"] += record.get("detection_count", len(record.get("detections", [])))
        bucket["来源"].add(record.get("source", "未知来源"))
    daily_rows = [{**item, "来源": "、".join(sorted(item["来源"]))} for _, item in sorted(daily.items(), reverse=True)]
    st.markdown("#### 每日归档")
    st.dataframe(daily_rows, use_container_width=True, hide_index=True)

    record_rows = []
    for record in filtered:
        detections = record.get("detections", [])
        classes = sorted({det.get("CLIP细分") or det.get("类别", "unknown") for det in detections})
        severe = sum(det.get("严重等级") == "严重" for det in detections)
        record_rows.append({
            "记录编号": record.get("id", ""),
            "检测时间": record.get("timestamp", ""),
            "来源": record.get("source", ""),
            "目标数": record.get("detection_count", len(detections)),
            "病害类型": "、".join(classes) if classes else "未检出",
            "严重目标": severe,
        })
    st.markdown("#### 检测记录")
    st.dataframe(record_rows, use_container_width=True, hide_index=True)

    if filtered:
        selected = st.selectbox("查看记录明细", [item.get("id") for item in filtered], key="history_selected")
        record = next(item for item in filtered if item.get("id") == selected)
        left, right = st.columns([1, 1.25])
        with left:
            if record.get("image_b64"):
                try:
                    image = b64_to_image(record["image_b64"])
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"{selected} 检测结果", use_container_width=True)
                except Exception:
                    st.info("该记录的结果图片暂不可读取，结构化检测数据仍然完整。")
            else:
                empty_panel("无结果图片", "视频汇总或旧记录可能只保存结构化明细。")
        with right:
            st.markdown(f"#### {selected}")
            st.caption(f"{record.get('timestamp', '')} · {record.get('source', '')}")
            detections = record.get("detections", [])
            if detections:
                st.dataframe(detections, use_container_width=True, hide_index=True)
            else:
                st.info("本次检测未发现病害目标。")

    actions = st.columns(2)
    actions[0].download_button("下载当前筛选 CSV", export_csv(filtered), "historical_detection_data.csv", "text/csv", use_container_width=True)
    actions[1].download_button("下载完整 JSON 备份", export_state_backup(), "road_inspection_backup.json", "application/json", use_container_width=True)
    with st.expander("从 JSON 备份恢复历史数据"):
        st.caption("用于部署更新或迁移后的完整恢复，会以备份中的记录、任务和档案替换当前数据。")
        backup = st.file_uploader("选择备份文件", type=["json"], key="history_restore")
        if st.button("恢复备份", disabled=backup is None, key="history_restore_button"):
            try:
                restore_state_backup(backup.getvalue())
                st.success("历史数据、任务和病害档案已恢复。")
                st.rerun()
            except Exception as exc:
                st.error(f"备份恢复失败：{exc}")


def render_archive():
    _backfill_archive()
    page_header("DEFECT REGISTRY", "病害档案", "按目标建立独立档案，支持日期与状态筛选、人工复核、分类修正和来源追溯。", "持久保存")
    archive = st.session_state["archives"]
    if not archive:
        empty_panel("暂无病害档案", "完成检测后，每个病害目标会自动拆分为独立档案。")
        return
    records_by_id = {record.get("id"): record for record in get_records()}
    for item in archive:
        if not item.get("检测时间"):
            item["检测时间"] = records_by_id.get(item.get("记录编号"), {}).get("timestamp", item.get("更新时间", ""))

    c1, c2, c3 = st.columns(3)
    c1.metric("档案总数", len(archive))
    c2.metric("已复核", sum(i["复核状态"] == "已复核" for i in archive))
    c3.metric("发生修正", sum(i["原始分类"] != i["当前分类"] for i in archive))

    valid_days = [item.get("检测时间", "")[:10] for item in archive if len(item.get("检测时间", "")) >= 10]
    parsed_days = [datetime.strptime(day, "%Y-%m-%d").date() for day in valid_days]
    min_day = min(parsed_days) if parsed_days else datetime.now().date()
    max_day = max(parsed_days) if parsed_days else datetime.now().date()
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1.2])
    date_from = f1.date_input("开始日期", value=min_day, min_value=min_day, max_value=max_day, key="archive_from")
    date_to = f2.date_input("结束日期", value=max_day, min_value=min_day, max_value=max_day, key="archive_to")
    review_status = f3.selectbox("复核状态", ["全部", "待复核", "已复核"], key="archive_status")
    class_options = sorted({item.get("当前分类", "unknown") for item in archive})
    selected_class = f4.selectbox("病害分类", ["全部", *class_options], key="archive_class")

    filtered = []
    for item in archive:
        day = item.get("检测时间", item.get("更新时间", ""))[:10]
        if day and not (date_from.isoformat() <= day <= date_to.isoformat()):
            continue
        if review_status != "全部" and item.get("复核状态") != review_status:
            continue
        if selected_class != "全部" and item.get("当前分类") != selected_class:
            continue
        filtered.append(item)
    if not filtered:
        empty_panel("没有符合条件的档案", "请调整日期、复核状态或病害分类筛选条件。")
        return

    selected_id = st.selectbox("选择档案", [item["档案编号"] for item in filtered])
    item = next(x for x in filtered if x["档案编号"] == selected_id)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 来源追溯")
        trace_keys = ["档案编号", "记录编号", "目标序号", "检测时间", "来源", "原始分类", "置信度", "更新时间"]
        st.json({key: item.get(key, "") for key in trace_keys}, expanded=True)
        source_record = records_by_id.get(item.get("记录编号"), {})
        if source_record.get("image_b64"):
            try:
                source_image = b64_to_image(source_record["image_b64"])
                st.image(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB), caption="来源检测结果", use_container_width=True)
            except Exception:
                pass
    with right:
        st.markdown("#### 人工复核")
        categories = ["纵向裂缝", "横向裂缝", "网状裂缝", "轻微裂缝", "严重裂缝", "小型坑洼", "大型坑洼", "骨料外露", "路面沉陷", "其他"]
        current_index = categories.index(item["当前分类"]) if item["当前分类"] in categories else 0
        corrected = st.selectbox("修正分类", categories, index=current_index)
        severity = st.select_slider("严重程度", ["轻微", "中等", "严重"], value=item["严重程度"] if item["严重程度"] in ["轻微", "中等", "严重"] else "中等")
        note = st.text_area("复核说明", value=item.get("修正说明", ""), placeholder="记录修正依据或处置建议")
        if st.button("保存复核结果", type="primary"):
            item.update({"当前分类": corrected, "严重程度": severity, "复核状态": "已复核", "修正说明": note, "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M")})
            if persist_session_state():
                st.success("复核结果已持久保存，刷新页面后仍可查询。")
            else:
                st.error("复核结果暂未写入持久化文件，请先下载备份。")
    st.markdown(f"#### 档案列表 · {len(filtered)} 条")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    archive_bytes = pd.DataFrame(filtered).to_csv(index=False).encode("utf-8-sig")
    st.download_button("导出当前档案 CSV", archive_bytes, "road_defect_archive_filtered.csv", "text/csv")


def render_dashboard():
    page_header("ANALYTICS", "数据看板", "从检测量、病害分布、复核状态与变化趋势观察道路风险。", "实时汇总")
    stats = get_statistics()
    archive = st.session_state["archives"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计检测量", stats["total_detections"])
    c2.metric("严重病害", stats["severity_counts"].get("严重", 0))
    c3.metric("复核完成率", f"{sum(i['复核状态'] == '已复核' for i in archive) / len(archive):.0%}" if archive else "—")
    c4.metric("今日新增", stats["today_count"])
    left, right = st.columns(2)
    with left:
        st.markdown("#### 病害分布")
        if stats["class_counts"]:
            st.bar_chart(pd.DataFrame({"病害类型": list(stats["class_counts"]), "数量": list(stats["class_counts"].values())}).set_index("病害类型"), color="#19c2b1")
        else:
            empty_panel("暂无分布数据", "完成检测后自动生成。")
    with right:
        st.markdown("#### 复核状态")
        review = pd.DataFrame({"状态": ["待复核", "已复核"], "数量": [sum(i["复核状态"] == "待复核" for i in archive), sum(i["复核状态"] == "已复核" for i in archive)]}).set_index("状态")
        st.bar_chart(review, color="#38bdf8")
    st.markdown("#### 近 7 日变化趋势")
    daily = {}
    for record in get_records():
        day = record.get("timestamp", "")[:10]
        daily[day] = daily.get(day, 0) + record.get("detection_count", 0)
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    trend = pd.DataFrame({"日期": dates, "检出量": [daily.get(d, 0) for d in dates]}).set_index("日期")
    st.line_chart(trend, color="#19c2b1")


def render_models():
    model, model_path = _model_state()
    page_header("MODEL LAB", "模型与实验", "管理网络结构、训练结果、对比实验与部署版本，确保线上模型来源清晰。", "生产版本 v2.1")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("生产模型", "YOLOv11n")
    c2.metric("检测类别", "2 类" if model else "—")
    c3.metric("语义模型", "CLIP ViT-B/32")
    c4.metric("权重状态", "已加载" if model else "未发现")
    architecture, results, experiments, versions = st.tabs(["网络结构", "训练结果", "对比实验", "模型版本"])
    with architecture:
        st.markdown("#### 两阶段推理架构")
        st.code("巡检影像 → YOLOv11 病害定位 → 目标裁剪 → CLIP 细分类 / 严重度 → 档案与报告", language="text")
        cols = st.columns(3)
        with cols[0]:
            info_card("01", "目标定位", "YOLOv11 检出 crack 与 pothole，保留现有 best.pt 权重和推理参数。")
        with cols[1]:
            info_card("02", "语义增强", "CLIP 对目标区域执行零样本细分类，并为相似案例生成归一化特征。")
        with cols[2]:
            info_card("03", "治理闭环", "结果进入任务、档案、看板与报告，人工修正不会改写原始预测。")
    with results:
        st.info("训练与验证脚本已保留。当前仓库未提供结构化指标文件，因此不展示推测数值。")
        st.dataframe([
            {"资产": "best.pt", "用途": "生产推理", "状态": "可用" if model_path else "缺失", "位置": os.path.basename(model_path) if model_path else "—"},
            {"资产": "evaluate_all.py", "用途": "批量评估", "状态": "已纳入仓库", "位置": "评估工具"},
            {"资产": "train.py", "用途": "训练入口", "状态": "已纳入仓库", "位置": "训练工具"},
        ], use_container_width=True, hide_index=True)
    with experiments:
        st.markdown("#### 对比实验登记")
        st.dataframe([
            {"实验": "YOLOv11 基线", "定位": "基础目标检测", "CLIP": "关闭", "状态": "可复现"},
            {"实验": "YOLOv11 + CLIP", "定位": "目标检测 + 细分类", "CLIP": "开启", "状态": "生产方案"},
            {"实验": "开放词汇扫描", "定位": "CLIP 多尺度窗口", "CLIP": "开启", "状态": "创新能力"},
        ], use_container_width=True, hide_index=True)
    with versions:
        st.dataframe([
            {"版本": "v2.1", "角色": "当前生产", "主要能力": "统一平台导航、CLIP 三项流程、任务与复核闭环"},
            {"版本": "v2.0", "角色": "历史版本", "主要能力": "图片、视频、开放词汇与报告"},
            {"版本": "v1.0", "角色": "基础版本", "主要能力": "YOLOv11 图片检测"},
        ], use_container_width=True, hide_index=True)


def render_reports():
    page_header("REPORTING", "报告中心", "按单次任务或批次生成报告，并导出结构化统计数据。", "导出就绪")
    records = get_records()
    if not records:
        empty_panel("暂无可报告数据", "完成一次检测后即可生成单次或批次报告。")
        return
    single, batch, export_tab = st.tabs(["单次检测报告", "批次报告", "统计导出"])
    with single:
        selected = st.selectbox("选择检测记录", [r["id"] for r in reversed(records)])
        record = next(r for r in records if r["id"] == selected)
        title = st.text_input("报告标题", value=f"{selected} 道路病害检测报告", key="single_report_title")
        report = generate_report([record], title)
        st.code(report, language="text")
        st.download_button("下载单次报告", report.encode("utf-8"), f"{selected}.txt", "text/plain")
    with batch:
        selected_ids = st.multiselect("选择批次记录", [r["id"] for r in reversed(records)], default=[r["id"] for r in records[-min(5, len(records)):]])
        selected_records = [r for r in records if r["id"] in selected_ids]
        if selected_records:
            report = generate_report(selected_records, "道路病害批次检测报告")
            st.code(report, language="text")
            st.download_button("下载批次报告", report.encode("utf-8"), "batch_inspection_report.txt", "text/plain")
    with export_tab:
        st.markdown("#### 结构化统计导出")
        st.download_button("导出全部检测数据 CSV", export_csv(records), "road_defect_statistics.csv", "text/csv", use_container_width=True)
        archive = st.session_state["archives"]
        if archive:
            archive_bytes = pd.DataFrame(archive).to_csv(index=False).encode("utf-8-sig")
            st.download_button("导出人工复核档案 CSV", archive_bytes, "road_defect_archive.csv", "text/csv", use_container_width=True)


def render_system():
    page_header("SYSTEM CONTROL", "系统管理", "统一查看账号权限、模型配置、设备接入和系统运行状态。", "服务健康")
    accounts, models, devices = st.tabs(["账号权限", "模型管理", "设备与运行状态"])
    with accounts:
        st.markdown("#### 角色权限矩阵")
        st.dataframe([
            {"角色": "系统管理员", "检测": "读写", "复核": "读写", "模型": "管理", "报告": "导出", "账号": "管理"},
            {"角色": "复核专家", "检测": "查看", "复核": "读写", "模型": "查看", "报告": "导出", "账号": "无"},
            {"角色": "巡检人员", "检测": "执行", "复核": "提交", "模型": "无", "报告": "查看", "账号": "无"},
            {"角色": "访客", "检测": "查看", "复核": "无", "模型": "无", "报告": "查看", "账号": "无"},
        ], use_container_width=True, hide_index=True)
        st.caption("当前版本采用本地会话权限展示；接入企业身份系统后可映射真实账号。")
    with models:
        cfg = _settings()
        model, path = _model_state()
        c1, c2 = st.columns(2)
        conf = c1.slider("默认置信度", 0.10, 0.95, float(cfg["confidence"]), 0.05)
        iou = c2.slider("默认 IoU", 0.10, 0.95, float(cfg["iou"]), 0.05)
        enable_clip = st.toggle("默认启用 CLIP 细分类", value=bool(cfg["enable_clip"]))
        enable_grad = st.toggle("默认启用 Grad-CAM", value=bool(cfg["enable_grad_cam"]))
        if st.button("保存模型配置", type="primary"):
            cfg.update({"confidence": conf, "iou": iou, "enable_clip": enable_clip, "enable_grad_cam": enable_grad})
            persist_session_state()
            st.success("模型参数已持久保存。")
        st.dataframe([{"模型": "YOLOv11", "权重": os.path.basename(path) if path else "未找到", "状态": "在线" if model else "离线"}, {"模型": "CLIP ViT-B/32", "权重": "按需缓存", "状态": "待调用"}], use_container_width=True, hide_index=True)
    with devices:
        import torch

        device = "CUDA" if torch.cuda.is_available() else "Apple MPS" if torch.backends.mps.is_available() else "CPU"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("计算设备", device)
        c2.metric("Python", platform.python_version())
        c3.metric("Streamlit", st.__version__)
        c4.metric("会话任务", len(st.session_state["tasks"]))
        st.markdown("#### 设备接入状态")
        st.dataframe([
            {"设备": "浏览器摄像头", "通道": "Web Camera", "状态": "按需授权", "用途": "电脑 / 手机实时抓拍"},
            {"设备": "网络摄像头", "通道": "RTSP / HTTP", "状态": "待配置", "用途": "固定点位巡检"},
            {"设备": "上传队列", "通道": "Image / Video", "状态": "在线", "用途": "离线批量分析"},
        ], use_container_width=True, hide_index=True)


PAGE_RENDERERS = {
    "数据总览": render_overview,
    "检测中心": render_detection_center,
    "实时检测": render_realtime,
    "CLIP 创新中心": render_clip_center,
    "检测任务": render_tasks,
    "往期数据": render_history,
    "病害档案": render_archive,
    "数据看板": render_dashboard,
    "模型与实验": render_models,
    "报告中心": render_reports,
    "系统管理": render_system,
}
