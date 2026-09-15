"""
优化版训练脚本 v2 - 提升mAP
改进点：
1. 全量数据集(16634张)代替5k子集
2. 更低学习率 + cosine退火
3. 更多数据增强(mosaic/mixup)
4. 更长训练 + 更大patience
用法: nohup python train_server_v2.py > train_log_v2.txt 2>&1 &
"""
from ultralytics import YOLO
import torch

device = 'cuda:0'
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 用上次训练的best.pt作为起点(迁移学习)，比从零开始更好
model = YOLO('runs/detect/train/weights/best.pt')

results = model.train(
    data='my_data_server_full.yaml',
    epochs=200,            # 更多轮次
    imgsz=640,
    batch=32,
    workers=8,
    device=device,
    project='runs/detect',
    name='train_v2',

    # === 学习率调优 ===
    lr0=0.001,             # 初始学习率降低(原默认0.01)
    lrf=0.01,              # 最终学习率 = lr0 * lrf
    cos_lr=True,           # cosine退火(比线性衰减更好)
    warmup_epochs=5,       # 预热5轮

    # === 数据增强 ===
    mosaic=1.0,            # mosaic增强(默认1.0，保持不变)
    mixup=0.1,             # mixup增强(新增，提升泛化)
    copy_paste=0.1,        # copy-paste增强
    degrees=10,            # 旋转角度
    scale=0.5,             # 缩放增强
    hsv_h=0.015,           # 色调增强
    hsv_s=0.5,             # 饱和度增强
    hsv_v=0.3,             # 明度增强

    # === 正则化 ===
    weight_decay=0.001,    # 权重衰减(防过拟合)
    label_smoothing=0.1,   # 标签平滑

    # === 其他 ===
    patience=30,           # 30轮无改善才停(给更多耐心)
    save=True,
    save_period=10,
    plots=True,
    amp=True,
)

print("v2训练完成!")
print(f"最佳模型: runs/detect/train_v2/weights/best.pt")
