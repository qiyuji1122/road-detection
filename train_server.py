"""
服务器版训练脚本 - NYIST GPU服务器 (4×A40 46GB)
使用方法: nohup python train_server.py > train_log.txt 2>&1 &
"""
from ultralytics import YOLO
import torch

# 服务器使用 CUDA
assert torch.cuda.is_available(), "CUDA不可用，请检查驱动"
device = 'cuda:0'
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# 加载预训练模型
model = YOLO('yolo11n.pt')

# A40 46GB显存充足，可以用大batch
results = model.train(
    data='/data2/aipub/lijiaxin/road_detection/my_data_server.yaml',
    epochs=100,          # 充分训练
    imgsz=640,
    batch=32,            # A40可以跑32
    workers=8,           # 服务器CPU多核
    device=device,
    project='runs/detect',
    name='train',
    patience=20,         # 早停：20轮无改善停止
    save=True,
    save_period=10,
    plots=True,          # 生成训练曲线图
    amp=True,            # 混合精度加速
)

print("\n训练完成！")
print(f"最佳模型: runs/detect/train/weights/best.pt")
