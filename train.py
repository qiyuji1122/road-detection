from ultralytics import YOLO
import torch

# 检查可用加速设备
if torch.backends.mps.is_available():
    device = 'mps'
    print("使用 Apple MPS (GPU) 加速训练")
elif torch.cuda.is_available():
    device = 'cuda:0'
    print(f"使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("使用 CPU 训练（较慢）")

# 加载预训练模型（yolo11n.pt 已在项目目录中）
model = YOLO('yolo11n.pt')

# 训练模型
results = model.train(
    data='/Users/qiyuji/Documents/道路目标检测YOLOv11/my_data.yaml',
    epochs=10,           # 训练轮次（快速测试版）
    imgsz=640,           # 输入图像尺寸
    batch=8,             # 批次大小（MPS显存有限，用8较安全）
    workers=2,           # macOS下workers不宜太大
    device=device,
    project='runs/detect',
    name='train',
    patience=5,          # 早停：5轮无改善则停止
    save=True,           # 保存checkpoint
    save_period=10,      # 每10轮保存一次
)

print("\n训练完成！")
print(f"最佳模型保存在: runs/detect/train/weights/best.pt")
