"""
上传到服务器后先运行此脚本，检查环境是否就绪
用法: python check_env.py
"""
import sys

print(f"Python: {sys.version}")
print()

# 1. 检查PyTorch
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
    print(f"     CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
        print(f"     显存: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB")
    else:
        print("[错误] CUDA不可用！")
        sys.exit(1)
except ImportError:
    print("[错误] PyTorch未安装！尝试: pip install torch torchvision")
    sys.exit(1)

print()

# 2. 检查ultralytics
try:
    import ultralytics
    print(f"[OK] ultralytics {ultralytics.__version__}")
except ImportError:
    print("[错误] ultralytics未安装！")
    print("  学期内: pip install ultralytics -i https://mirrors.nyist.edu.cn/pypi/web/simple/")
    print("  或离线安装: pip install ultralytics-*.whl")
    sys.exit(1)

print()

# 3. 检查数据集
import os
data_dir = '/data2/aipub/lijiaxin/road_detection/dataset_rdd_5k'
for split in ['train/images', 'val/images']:
    p = os.path.join(data_dir, split)
    if os.path.exists(p):
        n = len([f for f in os.listdir(p) if f.endswith(('.jpg','.png','.jpeg'))])
        print(f"[OK] {split}: {n}张图片")
    else:
        print(f"[错误] 目录不存在: {p}")

print()

# 4. 检查预训练模型
if os.path.exists('/data2/aipub/lijiaxin/road_detection/yolo11n.pt'):
    print("[OK] yolo11n.pt 预训练权重已就绪")
else:
    print("[警告] yolo11n.pt 不存在，ultralytics会自动下载（需联网）")

print()
print("环境检查完毕！可以运行: nohup python train_server.py > train_log.txt 2>&1 &")
