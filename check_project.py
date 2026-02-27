import os
import sys

def check_file(path, description):
    exists = os.path.isfile(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_dir(path, description):
    exists = os.path.isdir(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def main():
    print("=" * 50)
    print("项目完整性检查")
    print("=" * 50)

    # 检查根目录
    root_ok = check_dir(".", "项目根目录")

    # 检查核心目录
    dirs = [
        ("models", "模型定义"),
        ("losses", "损失函数"),
        ("datasets", "数据集"),
        ("scripts", "训练脚本"),
        ("configs", "配置文件"),
        ("data", "数据目录"),
        ("docs", "文档"),
        ("experiments", "实验输出（运行后自动创建）"),
    ]
    dir_ok = all(check_dir(d, desc) for d, desc in dirs)

    # 检查关键文件
    files = [
        ("models/encoder.py", "编码器"),
        ("models/projection_head.py", "投影头"),
        ("losses/nt_xent.py", "NT-Xent损失"),
        ("datasets/simclr_augment.py", "SimCLR数据增强"),
        ("datasets/pretrain_dataset.py", "预训练数据集"),
        ("scripts/train_simclr.py", "训练脚本"),
        ("configs/simclr_cifar.yaml", "配置文件"),
        ("docs/progress.md", "进度日志"),
        ("docs/troubleshooting.md", "问题记录"),
    ]
    files_ok = all(check_file(f, desc) for f, desc in files)

    # 检查数据划分文件
    data_files = [
        ("data/splits/pretrain_indices.txt", "预训练索引"),
        ("data/splits/fewshot_5shot.json", "5-shot标注数据"),
    ]
    data_ok = all(check_file(f, desc) for f, desc in data_files)

    # 检查虚拟环境
    try:
        import torch
        import torchvision
        import numpy
        import matplotlib
        import tensorboard
        import tqdm
        import sklearn
        print("✅ 虚拟环境已安装所有必需库")
    except ImportError as e:
        print(f"❌ 虚拟环境缺少库: {e}")

    print("=" * 50)
    if dir_ok and files_ok and data_ok:
        print("✅ 所有核心文件检查通过！你可以开始运行训练了。")
        print("运行命令：python scripts/train_simclr.py --config configs/simclr_cifar.yaml")
    else:
        print("❌ 部分文件缺失，请根据上方提示补全后再运行。")
    print("=" * 50)

if __name__ == "__main__":
    main()