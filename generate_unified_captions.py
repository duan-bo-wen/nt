"""
生成统一的 train_captions.json 和 test_captions.json
使用与 preprocess.py 完全相同的过滤和划分逻辑，确保与 Original/Model1/Model3 使用的数据一致
"""

import json
import os
from typing import Dict, List, Tuple


def load_captions(caption_path: str) -> Dict[str, str]:
    """从 caption.json 加载标签数据"""
    with open(caption_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def split_train_test(
    items: List[Tuple[str, str]], train_ratio: float
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """划分训练集和测试集（与 preprocess.py 完全相同的逻辑）"""
    n_train = int(len(items) * train_ratio)
    train = items[:n_train]
    test = items[n_train:]
    return train, test


def generate_unified_captions(
    data_root: str = "data",
    caption_file: str = "caption.json",
    images_dir: str = "images",
    train_ratio: float = 0.8,
    output_train: str = "data/train_captions.json",
    output_test: str = "data/test_captions.json",
):
    """
    生成统一的 train_captions.json 和 test_captions.json
    
    使用与 preprocess.py 完全相同的逻辑：
    1. 从 caption.json 读取数据
    2. 过滤不存在的图片
    3. 按照相同的 train_ratio 划分训练/测试集
    4. 保存为 JSON 字典格式
    """
    caption_path = os.path.join(data_root, caption_file)
    images_path = os.path.join(data_root, images_dir)
    
    print(f"正在从 {caption_path} 加载数据...")
    captions = load_captions(caption_path)
    print(f"  原始标签数量: {len(captions):,}")
    
    # 过滤不存在的图片（与 preprocess.py 完全相同的逻辑）
    items: List[Tuple[str, str]] = []
    missing_count = 0
    for fname, sent in captions.items():
        img_path = os.path.join(images_path, fname)
        if os.path.isfile(img_path):
            items.append((fname, sent))
        else:
            missing_count += 1
    
    print(f"  存在的图片数量: {len(items):,}")
    print(f"  缺失的图片数量: {missing_count:,}")
    
    # 划分训练/测试集（与 preprocess.py 完全相同的逻辑）
    print(f"\n按照 train_ratio={train_ratio} 划分数据集...")
    train_items, test_items = split_train_test(items, train_ratio=train_ratio)
    print(f"  训练集: {len(train_items):,} 个样本")
    print(f"  测试集: {len(test_items):,} 个样本")
    
    # 转换为字典格式
    train_dict = {fname: sent for fname, sent in train_items}
    test_dict = {fname: sent for fname, sent in test_items}
    
    # 保存文件
    print(f"\n保存文件...")
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    
    with open(output_train, "w", encoding="utf-8") as f:
        json.dump(train_dict, f, ensure_ascii=False, indent=1)
    print(f"  ✓ {output_train} ({len(train_dict):,} 个样本)")
    
    with open(output_test, "w", encoding="utf-8") as f:
        json.dump(test_dict, f, ensure_ascii=False, indent=1)
    print(f"  ✓ {output_test} ({len(test_dict):,} 个样本)")
    
    # 验证：检查是否与 preprocess.py 生成的 image_paths 一致
    output_dir = os.path.join("data", "output")
    image_paths_train_file = os.path.join(output_dir, "image_paths_train.json")
    image_paths_test_file = os.path.join(output_dir, "image_paths_test.json")
    
    if os.path.exists(image_paths_train_file) and os.path.exists(image_paths_test_file):
        print(f"\n验证数据一致性...")
        with open(image_paths_train_file, "r", encoding="utf-8") as f:
            paths_train = json.load(f)
        with open(image_paths_test_file, "r", encoding="utf-8") as f:
            paths_test = json.load(f)
        
        # 提取文件名（去掉路径前缀）
        def extract_fname(path):
            return os.path.basename(path)
        
        preprocess_train_fnames = set(extract_fname(p) for p in paths_train)
        preprocess_test_fnames = set(extract_fname(p) for p in paths_test)
        new_train_fnames = set(train_dict.keys())
        new_test_fnames = set(test_dict.keys())
        
        train_match = len(preprocess_train_fnames & new_train_fnames)
        test_match = len(preprocess_test_fnames & new_test_fnames)
        
        print(f"  训练集匹配: {train_match}/{len(train_items)} ({train_match/len(train_items)*100:.1f}%)")
        print(f"  测试集匹配: {test_match}/{len(test_items)} ({test_match/len(test_items)*100:.1f}%)")
        
        if train_match == len(train_items) and test_match == len(test_items):
            print("  ✓ 数据完全一致！")
        elif train_match == len(preprocess_train_fnames) and test_match == len(preprocess_test_fnames):
            print("  ✓ 数据完全一致！")
        else:
            print("  ⚠ 警告：数据不完全一致，但可能是由于文件路径格式不同")
    
    print(f"\n完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成统一的 train_captions.json 和 test_captions.json")
    parser.add_argument("--data_root", type=str, default="data", help="数据根目录")
    parser.add_argument("--caption_file", type=str, default="caption.json", help="caption.json 文件名")
    parser.add_argument("--images_dir", type=str, default="images", help="图片目录名")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--output_train", type=str, default="data/train_captions.json", help="输出训练集文件")
    parser.add_argument("--output_test", type=str, default="data/test_captions.json", help="输出测试集文件")
    
    args = parser.parse_args()
    
    # 备份现有文件
    if os.path.exists(args.output_train):
        backup_train = args.output_train + ".backup"
        print(f"备份现有文件: {args.output_train} -> {backup_train}")
        os.rename(args.output_train, backup_train)
    
    if os.path.exists(args.output_test):
        backup_test = args.output_test + ".backup"
        print(f"备份现有文件: {args.output_test} -> {backup_test}")
        os.rename(args.output_test, backup_test)
    
    generate_unified_captions(
        data_root=args.data_root,
        caption_file=args.caption_file,
        images_dir=args.images_dir,
        train_ratio=args.train_ratio,
        output_train=args.output_train,
        output_test=args.output_test,
    )

