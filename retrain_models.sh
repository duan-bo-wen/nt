#!/bin/bash
# 重新训练 Model2 和 BLIP 的脚本
# 使用统一后的完整数据集

echo "=========================================="
echo "重新训练 Model2 和 BLIP（使用统一数据集）"
echo "=========================================="
echo ""

# 备份现有权重文件（可选）
echo "【可选】备份现有权重文件..."
mkdir -p data/output/weights/backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null
BACKUP_DIR="data/output/weights/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -f "data/output/weights/model2_transformer.pth" ]; then
    cp data/output/weights/model2_transformer.pth "$BACKUP_DIR/"
    echo "  ✓ 已备份 model2_transformer.pth"
fi

if [ -f "data/output/weights/blip_finetuned.pth" ]; then
    cp data/output/weights/blip_finetuned.pth "$BACKUP_DIR/"
    echo "  ✓ 已备份 blip_finetuned.pth"
fi

echo ""
echo "数据集统计："
python3 -c "import json; train=json.load(open('data/train_captions.json')); test=json.load(open('data/test_captions.json')); print(f'  训练集: {len(train):,} 个样本'); print(f'  测试集: {len(test):,} 个样本'); print(f'  总计: {len(train)+len(test):,} 个样本')"
echo ""

# 训练 Model2 Transformer
echo "=========================================="
echo "开始训练 Model2 Transformer..."
echo "=========================================="
echo "预计训练时间：约 20 epochs，数据集较大，可能需要数小时"
echo ""
read -p "是否继续训练 Model2? (y/n): " train_model2

if [ "$train_model2" = "y" ] || [ "$train_model2" = "Y" ]; then
    cd Model2_Transformer
    python3 train_eval.py --mode train
    cd ..
    echo ""
    echo "✓ Model2 训练完成！"
else
    echo "跳过 Model2 训练"
fi

echo ""

# 训练 BLIP
echo "=========================================="
echo "开始训练 BLIP..."
echo "=========================================="
echo "预计训练时间：约 15 epochs，数据集较大，可能需要数小时"
echo ""
read -p "是否继续训练 BLIP? (y/n): " train_blip

if [ "$train_blip" = "y" ] || [ "$train_blip" = "Y" ]; then
    cd Ex1_BLIP
    python3 train_blip.py --mode train --epochs 15 --batch_size 6
    cd ..
    echo ""
    echo "✓ BLIP 训练完成！"
else
    echo "跳过 BLIP 训练"
fi

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "备份文件保存在: $BACKUP_DIR"

