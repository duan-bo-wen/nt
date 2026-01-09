import os
import json
import sys
from typing import Dict, List

# 添加项目根目录到 Python 路径，以便导入 metrics 模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 设置环境变量，消除 tokenizers 并行性警告（在使用 DataLoader 多进程时）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, get_linear_schedule_with_warmup

from metrics import compute_meteor, compute_rouge_l, compute_cider


class CaptionJsonDataset(Dataset):
    """
    基于 train_captions.json / test_captions.json 的简单 Dataset。
    """

    def __init__(self, json_path: str, images_dir: str = None):
        # 如果 images_dir 未指定，使用项目根目录下的 data/images
        if images_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            images_dir = os.path.join(project_root, "data", "images")
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = list(json.load(f).items())  # (fname, caption)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        fname, caption = self.data[idx]
        img_path = os.path.join(self.images_dir, fname)
        image = Image.open(img_path).convert("RGB")
        return image, caption


def collate_fn_blip(batch):
    """
    自定义 collate_fn，用于处理包含 PIL Image 和字符串的 batch。
    """
    images, captions = zip(*batch)
    return list(images), list(captions)


def train_blip(
    train_json: str = None,
    output_path: str = None,
    batch_size: int = 6,
    epochs: int = 15,  # 增加训练轮数，给模型更多学习机会
    lr: float = 3e-5,  # 提高基础学习率
    vision_lr_multiplier: float = 0.1,  # 视觉编码器使用更小的学习率（预训练权重需要更保守）
    text_lr_multiplier: float = 1.0,  # 文本解码器使用标准学习率
    warmup_steps: int = 1000,  # 增加warmup步数，更平滑的启动
    max_length: int = 60,
    weight_decay: float = 0.01,
    val_ratio: float = 0.1,
    early_stop_patience: int = 5,  # 增加早停耐心值，给模型更多机会
    unfreeze_vision_layers: int = 6,  # 解冻更多视觉层（从3增加到6）
    unfreeze_text_decoder: bool = True,  # 解冻文本解码器
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # 设置默认路径（基于项目根目录）
    if train_json is None:
        train_json = os.path.join(PROJECT_ROOT, "data", "train_captions.json")
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", "output", "weights", "blip_finetuned.pth")
    
    # 确保路径是绝对路径
    train_json = os.path.abspath(train_json)
    output_path = os.path.abspath(output_path)
    
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    # 优化的解冻策略：更激进的微调以适应结构化caption
    # 1. 解冻视觉编码器的更多层
    # 2. 解冻文本解码器（如果启用）
    # 3. 使用差异化学习率
    
    vision_params = []
    text_params = []
    other_params = []
    
    if hasattr(model, "vision_model"):
        # 先冻结所有视觉层
        for param in model.vision_model.parameters():
            param.requires_grad = False
        
        # 解冻视觉编码器的更多层
        try:
            if hasattr(model.vision_model, "encoder"):
                encoder = model.vision_model.encoder
                if hasattr(encoder, "layer"):
                    layers = encoder.layer
                    num_layers_to_unfreeze = min(unfreeze_vision_layers, len(layers))
                    for layer in layers[-num_layers_to_unfreeze:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                            vision_params.append(param)
                    print(f"Unfroze last {num_layers_to_unfreeze} vision encoder layers out of {len(layers)} total layers")
            elif hasattr(model.vision_model, "layers"):
                layers = model.vision_model.layers
                if len(layers) >= 2:
                    num_layers_to_unfreeze = min(unfreeze_vision_layers, len(layers))
                    for param in layers[-num_layers_to_unfreeze:].parameters():
                        param.requires_grad = True
                        vision_params.append(param)
                    print(f"Unfroze last {num_layers_to_unfreeze} vision encoder layers out of {len(layers)} total layers")
        except Exception as e:
            print(f"Warning: Could not partially unfreeze vision model: {e}. Unfreezing all vision layers.")
            for param in model.vision_model.parameters():
                param.requires_grad = True
                vision_params.append(param)
            print("All vision encoder layers are now trainable")
    
    # 解冻文本解码器（如果启用）
    if unfreeze_text_decoder:
        if hasattr(model, "text_decoder"):
            for param in model.text_decoder.parameters():
                param.requires_grad = True
                text_params.append(param)
            print("Text decoder is trainable")
        elif hasattr(model, "language_model"):  # 某些BLIP版本使用language_model
            for param in model.language_model.parameters():
                        param.requires_grad = True
                text_params.append(param)
            print("Language model (text decoder) is trainable")
    
    # 收集其他可训练参数（跨模态层等）
    # 使用集合来避免参数比较问题
    vision_param_set = set(id(p) for p in vision_params)
    text_param_set = set(id(p) for p in text_params)
    
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in vision_param_set and id(param) not in text_param_set:
            other_params.append(param)
    
    # 打印可训练参数统计
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  - Vision params: {sum(p.numel() for p in vision_params):,}")
    print(f"  - Text params: {sum(p.numel() for p in text_params):,}")
    print(f"  - Other params: {sum(p.numel() for p in other_params):,}")

    # 加载数据集并划分训练/验证集
    full_dataset = CaptionJsonDataset(train_json)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    
    # 随机划分训练集和验证集
    indices = list(range(dataset_size))
    import random
    random.seed(42)  # 固定随机种子，确保可复现
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 创建子数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 诊断：打印第一个样本的处理结果（仅第一个epoch）
    if len(full_dataset) > 0:
        sample_image, sample_caption = full_dataset[0]
        print(f"\n=== 数据格式诊断 ===")
        print(f"Sample caption: {sample_caption[:100]}...")
        sample_text_inputs = processor(text=[sample_caption], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        decoded = processor.decode(sample_text_inputs.input_ids[0], skip_special_tokens=True)
        print(f"Processor decoded: {decoded[:100]}...")
        print(f"Number of tokens: {sample_text_inputs.input_ids.shape[1]}")
        print("=" * 50 + "\n")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn_blip
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_blip
    )

    # 使用差异化学习率的优化器
    # 视觉编码器使用更小的学习率（预训练权重需要保守微调）
    # 文本解码器使用标准学习率（需要更多适应）
    param_groups = []
    
    if vision_params:
        param_groups.append({
            "params": vision_params,
            "lr": lr * vision_lr_multiplier,
            "weight_decay": weight_decay,
            "name": "vision_encoder"
        })
        print(f"Vision encoder LR: {lr * vision_lr_multiplier:.2e}")
    
    if text_params:
        param_groups.append({
            "params": text_params,
            "lr": lr * text_lr_multiplier,
            "weight_decay": weight_decay,
            "name": "text_decoder"
        })
        print(f"Text decoder LR: {lr * text_lr_multiplier:.2e}")
    
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": "other"
        })
        print(f"Other components LR: {lr:.2e}")
    
    # 如果没有分组，使用统一学习率（fallback）
    if not param_groups:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr, "weight_decay": weight_decay}]
    
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 早停和最佳模型跟踪
    best_val_score = -1.0
    best_epoch = 0
    no_improve = 0
    
    # 验证评估函数：同时计算loss和生成质量
    def evaluate_val(model, processor, val_loader, device, max_length, max_samples=100, max_batches=50):
        """
        验证评估函数：
        - max_samples: 用于生成质量评估的样本数量（至少需要50-100个样本才可靠）
        - max_batches: 用于loss计算的batch数量
        """
        model.eval()
        total_loss = 0.0
        references = []
        hypotheses = []
        count = 0
        samples_evaluated = 0
        
        with torch.no_grad():
            for images, captions in val_loader:
                # 使用processor同时处理图像和文本（标准BLIP方式）
                inputs = processor(
                    images=images,
                    text=list(captions),
                    padding=True,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                ).to(device)
                
                # 计算loss（所有batch都计算）
                outputs = model(**inputs, labels=inputs.input_ids)
                total_loss += outputs.loss.item()
                
                # 生成caption用于质量评估（评估足够的样本）
                if samples_evaluated < max_samples:
                    batch_size = len(images)
                    samples_to_eval = min(batch_size, max_samples - samples_evaluated)
                    
                    for i in range(samples_to_eval):
                        references.append(captions[i])
                        img_input = processor(images=images[i], return_tensors="pt").to(device)
                        # 使用与最终评估相同的生成参数，确保一致性
                        generated = model.generate(
                            **img_input,
                            max_length=max_length,
                            num_beams=5,  # 与eval_blip一致
                            no_repeat_ngram_size=2,
                            length_penalty=1.0,
                            early_stopping=True,
                            repetition_penalty=1.1,
                            do_sample=False,
                        )
                        text = processor.decode(generated[0], skip_special_tokens=True)
                        hypotheses.append(text)
                    
                    samples_evaluated += samples_to_eval
                
                count += 1
                # 如果已经评估了足够的样本且计算了足够的loss，可以提前结束
                if count >= max_batches and samples_evaluated >= max_samples:
                    break
        
        model.train()
        avg_loss = total_loss / max(count, 1)
        
        # 计算生成质量指标（如果有样本）
        if len(hypotheses) > 0:
            meteor = compute_meteor(references, hypotheses)
            rouge_l = compute_rouge_l(references, hypotheses)
            cider = compute_cider(references, hypotheses)
            return avg_loss, meteor, rouge_l, cider
        else:
            return avg_loss, 0.0, 0.0, 0.0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, captions in tqdm(train_loader, desc=f"BLIP finetune Epoch {epoch+1}/{epochs}"):
            # 关键修复：使用processor同时处理图像和文本（BLIP标准方式）
            # 这样可以确保多模态对齐和特殊token的正确添加
            inputs = processor(
                images=images,
                text=list(captions),
                padding=True,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(device)

            # 训练时：BLIP会自动处理labels的shift和padding忽略
            # labels=inputs.input_ids 是BLIP的标准用法
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸（提高阈值，允许更大的梯度更新）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        
        # 验证集评估（同时计算loss和生成质量，评估100个样本）
        val_loss, val_meteor, val_rouge, val_cider = evaluate_val(
            model, processor, val_loader, device, max_length, max_samples=100
        )
        
        # 使用综合指标作为score（CIDEr最重要，其次是METEOR和ROUGE-L）
        # 如果生成了样本，使用综合指标；否则只用loss
        if val_cider > 0:
            val_score = (val_meteor + val_rouge + val_cider) / 3.0
            print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"val_METEOR={val_meteor:.4f}, val_ROUGE-L={val_rouge:.4f}, val_CIDEr={val_cider:.4f}")
        else:
            val_score = -val_loss  # 如果没有生成样本，只用loss
            print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # 检查是否有改进
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch + 1
            no_improve = 0
            
            # 保存最佳模型
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_dict = {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }
            if val_cider > 0:
                save_dict.update({
                    "val_meteor": val_meteor,
                    "val_rouge": val_rouge,
                    "val_cider": val_cider,
                })
            torch.save(save_dict, output_path)
            print(f"  ✓ Best model saved (score={val_score:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve} epoch(s)")
            
            # 早停
            if no_improve >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1}. Best epoch was {best_epoch} with score={best_val_score:.4f}")
                break

    print(f"\nTraining completed. Best epoch: {best_epoch}, Best score: {best_val_score:.4f}")
    print(f"Best checkpoint saved to {output_path}")


def eval_blip(
    test_json: str = None,
    ckpt_path: str = None,
    max_samples: int = 200,
    max_length: int = 60,  # 与训练时的 max_length 保持一致，允许生成更长的描述
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    zero_shot: bool = False,  # 如果True，不加载checkpoint，使用原始预训练模型
) -> Dict[str, float]:
    # 设置默认路径（基于项目根目录）
    if test_json is None:
        test_json = os.path.join(PROJECT_ROOT, "data", "test_captions.json")
    if ckpt_path is None:
        ckpt_path = os.path.join(PROJECT_ROOT, "data", "output", "weights", "blip_finetuned.pth")
    
    # 确保路径是绝对路径
    test_json = os.path.abspath(test_json)
    ckpt_path = os.path.abspath(ckpt_path)
    
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    if not zero_shot and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded fine-tuned checkpoint from {ckpt_path}")
    elif zero_shot:
        print("Using zero-shot BLIP (no fine-tuning)")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found, using zero-shot BLIP")
    
    model.to(device)
    model.eval()

    dataset = CaptionJsonDataset(test_json)
    indices = list(range(len(dataset)))
    if max_samples < len(indices):
        indices = indices[:max_samples]

    references: List[str] = []
    hypotheses: List[str] = []

    for idx in tqdm(indices, desc="Evaluating BLIP"):
        image, caption = dataset[idx]
        references.append(caption)

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            # 优化的解码策略：平衡质量和长度
            out = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,  # 适中的beam size
                no_repeat_ngram_size=2,  # 防止重复短语
                length_penalty=1.0,  # 降低长度惩罚，避免生成过长文本
                early_stopping=True,  # 启用early stopping，遇到结束符即停止
                repetition_penalty=1.1,  # 适度的重复惩罚
                do_sample=False,  # 使用确定性beam search
            )
        text = processor.decode(out[0], skip_special_tokens=True)
        hypotheses.append(text)

    meteor = compute_meteor(references, hypotheses)
    rouge_l = compute_rouge_l(references, hypotheses)
    cider = compute_cider(references, hypotheses)
    print(f"BLIP eval: METEOR={meteor:.4f}, ROUGE-L={rouge_l:.4f}, CIDEr-D={cider:.4f}")
    return {"METEOR": meteor, "ROUGE-L": rouge_l, "CIDEr-D": cider}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--train_json", type=str, default=None, help="Path to train_captions.json (default: PROJECT_ROOT/data/train_captions.json)")
    parser.add_argument("--test_json", type=str, default=None, help="Path to test_captions.json (default: PROJECT_ROOT/data/test_captions.json)")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (default: PROJECT_ROOT/data/output/weights/blip_finetuned.pth)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=6)
    args = parser.parse_args()

    if args.mode == "train":
        train_blip(
            train_json=args.train_json,
            output_path=args.ckpt,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        eval_blip(
            test_json=args.test_json,
            ckpt_path=args.ckpt,
        )

