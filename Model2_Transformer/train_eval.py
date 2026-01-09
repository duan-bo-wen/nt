import os
import sys
import json
import math
from typing import Dict

# 添加项目根目录到 Python 路径，以便导入模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from Model2_Transformer.model import TransformerCaptionModel, greedy_decode, beam_search_decode
from metrics import compute_meteor, compute_rouge_l, compute_cider


class TransformerCaptionDataset(Dataset):
    """
    使用 train_captions.json / test_captions.json + images/ 进行 BERT tokenizer 预处理。
    """

    def __init__(
        self,
        split: str,
        data_root: str = "data",
        images_dir: str = "images",
    ) -> None:
        assert split in {"train", "test"}
        self.split = split

        with open(
            os.path.join(data_root, f"{split}_captions.json"),
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        self.items = list(data.items())  # (fname, caption)
        self.images_root = os.path.join(data_root, images_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, caption = self.items[idx]
        img_path = os.path.join(self.images_root, fname)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, caption


def get_image_encoder(device: str):
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    modules = list(resnet.children())[:-2]
    backbone = torch.nn.Sequential(*modules).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def extract_feats(backbone, images: torch.Tensor):
    with torch.no_grad():
        feats = backbone(images)  # (B, 2048, H, W)
    B, C, H, W = feats.shape
    feats = feats.view(B, C, -1).permute(0, 2, 1)  # (B, L, 2048)
    return feats


def train_model2(
    epochs: int = 20,  # 增加训练轮数
    batch_size: int = 16,
    lr: float = 1e-4,  # 提高初始学习率
    weight_decay: float = 1e-5,
    warmup_steps: int = 1000,  # warmup步数
    label_smoothing: float = 0.1,  # label smoothing
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ckpt_path: str = os.path.join("data", "output", "weights", "model2_transformer.pth"),
) -> None:
    dataset = TransformerCaptionDataset(split="train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = TransformerCaptionModel().to(device)
    tokenizer = model.tokenizer
    backbone = get_image_encoder(device)

    # 使用label smoothing的CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=label_smoothing
    )
    # 训练decoder、投影层和图像编码器（不训练BERT）
    trainable_params = [p for name, p in model.named_parameters() if 'bert' not in name]
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    
    # Warmup + Cosine Annealing学习率调度
    total_steps = len(loader) * epochs
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"训练参数数量: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
    print(f"总参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for images, captions in tqdm(loader, desc=f"Model2 Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            tokenized = tokenizer(
                list(captions),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=60,  # 增加最大长度
            )
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)

            img_feats = extract_feats(backbone, images)

            optimizer.zero_grad()
            # 训练时也使用 embedding 而不是完整 BERT 编码，强制模型依赖图像
            # 这样可以避免模型只依赖文本信息
            logits = model(
                img_feats=img_feats, 
                input_ids=input_ids, 
                attention_mask=attention_mask,
                use_bert_embedding=False,  # 训练时也强制依赖图像
            )

            # 预测下一个词：shift
            # logits: (B, T, vocab_size) -> (B, T-1, vocab_size)
            # targets: (B, T) -> (B, T-1)
            logits = logits[:, :-1, :].contiguous()
            targets = input_ids[:, 1:].contiguous()
            
            # 计算 loss，忽略 padding
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # 诊断：打印第一个 batch 的 token 分布（仅第一个 epoch 的第一个 batch）
            if epoch == 0 and batch_count == 0:
                with torch.no_grad():
                    pred_tokens = logits[0, :5, :].argmax(dim=-1)  # 前5个位置的预测
                    true_tokens = targets[0, :5]
                    print(f"\n[诊断] 第一个 batch 的 token 预测:")
                    print(f"  真实 tokens: {true_tokens.tolist()}")
                    print(f"  预测 tokens: {pred_tokens.tolist()}")
                    print(f"  真实文本: {tokenizer.decode(true_tokens, skip_special_tokens=True)[:50]}")
                    print(f"  预测文本: {tokenizer.decode(pred_tokens, skip_special_tokens=True)[:50]}")
                    print(f"  Loss: {loss.item():.4f}\n")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)  # 降低梯度裁剪阈值
            optimizer.step()
            scheduler.step()  # 每个batch后更新学习率（warmup需要）

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(len(loader), 1)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Model2 Epoch {epoch+1}: loss={avg_loss:.4f}, lr={current_lr:.6f}")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model2 saved to {ckpt_path}")


def load_model2(
    ckpt_path: str = os.path.join("data", "output", "weights", "model2_transformer.pth"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = TransformerCaptionModel().to(device)
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # 兼容 {"model": state_dict} 或纯 state_dict
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state)
    model.eval()
    backbone = get_image_encoder(device)
    return model, backbone


def generate_caption_model2(
    image_path: str,
    ckpt_path: str = os.path.join("data", "output", "weights", "model2_transformer.pth"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    model, backbone = load_model2(ckpt_path, device=device)
    tokenizer = model.tokenizer

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    img_feats = extract_feats(backbone, image)

    # 使用beam search
    input_ids, attn_mask = beam_search_decode(model, img_feats, beam_size=5, max_len=60)
    text = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return text


def eval_model2(
    ckpt_path: str = os.path.join("data", "output", "weights", "model2_transformer.pth"),
    num_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    dataset = TransformerCaptionDataset(split="test")
    indices = list(range(len(dataset)))
    if num_samples < len(indices):
        indices = indices[:num_samples]

    model, backbone = load_model2(ckpt_path, device=device)
    tokenizer = model.tokenizer

    references = []
    hypotheses = []

    for i in tqdm(indices, desc="Evaluating Model2"):
        image, caption = dataset[i]
        references.append(caption)

        img = image.unsqueeze(0).to(device)
        img_feats = extract_feats(backbone, img)
        # 使用beam search代替greedy decode
        input_ids, _ = beam_search_decode(model, img_feats, beam_size=5, max_len=60)
        text = tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        hypotheses.append(text)

    # 诊断：打印前几个生成示例
    print("\n=== 生成文本诊断（前5个样本）===")
    for i in range(min(5, len(hypotheses))):
        print(f"\n样本 {i+1}:")
        print(f"  参考: {references[i][:100]}...")  # 只显示前100字符
        print(f"  生成: {hypotheses[i][:100]}...")
        print(f"  生成长度: {len(hypotheses[i])} 字符")
    
    meteor = compute_meteor(references, hypotheses)
    rouge_l = compute_rouge_l(references, hypotheses)
    cider = compute_cider(references, hypotheses)
    print(f"\nModel2 eval: METEOR={meteor:.4f}, ROUGE-L={rouge_l:.4f}, CIDEr-D={cider:.4f}")
    return {"METEOR": meteor, "ROUGE-L": rouge_l, "CIDEr-D": cider}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join("data", "output", "weights", "model2_transformer.pth"),
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_model2()
    else:
        eval_model2(args.ckpt)


