import os
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils_data import get_loader, load_vocab, CaptionDataset
from Model3_CNN_GRU.models import GlobalImageEncoder, GlobalGRUDecoder, generate_by_beam_search
from metrics import compute_meteor, compute_rouge_l, compute_cider


def train_cnn_gru(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = os.path.join("data", "output", "weights"),
) -> None:
    """
    训练 CNN 整体表示 + GRU 模型。
    """
    os.makedirs(output_dir, exist_ok=True)

    loader = get_loader("train", batch_size=batch_size, shuffle=True)
    vocab = load_vocab()
    vocab_size = len(vocab)

    encoder = GlobalImageEncoder().to(device)
    decoder = GlobalGRUDecoder(vocab_size=vocab_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    params = list(decoder.parameters()) + [p for p in encoder.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, captions, caplens in tqdm(loader, desc=f"[CNN+GRU] Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            captions = captions.to(device)
            if not isinstance(caplens, torch.Tensor):
            caplens = torch.tensor(caplens, dtype=torch.long, device=device)
            else:
                caplens = caplens.to(device)

            optimizer.zero_grad()

            global_feats = encoder(images)  # (B, D)
            logits, _ = decoder(global_feats, captions, caplens)  # (B, T, vocab)

            max_decode = logits.size(1)
            targets = captions[:, 1 : 1 + max_decode]

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"[CNN+GRU] Epoch {epoch+1}: loss={avg_loss:.4f}")

    ckpt = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "vocab": vocab,
    }
    torch.save(ckpt, os.path.join(output_dir, "cnn_gru.pth"))


def load_cnn_gru_checkpoint(
    ckpt_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt.get("vocab", load_vocab())
    vocab_size = len(vocab)

    encoder = GlobalImageEncoder().to(device)
    decoder = GlobalGRUDecoder(vocab_size=vocab_size).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    encoder.eval()
    decoder.eval()
    return encoder, decoder, vocab


def idx_to_words(indices, inv_vocab):
    return [inv_vocab.get(int(i), "<unk>") for i in indices]


def generate_caption_cnn_gru(
    image_path: str,
    ckpt_path: str,
    beam_size: int = 3,
    max_len: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    from PIL import Image
    from torchvision import transforms

    encoder, decoder, vocab = load_cnn_gru_checkpoint(ckpt_path, device=device)
    inv_vocab = {v: k for k, v in vocab.items()}
    start_idx = vocab["<start>"]
    end_idx = vocab["<end>"]

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
    image = transform(image).to(device)

    seq = generate_by_beam_search(
        encoder,
        decoder,
        image,
        start_idx=start_idx,
        end_idx=end_idx,
        beam_size=beam_size,
        max_len=max_len,
    )
    words = idx_to_words(seq, inv_vocab)
    words = [w for w in words if w not in {"<pad>", "<start>", "<end>", "<unk>"}]
    return " ".join(words)


def eval_cnn_gru(
    ckpt_path: str,
    num_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """
    简单评估：随机取若干 test 样本，生成 caption，与 gt 文本做 METEOR / ROUGE-L / CIDEr-D。
    """
    dataset = CaptionDataset(split="test")
    indices = list(range(len(dataset)))
    if num_samples < len(indices):
        indices = indices[:num_samples]

    encoder, decoder, vocab = load_cnn_gru_checkpoint(ckpt_path, device=device)
    inv_vocab = {v: k for k, v in vocab.items()}
    start_idx = vocab["<start>"]
    end_idx = vocab["<end>"]

    references = []
    hypotheses = []

    for idx in indices:
        image, caption_idx, caplen = dataset[idx]
        image = image.to(device)

        # 真实 caption
        real_tokens = caption_idx[:caplen].tolist()
        real_words = idx_to_words(real_tokens[1:-1], inv_vocab)  # 去掉 <start>/<end>
        references.append(" ".join(real_words))

        seq = generate_by_beam_search(
            encoder,
            decoder,
            image,
            start_idx=start_idx,
            end_idx=end_idx,
            beam_size=3,
            max_len=64,
        )
        gen_words = idx_to_words(seq, inv_vocab)
        gen_words = [w for w in gen_words if w not in {"<pad>", "<start>", "<end>", "<unk>"}]
        hypotheses.append(" ".join(gen_words))

    meteor = compute_meteor(references, hypotheses)
    rouge_l = compute_rouge_l(references, hypotheses)
    cider = compute_cider(references, hypotheses)
    print(f"[CNN+GRU] eval: METEOR={meteor:.4f}, ROUGE-L={rouge_l:.4f}, CIDEr-D={cider:.4f}")
    return {"METEOR": meteor, "ROUGE-L": rouge_l, "CIDEr-D": cider}



