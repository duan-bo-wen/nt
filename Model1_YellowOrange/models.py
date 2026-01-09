import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageEncoder(nn.Module):
    """
    ResNet101 网格特征编码器，输出 (B, L, D)
    """

    def __init__(self, encoded_size: int = 14, fine_tune: bool = False):
        super().__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]  # 去掉 avgpool & fc
        self.backbone = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))

        self.fine_tune = fine_tune
        for p in self.backbone.parameters():
            p.requires_grad = fine_tune

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, 2048, H', W')
        features = self.backbone(images)
        features = self.adaptive_pool(features)  # (B, 2048, enc, enc)
        features = features.permute(0, 2, 3, 1)  # (B, enc, enc, 2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (B, L, D)
        return features


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(
        self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_out: (B, L, D), decoder_hidden: (B, decoder_dim)
        att1 = self.encoder_att(encoder_out)  # (B, L, A)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (B, 1, A)
        att = torch.tanh(att1 + att2)
        e = self.full_att(att).squeeze(-1)  # (B, L)
        alpha = F.softmax(e, dim=1)  # (B, L)
        context = (encoder_out * alpha.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return context, alpha


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 2048,
        decoder_dim: int = 512,
        attention_dim: int = 512,
        embed_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = AdditiveAttention(encoder_dim, decoder_dim, attention_dim)
        self.gru = nn.GRU(
            input_size=encoder_dim + embed_dim,
            hidden_size=decoder_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        prev_word: torch.Tensor,
        prev_hidden: torch.Tensor,
        encoder_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单步解码
        prev_word: (B,)
        prev_hidden: (1, B, decoder_dim)
        encoder_out: (B, L, encoder_dim)
        """
        emb = self.embedding(prev_word)  # (B, E)
        context, alpha = self.attention(encoder_out, prev_hidden[-1])  # (B, D), (B, L)
        gru_input = torch.cat([emb, context], dim=-1).unsqueeze(1)  # (B, 1, E+D)
        output, hidden = self.gru(gru_input, prev_hidden)  # output: (B, 1, H)
        output = self.dropout(output.squeeze(1))
        logits = self.fc(output)  # (B, vocab)
        return logits, hidden, alpha

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoded_captions: torch.Tensor,
        caption_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher forcing 训练
        encoder_out: (B, L, D)
        encoded_captions: (B, max_len)
        caption_lengths: (B,)
        """
        batch_size, max_len = encoded_captions.size()
        device = encoded_captions.device

        # 初始化 hidden，用全零即可
        hidden = torch.zeros(1, batch_size, self.decoder_dim, device=device)

        # 逐步解码（忽略 <pad> 之后的 loss）
        logits_list = []
        for t in range(max_len - 1):  # 预测下一个词
            prev_word = encoded_captions[:, t]
            logits, hidden, _ = self.forward_step(prev_word, hidden, encoder_out)
            logits_list.append(logits.unsqueeze(1))

        logits = torch.cat(logits_list, dim=1)  # (B, max_len-1, vocab)
        return logits, caption_lengths - 1


def generate_by_beam_search(
    encoder: ImageEncoder,
    decoder: AttentionDecoder,
    image: torch.Tensor,
    start_idx: int,
    end_idx: int,
    beam_size: int = 3,
    max_len: int = 64,
) -> torch.Tensor:
    """
    对单张图片进行 beam search，返回 token id 序列（不含 <start>）
    """
    device = image.device
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        encoder_out = encoder(image.unsqueeze(0))  # (1, L, D)
        L = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        encoder_out = encoder_out.expand(beam_size, L, encoder_dim)

        k = beam_size
        prev_words = torch.full((k,), start_idx, dtype=torch.long, device=device)
        seqs = prev_words.unsqueeze(1)  # (k, 1)
        top_k_scores = torch.zeros(k, 1, device=device)

        hidden = torch.zeros(1, k, decoder.decoder_dim, device=device)

        complete_seqs = []
        complete_scores = []

        for _ in range(max_len):
            logits, hidden, _ = decoder.forward_step(prev_words, hidden, encoder_out)
            scores = F.log_softmax(logits, dim=-1)  # (k, vocab)

            scores = top_k_scores + scores  # 累积 log prob
            top_k_scores, top_k_words = scores.view(-1).topk(k, dim=-1)

            prev_k = top_k_words // decoder.vocab_size
            next_words = top_k_words % decoder.vocab_size

            seqs = torch.cat([seqs[prev_k], next_words.unsqueeze(1)], dim=1)
            prev_words = next_words

            incomplete_idx = []
            new_complete_seqs = []
            new_complete_scores = []
            for i, w in enumerate(next_words):
                if w.item() == end_idx:
                    new_complete_seqs.append(seqs[i].clone())
                    new_complete_scores.append(top_k_scores[i].item())
                else:
                    incomplete_idx.append(i)

            if len(new_complete_seqs) > 0:
                complete_seqs.extend(new_complete_seqs)
                complete_scores.extend(new_complete_scores)

            if len(incomplete_idx) == 0:
                break

            seqs = seqs[incomplete_idx]
            encoder_out = encoder_out[incomplete_idx]
            hidden = hidden[:, incomplete_idx, :]
            top_k_scores = top_k_scores[incomplete_idx].unsqueeze(1)
            prev_words = prev_words[incomplete_idx]
            k = len(incomplete_idx)

        if len(complete_seqs) == 0:
            complete_seqs = [seqs[0]]
            complete_scores = [top_k_scores[0].item()]

        best_idx = int(torch.tensor(complete_scores).argmax())
        best_seq = complete_seqs[best_idx]

        # 去掉开头的 <start>
        best_seq = best_seq[1:]
        return best_seq


