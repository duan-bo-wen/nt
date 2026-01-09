import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple


class GlobalImageEncoder(nn.Module):
    """
    使用 ResNet101 提取整图全局特征，输出 (B, D)，对应“CNN 整体表示”。
    """

    def __init__(self, fine_tune: bool = False):
        super().__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # 保留到 avgpool，去掉最后的全连接层
        modules = list(resnet.children())[:-1]  # (B, 2048, 1, 1)
        self.backbone = nn.Sequential(*modules)

        for p in self.backbone.parameters():
            p.requires_grad = fine_tune

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, 2048, 1, 1)
        feats = self.backbone(images)
        feats = feats.view(feats.size(0), -1)  # (B, 2048)
        return feats


class GlobalGRUDecoder(nn.Module):
    """
    纯 GRU 解码器：用整图全局特征初始化 hidden，不再做网格/区域注意力。
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 2048,
        decoder_dim: int = 512,
        embed_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 用整图特征生成初始 hidden
        self.init_hidden_fc = nn.Linear(encoder_dim, decoder_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=decoder_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, global_feats: torch.Tensor) -> torch.Tensor:
        """
        global_feats: (B, encoder_dim)
        return: (1, B, decoder_dim)
        """
        h = torch.tanh(self.init_hidden_fc(global_feats))  # (B, decoder_dim)
        return h.unsqueeze(0)

    def forward(
        self,
        global_feats: torch.Tensor,
        encoded_captions: torch.Tensor,
        caption_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher forcing 训练
        global_feats: (B, encoder_dim)
        encoded_captions: (B, max_len)
        caption_lengths: (B,)
        """
        batch_size, max_len = encoded_captions.size()
        _ = batch_size  # unused, for clarity
        device = encoded_captions.device
        _ = device

        hidden = self.init_hidden(global_feats)  # (1, B, decoder_dim)

        logits_list = []
        for t in range(max_len - 1):  # 预测下一个词
            prev_word = encoded_captions[:, t]  # (B,)
            emb = self.embedding(prev_word).unsqueeze(1)  # (B, 1, E)
            output, hidden = self.gru(emb, hidden)  # output: (B, 1, H)
            output = self.dropout(output.squeeze(1))  # (B, H)
            logits = self.fc(output)  # (B, vocab)
            logits_list.append(logits.unsqueeze(1))

        logits = torch.cat(logits_list, dim=1)  # (B, max_len-1, vocab)
        return logits, caption_lengths - 1


def generate_by_beam_search(
    encoder: GlobalImageEncoder,
    decoder: GlobalGRUDecoder,
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
        global_feat = encoder(image.unsqueeze(0))  # (1, D)

        k = beam_size
        prev_words = torch.full((k,), start_idx, dtype=torch.long, device=device)
        seqs = prev_words.unsqueeze(1)  # (k, 1)
        top_k_scores = torch.zeros(k, 1, device=device)

        # 为每个 beam 复制全局特征
        global_feats = global_feat.expand(k, -1)  # (k, D)
        hidden = decoder.init_hidden(global_feats)  # (1, k, H)

        complete_seqs = []
        complete_scores = []

        for _ in range(max_len):
            emb = decoder.embedding(prev_words).unsqueeze(1)  # (k, 1, E)
            output, hidden = decoder.gru(emb, hidden)  # (k, 1, H)
            output = decoder.dropout(output.squeeze(1))  # (k, H)
            logits = decoder.fc(output)  # (k, vocab)

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








