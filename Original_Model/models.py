import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class OriginalEncoder(nn.Module):
    """
    简化版 Original：仍然使用 ResNet101 网格特征，但后续解码结构可以和 Model1 略有不同。
    """

    def __init__(self, encoded_size: int = 14, fine_tune: bool = False):
        super().__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))

        for p in self.backbone.parameters():
            p.requires_grad = fine_tune

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        feats = self.adaptive_pool(feats)
        feats = feats.view(feats.size(0), feats.size(1), -1).permute(0, 2, 1)  # (B, L, D)
        return feats


class OriginalDecoder(nn.Module):
    """
    简单注意力 + GRU 解码（比 Model1 去掉一些组件，作为“Original”版本）。
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 2048,
        decoder_dim: int = 512,
        embed_dim: int = 512,
        attention_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.gru = nn.GRU(embed_dim + encoder_dim, decoder_dim, batch_first=True)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.decoder_dim = decoder_dim

    def attention(self, encoder_out, hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(hidden).unsqueeze(1)
        att = torch.tanh(att1 + att2)
        e = self.full_att(att).squeeze(-1)
        alpha = F.softmax(e, dim=1)
        context = (encoder_out * alpha.unsqueeze(-1)).sum(dim=1)
        return context, alpha

    def forward(self, encoder_out, captions, caplens):
        batch_size, max_len = captions.size()
        device = captions.device
        hidden = torch.zeros(1, batch_size, self.decoder_dim, device=device)

        logits_list = []
        for t in range(max_len - 1):
            prev_word = captions[:, t]
            emb = self.embedding(prev_word)
            context, _ = self.attention(encoder_out, hidden[-1])
            gru_input = torch.cat([emb, context], dim=-1).unsqueeze(1)
            out, hidden = self.gru(gru_input, hidden)
            out = self.dropout(out.squeeze(1))
            logits = self.fc(out)
            logits_list.append(logits.unsqueeze(1))

        logits = torch.cat(logits_list, dim=1)
        return logits, caplens - 1


def greedy_generate(
    encoder: OriginalEncoder,
    decoder: OriginalDecoder,
    image: torch.Tensor,
    start_idx: int,
    end_idx: int,
    max_len: int = 64,
):
    device = image.device
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        enc_out = encoder(image.unsqueeze(0))
        hidden = torch.zeros(1, 1, decoder.decoder_dim, device=device)
        prev_word = torch.tensor([start_idx], dtype=torch.long, device=device)
        seq = []
        for _ in range(max_len):
            emb = decoder.embedding(prev_word)
            context, _ = decoder.attention(enc_out, hidden[-1])
            gru_input = torch.cat([emb, context], dim=-1).unsqueeze(1)
            out, hidden = decoder.gru(gru_input, hidden)
            out = decoder.dropout(out.squeeze(1))
            logits = decoder.fc(out)
            next_word = logits.argmax(dim=-1)
            if next_word.item() == end_idx:
                break
            seq.append(next_word.item())
            prev_word = next_word
        return torch.tensor(seq, device=device)


